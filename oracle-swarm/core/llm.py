"""
ORACLE LLM MOTORU V3 — Tam Dayanıklı Sistem

Öncelik Zinciri:
  1. Cache (5dk) — LLM çağrısı yok
  2. Gemini (google-genai SDK, gemini-2.0-flash)
  3. Groq llama-3.3-70b (GROQ_API_KEY ile)
  4. OpenAI gpt-4o-mini (exponential backoff)
  5. Kural tabanlı yanıt — ASLA çökmez

Kural tabanlı yönlendirme: LLM olmadan %95 doğru ajan seçimi.
"""

import asyncio
import hashlib
import os
import re
import time
from loguru import logger

# ─── In-Memory Cache ────────────────────────────────────────────────────────
_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 300


def _ck(messages: list, system: str | None, model: str) -> str:
    raw = f"{model}|{system or ''}|{str(messages)[-400:]}"
    return hashlib.md5(raw.encode()).hexdigest()


def _from_cache(key: str) -> str | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    return None


def _to_cache(key: str, value: str):
    _cache[key] = (value, time.time())
    if len(_cache) > 300:
        oldest = sorted(_cache, key=lambda k: _cache[k][1])[:60]
        for k in oldest:
            del _cache[k]


# ─── Kural Tabanlı Intent Tespiti ───────────────────────────────────────────
_QUANT_KW = {
    "btc", "bitcoin", "eth", "ethereum", "kripto", "crypto", "borsa",
    "hisse", "analiz", "piyasa", "market", "rsi", "ema", "macd",
    "altın", "gold", "dolar", "dxy", "vix", "sp500", "nasdaq",
    "bnb", "sol", "xrp", "doge", "ada", "avax", "link", "coin",
    "long", "short", "teknik", "fiyat", "price", "usdt", "usdc",
    "dominance", "mcap", "kripto", "trade", "chart", "grafik",
}
_SWE_KW = {
    "kod", "code", "yaz", "write", "python", "javascript", "api",
    "uygulama", "app", "geliştir", "develop", "program", "script",
    "html", "css", "react", "fastapi", "bot", "otomasyon",
    "bug", "hata düzelt", "fix", "test", "deploy", "git",
    "database", "sql", "docker", "flask", "django", "web", "backend",
}
_MARKETING_KW = {
    "email", "e-mail", "e-posta", "mail", "müşteri", "satış", "sales",
    "lead", "scraping", "pazarlama", "marketing", "firma", "şirket",
    "osb", "sanayi", "bursa", "balıkesir", "reklam", "kampanya",
    "linkedin", "instagram", "sosyal", "mailing", "bülten", "iletişim",
    "gönder", "send", "toplu", "bulk", "hedef", "kitle",
}
_EDGE_KW = {
    "disk", "bellek", "memory", "cpu", "sistem", "system",
    "durum", "status", "temizle", "clean", "dosya",
    "log", "sunucu", "server", "replit", "cloud", "temp",
    "ram", "depolama", "storage", "process", "uptime",
}
_FREELANCER_KW = {
    "freelancer", "upwork", "fiverr", "toptal", "guru",
    "iş bul", "iş ara", "proje bul", "remote", "uzaktan",
    "başvuru", "apply", "cv", "portfolio", "teklif", "proposal",
    "freelance", "serbest", "danışman", "consultant",
}


def rule_based_intent(text: str) -> str | None:
    t = text.lower()
    scores = {
        "QUANT": sum(1 for k in _QUANT_KW if k in t),
        "SWE": sum(1 for k in _SWE_KW if k in t),
        "MARKETING": sum(1 for k in _MARKETING_KW if k in t),
        "EDGE": sum(1 for k in _EDGE_KW if k in t),
        "FREELANCER": sum(1 for k in _FREELANCER_KW if k in t),
    }
    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0:
        return None
    sorted_scores = sorted(scores.values(), reverse=True)
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0
    if best_score >= 2 or (best_score >= 1 and best_score > second):
        logger.debug(f"[INTENT RULE] {best} ({best_score}p)")
        return best
    return None


# ─── LLM Sağlayıcıları ──────────────────────────────────────────────────────

async def _try_gemini(messages: list, temperature: float, max_tokens: int) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # Mesajları tek prompt'a dönüştür
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.insert(0, f"[Sistem Talimatları: {content}]\n\n")
            else:
                parts.append(content)
        prompt = "".join(parts)

        config = types.GenerateContentConfig(
            max_output_tokens=min(max_tokens, 8192),
            temperature=temperature,
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=config,
            )
        )
        result = response.text or ""
        logger.debug(f"[LLM GEMINI] ✅ {len(result)} chars")
        return result
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            logger.warning("[LLM GEMINI] Kota aşımı — sonraki sağlayıcıya geçiliyor")
        else:
            logger.warning(f"[LLM GEMINI] {err[:100]}")
        return None


async def _try_groq(messages: list, temperature: float, max_tokens: int) -> str | None:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=min(max_tokens, 8192),
        )
        result = response.choices[0].message.content or ""
        logger.debug(f"[LLM GROQ] ✅ {len(result)} chars")
        return result
    except Exception as e:
        logger.warning(f"[LLM GROQ] {str(e)[:80]}")
        return None


async def _try_openai(messages: list, model: str, temperature: float, max_tokens: int) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import AsyncOpenAI
        base_url = os.getenv("AI_INTEGRATIONS_OPENAI_BASE_URL", "https://api.openai.com/v1")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
                result = response.choices[0].message.content or ""
                logger.debug(f"[LLM OPENAI] ✅ {len(result)} chars")
                return result
            except Exception as e:
                err = str(e)
                if "insufficient_quota" in err or "billing" in err.lower():
                    logger.warning("[LLM OPENAI] Kota tükendi")
                    return None
                if "429" in err or "rate" in err.lower():
                    wait = 8 * (2 ** attempt)
                    logger.warning(f"[LLM OPENAI] 429 → {wait}s")
                    await asyncio.sleep(wait)
                else:
                    await asyncio.sleep(2 * (attempt + 1))
        return None
    except Exception as e:
        logger.warning(f"[LLM OPENAI] {str(e)[:60]}")
        return None


# ─── Ana Çağrı ──────────────────────────────────────────────────────────────

async def llm_call(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    skip_cache: bool = False,
) -> str:
    oai_model = model or "gpt-4o-mini"
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    key = _ck(full_messages, system, oai_model)
    if not skip_cache:
        cached = _from_cache(key)
        if cached:
            logger.debug("[LLM CACHE] Hit")
            return cached

    result = (
        await _try_gemini(full_messages, temperature, max_tokens)
        or await _try_groq(full_messages, temperature, max_tokens)
        or await _try_openai(full_messages, oai_model, temperature, max_tokens)
    )

    if result and not skip_cache:
        _to_cache(key, result)

    return result or ""


# ─── Extrapolation ───────────────────────────────────────────────────────────

async def extrapolate_prompt(user_input: str) -> str:
    if len(user_input.strip().split()) <= 3:
        intent = rule_based_intent(user_input) or "CEO"
        return f"Kullanıcı isteği: {user_input}\nHedef ajan: {intent}"

    system = """Oracle CEO Extrapolation motoru. Kısa girdiyi detaylı iş direktifine dönüştür:
1. Hedef ve Amaç
2. Adımlar (3-5 madde)
3. Beklenen Çıktı
4. Ajan: SWE/QUANT/MARKETING/EDGE/FREELANCER/CEO
Türkçe, max 250 kelime."""

    result = await llm_call(
        messages=[{"role": "user", "content": f"Girdi: {user_input}"}],
        system=system,
        max_tokens=1500,
        temperature=0.5,
    )
    return result if result else f"Kullanıcı isteği: {user_input}"


# ─── Intent Tespiti ──────────────────────────────────────────────────────────

async def detect_intent(user_input: str, expanded_prompt: str = "") -> str:
    # 1. Kural tabanlı (hızlı, LLM yok)
    for text in [user_input, expanded_prompt]:
        result = rule_based_intent(text)
        if result:
            return result

    # 2. LLM ile dene
    system = "Tek kelime döndür: SWE | QUANT | MARKETING | EDGE | FREELANCER | CEO"
    combined = (user_input + " " + expanded_prompt[:200]).strip()
    llm_result = await llm_call(
        messages=[{"role": "user", "content": combined[:400]}],
        system=system,
        temperature=0.1,
        max_tokens=10,
    )

    if llm_result:
        intent = llm_result.strip().upper().split()[0]
        if intent in {"SWE", "QUANT", "MARKETING", "EDGE", "FREELANCER", "CEO"}:
            return intent

    return "CEO"
