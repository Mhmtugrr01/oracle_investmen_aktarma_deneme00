"""
ORACLE LLM MOTORU — Çok Katmanlı LLM İstemcisi
Öncelik sırası:
  1. Groq (Llama-3.3-70b) — Ücretsiz, hızlı, GROQ_API_KEY gerekli
  2. OpenAI (gpt-4o-mini)  — Ücretli yedek
Cache + Retry + Rate Limit koruması dahil.
"""
import asyncio
import hashlib
import time
from loguru import logger
from core.config import settings

# ─── Basit In-Memory Cache ──────────────────────────────────────────────────
_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 300  # 5 dakika


def _cache_key(messages: list, system: str | None, model: str) -> str:
    raw = f"{model}|{system}|{messages}"
    return hashlib.md5(raw.encode()).hexdigest()


def _from_cache(key: str) -> str | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    return None


def _to_cache(key: str, value: str):
    _cache[key] = (value, time.time())
    if len(_cache) > 200:
        oldest = sorted(_cache, key=lambda k: _cache[k][1])[:50]
        for k in oldest:
            del _cache[k]


# ─── Groq İstemcisi (Ücretsiz) ─────────────────────────────────────────────
def _get_groq_client():
    try:
        from groq import AsyncGroq
        key = settings.groq_api_key
        if not key:
            return None
        return AsyncGroq(api_key=key)
    except ImportError:
        return None


# ─── OpenAI İstemcisi ───────────────────────────────────────────────────────
def _get_openai_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        api_key=settings.openai_api_key or "dummy",
        base_url=settings.openai_base_url,
    )


# ─── Ana LLM Çağrı Fonksiyonu ───────────────────────────────────────────────
async def llm_call(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    skip_cache: bool = False,
) -> str:
    """
    LLM çağrısı yapar.
    1. Cache kontrol et
    2. Groq dene (ücretsiz)
    3. OpenAI yedek (retry ile)
    """
    final_model = model or settings.llm_model

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    cache_key = _cache_key(messages, system, final_model)
    if not skip_cache:
        cached = _from_cache(cache_key)
        if cached:
            logger.debug(f"[LLM CACHE] Hit ({len(cached)} chars)")
            return cached

    result = await _try_groq(full_messages, temperature, max_tokens)

    if result is None:
        result = await _try_openai_with_retry(full_messages, final_model, temperature, max_tokens)

    if not skip_cache and result:
        _to_cache(cache_key, result)

    return result or ""


async def _try_groq(messages: list, temperature: float, max_tokens: int) -> str | None:
    """Groq ücretsiz API denemesi."""
    client = _get_groq_client()
    if client is None:
        return None

    groq_model = settings.groq_model

    try:
        response = await client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=min(max_tokens, 8192),
        )
        result = response.choices[0].message.content or ""
        logger.debug(f"[LLM GROQ] ✅ {len(result)} chars via {groq_model}")
        return result
    except Exception as e:
        logger.warning(f"[LLM GROQ] Failed: {e}")
        return None


async def _try_openai_with_retry(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
) -> str:
    """OpenAI çağrısı — exponential backoff retry ile."""
    client = _get_openai_client()
    last_error = None

    for attempt in range(retries):
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
            last_error = e
            err_str = str(e)

            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                wait = 10 * (2 ** attempt)
                logger.warning(f"[LLM OPENAI] 429 rate limit — {wait}s bekleniyor (deneme {attempt+1}/{retries})")
                await asyncio.sleep(wait)
            elif "insufficient_quota" in err_str:
                logger.error("[LLM OPENAI] ❌ Kota tükendi — GROQ_API_KEY ekleyin (ücretsiz: console.groq.com)")
                return _fallback_response(messages)
            else:
                wait = 2 * (2 ** attempt)
                logger.warning(f"[LLM OPENAI] Hata: {e} — {wait}s bekleniyor")
                await asyncio.sleep(wait)

    logger.error(f"[LLM] Tüm denemeler başarısız: {last_error}")
    return _fallback_response(messages)


def _fallback_response(messages: list) -> str:
    """LLM tamamen çalışmadığında kural tabanlı yedek yanıt."""
    last_msg = messages[-1].get("content", "") if messages else ""
    last_lower = last_msg.lower()

    if any(w in last_lower for w in ["btc", "bitcoin", "eth", "kripto", "borsa", "analiz"]):
        return (
            "⚠️ *LLM Servisi Geçici Olarak Kullanılamıyor*\n\n"
            "Teknik analiz verileri yfinance üzerinden hâlâ alınabilir.\n"
            "/quant komutu ile sembol belirterek deneyin.\n\n"
            "Kalıcı çözüm için: console.groq.com adresinden ücretsiz Groq API key alın, "
            "GROQ_API_KEY olarak Replit Secrets'a ekleyin."
        )
    elif any(w in last_lower for w in ["freelancer", "iş", "upwork"]):
        return (
            "⚠️ *LLM Servisi Geçici Olarak Kullanılamıyor*\n\n"
            "Ücretsiz Groq LLM için: console.groq.com → API Keys → Key oluştur\n"
            "Replit'te Secrets → GROQ_API_KEY olarak ekle → Bot otomatik kullanacak."
        )

    return (
        "⚠️ *Oracle Geçici Hata*\n\n"
        "OpenAI kotası tükendi. Ücretsiz yedek için:\n"
        "1. console.groq.com → Ücretsiz kayıt\n"
        "2. API Keys → Yeni key oluştur\n"
        "3. Replit Secrets → GROQ_API_KEY = key_değeri\n"
        "4. Bot otomatik Groq'u kullanacak ✅"
    )


# ─── Extrapolation & Intent (aynı llm_call üzerinden) ─────────────────────
async def extrapolate_prompt(user_input: str) -> str:
    system = """Sen Oracle CEO Ajan sisteminin Extrapolation motorusun.
Kullanıcıdan 1-10 kelimelik kısa bir fikir alıyorsun.
Bu kısa girdiyi KAPSAMLI, DETAYLI ve UYGULANABILIR bir iş direktifine dönüştür:
1. Hedef ve Amaç
2. Yapılacak Adımlar (en az 5 adım)
3. Beklenen Çıktı
4. Başarı Kriterleri
5. Hangi ajana yönlendirileceği (SWE/QUANT/MARKETING/EDGE/FREELANCER)
Türkçe yanıt ver. Profesyonel ve net ol."""

    return await llm_call(
        messages=[{"role": "user", "content": f"Kısa girdi: {user_input}"}],
        system=system,
        max_tokens=4096,
    )


async def detect_intent(expanded_prompt: str) -> str:
    system = """Aşağıdaki iş direktifini analiz et. Sadece şu değerlerden birini döndür:
SWE | QUANT | MARKETING | EDGE | FREELANCER | CEO

SWE: Yazılım, kod, uygulama geliştirme
QUANT: Borsa, kripto, BTC, ETH, piyasa analizi
MARKETING: Scraping, email, satış, lead generation
EDGE: Sistem durumu, disk, bellek
FREELANCER: İş arama, Upwork, LinkedIn iş, başvuru
CEO: Genel sorular, strateji"""

    result = await llm_call(
        messages=[{"role": "user", "content": expanded_prompt[:500]}],
        system=system,
        temperature=0.1,
        max_tokens=10,
        skip_cache=False,
    )
    intent = result.strip().upper().split()[0] if result.strip() else "CEO"
    valid = {"SWE", "QUANT", "MARKETING", "EDGE", "FREELANCER", "CEO"}
    return intent if intent in valid else "CEO"
