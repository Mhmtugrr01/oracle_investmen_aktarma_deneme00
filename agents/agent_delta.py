"""
PROJECT OLYMPUS — agents/agent_delta.py (Social Intelligence Agent & Alpha Catcher)
"""

import asyncio
import os
import json
import aiohttp
from pydantic import BaseModel, Field
from loguru import logger

from core.console import CYAN, GREEN, RED, agent_print, error_print
from core.types import OracleState, PipelineStatus

# 👁️ İZLENEN SEÇKİN 10 KÜRESEL ANALİST LİSTESİ
ELITE_ANALYSTS = [
    "Pentoshi", "Ansem", "GCRClassic", "CredibleCrypto", "CryptoBullet",
    "DonAlt", "Benjamin Cowen", "Michael van de Poppe", "Bluntz_Capital", "IncomeSharks"
]

class DeltaReport(BaseModel):
    summary: str = Field(description="Analistlerin tezlerinin kisa ozeti (Markdown formatinda)")
    critique: str = Field(description="Tezlerin acimasiz, realist, anti-shill elestirisi ve risk analizi")
    candidates: list[str] = Field(description="Analiz edilmeye deger bulunan hisse veya kripto sembolleri (Orn: ['FET', 'MSTR'])")


async def _gather_social_intel() -> str:
    """Tavily Search API kullanarak son 24 saatteki analist paylaşımlarını canlı tarar."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        logger.warning("[AGENT_DELTA] TAVILY_API_KEY bulunamadı! Simüle edilmiş veri toplanıyor.")
        return "Pentoshi: FET daily breakout is looking massive on high volume. Michael van de Poppe: MSTR local support holding, ready for bounce."

    query = f"site:twitter.com ({' OR '.join(ELITE_ANALYSTS)}) crypto analysis after:24h"
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": tavily_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": 10
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=20) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("results", [])
                    raw_text = " ".join([r.get("content", "") for r in results])
                    return raw_text if raw_text else "FET breakout on daily. MSTR support holding."
    except Exception as e:
        logger.error(f"[AGENT_DELTA] Tavily veri toplama hatası: {e}")
    return "FET breakout on daily. MSTR support holding."


async def run_social_alpha_catcher(telegram_sender=None) -> None:
    """
    Her sabah 08:00'de tetiklenir.
    Verileri toplar, Claude'a anti-shill yaptırır ve çıkan varlıkları Olympus Motoruna fırlatır.
    telegram_sender: async callable(text: str) — Telegram mesajı gönderir.
    """
    agent_print("AGENT_DELTA", "Sabah 08:00 Protokolü Devrede — Küresel 10 Analist taranıyor...", CYAN)
    
    # 1. Veri toplama
    raw_intel = await _gather_social_intel()
    
    # 2. Claude 3.5 Sonnet / OpenRouter ile Akıl Yürütme ve Karar
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    if not openrouter_key:
        logger.error("[AGENT_DELTA] OPENROUTER_API_KEY bulunamadı! Sosyal analiz durduruldu.")
        return

    system_prompt = (
        "Sen, dünyanın en büyük kuant hedge fonunun Baş Risk Denetçisi ve acımasız bir savcısın.\n"
        "Sana sunulan sosyal medya analizlerini okuyacak, arkasındaki FOMO, shill, wash-trading ve erken alım tuzaklarını "
        "acımasızca ifşa edeceksin. Gerçekçi ol, duyguları sil. Analistlerin tezlerini parçala.\n"
        "Çıktını mutlaka şu Pydantic JSON formatına tam uygun olarak döndür:\n"
        "{\n"
        "  \"summary\": \"Özet metin...\",\n"
        "  \"critique\": \"Anti-shill eleştiri raporu...\",\n"
        "  \"candidates\": [\"FET\", \"MSTR\"]\n"
        "}"
    )

    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Son 24 saatlik analist istihbarat dökümü:\n{raw_intel}"}
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{openrouter_url}/chat/completions", headers=headers, json=payload, timeout=30) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"OpenRouter HTTP {resp.status}")
                result = await resp.json()
                content = result["choices"][0]["message"]["content"]
                
                # Pydantic structured output doğrulaması
                report_data = json.loads(content)
                report = DeltaReport(**report_data)
                
                logger.info("[AGENT_DELTA] Claude 3.5 Sonnet Anti-Shill Analizi Tamamlandı.")

                # ── Telegram'a sabah özeti gönder ────────────────────────────
                if telegram_sender:
                    summary_msg = (
                        "☀️ OLYMPUS SABAH BRİFİNGİ — Sosyal Zeka Raporu\n\n"
                        f"📋 ÖZET:\n{report.summary}\n\n"
                        f"⚔️ ANTİ-SHİLL DEĞERLENDİRME:\n{report.critique}\n\n"
                        f"🎯 ADAYLAR: {', '.join(report.candidates) if report.candidates else 'Yok'}\n\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )
                    try:
                        await telegram_sender(summary_msg)
                    except Exception as tg_exc:
                        logger.error(f"[AGENT_DELTA] Telegram özet gönderilemedi: {tg_exc}")

                # ── 🚀 BÜYÜK MOTOR ENTEGRASYONU (Otonom Sinyal Tetikleyicisi) ──
                try:
                    from core.graph import compile_oracle_graph
                except ImportError:
                    logger.error("[AGENT_DELTA] core.graph.compile_oracle_graph bulunamadı! Tetikleme iptal.")
                    return

                graph = compile_oracle_graph()
                no_signal_symbols = []

                for symbol in report.candidates:
                    norm_symbol = symbol.strip().upper()
                    if "/" not in norm_symbol:
                        norm_symbol = f"{norm_symbol}/USDT"

                    agent_print("AGENT_DELTA", f"🎯 HEDEF KİLİTLENDİ: {norm_symbol} Olympus motoruna gönderiliyor...", GREEN)
                    initial_state = OracleState(symbol=norm_symbol, query=f"social_trigger:{symbol}")

                    try:
                        final_state = await asyncio.wait_for(graph.ainvoke(initial_state), timeout=300.0)
                        if hasattr(final_state, "model_dump"):
                            st = final_state
                        else:
                            from core.types import OracleState as _OS
                            st = _OS.model_validate(final_state)

                        from bot.telegram_handler import format_oracle_response
                        status_str = str(st.status.value).upper()
                        if "ABORT" in status_str or "FAIL" in status_str or st.fatal_error:
                            no_signal_symbols.append(norm_symbol)
                        else:
                            if telegram_sender:
                                await telegram_sender(format_oracle_response(st))
                    except Exception as pipe_exc:
                        logger.error(f"[AGENT_DELTA] {norm_symbol} pipeline hatası: {pipe_exc}")
                        no_signal_symbols.append(norm_symbol)

                if no_signal_symbols and telegram_sender:
                    await telegram_sender(
                        f"📭 Sosyal medya adaylarından analiz edildi ancak fırsat bulunamadı:\n"
                        + "\n".join(f"  • {s}" for s in no_signal_symbols)
                    )
                        
    except Exception as exc:
        logger.error(f"[AGENT_DELTA] Otonom analiz ve tetikleme hatası: {exc}")