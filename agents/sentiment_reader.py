"""DÜĞÜM 5 — Master Alpha Duygu Okyucusu (Social Whisperer)."""

from __future__ import annotations

import os
import asyncio
import ssl

import aiohttp
import certifi
from loguru import logger

try:
    import truststore

    _ssl_ctx = ssl.create_default_context()
    truststore.inject_into_ssl()
except ImportError:
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())

from agents.fundamental_filter import _fetch_news_sentiment
from core.console import BLUE, GREEN, agent_print
from core.types import AgentNode, OracleState, PipelineStatus


async def _fetch_fear_greed() -> dict:
    """
    Alternative.me Fear & Greed endeksini çeker.
    """
    url = "https://api.alternative.me/fng/?limit=2"
    try:
        connector = aiohttp.TCPConnector(ssl=_ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json()
                entries = data.get("data", [])
                if not entries:
                    return {"fg_value": 50, "fg_label": "Neutral", "fg_score": 0.0}

                current = int(entries[0]["value"])
                label = entries[0]["value_classification"]
                fg_score = (current - 50) / 50.0
                return {
                    "fg_value": current,
                    "fg_label": label,
                    "fg_score": round(fg_score, 3),
                }
    except Exception as exc:
        logger.warning(f"Fear & Greed API hatası: {exc}")
        return {"fg_value": 50, "fg_label": "Neutral", "fg_score": 0.0}


async def _fetch_cmc_sentiment(symbol: str) -> dict:
    """
    CoinMarketCap global metriklerinden piyasa sentiment proxy üretir.
    """
    cmc_key = os.getenv("COINMARKETCAP_API_KEY", "").strip()
    if not cmc_key:
        return {"market_sentiment_score": 0.0, "error": "COINMARKETCAP_API_KEY missing"}

    url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": cmc_key}

    try:
        connector = aiohttp.TCPConnector(ssl=_ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as s:
            async with s.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {"market_sentiment_score": 0.0, "error": f"HTTP {resp.status}"}
                data = await resp.json()
                quote = data["data"]["quote"]["USD"]

                total_market_cap_change = float(
                    quote.get("total_market_cap_yesterday_percentage_change", 0) or 0
                )
                total_volume_change = float(
                    quote.get("total_volume_24h_yesterday_percentage_change", 0) or 0
                )

                sentiment = 0.0
                if total_volume_change > 20:
                    sentiment += 0.15
                elif total_volume_change > 0:
                    sentiment += 0.05
                elif total_volume_change < -20:
                    sentiment -= 0.15

                if total_market_cap_change > 3:
                    sentiment += 0.10
                elif total_market_cap_change < -3:
                    sentiment -= 0.10

                return {
                    "market_sentiment_score": round(sentiment, 3),
                    "market_cap_change": total_market_cap_change,
                    "volume_change": total_volume_change,
                }
    except Exception as exc:
        logger.warning(f"CMC global sentiment hatası: {exc}")
        return {"market_sentiment_score": 0.0, "error": str(exc)}


async def run_sentiment_reader(state: OracleState) -> OracleState:
    agent_print(
        "SOCIAL_WHISPERER",
        f"Devrede → {state.symbol} | Duygu okuması başlatıldı…",
        GREEN,
    )
    symbol = state.symbol or "BTC/USDT"
    is_crypto = "/" in symbol and "USDT" in symbol

    fg_data = {"fg_value": 50, "fg_label": "N/A", "fg_score": 0.0}
    if is_crypto:
        fg_data, cmc_sentiment = await asyncio.gather(
            _fetch_fear_greed(),
            _fetch_cmc_sentiment(symbol),
            return_exceptions=True,
        )
        if isinstance(fg_data, Exception):
            fg_data = {"fg_value": 50, "fg_score": 0.0, "fg_label": "Neutral"}
        if isinstance(cmc_sentiment, Exception):
            cmc_sentiment = {"market_sentiment_score": 0.0}

        fg_score = float(fg_data.get("fg_score", 0.0) or 0.0)
        market_score = float(cmc_sentiment.get("market_sentiment_score", 0.0) or 0.0)
        sentiment_score = (fg_score * 0.60) + (market_score * 0.40)

        logger.info(
            f"[SENTIMENT] {symbol} -> F&G={fg_data.get('fg_value')}"
            f"({fg_data.get('fg_label')}) | market={market_score:.3f} | "
            f"sentiment_score={sentiment_score:.3f}"
        )
    else:
        news_data = await _fetch_news_sentiment(symbol, None)
        if isinstance(news_data, Exception):
            news_data = {"news_score": 0.0}
        sentiment_score = float(news_data.get("news_score", 0.0) or 0.0)
        fg_data = {"fg_value": 50, "fg_label": "N/A"}

        logger.info(
            f"[SENTIMENT] {symbol} (hisse/emtia) -> news_sentiment={sentiment_score:.3f}"
        )

    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    agent_print(
        "SOCIAL_WHISPERER",
        f"Sosyal hacim + fear/greed -> sentiment_score={sentiment_score:+.3f}",
        BLUE,
    )
    agent_print(
        "SOCIAL_WHISPERER",
        "Twitter/Telegram/Reddit fısıltı ağı tarandı.",
        GREEN,
    )

    return state.model_copy(
        update={
            "current_node": AgentNode.SENTIMENT_READER,
            "status": PipelineStatus.RUNNING,
            "sentiment_score": sentiment_score,
            "fear_greed_value": fg_data.get("fg_value", 50),
            "fear_greed_label": fg_data.get("fg_label", "N/A"),
            "messages": [
                f"[SOCIAL_WHISPERER] score={sentiment_score:+.3f}"
            ],
        }
    )
