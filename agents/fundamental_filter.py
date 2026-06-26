"""DÜĞÜM 4 — Temel / On-Chain Süzgeci (Fundamental Miner)."""

from __future__ import annotations

import os
import asyncio
import ssl
import time as _time
from datetime import datetime, timedelta, timezone

import aiohttp
import certifi
from loguru import logger

try:
    import truststore

    _ssl_ctx = ssl.create_default_context()
    truststore.inject_into_ssl()
except ImportError:
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())

from core.console import BLUE, CYAN, agent_print
from core.types import AgentNode, OracleState, PipelineStatus

_NEWS_CACHE: dict[str, tuple[float, dict]] = {}
_NEWS_CACHE_TTL = 6 * 3600  # 6 saat


def _build_news_query(symbol: str) -> str:
    search_terms = {
        "BTC/USDT": "Bitcoin BTC",
        "ETH/USDT": "Ethereum ETH",
        "INJ/USDT": "Injective INJ",
        "RNDR/USDT": "Render RNDR",
        "FET/USDT": "Fetch.ai FET ASI",
        "NVDA": "NVIDIA NVDA",
        "TSLA": "Tesla TSLA",
        "MSTR": "MicroStrategy MSTR",
        "COIN": "Coinbase COIN",
        "INTC": "Intel INTC",
        "AMD": "AMD semiconductor",
        "AMAT": "Applied Materials AMAT",
        "SPY": "S&P 500 stock market",
        "GC=F": "gold price XAU",
        "SI=F": "silver price XAG",
        "CL=F": "crude oil WTI",
        "ASTOR.IS": "Astor Energy Turkey OR Astor Enerji",
        "EUPWR.IS": "Europower Energy Turkey OR EUPWR",
        "THYAO.IS": "Turkish Airlines THY OR Turk Hava Yollari",
        "GARAN.IS": "Garanti Bank Turkey OR Garanti BBVA",
        "EREGL.IS": "Eregli Steel Turkey OR Erdemir OR EREGL",
    }
    if symbol in search_terms:
        return search_terms[symbol]
    if "/" in symbol:
        return symbol.split("/", 1)[0]
    return symbol.split(".", 1)[0]


async def _fetch_news_sentiment(symbol: str, session=None) -> dict:
    """
    NewsAPI üzerinden son 48 saatin haberlerini çek.
    Döndür: article sayısı ve normalize haber skoru.
    """
    api_key = os.getenv("NEWS_API_KEY", "").strip()
    if not api_key:
        logger.warning("NEWS_API_KEY tanımlı değil, haber sentiment nötr döndürüldü.")
        return {"news_score": 0.0, "article_count": 0, "error": "NEWS_API_KEY missing"}

    query = _build_news_query(symbol)
    from_date = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 20,
        "apiKey": api_key,
    }

    try:
        connector = aiohttp.TCPConnector(ssl=_ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as s:
            async with s.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {"news_score": 0.0, "article_count": 0, "error": f"HTTP {resp.status}"}
                data = await resp.json()
                articles = data.get("articles", [])
    except Exception as exc:
        logger.warning(f"NewsAPI hatası {symbol}: {exc}")
        return {"news_score": 0.0, "article_count": 0, "error": str(exc)}

    if not articles:
        return {"news_score": 0.0, "article_count": 0}

    positive_kw = [
        "surge", "rally", "bullish", "breakout", "record", "growth",
        "adoption", "partnership", "upgrade", "beat", "strong", "rise",
        "gain", "boost", "yukselis", "artis", "pozitif",
    ]
    negative_kw = [
        "crash", "dump", "bearish", "breakdown", "hack", "ban",
        "regulation", "lawsuit", "miss", "weak", "fall", "decline",
        "drop", "loss", "dusus", "kayip", "negatif", "iflas",
    ]

    pos, neg, neu = 0, 0, 0
    for article in articles:
        text = ((article.get("title") or "") + " " + (article.get("description") or "")).lower()
        pos_hits = sum(1 for kw in positive_kw if kw in text)
        neg_hits = sum(1 for kw in negative_kw if kw in text)

        if pos_hits > neg_hits:
            pos += 1
        elif neg_hits > pos_hits:
            neg += 1
        else:
            neu += 1

    total = len(articles)
    news_score = (pos - neg) / total if total > 0 else 0.0
    return {
        "news_score": round(news_score, 3),
        "article_count": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
    }


async def _fetch_news_sentiment_cached(symbol: str, session=None) -> dict:
    """6 saatlik önbellekle NewsAPI çağrısı."""
    now = _time.time()
    if symbol in _NEWS_CACHE:
        ts, cached = _NEWS_CACHE[symbol]
        if now - ts < _NEWS_CACHE_TTL:
            return cached
    result = await _fetch_news_sentiment(symbol, session)
    _NEWS_CACHE[symbol] = (now, result)
    return result


async def _fetch_bist_news(symbol: str) -> dict:
    return {"news_score": 0.0, "article_count": 0}


async def _fetch_crypto_fundamentals(symbol: str) -> dict:
    """
    CoinMarketCap API üzerinden kripto temel veriler.
    """
    cmc_key = os.getenv("COINMARKETCAP_API_KEY", "").strip()
    if not cmc_key:
        return {"fundamental_boost": 0.0, "error": "COINMARKETCAP_API_KEY missing"}

    cmc_symbols = {
        "BTC/USDT": "BTC",
        "ETH/USDT": "ETH",
        "INJ/USDT": "INJ",
        "RNDR/USDT": "RENDER",
        "FET/USDT": "FET",
    }
    ticker = cmc_symbols.get(symbol)
    if not ticker:
        return {"fundamental_boost": 0.0}

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": cmc_key}
    params = {"symbol": ticker, "convert": "USD"}

    try:
        connector = aiohttp.TCPConnector(ssl=_ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as s:
            async with s.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return {"fundamental_boost": 0.0, "error": f"HTTP {resp.status}"}
                data = await resp.json()

        quote = data["data"][ticker]["quote"]["USD"]
        volume_change = float(quote.get("volume_change_24h", 0) or 0)
        pct_7d = float(quote.get("percent_change_7d", 0) or 0)
        pct_30d = float(quote.get("percent_change_30d", 0) or 0)

        boost = 0.0
        if volume_change > 50:
            boost += 0.10
        elif volume_change > 20:
            boost += 0.05

        if pct_7d > 10:
            boost += 0.08
        elif pct_7d > 0:
            boost += 0.03
        elif pct_7d < -15:
            boost -= 0.08

        if pct_30d > 20:
            boost += 0.05

        return {
            "fundamental_boost": round(boost, 3),
            "volume_change_24h": volume_change,
            "pct_7d": pct_7d,
            "pct_30d": pct_30d,
        }
    except Exception as exc:
        logger.warning(f"CoinMarketCap hatası {symbol}: {exc}")
        return {"fundamental_boost": 0.0, "error": str(exc)}


async def run_fundamental_filter(state: OracleState) -> OracleState:
    agent_print(
        "FUNDAMENTAL_MINER",
        f"Devrede → {state.symbol} | Temel veri madenciliği…",
        CYAN,
    )
    symbol = state.symbol or "BTC/USDT"

    news_data, cmc_data = await asyncio.gather(
        _fetch_news_sentiment_cached(symbol, None),
        _fetch_crypto_fundamentals(symbol),
        return_exceptions=True,
    )

    if isinstance(news_data, Exception):
        logger.warning(f"Haber verisi alınamadı: {news_data}")
        news_data = {"news_score": 0.0, "article_count": 0}
    if isinstance(cmc_data, Exception):
        logger.warning(f"CMC verisi alınamadı: {cmc_data}")
        cmc_data = {"fundamental_boost": 0.0}

    if symbol.endswith(".IS"):
        bist_data = await _fetch_bist_news(symbol)
        news_data = {
            "news_score": float(news_data.get("news_score", 0.0) or 0.0)
            + float(bist_data.get("news_score", 0.0) or 0.0),
            "article_count": int(news_data.get("article_count", 0) or 0)
            + int(bist_data.get("article_count", 0) or 0),
        }

    news_score = float(news_data.get("news_score", 0.0) or 0.0)
    cmc_boost = float(cmc_data.get("fundamental_boost", 0.0) or 0.0)
    article_count = int(news_data.get("article_count", 0) or 0)

    fundamental_score = (news_score * 0.70) + (cmc_boost * 0.30)
    fundamental_score = max(-1.0, min(1.0, fundamental_score))

    data_confidence = min(article_count / 10.0, 1.0)

    logger.info(
        f"[FUNDAMENTAL] {symbol} -> haber={article_count} adet, "
        f"news_score={news_score:.3f}, cmc_boost={cmc_boost:.3f}, "
        f"fundamental_score={fundamental_score:.3f}, guven={data_confidence:.2f}"
    )

    agent_print(
        "FUNDAMENTAL_MINER",
        f"Tokenomics + haber akışı -> fundamental_score={fundamental_score:+.3f}",
        BLUE,
    )
    agent_print(
        "FUNDAMENTAL_MINER",
        "On-chain metrik süzgeci tamamlandı.",
        CYAN,
    )

    return state.model_copy(
        update={
            "current_node": AgentNode.FUNDAMENTAL_FILTER,
            "status": PipelineStatus.RUNNING,
            "fundamental_score": fundamental_score,
            "fundamental_data_confidence": data_confidence,
            "news_article_count": article_count,
            "news_sentiment": news_score,
            "messages": [
                (
                    f"[FUNDAMENTAL_MINER] score={fundamental_score:+.3f} "
                    f"news={news_score:+.3f} cmc={cmc_boost:+.3f} articles={article_count}"
                )
            ],
        }
    )
