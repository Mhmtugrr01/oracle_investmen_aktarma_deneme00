"""DÜĞÜM 3 — Sembiyotik Balina Avcısı (Symbiotic Hunter)."""

from __future__ import annotations

import os
import ssl
import asyncio
from typing import Any

import aiohttp
import certifi
import numpy as np
import pandas as pd
from loguru import logger

try:
    import truststore

    _ssl_ctx = ssl.create_default_context()
    truststore.inject_into_ssl()
except ImportError:
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())

from core.config import load_oracle_config
from core.console import BLUE, MAGENTA, agent_print
from core.indicators import normalized_from_score
from core.types import AgentNode, OracleState, PipelineStatus
from tools.market_data import fetch_crypto_ohlcv


def _safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _detect_liquidity_sweep(
    df: pd.DataFrame,
    lookback: int,
    wick_ratio_threshold: float,
    body_ratio_threshold: float,
) -> dict[str, Any]:
    recent = df.tail(lookback + 1).reset_index(drop=True)
    if len(recent) < lookback + 1:
        return {
            "event": "none",
            "score": 0.0,
            "detail": "Sweep analizi icin veri yetersiz.",
        }

    current = recent.iloc[-1]
    history = recent.iloc[:-1]

    resistance = float(history["high"].max())
    support = float(history["low"].min())

    bar_range = max(float(current["high"] - current["low"]), 1e-9)
    body_size = abs(float(current["close"] - current["open"]))
    upper_wick = float(current["high"] - max(current["open"], current["close"]))
    lower_wick = float(min(current["open"], current["close"]) - current["low"])

    upper_wick_ratio = _safe_ratio(upper_wick, bar_range)
    lower_wick_ratio = _safe_ratio(lower_wick, bar_range)
    body_ratio = _safe_ratio(body_size, bar_range)

    up_sweep = (
        float(current["high"]) > resistance
        and float(current["close"]) < resistance
        and upper_wick_ratio >= wick_ratio_threshold
        and body_ratio <= body_ratio_threshold
    )
    down_sweep = (
        float(current["low"]) < support
        and float(current["close"]) > support
        and lower_wick_ratio >= wick_ratio_threshold
        and body_ratio <= body_ratio_threshold
    )

    if up_sweep:
        return {
            "event": "distribution_sweep",
            "score": -24.0,
            "detail": (
                f"Likidite avı (direnc ustu igne): H={float(current['high']):.6f} > "
                f"res={resistance:.6f}, kapanis geri dondu."
            ),
        }

    if down_sweep:
        return {
            "event": "accumulation_sweep",
            "score": 24.0,
            "detail": (
                f"Likidite avı (destek alti igne): L={float(current['low']):.6f} < "
                f"sup={support:.6f}, kapanis geri dondu."
            ),
        }

    return {
        "event": "none",
        "score": 0.0,
        "detail": "Belirgin sweep sinyali yok.",
    }


def _detect_cvd_divergence(df: pd.DataFrame, lookback: int) -> dict[str, Any]:
    recent = df.tail(lookback).copy()
    if len(recent) < 10:
        return {
            "event": "none",
            "score": 0.0,
            "detail": "CVD analizi icin veri yetersiz.",
        }

    delta = np.where(recent["close"] >= recent["open"], recent["volume"], -recent["volume"])
    cvd = np.cumsum(delta)

    x = np.arange(len(recent), dtype=float)
    price_slope = float(np.polyfit(x, recent["close"].to_numpy(dtype=float), 1)[0])
    cvd_slope = float(np.polyfit(x, cvd.astype(float), 1)[0])

    if price_slope > 0 and cvd_slope < 0:
        return {
            "event": "bearish_cvd_divergence",
            "score": -18.0,
            "detail": f"Fiyat trend yukari ({price_slope:.6f}), CVD asagi ({cvd_slope:.2f}).",
        }
    if price_slope < 0 and cvd_slope > 0:
        return {
            "event": "bullish_cvd_divergence",
            "score": 18.0,
            "detail": f"Fiyat trend asagi ({price_slope:.6f}), CVD yukari ({cvd_slope:.2f}).",
        }

    return {
        "event": "none",
        "score": 0.0,
        "detail": f"CVD uyumsuzlugu yok (price={price_slope:.6f}, cvd={cvd_slope:.2f}).",
    }


def _build_whale_score_0_100(
    sweep: dict[str, Any],
    cvd: dict[str, Any],
    futures_score: float,
) -> tuple[float, list[str], str]:
    raw = 50.0 + float(sweep["score"]) + float(cvd["score"]) + (futures_score * 30.0)
    score = round(float(np.clip(raw, 0.0, 100.0)), 2)
    notes = [sweep["detail"], cvd["detail"]]

    if score >= 62:
        regime = "accumulation"
    elif score <= 38:
        regime = "distribution"
    else:
        regime = "neutral"
    return score, notes, regime


async def _fetch_binance_futures_data(symbol: str) -> dict:
    """
    Binance Public API (API key gerekmez).
    Kripto futures: OI (Açık Pozisyon) ve Funding Rate.
    """
    if "/" not in symbol or "USDT" not in symbol:
        return {"oi_change": 0.0, "funding_rate": 0.0, "futures_score": 0.0}

    futures_symbol = symbol.replace("/", "")
    results: dict[str, float | str] = {}

    try:
        connector = aiohttp.TCPConnector(ssl=_ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as s:
            funding_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            async with s.get(
                funding_url,
                params={"symbol": futures_symbol},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results["funding_rate"] = float(data.get("lastFundingRate", 0))
                else:
                    results["funding_rate"] = 0.0

            oi_url = "https://fapi.binance.com/fapi/v1/openInterest"
            async with s.get(
                oi_url,
                params={"symbol": futures_symbol},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results["open_interest"] = float(data.get("openInterest", 0))
                else:
                    results["open_interest"] = 0.0

            ls_url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
            async with s.get(
                ls_url,
                params={"symbol": futures_symbol, "period": "4h", "limit": 2},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and len(data) >= 1:
                        results["long_short_ratio"] = float(data[0].get("longShortRatio", 1.0))
                    else:
                        results["long_short_ratio"] = 1.0
                else:
                    results["long_short_ratio"] = 1.0
    except Exception as exc:
        logger.warning(f"Binance futures hatası {symbol}: {exc}")
        results.setdefault("funding_rate", 0.0)
        results.setdefault("open_interest", 0.0)
        results.setdefault("long_short_ratio", 1.0)

    funding = float(results.get("funding_rate", 0.0) or 0.0)
    ls_ratio = float(results.get("long_short_ratio", 1.0) or 1.0)

    futures_score = 0.0
    if funding > 0.01:
        futures_score -= 0.15
        results["funding_signal"] = "ASIRI_LONG_RISK"
    elif funding > 0.005:
        futures_score -= 0.05
        results["funding_signal"] = "HAFIF_LONG_BASKISI"
    elif funding < -0.005:
        futures_score += 0.10
        results["funding_signal"] = "SHORT_SQUEEZE_POTANSIYELI"
    else:
        results["funding_signal"] = "NOTR"

    if ls_ratio > 2.0:
        futures_score -= 0.10
        results["ls_signal"] = "ASIRI_LONG_TUZAGI"
    elif ls_ratio < 0.7:
        futures_score += 0.10
        results["ls_signal"] = "SHORT_SQUEEZE"
    else:
        results["ls_signal"] = "DENGELI"

    results["futures_score"] = round(futures_score, 3)
    results["oi_change"] = 0.0
    return results


async def run_whale_hunter(state: OracleState) -> OracleState:
    agent_print(
        "SYMBIOTIC_HUNTER",
        f"Devrede → {state.symbol} | Balina radarı aktif…",
        MAGENTA,
    )
    try:
        conf = await load_oracle_config()
        whale_conf = conf.whale
        df = await fetch_crypto_ohlcv(
            state.symbol,
            timeframe=whale_conf.timeframe,
            limit=whale_conf.ohlcv_limit,
        )

        sweep = _detect_liquidity_sweep(
            df,
            lookback=whale_conf.sweep_lookback_bars,
            wick_ratio_threshold=whale_conf.wick_ratio_threshold,
            body_ratio_threshold=whale_conf.body_ratio_threshold,
        )
        cvd = _detect_cvd_divergence(df, lookback=whale_conf.cvd_lookback_bars)

        futures_data = await _fetch_binance_futures_data(state.symbol)
        futures_score = float(futures_data.get("futures_score", 0.0) or 0.0)

        score_0_100, notes, regime = _build_whale_score_0_100(sweep, cvd, futures_score)
        whale_score = normalized_from_score(score_0_100)

        notes.append(
            (
                f"Futures: funding={float(futures_data.get('funding_rate', 0.0) or 0.0):+.6f} "
                f"({futures_data.get('funding_signal', 'NOTR')}), "
                f"L/S={float(futures_data.get('long_short_ratio', 1.0) or 1.0):.3f} "
                f"({futures_data.get('ls_signal', 'DENGELI')})"
            )
        )

        agent_print(
            "SYMBIOTIC_HUNTER",
            f"Whale Skor={score_0_100:.1f}/100 -> {whale_score:+.3f} | rejim={regime}",
            BLUE,
        )
        for note in notes:
            agent_print("SYMBIOTIC_HUNTER", note, MAGENTA)

        return state.model_copy(
            update={
                "current_node": AgentNode.WHALE_HUNTER,
                "status": PipelineStatus.RUNNING,
                "whale_score": whale_score,
                "funding_rate": float(futures_data.get("funding_rate", 0.0) or 0.0),
                "funding_signal": futures_data.get("funding_signal", "NOTR"),
                "long_short_ratio": float(futures_data.get("long_short_ratio", 1.0) or 1.0),
                "ls_signal": futures_data.get("ls_signal", "DENGELI"),
                "open_interest": float(futures_data.get("open_interest", 0.0) or 0.0),
                "messages": [
                    (
                        f"[WHALE_HUNTER] score={score_0_100:.1f} norm={whale_score:+.3f} "
                        f"regime={regime} futures={futures_score:+.3f}"
                    )
                ],
            }
        )
    except Exception as exc:
        msg = f"Whale Hunter hata: {exc}"
        return state.model_copy(
            update={
                "current_node": AgentNode.WHALE_HUNTER,
                "status": PipelineStatus.RUNNING,
                "fatal_error": msg,
                "messages": [f"[WHALE_HUNTER] ERROR {msg}"],
            }
        )
