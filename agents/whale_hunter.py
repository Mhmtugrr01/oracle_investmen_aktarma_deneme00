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
from tools.market_data import build_ssl_context, fetch_crypto_ohlcv


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
    Kripto futures: OI değişimi, funding rate trend, long/short oranı.
    """
    if "/" not in symbol or "USDT" not in symbol:
        return {
            "oi_change": 0.0,
            "oi_change_pct": 0.0,
            "oi_signal": "UNAVAILABLE",
            "funding_rate": 0.0,
            "funding_trend": "NOTR",
            "futures_score": 0.0,
            "futures_available": False,
        }

    futures_symbol = symbol.replace("/", "")
    results: dict[str, float | str] = {}

    async def _pull(verify_ssl: bool) -> None:
        connector = aiohttp.TCPConnector(ssl=build_ssl_context(verify_ssl))
        async with aiohttp.ClientSession(connector=connector) as s:
            # 1. Anlık Funding Rate
            funding_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            async with s.get(
                funding_url,
                params={"symbol": futures_symbol},
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results["funding_rate"] = float(data.get("lastFundingRate", 0))
                else:
                    results["funding_rate"] = 0.0

            # 2. Tarihsel Funding Rate (son 8 dönem = 48 saat trend)
            funding_hist_url = "https://fapi.binance.com/fapi/v1/fundingRate"
            async with s.get(
                funding_hist_url,
                params={"symbol": futures_symbol, "limit": 8},
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and len(data) >= 3:
                        rates = [float(d.get("fundingRate", 0)) for d in data]
                        # Trend: son 3 periyot önceki 3 perioda kıyasla artıyor mu?
                        recent_avg = sum(rates[:3]) / 3
                        older_avg = sum(rates[4:7]) / 3 if len(rates) >= 7 else recent_avg
                        if recent_avg > older_avg * 1.3:
                            results["funding_trend"] = "YUKSELIYOR"    # Long kalabalık oluyor = risk
                        elif recent_avg < older_avg * 0.7:
                            results["funding_trend"] = "DUSUYOR"       # Short squeeze potansiyeli
                        else:
                            results["funding_trend"] = "NOTR"
                    else:
                        results["funding_trend"] = "NOTR"
                else:
                    results["funding_trend"] = "NOTR"

            # 3. OI Tarihsel (son 4 nokta ile gerçek OI değişimi)
            oi_hist_url = "https://fapi.binance.com/futures/data/openInterestHist"
            async with s.get(
                oi_hist_url,
                params={"symbol": futures_symbol, "period": "4h", "limit": 4},
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and len(data) >= 2:
                        oi_latest = float(data[-1].get("sumOpenInterestValue", 0))
                        oi_prev   = float(data[0].get("sumOpenInterestValue", 0))
                        results["open_interest"] = oi_latest
                        if oi_prev > 0:
                            results["oi_change_pct"] = round((oi_latest - oi_prev) / oi_prev * 100.0, 2)
                        else:
                            results["oi_change_pct"] = 0.0
                    else:
                        results["open_interest"] = 0.0
                        results["oi_change_pct"] = 0.0
                else:
                    results["open_interest"] = 0.0
                    results["oi_change_pct"] = 0.0

            # 4. Long/Short oranı
            ls_url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
            async with s.get(
                ls_url,
                params={"symbol": futures_symbol, "period": "4h", "limit": 2},
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and len(data) >= 1:
                        results["long_short_ratio"] = float(data[0].get("longShortRatio", 1.0))
                    else:
                        results["long_short_ratio"] = 1.0
                else:
                    results["long_short_ratio"] = 1.0

    try:
        try:
            await _pull(True)
        except aiohttp.ClientConnectorCertificateError:
            logger.warning("[WHALE_HUNTER] Binance futures SSL doğrulaması başarısız, fallback aktif.")
            await _pull(False)
    except Exception as exc:
        logger.warning(f"[TIMEOUT/BYPASS] Futures Hata: Sistem donmasın diye gecirildi! {exc}")
        return {
            "oi_change": 0.0,
            "oi_change_pct": 0.0,
            "oi_signal": "UNAVAILABLE",
            "funding_rate": 0.0,
            "funding_trend": "NOTR",
            "futures_score": 0.0,
            "long_short_ratio": 1.0,
            "futures_available": False,
        }

    funding = float(results.get("funding_rate", 0.0) or 0.0)
    ls_ratio = float(results.get("long_short_ratio", 1.0) or 1.0)
    oi_chg_pct = float(results.get("oi_change_pct", 0.0) or 0.0)
    funding_trend = str(results.get("funding_trend", "NOTR"))

    futures_score = 0.0

    # ── OI Değişim Analizi (Fiyat + OI birlikteliği için CEO'ya mesaj gönder) ──
    oi_signal = "NOTR"
    if oi_chg_pct > 3.0:
        oi_signal = "OI_YUKSELIYOR"      # Yeni pozisyon geliyor — trend güçlü olabilir
        futures_score += 0.05
    elif oi_chg_pct < -3.0:
        oi_signal = "OI_DUSUYOR"         # Pozisyon kapanıyor — momentum zayıflıyor
        futures_score -= 0.05
    results["oi_signal"] = oi_signal
    results["oi_change"] = oi_chg_pct   # artık 0.0 değil, gerçek değer

    # ── Funding Rate Anlık + Trend Birlikte Değerlendir ──
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

    # Funding giderek artıyorsa ek ceza
    if funding_trend == "YUKSELIYOR" and funding > 0.003:
        futures_score -= 0.05   # Long tarafı git gide kalabalıklaşıyor = tehlike

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
    results["futures_available"] = True
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

        fallback_mode = str(getattr(whale_conf, "whale_ssl_blocked_fallback", "cvd_only") or "cvd_only").lower()
        futures_available = bool(futures_data.get("futures_available", False))

        if not futures_available and fallback_mode == "cvd_only":
            # Futures katmanı yoksa CVD/sweep etkisini daha görünür hale getir.
            boosted_cvd = cvd.copy()
            boosted_cvd["score"] = float(np.clip(float(cvd.get("score", 0.0)) * 1.35, -24.0, 24.0))
            score_0_100, notes, regime = _build_whale_score_0_100(sweep, boosted_cvd, 0.0)
            notes.append("Futures unavailable -> cvd_only fallback aktif (SSL/erişim blok).")
        else:
            score_0_100, notes, regime = _build_whale_score_0_100(sweep, cvd, futures_score)

        whale_score = normalized_from_score(score_0_100)

        # ── Remora / Çakal Strateji Tespiti ───────────────────────────────────
        # Sweep tespit edildiyse ve regime birikim ise: Çakal (panik sonrası al)
        # Sweep tespit + CVD uyumsuzluk + dip seviyesi: Remora (dip öncesi kuyu)
        whale_strategy = "IZLE"
        whale_strategy_note = ""
        if sweep["event"] == "accumulation_sweep":
            if cvd["event"] == "bullish_cvd_divergence":
                whale_strategy = "REMORA"
                _sweep_low = float(df["low"].tail(whale_conf.sweep_lookback_bars + 1).min())
                _remora_entry = round(_sweep_low * 0.995, 8)
                whale_strategy_note = (
                    f"REMORA STRATEJİSİ: Likidite tuzağı {_sweep_low:.4f} altında. "
                    f"Balina avını bekle, hedef giriş {_remora_entry:.4f} bölgesi."
                )
            else:
                whale_strategy = "CAKAL"
                _sweep_low = float(df["low"].tail(whale_conf.sweep_lookback_bars + 1).min())
                _cakal_entry = round(_sweep_low * 1.005, 8)
                whale_strategy_note = (
                    f"ÇAKAL STRATEJİSİ: Likidite avı {_sweep_low:.4f} altında. "
                    f"Panik satışı sonrası {_cakal_entry:.4f} bölgesinde al."
                )
        elif sweep["event"] == "distribution_sweep":
            whale_strategy = "CAKAL_SHORT"
            whale_strategy_note = (
                f"ÇAKAL SHORT: Dağıtım sweep tespit edildi. "
                f"Panik alımı sonrası kısa pozisyon değerlendirilebilir."
            )

        if whale_strategy_note:
            agent_print("SYMBIOTIC_HUNTER", whale_strategy_note, MAGENTA)

        notes.append(
            (
                f"Futures: funding={float(futures_data.get('funding_rate', 0.0) or 0.0):+.6f} "
                f"({futures_data.get('funding_signal', 'NOTR')}) "
                f"trend={futures_data.get('funding_trend', 'NOTR')}, "
                f"L/S={float(futures_data.get('long_short_ratio', 1.0) or 1.0):.3f} "
                f"({futures_data.get('ls_signal', 'DENGELI')}), "
                f"OI_degisim={futures_data.get('oi_change', 0.0):+.2f}% ({futures_data.get('oi_signal', 'NOTR')})"
            )
        )

        agent_print(
            "SYMBIOTIC_HUNTER",
            f"Whale Skor={score_0_100:.1f}/100 -> {whale_score:+.3f} | rejim={regime}",
            BLUE,
        )
        for note in notes:
            agent_print("SYMBIOTIC_HUNTER", note, MAGENTA)

        whale_messages = [
            (
                f"[WHALE_HUNTER] score={score_0_100:.1f} norm={whale_score:+.3f} "
                f"regime={regime} futures={futures_score:+.3f} strategy={whale_strategy}"
            )
        ]
        if whale_strategy_note:
            whale_messages.append(f"[WHALE_STRATEGY] {whale_strategy}: {whale_strategy_note}")

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
                "messages": whale_messages,
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
