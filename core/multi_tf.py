"""
KATMAN 1.5 — MULTI-TIMEFRAME ANALİZ (Varlık Özelinde)
=====================================================
Bir varlığın 5m / 15m / 1h / 4h / 1d / 1w zaman dilimlerindeki yönlülüğünü
ayrı ayrı hesaplar (quant_engine._compute_tf_indicators ile aynı göstergeler),
ardından:
  - DÜŞÜK TF (5m/15m)  => GİRİŞ zamanlaması  (now / wait / avoid)
  - YÜKSEK TF (1h/4h/1d/1w) => SİNYAL GEÇERLİLİK alanı (kaç TF hizalı)

Çıktı `MultiTFAnalysis`; scanner digest v2 ve WATCHLIST_PREMIUM koşulunda
kullanılır. Bellek disiplini: veriler toplanıp serbest bırakılır.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agents.quant_engine import _compute_tf_indicators
from core.asset_classifier import is_crypto
from tools.market_data import fetch_crypto_ohlcv, fetch_stock_macro_data

logger = logging.getLogger(__name__)

MTF_TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d", "1w")

# Kripto dışı varlıklar için yfinance period/interval eşlemesi
_STOCK_TF_MAP: dict[str, tuple[str, str]] = {
    "5m": ("5d", "5m"),
    "15m": ("1mo", "15m"),
    "1h": ("1mo", "1h"),
    "1d": ("6mo", "1d"),
    "1w": ("2y", "1wk"),
}

_BULLISH_BIASES = ("BULLISH", "OVERSOLD", "ACCUMULATING")
_BEARISH_BIASES = ("BEARISH", "OVERBOUGHT", "DISTRIBUTING")

_FETCH_TIMEOUT_SEC = 30.0


@dataclass
class MultiTFAnalysis:
    """Bir varlığın MTF analiz özeti."""

    symbol: str
    points: dict[str, dict[str, Any]] = field(default_factory=dict)
    signal_bias: str = "NEUTRAL"  # BULLISH | BEARISH | NEUTRAL
    aligned_count: int = 0
    total_count: int = 0
    entry_timing: str = "WAIT"  # NOW | WAIT | AVOID
    validity_tf: str = "1h"
    validity_text: str = ""
    notes: list[str] = field(default_factory=list)
    partial: bool = False  # bazı TF'ler çekilemedi mi?
    # ── FASE E: teknik analiz teyit sayacı (divergence/hook/breakout/CHoCH) ──
    bull_confirmations: int = 0
    bear_confirmations: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  VERİ ÇEKME
# ═══════════════════════════════════════════════════════════════════════════════

def _resample_4h_from_1h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """1 saatlik veriyi 4 saatlik barlara agrega eder (OHLCV)."""
    d = df_1h.copy()
    if "timestamp" in d.columns:
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.set_index("timestamp")
    resampled = d.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    out = resampled.reset_index()
    out = out.rename(columns={"index": "timestamp"})
    return out


async def _fetch_one_tf(
    symbol: str,
    crypto: bool,
    tf: str,
    sem: asyncio.Semaphore,
) -> tuple[str, pd.DataFrame | Exception]:
    async with sem:
        try:
            if crypto:
                df = await asyncio.wait_for(
                    fetch_crypto_ohlcv(symbol, timeframe=tf, limit=220),
                    timeout=_FETCH_TIMEOUT_SEC,
                )
            else:
                if tf == "4h":
                    df_1h = await asyncio.wait_for(
                        fetch_stock_macro_data(symbol, period="1mo", interval="1h"),
                        timeout=_FETCH_TIMEOUT_SEC,
                    )
                    df = _resample_4h_from_1h(df_1h)
                else:
                    period, interval = _STOCK_TF_MAP.get(tf, ("6mo", "1d"))
                    df = await asyncio.wait_for(
                        fetch_stock_macro_data(symbol, period=period, interval=interval),
                        timeout=_FETCH_TIMEOUT_SEC,
                    )
            return tf, df
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[MTF] {symbol} {tf} çekilemedi: {exc}")
            return tf, exc


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA ANALİZ
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_bias(bias: str) -> str:
    if bias in _BULLISH_BIASES:
        return "BULL"
    if bias in _BEARISH_BIASES:
        return "BEAR"
    return "NEUTRAL"


async def analyze_multi_tf(
    symbol: str,
    crypto: bool | None = None,
    timeframes: tuple[str, ...] = MTF_TIMEFRAMES,
    max_concurrency: int = 2,
) -> MultiTFAnalysis | None:
    """
    Varlığın MTF bias matrisini hesaplar.
    crypto=None ise `core.asset_classifier.is_crypto` ile otomatik algılanır.
    """
    if crypto is None:
        crypto = is_crypto(symbol)

    sem = asyncio.Semaphore(max_concurrency)
    results = await asyncio.gather(
        *[_fetch_one_tf(symbol, crypto, tf, sem) for tf in timeframes],
        return_exceptions=True,
    )

    points: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for item in results:
        if isinstance(item, BaseException):
            failed.append("?")
            continue
        tf, df = item
        if isinstance(df, Exception):
            failed.append(tf)
            continue
        try:
            if df is None or len(df) < 25:
                failed.append(tf)
                continue
            points[tf] = _compute_tf_indicators(df)
            del df
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[MTF] {symbol} {tf} gösterge hatası: {exc}")
            failed.append(tf)

    if not points:
        return None

    analysis = aggregate_mtf_points(symbol, points, failed)

    # ── Notlar ──────────────────────────────────────────────────────────────
    if failed:
        analysis.notes.append(f"Eksik TF'ler: {', '.join(failed)}")
    if points.get("5m", {}).get("choch_detected"):
        analysis.notes.append(
            f"5m CHoCH: {points['5m'].get('choch_direction', '?')} "
            f"({points['5m'].get('choch_strength', '?')})"
        )
    if points.get("15m", {}).get("rsi_trendline_break"):
        analysis.notes.append(
            f"15m RSI Trendline Break: {points['15m'].get('rsi_break_direction', '?')}"
        )
    # ── FASE E: divergence / hook / breakout notları ────────────────────────
    for tf in ("5m", "15m", "1h", "4h", "1d"):
        p = points.get(tf, {})
        div = p.get("divergence", "NONE")
        if div and div != "NONE":
            analysis.notes.append(
                f"{tf} {div.replace('_DIVERGENCE', '')} "
                f"({p.get('divergence_strength', '?')})"
            )
        if p.get("rsi_hook"):
            analysis.notes.append(f"{tf} RSI hook (30 geri alımı)")
        if p.get("price_breakout"):
            analysis.notes.append(f"{tf} düşeni kırdı (hacim teyitli)")

    return analysis


def aggregate_mtf_points(
    symbol: str,
    points: dict[str, dict[str, Any]],
    failed: list[str] | None = None,
) -> MultiTFAnalysis:
    """
    SAF AGREGASYON (test edilebilir): TF bias matrisinden yön, giriş zamanlaması
    ve geçerlilik alanını hesaplar. Ağ çağrısı yoktur.
    """
    failed = failed or []
    analysis = MultiTFAnalysis(symbol=symbol, points=points, partial=bool(failed))

    # ── Yön özeti (tüm TF'lerin çoğunluğu) ──────────────────────────────────
    counts: dict[str, int] = Counter(_normalize_bias(p["bias"]) for p in points.values())
    bull = counts.get("BULL", 0)
    bear = counts.get("BEAR", 0)
    analysis.total_count = len(points)
    analysis.aligned_count = max(bull, bear)

    if bull > bear and bull >= 2:
        analysis.signal_bias = "BULLISH"
    elif bear > bull and bear >= 2:
        analysis.signal_bias = "BEARISH"
    else:
        analysis.signal_bias = "NEUTRAL"

    # ── FASE E: teknik teyit sayacı (divergence/hook/breakout/CHoCH) ────────
    bull_c = bear_c = 0
    for p in points.values():
        div = p.get("divergence", "NONE")
        if div in ("POSITIVE_DIVERGENCE", "HIDDEN_BULLISH_DIVERGENCE"):
            bull_c += 2 if p.get("divergence_strength") == "STRONG" else 1
        elif div in ("NEGATIVE_DIVERGENCE", "HIDDEN_BEARISH_DIVERGENCE"):
            bear_c += 2 if p.get("divergence_strength") == "STRONG" else 1
        if p.get("rsi_hook"):
            bull_c += 1
        if p.get("price_breakout"):
            bull_c += 1
        if p.get("rsi_breakout"):
            bull_c += 1
        if p.get("choch_detected"):
            if p.get("choch_direction") == "BULLISH":
                bull_c += 2 if p.get("choch_strength") == "STRONG" else 1
            elif p.get("choch_direction") == "BEARISH":
                bear_c += 2 if p.get("choch_strength") == "STRONG" else 1
        if p.get("rsi_trendline_break"):
            if p.get("rsi_break_direction") == "BULLISH":
                bull_c += 1
            elif p.get("rsi_break_direction") == "BEARISH":
                bear_c += 1
    analysis.bull_confirmations = bull_c
    analysis.bear_confirmations = bear_c

    # ── GİRİŞ ZAMANLAMASI (düşük TF) ────────────────────────────────────────
    low_bias = []
    for tf in ("5m", "15m"):
        if tf in points:
            low_bias.append(_normalize_bias(points[tf]["bias"]))
    if low_bias:
        if all(b == "BULL" for b in low_bias):
            analysis.entry_timing = "NOW"
        elif any(b == "BULL" for b in low_bias):
            analysis.entry_timing = "WAIT"
        elif any(b == "BEAR" for b in low_bias):
            analysis.entry_timing = "AVOID"
        else:
            analysis.entry_timing = "WAIT"

    # ── FASE E: düşük TF'de boğa teyidi → WAIT'i NOW'a yükselt ──────────────
    if analysis.entry_timing == "WAIT":
        for tf in ("5m", "15m"):
            p = points.get(tf, {})
            if (
                p.get("divergence") in ("POSITIVE_DIVERGENCE", "HIDDEN_BULLISH_DIVERGENCE")
                or p.get("rsi_hook")
                or p.get("price_breakout")
            ):
                analysis.entry_timing = "NOW"
                analysis.notes.append(
                    f"{tf} boğa teyidi (divergence/hook/breakout) → giriş NOW"
                )
                break

    # ── GEÇERLİLİK ALANI (yüksek TF hizalaması) ─────────────────────────────
    higher = [tf for tf in ("1h", "4h", "1d", "1w") if tf in points]
    aligned_hi = 0
    for tf in higher:
        if _normalize_bias(points[tf]["bias"]) == _normalize_bias(analysis.signal_bias) and analysis.signal_bias != "NEUTRAL":
            aligned_hi += 1
    if analysis.signal_bias != "NEUTRAL" and higher:
        if aligned_hi >= 2:
            analysis.validity_tf = "4h" if "4h" in points and _normalize_bias(points["4h"]["bias"]) == _normalize_bias(analysis.signal_bias) else "1d"
            analysis.validity_text = (
                f"Yüksek TF'ler sinyalle hizalı ({aligned_hi}/{len(higher)}) — sinyal "
                f"{analysis.validity_tf} seviyesine kadar geçerli."
            )
        elif aligned_hi == 1:
            analysis.validity_tf = "1h"
            analysis.validity_text = "Yalnızca 1 yüksek TF hizalı — sinyal kısa vadeli, 1h teyidi şart."
        else:
            analysis.validity_tf = "15m"
            analysis.validity_text = "Yüksek TF'ler ters — sinyal yalnızca 5-15m tepki olarak kullanılabilir."
    else:
        analysis.validity_tf = "1h"
        analysis.validity_text = "TF'ler karışık — net yön yok, yüksek TF teyidi bekle."

    # ── FASE E: teyit sayacı geçerlilik metnine eklenir ─────────────────────
    if analysis.signal_bias != "NEUTRAL":
        if analysis.bull_confirmations > analysis.bear_confirmations:
            analysis.validity_text += (
                f" Teknik teyit: {analysis.bull_confirmations} boğa "
                f"vs {analysis.bear_confirmations} ayı sinyali."
            )
        elif analysis.bear_confirmations > analysis.bull_confirmations:
            analysis.validity_text += (
                f" Teknik teyit: {analysis.bear_confirmations} ayı "
                f"vs {analysis.bull_confirmations} boğa sinyali."
            )

    return analysis


def format_mtf_summary(analysis: MultiTFAnalysis | None, symbol: str) -> str:
    """Digest için tek satırlık MTF özeti üretir."""
    if analysis is None:
        return f"{symbol}: MTF verisi yok"
    emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(analysis.signal_bias, "⚪")
    tf_line = " | ".join(
        f"{tf}:{_tf_emoji(p['bias'])}" for tf, p in analysis.points.items()
    )
    return (
        f"{emoji} {symbol} MTF: {tf_line}\n"
        f"   Yön: {analysis.signal_bias} ({analysis.aligned_count}/{analysis.total_count} TF) · "
        f"Giriş: {analysis.entry_timing} · Geçerlilik: {analysis.validity_text}"
    )


def _tf_emoji(bias: str) -> str:
    if bias in _BULLISH_BIASES:
        return "▲"
    if bias in _BEARISH_BIASES:
        return "▼"
    return "—"
