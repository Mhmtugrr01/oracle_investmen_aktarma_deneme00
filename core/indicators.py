"""
PROJECT OLYMPUS — Kantitatif teknik analiz motoru.
Saf pandas / pandas-ta; LLM tahmini yok.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import pandas_ta as ta

Side = Literal["long", "short"]


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> dict[str, Any]:
    """RSI hesaplama + asiri alim/satim bolgesi."""
    if len(df) < period + 2:
        raise ValueError(f"RSI icin yetersiz veri: {len(df)} bar")

    rsi_series = ta.rsi(df["close"], length=period)
    if rsi_series is None or rsi_series.dropna().empty:
        raise ValueError("RSI hesaplanamadi.")

    current = float(rsi_series.iloc[-1])
    prev = float(rsi_series.iloc[-2])

    if current >= 70:
        zone = "overbought"
    elif current <= 30:
        zone = "oversold"
    elif current >= 60:
        zone = "bullish"
    elif current <= 40:
        zone = "bearish"
    else:
        zone = "neutral"

    return {
        "rsi": round(current, 2),
        "rsi_prev": round(prev, 2),
        "zone": zone,
        "overbought": current >= 70,
        "oversold": current <= 30,
        "momentum": "rising" if current > prev else "falling",
        "series": rsi_series,
    }


def ma_ema_cross(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
    ema: bool = True,
) -> dict[str, Any]:
    """MA veya EMA kesisim (golden/death cross) analizi."""
    if len(df) < slow_period + 2:
        raise ValueError("MA/EMA cross icin yetersiz veri.")

    close = df["close"]
    if ema:
        fast = ta.ema(close, length=fast_period)
        slow = ta.ema(close, length=slow_period)
        kind = "EMA"
    else:
        fast = ta.sma(close, length=fast_period)
        slow = ta.sma(close, length=slow_period)
        kind = "SMA"

    if fast is None or slow is None:
        raise ValueError(f"{kind} hesaplanamadi.")

    f_now, f_prev = float(fast.iloc[-1]), float(fast.iloc[-2])
    s_now, s_prev = float(slow.iloc[-1]), float(slow.iloc[-2])

    if f_prev <= s_prev and f_now > s_now:
        signal = "golden_cross"
    elif f_prev >= s_prev and f_now < s_now:
        signal = "death_cross"
    elif f_now > s_now:
        signal = "bullish"
    else:
        signal = "bearish"

    return {
        "signal": signal,
        "fast": round(f_now, 6),
        "slow": round(s_now, 6),
        "fast_period": fast_period,
        "slow_period": slow_period,
        "type": kind,
        "spread_pct": round(((f_now - s_now) / s_now) * 100, 4) if s_now else 0.0,
    }


def _find_pivots(series: pd.Series, window: int, mode: str) -> list[tuple[int, float]]:
    values = series.values
    pivots: list[tuple[int, float]] = []
    for i in range(window, len(values) - window):
        segment = values[i - window : i + window + 1]
        if mode == "low" and values[i] == segment.min():
            pivots.append((i, float(values[i])))
        elif mode == "high" and values[i] == segment.max():
            pivots.append((i, float(values[i])))
    return pivots


def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_series: pd.Series | None = None,
    period: int = 14,
    lookback: int = 40,
    pivot_window: int = 3,
) -> dict[str, Any]:
    """
    RSI pozitif/negatif uyumsuzluk tarayici.
    Fiyat yeni dip/tepe yaparken RSI zit yonde hareket ediyorsa saptar.
    """
    if rsi_series is None:
        rsi_data = calculate_rsi(df, period)
        rsi_series = rsi_data["series"]

    subset = df.tail(lookback).reset_index(drop=True)
    rsi_sub = rsi_series.tail(lookback).reset_index(drop=True)

    price_lows = _find_pivots(subset["close"], pivot_window, "low")
    price_highs = _find_pivots(subset["close"], pivot_window, "high")

    divergence = "none"
    detail = ""

    if len(price_lows) >= 2:
        i1, v1 = price_lows[-2]
        i2, v2 = price_lows[-1]
        r1 = float(rsi_sub.iloc[i1])
        r2 = float(rsi_sub.iloc[i2])
        if v2 < v1 and r2 > r1:
            divergence = "bullish"
            detail = f"Fiyat dip: {v1:.4f}->{v2:.4f}, RSI: {r1:.1f}->{r2:.1f}"

    if len(price_highs) >= 2:
        i1, v1 = price_highs[-2]
        i2, v2 = price_highs[-1]
        r1 = float(rsi_sub.iloc[i1])
        r2 = float(rsi_sub.iloc[i2])
        if v2 > v1 and r2 < r1:
            divergence = "bearish"
            detail = f"Fiyat tepe: {v1:.4f}->{v2:.4f}, RSI: {r1:.1f}->{r2:.1f}"

    return {
        "divergence": divergence,
        "detail": detail or "Uyumsuzluk tespit edilmedi.",
        "bullish": divergence == "bullish",
        "bearish": divergence == "bearish",
    }


def calculate_fibonacci_levels(
    high: float,
    low: float,
    trend: Literal["up", "down"] = "up",
) -> dict[str, float]:
    """Fibonacci geri cekilme seviyeleri."""
    if high <= low:
        raise ValueError(f"Gecersiz high/low: {high}/{low}")

    diff = high - low
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    levels: dict[str, float] = {}
    for r in ratios:
        if trend == "up":
            price = high - diff * r
        else:
            price = low + diff * r
        key = f"fib_{str(r).replace('.', '')}"
        levels[key] = round(price, 8)

    levels["swing_high"] = round(high, 8)
    levels["swing_low"] = round(low, 8)
    return levels


def calculate_atr_stop_loss(
    df: pd.DataFrame,
    period: int = 14,
    multiplier: float = 2.0,
    side: Side = "long",
) -> dict[str, Any]:
    """ATR tabanli volatilite stop-loss ve hedef mesafesi."""
    if len(df) < period + 2:
        raise ValueError("ATR icin yetersiz veri.")

    atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
    if atr_series is None or atr_series.dropna().empty:
        raise ValueError("ATR hesaplanamadi.")

    atr = float(atr_series.iloc[-1])
    entry = float(df["close"].iloc[-1])

    if side == "long":
        stop_loss = entry - atr * multiplier
        take_profit = entry + atr * multiplier * 2.0
    else:
        stop_loss = entry + atr * multiplier
        take_profit = entry - atr * multiplier * 2.0

    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    return {
        "atr": round(atr, 8),
        "entry": round(entry, 8),
        "stop_loss": round(stop_loss, 8),
        "take_profit": round(take_profit, 8),
        "risk_reward": rr,
        "multiplier": multiplier,
        "side": side,
    }


def nearest_fib_level(
    price: float,
    fib_levels: dict[str, float],
) -> tuple[str, float, float]:
    """Fiyata en yakin fib seviyesini ve uzakligi (%) dondurur."""
    core = {k: v for k, v in fib_levels.items() if k.startswith("fib_")}
    if not core:
        return "none", price, 0.0

    nearest_key = min(core, key=lambda k: abs(core[k] - price))
    nearest_price = core[nearest_key]
    dist_pct = abs(price - nearest_price) / price * 100 if price else 0.0
    return nearest_key, nearest_price, round(dist_pct, 4)


def build_quant_score(
    rsi: dict[str, Any],
    cross: dict[str, Any],
    divergence: dict[str, Any],
    fib_proximity_pct: float,
    atr_rr: float,
) -> tuple[float, list[str]]:
    """
    0-100 arasi teknik skor uretir.
    50 = notr, >50 bullish, <50 bearish.
    """
    score = 50.0
    notes: list[str] = []

    if rsi["oversold"]:
        score += 12
        notes.append(f"RSI oversold ({rsi['rsi']})")
    elif rsi["overbought"]:
        score -= 12
        notes.append(f"RSI overbought ({rsi['rsi']})")
    elif rsi["zone"] == "bullish":
        score += 5
    elif rsi["zone"] == "bearish":
        score -= 5

    cross_map = {
        "golden_cross": 18,
        "bullish": 8,
        "bearish": -8,
        "death_cross": -18,
    }
    delta = cross_map.get(cross["signal"], 0)
    score += delta
    notes.append(f"{cross['type']} {cross['signal']} ({delta:+d})")

    if divergence["bullish"]:
        score += 14
        notes.append("RSI bullish divergence")
    elif divergence["bearish"]:
        score -= 14
        notes.append("RSI bearish divergence")

    if fib_proximity_pct < 1.5:
        score += 8
        notes.append(f"Fib yakini (%{fib_proximity_pct})")

    if atr_rr >= 2.0:
        score += 6
        notes.append(f"ATR R:R={atr_rr}")
    elif atr_rr < 1.0:
        score -= 6
        notes.append(f"Dusuk ATR R:R={atr_rr}")

    return round(float(np.clip(score, 0, 100)), 2), notes


def normalized_from_score(score_0_100: float) -> float:
    """OracleState skoru: 0-100 -> -1..+1."""
    return round(max(-1.0, min(1.0, (score_0_100 - 50.0) / 50.0)), 4)
