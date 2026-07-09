"""
PROJECT OLYMPUS — DÜĞÜM 2: Kantitatif Teknik Motor (R04_MASTER)
Sıfır hata politikasına uygun, DatetimeIndex korumalı ve RSI + Hacim Confluence odaklı motor.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from loguru import logger

from core.config import load_oracle_config
from core.console import BLUE, GREEN, agent_print, error_print
from core.indicators import normalized_from_score
from core.types import AgentNode, OracleState, PipelineStatus
from tools.market_data import fetch_crypto_ohlcv


def _is_crypto(symbol: str) -> bool:
    """Kripto varlık mı? (CCXT formatı: BTC/USDT gibi / içerir)"""
    return "/" in symbol and any(symbol.endswith(f"/{q}") for q in ["USDT", "BTC", "ETH", "BUSD"])


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    data = data.rename(columns=col_map)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in data.columns:
            data[col] = 0.0

    data = data[["open", "high", "low", "close", "volume"]].dropna()
    return data


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h is None or df_1h.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    
    # df_1h'nin DatetimeIndex olduğundan emin oluyoruz
    df_local = df_1h.copy()
    if not isinstance(df_local.index, pd.DatetimeIndex):
        df_local.index = pd.to_datetime(df_local.index, utc=True)
        
    return (
        df_local.resample("4h")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
    )


def _get_atr_multipliers(symbol: str, risk_conf: Any) -> dict[str, Any]:
    """Varlik tipine gore ATR carpanlari, config tabanlı minimum R:R kalibrasyonu ile."""
    stop_mult = float(getattr(risk_conf.atr, "stop_loss_multiplier", 1.0) or 1.0)
    tp_mult = float(getattr(risk_conf.atr, "take_profit_multiplier", 3.0) or 3.0)
    min_rr = float(getattr(risk_conf, "min_risk_reward_ratio", 3.0) or 3.0)
    required_t1 = max(tp_mult, stop_mult * min_rr)

    if _is_crypto(symbol):
        t1 = max(4.5, required_t1)
        return {
            "stop": stop_mult,
            "t1": t1,
            "t2": max(7.0, t1 * 1.55),
            "t3": max(10.0, t1 * 2.20),
            "atr_period": 14,
            "atr_timeframe": "4h",
        }

    t1 = max(3.6, required_t1)
    return {
        "stop": stop_mult,
        "t1": t1,
        "t2": max(5.2, t1 * 1.50),
        "t3": max(7.8, t1 * 2.10),
        "atr_period": 14,
        "atr_timeframe": "1d",
    }


async def _download_yf(symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = symbol.replace("/USDT", "").replace("/USD", "")

    def _run() -> pd.DataFrame:
        return yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

    raw = await asyncio.to_thread(_run)
    df_norm = _normalize_ohlcv_columns(raw)
    
    # ── MUTLAK RANGEINDEX COZUMU: DatetimeIndex Korunuyor, dropna ve ffill ekleniyor ──
    df_filled = df_norm.ffill().bfill().dropna()
    return df_filled


async def _fetch_timeframe_data(symbol: str, tf: str, limit: int) -> pd.DataFrame:
    if _is_crypto(symbol):
        return (await fetch_crypto_ohlcv(symbol, timeframe=tf, limit=limit)).ffill().dropna()

    if tf == "4h":
        df_1h = await _download_yf(symbol, period="60d", interval="1h")
        return _resample_to_4h(df_1h).ffill().dropna()
    if tf == "1h":
        return (await _download_yf(symbol, period="60d", interval="1h")).ffill().dropna()
    if tf == "1d":
        return (await _download_yf(symbol, period="5y", interval="1d")).ffill().dropna()
    if tf == "1w":
        return (await _download_yf(symbol, period="5y", interval="1wk")).ffill().dropna()

    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
    if atr_series is None:
        return pd.Series(dtype=float)
    return atr_series.dropna()


def _classify_bias(price: float, ema50: float, sma200: float, rsi: float) -> str:
    if price > ema50 > sma200 and 50 <= rsi <= 70:
        return "BULLISH"
    if price < ema50 < sma200 and 30 <= rsi <= 50:
        return "BEARISH"
    if rsi < 35:
        return "OVERSOLD"
    if rsi > 65:
        return "OVERBOUGHT"
    if price > ema50 and rsi < 48:
        return "ACCUMULATING"
    if price < ema50 and rsi > 52:
        return "DISTRIBUTING"
    return "NEUTRAL"


def calculate_rsi_volume_confluence(df, rsi_value):
    avg_volume_20 = df['volume'].tail(20).mean()
    current_volume = df['volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
    if rsi_value <= 30 or rsi_value >= 70:
        if volume_ratio > 1.2:
            return 1.4  # RSI ekstrem + hacim teyidi = güçlü
        elif volume_ratio < 0.8:
            return 0.7  # RSI ekstrem ama hacim yok = zayıf/şüpheli
    return 1.0  # nötr bölge, çarpan yok


def _compute_tf_indicators(df: pd.DataFrame) -> dict[str, Any]:
    # ── MUTLAK RANGEINDEX COZUMU: DatetimeIndex Güvencesi ──
    df_local = df.copy()
    if "timestamp" in df_local.columns:
        df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], utc=True)
        df_local.set_index("timestamp", inplace=True)
    elif not isinstance(df_local.index, pd.DatetimeIndex):
        df_local.index = pd.to_datetime(df_local.index, utc=True)

    close = df_local["close"]
    volume = df_local["volume"]

    rsi_series = ta.rsi(close, length=14)
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    bb_df = ta.bbands(close, length=20, std=2.0)
    ema21 = ta.ema(close, length=21)
    ema50 = ta.ema(close, length=50)
    sma50 = ta.sma(close, length=50)
    sma200 = ta.sma(close, length=200)
    obv = ta.obv(close, volume)

    # SMA200 hesabı verisizlikten None dönerse SMA50'yi veya mevcut son fiyatı yedeğe al.
    fallback_sma200 = False
    if sma200 is None or sma200.dropna().empty:
        fallback_sma200 = True

    if (
        rsi_series is None
        or macd_df is None
        or bb_df is None
        or ema21 is None
        or ema50 is None
        or obv is None
    ):
        raise ValueError("Temel Timeframe indicator hesaplaması başarısız.")

    price = float(close.iloc[-1])
    rsi = float(rsi_series.iloc[-1])
    macd_hist_col = [c for c in macd_df.columns if "h" in c.lower()]
    macd_hist = float(macd_df[macd_hist_col[0]].iloc[-1]) if macd_hist_col else 0.0

    bbl_col = [c for c in bb_df.columns if c.lower().startswith("bbl_")]
    bbu_col = [c for c in bb_df.columns if c.lower().startswith("bbu_")]
    bbl = float(bb_df[bbl_col[0]].iloc[-1]) if bbl_col else price
    bbu = float(bb_df[bbu_col[0]].iloc[-1]) if bbu_col else price
    bb_pos = 0.5 if abs(bbu - bbl) < 1e-9 else float(np.clip((price - bbl) / (bbu - bbl), 0.0, 1.0))

    ema21_v = float(ema21.iloc[-1])
    ema50_v = float(ema50.iloc[-1])
    if fallback_sma200:
        if sma50 is not None and not sma50.dropna().empty:
            sma200_v = float(sma50.iloc[-1])
        else:
            sma200_v = float(price)
    else:
        sma200_v = float(sma200.iloc[-1])

    obv_trend = "UP" if float(obv.iloc[-1]) >= float(obv.iloc[-5]) else "DOWN"
    bias = _classify_bias(price=price, ema50=ema50_v, sma200=sma200_v, rsi=rsi)

    return {
        "price": price,
        "rsi": round(rsi, 2),
        "macd_hist": round(macd_hist, 6),
        "bb_position": round(bb_pos, 4),
        "ema21": round(ema21_v, 6),
        "ema50": round(ema50_v, 6),
        "sma200": round(sma200_v, 6),
        "ma_fallback_used": fallback_sma200,
        "obv_trend": obv_trend,
        "bias": bias,
    }


def _alignment_score(biases: dict[str, str]) -> tuple[float, int]:
    normalized = []
    for b in biases.values():
        if b in ("BULLISH", "OVERSOLD", "ACCUMULATING"):
            normalized.append("BULL")
        elif b in ("BEARISH", "OVERBOUGHT", "DISTRIBUTING"):
            normalized.append("BEAR")
        else:
            normalized.append("NEUTRAL")

    counts = Counter(normalized)
    aligned = max(counts.values()) if counts else 0
    mapping = {4: 1.0, 3: 0.75, 2: 0.50}
    return mapping.get(aligned, 0.25), aligned


def _extract_macro_dominance(state: OracleState) -> tuple[float | None, float | None]:
    """Read BTC.D and USDT.D values emitted by MACRO_SENTINEL message stream."""
    macro_msg = next((m for m in state.messages if m.startswith("[MACRO_SENTINEL]")), "")
    if not macro_msg:
        return None, None

    btc_d: float | None = None
    usdt_d: float | None = None
    for part in macro_msg.split():
        if part.startswith("btc_d="):
            raw = part.split("=", 1)[1]
            if raw != "NA":
                try:
                    btc_d = float(raw)
                except ValueError:
                    btc_d = None
        elif part.startswith("usdt_d="):
            raw = part.split("=", 1)[1]
            if raw != "NA":
                try:
                    usdt_d = float(raw)
                except ValueError:
                    usdt_d = None
    return btc_d, usdt_d

def _detect_price_breakout(df: pd.DataFrame) -> bool:
    """
    Fiyat grafiğinde son 20 bardaki zirvelerden geçen düşen trend çizgisinin 
    yukarı yönlü kırılıp kırılmadığını (Düşeni Kırma) lineer regresyon ile doğrular.
    """
    if len(df) < 20:
        return False
        
    recent_highs = df["high"].tail(20).values
    x = np.arange(len(recent_highs))
    
    # En iyi uyum sağlayan direnç çizgisinin eğimini (slope) hesapla
    slope, intercept = np.polyfit(x, recent_highs, 1)
    
    # Eğim pozitifse zaten düşen trend yoktur, kırılım aranmaz
    if slope >= 0:
        return False
        
    # Son iki barın trend çizgisi değerleri
    prev_trend_val = slope * 18 + intercept
    curr_trend_val = slope * 19 + intercept
    
    prev_close = df["close"].iloc[-2]
    curr_close = df["close"].iloc[-1]
    
    # Kırılım Koşulu: Önceki bar trendin altındayken, güncel bar trendin üzerinde kapattı mı?
    if prev_close <= prev_trend_val and curr_close > curr_trend_val:
        # Hacim Teyidi: Son barın hacmi, son 10 barın hacim ortalamasının üzerinde olmalı!
        recent_volume_avg = df["volume"].tail(10).mean()
        if df["volume"].iloc[-1] > recent_volume_avg * 0.95:
            return True
            
    return False


def _detect_rsi_breakout(df: pd.DataFrame) -> bool:
    """
    RSI göstergesinin kendi düşen trend çizgisini (momentum kırılımını) 
    yukarı kırıp kırmadığını hesaplar.
    """
    if len(df) < 20:
        return False

    rsi_series = ta.rsi(df["close"], length=14)
    if rsi_series is None or rsi_series.dropna().empty:
        return False
    rsi_series = rsi_series.dropna()
    if len(rsi_series) < 20:
        return False

    recent_rsi = rsi_series.tail(20).values
    x = np.arange(len(recent_rsi))
    slope, intercept = np.polyfit(x, recent_rsi, 1)
    
    if slope >= 0:
        return False
        
    prev_trend_val = slope * 18 + intercept
    curr_trend_val = slope * 19 + intercept
    
    prev_rsi = float(rsi_series.iloc[-2])
    curr_rsi = float(rsi_series.iloc[-1])
    
    # RSI Düşen Trend Kırılım Teyidi
    if prev_rsi <= prev_trend_val and curr_rsi > curr_trend_val:
        return True
        
    return False


def _detect_rsi_hook(df: pd.DataFrame) -> bool:
    """RSI'ın 30 çukurundan kafasını yukarı kaldırdığını (hook/reclaim) doğrular."""
    try:
        if len(df) < 5:
            return False
        rsi_series = ta.rsi(df["close"], length=14)
        if rsi_series is None or rsi_series.dropna().empty:
            return False
        rsi_series = rsi_series.dropna()
        if len(rsi_series) < 2:
            return False
        prev_rsi = float(rsi_series.iloc[-2])
        curr_rsi = float(rsi_series.iloc[-1])
        return prev_rsi < 30.0 and curr_rsi >= 30.0
    except Exception:
        return False


def _decide_trade_type(
    weekly_bias: str, 
    daily_bias: str, 
    h4_bias: str, 
    h1_bias: str, 
    price_breakout: bool, 
    rsi_breakout: bool,
    rsi_hook: bool
) -> str:
    # ── 🛡️ MULTI-TIMEFRAME CONFLUENCE & TRENDLINE BREAKOUT REFORM (R06) ──
    all_biases = [weekly_bias, daily_bias, h4_bias, h1_bias]
    oversold_count = sum(1 for b in all_biases if b == "OVERSOLD")
    
    # Haftalık grafik Bearish (Trend Aşağı) ise alt zaman dilimlerindeki ham alımları KİLİTLE!
    # Sadece ve sadece fiyatta, RSI'da kırılım VEYA RSI Hook (çukurdan dönüş) geldiyse oyuna gir!
    has_breakout = price_breakout or rsi_breakout or rsi_hook
    
    # 3 veya daha fazla zaman dilimi oversold + kırılım varsa bu jenerasyonel bir fırsattır!
    if oversold_count >= 3 and has_breakout:
        return "STRONG_LONG_TERM_ENTRY"
        
    # Günlük grafik oversold ama henüz ne düşen kırılımı ne de RSI Hook var: ALMA, Pusuya yat! (COIN Tuzağı Kalkanı)
    if daily_bias == "OVERSOLD" and not has_breakout:
        return "AVOID_CONFLICTING_SIGNALS"

    # Günlük grafik oversold olmuş VEYA trend dönüşü teyit edilmiş + düşen trend kırılmışsa/RSI hook varsa: AL!
    if (daily_bias in ["OVERSOLD", "BULLISH"] or oversold_count >= 1) and has_breakout:
        if weekly_bias == "BEARISH":
            return "SHORT_TERM_BOUNCE_ONLY" # Haftalık düşerken günlük kırılım = tepki alımı
        return "ACCUMULATE_ZONE" # Haftalık yön yukarı/nötr ise güvenli toplama alanı

    if weekly_bias == "BULLISH" and daily_bias == "BULLISH":
        if h4_bias in ["BULLISH", "OVERSOLD"] and h1_bias in ["BULLISH", "OVERSOLD"]:
            return "STRONG_LONG_TERM_ENTRY"
        return "HOLD_EXISTING"

    if weekly_bias == "BULLISH" and daily_bias in ["NEUTRAL", "OVERBOUGHT"]:
        return "REDUCE_EXPOSURE"

    if weekly_bias == "BEARISH" and daily_bias == "BEARISH":
        return "STRONG_SELL_OR_SHORT"

    return "AVOID_CONFLICTING_SIGNALS"


def _detect_divergence(df: pd.DataFrame, pivot: int = 14) -> str:
    if len(df) < pivot + 2:
        return "NONE"
    close = df["close"]
    rsi_series = ta.rsi(close, length=14)
    if rsi_series is None or rsi_series.dropna().empty:
        return "NONE"

    price_now = float(close.iloc[-1])
    price_prev = float(close.iloc[-pivot])
    rsi_now = float(rsi_series.iloc[-1])
    rsi_prev = float(rsi_series.iloc[-pivot])

    if price_now < price_prev and rsi_now > rsi_prev:
        return "POSITIVE_DIVERGENCE"
    if price_now > price_prev and rsi_now < rsi_prev:
        return "NEGATIVE_DIVERGENCE"
    return "NONE"


def find_historical_levels(df: pd.DataFrame, lookback_days: int = 500, threshold: float = 0.02) -> tuple[list[float], bool]:
    sample = df.tail(lookback_days)
    closes = sample["close"].astype(float).to_list()
    levels: list[float] = []
    for p in closes:
        if any(abs(p - lv) / lv <= threshold for lv in levels if lv != 0):
            continue
        hits = sum(1 for x in closes if abs(x - p) / p <= threshold)
        if hits >= 3:
            levels.append(float(p))
    current = float(closes[-1]) if closes else 0.0
    near = any(abs(current - lv) / lv <= 0.03 for lv in levels if lv != 0)
    return levels[:8], near


def _compute_fibonacci_levels(df: pd.DataFrame, direction: str, lookback: int = 120) -> dict[str, float] | None:
    """Compute 0.382 / 0.500 / 0.618 retracement levels from recent swing."""
    if df is None or df.empty or len(df) < 20:
        return None
    sample = df.tail(min(max(lookback, 20), len(df)))
    swing_high = float(sample["high"].max())
    swing_low = float(sample["low"].min())
    if swing_high <= swing_low:
        return None

    span = swing_high - swing_low
    if direction == "SHORT":
        # Bearish leg retrace levels measured upward from recent low.
        fib_382 = swing_low + (span * 0.382)
        fib_500 = swing_low + (span * 0.500)
        fib_618 = swing_low + (span * 0.618)
    else:
        # Bullish leg pullback levels measured downward from recent high.
        fib_382 = swing_high - (span * 0.382)
        fib_500 = swing_high - (span * 0.500)
        fib_618 = swing_high - (span * 0.618)

    return {
        "fib_382": round(float(fib_382), 8),
        "fib_500": round(float(fib_500), 8),
        "fib_618": round(float(fib_618), 8),
    }


def find_similar_cycles(df: pd.DataFrame, current_window: int = 60, top_n: int = 3) -> tuple[float, str, str]:
    close = df["close"].astype(float).reset_index(drop=True)
    if len(close) < current_window + 120:
        return 50.0, "Yetersiz tarihsel veri", "HISTORICALLY_MIXED"

    current = close.iloc[-current_window:]
    c_mean = float(current.mean())
    c_std = float(current.std()) or 1.0
    current_norm = (current - c_mean) / c_std

    sims: list[tuple[float, int, float]] = []
    for i in range(0, len(close) - current_window - 90):
        win = close.iloc[i : i + current_window]
        w_std = float(win.std()) or 1.0
        win_norm = (win - float(win.mean())) / w_std
        corr = float(np.corrcoef(current_norm.values, win_norm.values)[0, 1])
        if corr > 0.80:
            future = close.iloc[i + current_window : i + current_window + 60]
            if len(future) < 60:
                continue
            change_60 = float((future.iloc[-1] - win.iloc[-1]) / win.iloc[-1] * 100.0)
            sims.append((corr, i, change_60))

    if not sims:
        return 40.0, "Benzer döngü bulunamadı", "HISTORICALLY_MIXED"

    top = sorted(sims, key=lambda x: x[0], reverse=True)[:top_n]
    bull = sum(1 for _, _, chg in top if chg > 0)
    bear = sum(1 for _, _, chg in top if chg < 0)
    ratio_bull = bull / len(top)
    ratio_bear = bear / len(top)

    if ratio_bull >= 0.70:
        bias = "HISTORICALLY_BULLISH"
    elif ratio_bear >= 0.70:
        bias = "HISTORICALLY_BEARISH"
    else:
        bias = "HISTORICALLY_MIXED"

    mean_corr = float(np.mean([c for c, _, _ in top]))
    similarity_score = max(0.0, min(100.0, 50.0 + (mean_corr - 0.8) * 200.0))
    summary = f"{len(top)} benzer döngü bulundu, ortalama korelasyon={mean_corr:.3f}"
    return round(similarity_score, 2), summary, bias


def calculate_trade_levels(
    df: pd.DataFrame,
    signal_direction: str,
    entry_price: float,
    atr: float,
    stop_loss_multiplier: float,
    t1_multiplier: float,
    t2_multiplier: float,
    t3_multiplier: float,
) -> dict[str, float]:
    window = min(max(20, len(df) // 8), 50)
    swing_high = float(df["high"].tail(window).max())
    swing_low = float(df["low"].tail(window).min())

    if signal_direction == "LONG":
        entry_zone_low = entry_price - (atr * 0.5)
        entry_zone_high = entry_price + (atr * 0.25)
        atr_stop = entry_price - atr * stop_loss_multiplier
        structural_stop = swing_low * 0.985
        stop_loss = max(atr_stop, structural_stop)
        t1 = entry_price + atr * t1_multiplier
        t2 = entry_price + atr * t2_multiplier
        t3 = entry_price + atr * t3_multiplier
        invalidation_level = swing_low
    else:
        entry_zone_high = entry_price + (atr * 0.5)
        entry_zone_low = entry_price - (atr * 0.25)
        atr_stop = entry_price + atr * stop_loss_multiplier
        structural_stop = swing_high * 1.015
        stop_loss = min(atr_stop, structural_stop)
        t1 = entry_price - atr * t1_multiplier
        t2 = entry_price - atr * t2_multiplier
        t3 = entry_price - atr * t3_multiplier
        invalidation_level = swing_high

    # GERÇEK KURUMSAL RISK ZIRHI (RISKI FİYATIN %1.5'UNDAN DAHA AZ OLARAK BAZ ALMAYARAK R:R İLLÜZYONUNU BİTİR!)
    raw_risk = abs(entry_price - stop_loss)
    risk = max(raw_risk, entry_price * 0.015)
    t1_rr = abs(t1 - entry_price) / risk if risk > 0 else 0.0
    t2_rr = abs(t2 - entry_price) / risk if risk > 0 else 0.0
    t3_rr = abs(t3 - entry_price) / risk if risk > 0 else 0.0

    # ── 🛡️ GEOMETRİK TRENDLINE PROJECTION (MSTR Peak / Tepe Saptama Devrimi) ──
    # Son 30 barın zirvelerinden geçen düşen trend çizgisinin gelecekteki kesişim fiyatını hesaplar.
    try:
        recent_highs = df["high"].tail(30).values
        x_indices = np.arange(len(recent_highs))
        slope, intercept = np.polyfit(x_indices, recent_highs, 1)
        # 5 bar sonrasını (index 34) projekte et
        projected_target = float(slope * 34 + intercept) if slope < 0 else float(entry_price * 1.15)
    except Exception:
        projected_target = float(entry_price * 1.15)

    return {
        "entry_zone_low": round(entry_zone_low, 8),
        "entry_zone_high": round(entry_zone_high, 8),
        "stop_loss": round(stop_loss, 8),
        "t1": round(t1, 8),
        "t1_rr": round(t1_rr, 2),
        "t2": round(t2, 8),
        "t2_rr": round(t2_rr, 2),
        "t3": round(t3, 8),
        "t3_rr": round(t3_rr, 2),
        "base_rr": round(t2_rr, 2), # R:R Hedefi T2 (Altın Oran) baz alınır!
        "invalidation_level": round(invalidation_level, 8),
        "dynamic_trendline_target": round(projected_target, 8), # Geometrik çıkış hedefimiz!
    }


def _technical_unit_from_timeframes(tf: dict[str, dict[str, Any]], divergence_bonus: float) -> float:
    """RSI+MACD+OBV çok zaman dilimi ağırlıklı skor. 0.0–1.0 çıktı."""
    score = 0.5
    weekly = tf["1w"]
    daily  = tf["1d"]
    h4     = tf["4h"]
    h1     = tf["1h"]

    # --- Haftalık bias (en ağır) ---
    if weekly["bias"] == "BULLISH":    score += 0.20
    elif weekly["bias"] == "BEARISH":  score -= 0.20
    elif weekly["bias"] == "OVERSOLD": score += 0.18   # Haftalık OVERSOLD = büyük fırsat
    elif weekly["bias"] == "OVERBOUGHT": score -= 0.18
    elif weekly["bias"] == "ACCUMULATING": score += 0.10
    elif weekly["bias"] == "DISTRIBUTING": score -= 0.10

    # --- Günlük bias ---
    if daily["bias"] == "BULLISH":    score += 0.12
    elif daily["bias"] == "BEARISH":  score -= 0.12
    elif daily["bias"] == "OVERSOLD": score += 0.10
    elif daily["bias"] == "OVERBOUGHT": score -= 0.10
    elif daily["bias"] == "ACCUMULATING": score += 0.06
    elif daily["bias"] == "DISTRIBUTING": score -= 0.06

    # --- RSI çok zaman dilimi uyumu (en güçlü sinyal) ---
    weekly_rsi = float(weekly.get("rsi", 50.0))
    daily_rsi  = float(daily.get("rsi", 50.0))

    # Her iki büyük dilim aynı anda aşırı satımda = asimetrik fırsat
    if weekly_rsi < 30 and daily_rsi < 30:
        score += 0.18
    elif weekly_rsi < 40 and daily_rsi < 40:
        score += 0.10
    elif weekly_rsi > 70 and daily_rsi > 70:
        score -= 0.15
    elif weekly_rsi > 60 and daily_rsi > 60:
        score -= 0.08
    else:
        # Tek dilim skorlaması (daha düşük ağırlık)
        if daily_rsi < 32: score += 0.09
        elif daily_rsi < 42: score += 0.05
        elif daily_rsi > 72: score -= 0.10
        elif daily_rsi > 62: score -= 0.05

    # --- MACD çok zaman dilimi uyumu ---
    h4_macd    = float(h4.get("macd_hist", 0.0))
    daily_macd = float(daily.get("macd_hist", 0.0))

    if h4_macd > 0 and daily_macd > 0:
        score += 0.09   # İki dilim MACD pozitif = momentum onayı
    elif h4_macd < 0 and daily_macd < 0:
        score -= 0.09   # İki dilim MACD negatif = düşüş baskısı
    else:
        score += 0.03 if h4_macd > 0 else -0.03

    # --- OBV (hacim yönü) ---
    score += 0.07 if h1["obv_trend"] == "UP" else -0.07

    # --- BB pozisyon ---
    score += 0.03 if 0.20 < h4["bb_position"] < 0.80 else -0.03

    # ── RSI UYUMSUZLUK BONUSU BURADA ENJEKTE EDİLİYOR ──
    score += divergence_bonus

    return float(np.clip(score, 0.0, 1.0))


def _compute_kinetic_score(
    tf: dict[str, dict[str, Any]],
    atr: float,
    entry_price: float,
    whale_score: float | None,
    divergence_bonus: float,
) -> float:
    """Compute the Kinetic composite score (0.0 - 1.0) using:
    (Momentum_Kinetiği × 0.35) + (Kurumsal_Ayak_İzi × 0.25) +
    (Uyumsuzluk_Kırılım × 0.25) + (Sıkışma_Volatilitesi × 0.15)
    """
    # Momentum_Kinetiği: use technical unit as momentum proxy
    momentum = _technical_unit_from_timeframes(tf, divergence_bonus)

    # Kurumsal_Ayak_İzi: normalize whale_score (0-100) to 0-1
    corp = float(whale_score or 0.0) / 100.0

    # Uyumsuzluk_Kırılım: normalize divergence_bonus (-0.12..+0.12) -> (-1..1)
    # map 0.12 => 1.0
    div_component = float(np.clip(divergence_bonus / 0.12, -1.0, 1.0))
    # shift to 0..1
    div_norm = (div_component + 1.0) / 2.0

    # Sıkışma_Volatilitesi: atr/entry ratio normalized (expected small, cap at 0.1)
    vol_ratio = 0.0
    try:
        vol_ratio = float(atr / entry_price) if entry_price and atr else 0.0
    except Exception:
        vol_ratio = 0.0
    vol_norm = float(np.clip(vol_ratio / 0.10, 0.0, 1.0))

    kinetic = (
        (momentum * 0.35)
        + (corp * 0.25)
        + (div_norm * 0.25)
        + (vol_norm * 0.15)
    )

    return float(np.clip(kinetic, 0.0, 1.0))


def _neutral_tf_metrics(df: pd.DataFrame | None = None) -> dict[str, Any]:
    price = 0.0
    if df is not None and not df.empty and "close" in df.columns:
        price = float(df["close"].iloc[-1])
    return {
        "price": price,
        "rsi": 50.0,
        "macd_hist": 0.0,
        "bb_position": 0.5,
        "ema21": price,
        "ema50": price,
        "sma200": price,
        "obv_trend": "DOWN",
        "bias": "NEUTRAL",
    }


def _calculate_confidence(state: dict[str, Any]) -> float:
    """Confidence = sistem kararina ne kadar guvenilebilir?"""
    alignment = float(state.get("timeframe_alignment_score", 0.5) or 0.5)
    composite = abs(float(state.get("composite_score", 0.0) or 0.0))
    consensus_variance = float(state.get("consensus_variance", 1.0) or 1.0)
    divergence_d = state.get("divergence_daily", "NONE")
    divergence_w = state.get("divergence_weekly", "NONE")
    hist_score = float(state.get("historical_similarity_score", 0.0) or 0.0)

    base = (alignment * 0.50) + (composite * 0.30)
    variance_penalty = min(consensus_variance * 0.08, 0.20)

    div_bonus = 0.0
    if divergence_d in ["POSITIVE_DIVERGENCE", "NEGATIVE_DIVERGENCE"]:
        div_bonus += 0.06
    if divergence_w in ["POSITIVE_DIVERGENCE", "NEGATIVE_DIVERGENCE"]:
        div_bonus += 0.08

    hist_bonus = (hist_score / 100.0) * 0.10
    confidence = base - variance_penalty + div_bonus + hist_bonus
    return round(float(np.clip(confidence, 0.0, 1.0)), 3)


async def run_quant_engine(state: OracleState) -> OracleState:
    agent_print(
        "QUANT_ENGINE",
        f"Devrede -> {state.symbol} | Gercek OHLCV + Sembiyotik Analiz...",
        GREEN,
    )

    pattern_outcome_bias: str = "HISTORICALLY_MIXED"
    historical_pattern: str = "NONE"

    try:
        conf = await load_oracle_config()
        quant_conf = conf.quant
        risk_conf = conf.risk

        timeframe_limits = {
            "1h": max(quant_conf.ohlcv_limit, 260),
            "4h": max(quant_conf.ohlcv_limit, 260),
            "1d": max(quant_conf.ohlcv_limit, 520),
            "1w": max(quant_conf.ohlcv_limit, 260),
        }

        tf_dfs: dict[str, pd.DataFrame] = {}
        for tf, limit in timeframe_limits.items():
            tf_dfs[tf] = await _fetch_timeframe_data(state.symbol, tf=tf, limit=limit)

        # Her timeframe için minimum bar sayısı kontrolü
        MIN_BARS = {"1h": 30, "4h": 20, "1d": 30, "1w": 20}
        tf_metrics: dict[str, dict[str, Any]] = {}
        biases: dict[str, str] = {}
        ma_fallback_used = False

        for tf, min_bars in MIN_BARS.items():
            df = tf_dfs.get(tf)
            bar_count = 0 if df is None else len(df)
            if df is None or bar_count < min_bars:
                logger.warning(
                    f"[QUANT] {state.symbol} {tf} için yetersiz veri "
                    f"({bar_count} bar, min {min_bars}). "
                    "Bu timeframe NEUTRAL olarak işaretlendi."
                )
                tf_metrics[tf] = _neutral_tf_metrics(df)
                biases[tf] = "NEUTRAL"
                continue

            try:
                metrics = _compute_tf_indicators(df)
                tf_metrics[tf] = metrics
                biases[tf] = metrics["bias"]
                ma_fallback_used = ma_fallback_used or bool(metrics.get("ma_fallback_used", False))
            except Exception as ind_exc:
                logger.warning(
                    f"[QUANT] {state.symbol} {tf} indicator hesaplanamadı: {ind_exc}. "
                    "Timeframe NEUTRAL olarak işaretlendi."
                )
                tf_metrics[tf] = _neutral_tf_metrics(df)
                biases[tf] = "NEUTRAL"

        alignment_score, aligned_count = _alignment_score(biases)

        weekly_bias = biases["1w"]
        daily_bias = biases["1d"]
        h4_bias = biases["4h"]
        h1_bias = biases["1h"]
        # Fiyat ve RSI grafiklerinin kendi düşen kırılımlarını (breakout) teyit et!
        # # ── 🛡️ DYNAMIC NAMESPACE RESOLVER (df_local Tanımsızlık Kalkanı) ──
        # # Değişken ismi localde ne olursa olsun bulur ve NameError oluşmasını engeller!
        df_local = tf_dfs.get("1d")
        if df_local is None or df_local.empty:
            df_local = tf_dfs.get("4h")

        price_breakout = _detect_price_breakout(df_local)
        rsi_breakout = _detect_rsi_breakout(df_local)
        rsi_hook = _detect_rsi_hook(df_local)
        
        trade_type = _decide_trade_type(
            weekly_bias, daily_bias, h4_bias, h1_bias,
            price_breakout=price_breakout, rsi_breakout=rsi_breakout,
            rsi_hook=rsi_hook
        )

        divergence_daily = _detect_divergence(tf_dfs["1d"], pivot=14) if len(tf_dfs["1d"]) >= 20 else "NONE"
        divergence_weekly = _detect_divergence(tf_dfs["1w"], pivot=8) if len(tf_dfs["1w"]) >= 20 else "NONE"
        # Evrensel Kalkan: 4 Saatlik Uyumsuzluğu da sisteme dahil et!
        divergence_h4 = _detect_divergence(tf_dfs["4h"], pivot=14) if len(tf_dfs["4h"]) >= 20 else "NONE"

        # RSI Uyumsuzluk Bonus Hesaplama (Multi-Timeframe Divergence Bonus)
        divergence_bonus = 0.0
        if "POSITIVE" in divergence_daily or "POSITIVE" in divergence_h4:
            divergence_bonus += 0.12
        elif "NEGATIVE" in divergence_daily or "NEGATIVE" in divergence_h4:
            divergence_bonus -= 0.12

        technical_unit = _technical_unit_from_timeframes(tf_metrics, divergence_bonus)
        quant_score = normalized_from_score(technical_unit * 100.0)

        h4_df = tf_dfs["4h"] if len(tf_dfs["4h"]) >= 20 else tf_dfs["1d"]
        daily_df = tf_dfs["1d"]
        atr_cfg = _get_atr_multipliers(state.symbol, risk_conf)
        atr_tf = str(atr_cfg["atr_timeframe"])
        atr_df = h4_df if atr_tf == "4h" else daily_df
        atr_series = _calculate_atr(atr_df, period=int(atr_cfg["atr_period"]))
        if atr_series.empty:
            raise ValueError("ATR hesaplanamadı.")
        atr = float(atr_series.iloc[-1])
        entry = float(h4_df["close"].iloc[-1])

        # Kinetic composite score (new): integrates momentum, whale footprint, divergence and squeeze
        whale_score = float(getattr(state, "whale_score", 0.0) or 0.0)
        kinetic_score = _compute_kinetic_score(tf_metrics, atr, entry, whale_score, divergence_bonus)

        # ── 🛡️ DUAL-CONCURRENCE LEVEL GENERATOR (R03 Phase 2) ──
        # Hem LONG hem SHORT yönleri için seviyeleri bağımsız ve paralel olarak hesaplıyoruz!
        long_levels = calculate_trade_levels(
            h4_df,
            signal_direction="LONG",
            entry_price=entry,
            atr=atr,
            stop_loss_multiplier=float(atr_cfg["stop"]),
            t1_multiplier=float(atr_cfg["t1"]),
            t2_multiplier=float(atr_cfg["t2"]),
            t3_multiplier=float(atr_cfg["t3"]),
        )
        
        short_levels = calculate_trade_levels(
            h4_df,
            signal_direction="SHORT",
            entry_price=entry,
            atr=atr,
            stop_loss_multiplier=float(atr_cfg["stop"]),
            t1_multiplier=float(atr_cfg["t1"]),
            t2_multiplier=float(atr_cfg["t2"]),
            t3_multiplier=float(atr_cfg["t3"]),
        )
        
        direction_for_fib = "SHORT" if trade_type in {"STRONG_SELL_OR_SHORT"} else "LONG"
        fib_levels = _compute_fibonacci_levels(h4_df, direction=direction_for_fib, lookback=120)

        import json
        levels_payload: dict[str, Any] = {"LONG": long_levels, "SHORT": short_levels}
        if fib_levels:
            levels_payload["FIB"] = fib_levels
        levels_shuttle = f"[LEVELS_DATA] {json.dumps(levels_payload)}"

        historical_df = tf_dfs["1d"] if len(tf_dfs["1d"]) >= 20 else h4_df
        levels, near_hist_level = find_historical_levels(historical_df, lookback_days=500, threshold=0.02)
        historical_similarity_score, historical_pattern, pattern_outcome_bias = find_similar_cycles(
            historical_df, current_window=60, top_n=3
        )
        confidence = _calculate_confidence(
            {
                "timeframe_alignment_score": alignment_score,
                "composite_score": state.composite_score,
                "consensus_variance": getattr(state, "consensus_variance", 1.0),
                "divergence_daily": divergence_daily,
                "divergence_weekly": divergence_weekly,
                "historical_similarity_score": historical_similarity_score,
            }
        )

        # USDT dominance ters korelasyon teyidi: kripto long kararlarında ek güvenlik uyarısı.
        dominance_warnings: list[str] = []
        if _is_crypto(state.symbol):
            _, usdt_d = _extract_macro_dominance(state)
            if usdt_d is not None and usdt_d >= 7.0 and trade_type in {
                "ACCUMULATE_ZONE",
                "STRONG_LONG_TERM_ENTRY",
                "SHORT_TERM_BOUNCE_ONLY",
            }:
                dominance_warnings.append(
                    f"USDT.D yüksek ({usdt_d:.2f}) -> LONG sinyalinde likidite baskısı riski."
                )

        agent_print("QUANT_ENGINE", f"Bias 1w/1d/4h/1h={weekly_bias}/{daily_bias}/{h4_bias}/{h1_bias}", BLUE)
        agent_print("QUANT_ENGINE", f"Alignment={alignment_score:.2f} ({aligned_count}/4) | TradeType={trade_type}", GREEN)
        agent_print("QUANT_ENGINE", f"Divergence D/W={divergence_daily}/{divergence_weekly}", BLUE)
        agent_print("QUANT_ENGINE", f"Historical similarity={historical_similarity_score:.1f}/100 | {pattern_outcome_bias}", GREEN)

        return state.model_copy(
            update={
                "current_node": AgentNode.QUANT_ENGINE,
                "status": PipelineStatus.RUNNING,
                "quant_score": quant_score,
                "kinetic_score": kinetic_score,
                "entry_price": entry,
                "entry_zone_low": long_levels["entry_zone_low"],
                "entry_zone_high": long_levels["entry_zone_high"],
                "stop_loss": long_levels["stop_loss"],
                "take_profit": long_levels["t1"],
                "t1": long_levels["t1"],
                "t1_rr": long_levels["t1_rr"],
                "t2": long_levels["t2"],
                "t2_rr": long_levels["t2_rr"],
                "t3": long_levels["t3"],
                "t3_rr": long_levels["t3_rr"],
                "fib_382": fib_levels["fib_382"] if fib_levels else None,
                "fib_500": fib_levels["fib_500"] if fib_levels else None,
                "fib_618": fib_levels["fib_618"] if fib_levels else None,
                "invalidation_level": long_levels["invalidation_level"],
                "base_rr": long_levels["base_rr"],
                "risk_reward_ratio": long_levels["base_rr"],
                "confidence": confidence,
                "trade_type": trade_type,
                "timeframe_alignment_score": alignment_score,
                "timeframe_biases": biases,
                "divergence_daily": divergence_daily,
                "divergence_weekly": divergence_weekly,
                "historical_similarity_score": historical_similarity_score,
                "historical_pattern": historical_pattern + (" | Yakın S/R var" if near_hist_level else ""),
                "pattern_outcome_bias": pattern_outcome_bias,
                "ma_fallback_used": ma_fallback_used,
                "messages": [
                    f"[QUANT_ENGINE] tf_bias={biases} align={alignment_score:.2f} trade={trade_type} "
                    f"base_rr={long_levels['base_rr']} hist_score={historical_similarity_score:.1f} "
                    f"levels={len(levels)} ma_fallback={ma_fallback_used}",
                    *[f"[QUANT_ENGINE] WARN {w}" for w in dominance_warnings],
                    f"[DYNAMIC_TARGET] {long_levels['dynamic_trendline_target']}", # Geometrik hedef şatılı
                    levels_shuttle # Veri Şatılı güvenli mesaj kuyruğuna yükleniyor!
                ],
            }
        )

    except Exception as exc:
        msg = f"Quant Hata Olustu: {exc}"
        error_print(msg)
        return state.model_copy(
            update={
                "current_node": AgentNode.QUANT_ENGINE,
                "status": PipelineStatus.RUNNING,
                "fatal_error": msg,
                "messages": [f"[QUANT_ENGINE] ERROR {msg}"],
            }
        )
