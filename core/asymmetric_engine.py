"""
PROJECT OLYMPUS — core/asymmetric_engine.py
============================================
FAZ 5 (KÖK HÜCRE REVİZYONU) — KATI VETO MOTORU (LONG & SHORT)

Bu motor, bir varlığın fırsat olarak onaylanması için TEK VE MUTLAK karar
mekanizmasıdır. Eski "puan toplama" (composite/quant/macro bonus) MANTIĞI
TAMAMEN SÖKÜLMÜŞTÜR: Bu motor "Sinyal Yok" derse varlık ANINDA elenir.

Tüm fonksiyonlar AĞSIZDIR (offline) — saf Pandas/NumPy, test edilebilir.

  🟢 LONG  (4 şart AYNI ANDA):
     1. Zirveden Uzaklık : current_price < 100_bar_high * 0.70
     2. RSI Pozitif Uyumsuzluk : fiyat Lower Low + RSI Higher Low
     3. FİYAT Düşeni Kırma : son 2 Swing_High'tan geçen DÜŞEN trend
        çizgisini (m<0) GÖVDELİ kapanış YUKARI kırar (Close > trendline)
     4. RSI Düşeni Kırma : RSI son 2 tepesinden geçen düşen çizgiyi
        YUKARI kırar

  🔴 SHORT (4 şart AYNI ANDA):
     1. Dipten Uzaklık  : current_price > 100_bar_low * 1.30
     2. RSI Negatif Uyumsuzluk : fiyat Higher High + RSI Lower High
     3. FİYAT Yükseleni Kırma : son 2 Swing_Low'dan geçen YÜKSELEN trend
        çizgisini (m>0) kapanış AŞAĞI kırar (Close < trendline)
     4. RSI Yükseleni Kırma : RSI son 2 dibinden geçen yükselen çizgiyi
        AŞAĞI kırar

Trend çizgileri numpy ile gerçek `y = mx + b` denklemi olarak kurulur.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  YARDIMCILAR (saf)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI — saf pandas (pandas_ta/ta-lib yok).

    Isınma dönemi (ilk `period` bar) NaN'dır — 50 ile doldurulmaz ki erken
    bar'lar yanlış tepe/dip gibi görünmesin.
    """
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    # Saf yükselişte (kayıp yok) RSI=100; saf düşüşte (kazanç yok) RSI=0;
    # tamamen düz (her ikisi de 0) → RSI=50 (belirsiz).
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR — saf pandas (True Range + EWM).

    KURAL 1 — dinamik mesafe: sabit yüzde yerine ATR ile ölçeklenen
    zirve/dip uzaklığı kullanılır (1h ile 1w kendi volatilitesine göre).
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


def _break_state(
    current: float,
    trendline_y: float,
    tolerance: float,
    direction: str,
) -> str:
    """KURAL 2 — kırılım durumu: "BREAK" | "NEAR" | "NO".

    direction "up"  → current > trendline = BREAK (LONG direnç kırılımı).
    direction "down"→ current < trendline = BREAK (SHORT destek kırılımı).
    Kırılım yoksa ama mesafe %tolerance içindeyse → "NEAR" (EŞİKTE → AI veto).
    """
    if trendline_y is None or not np.isfinite(trendline_y) or abs(trendline_y) < 1e-12:
        return "NO"
    dist = abs(current - trendline_y)
    ratio = dist / max(abs(trendline_y), 1e-9)
    if direction == "up":
        if current > trendline_y:
            return "BREAK"
        if ratio <= tolerance:
            return "NEAR"
    else:
        if current < trendline_y:
            return "BREAK"
        if ratio <= tolerance:
            return "NEAR"
    return "NO"


def find_swing_pivots(
    df: pd.DataFrame, window: int = 5
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Fraktal pivot bulucu (indeksler girdi df'ye göre 0-tabanlı).

    Swing High : high[i], sol/sağ `window` barın max'ı ise.
    Swing Low  : low[i], sol/sağ `window` barın min'i ise.

    Düz/düşey barlarda çoklu pivot üretmemek için en az `window` bar aralıklı
    ve sadece daha belirgin yeni seviyeler kaydedilir.

    Dönüş: (swing_highs, swing_lows) — [(idx, değer), ...]
    """
    if df is None or len(df) < window * 2 + 1:
        return [], []
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    for i in range(window, n - window):
        left_h = np.nanmax(highs[i - window : i])
        right_h = np.nanmax(highs[i + 1 : i + 1 + window])
        if highs[i] >= left_h and highs[i] >= right_h:
            if not swing_highs or i - swing_highs[-1][0] >= window or highs[i] > swing_highs[-1][1] + 1e-9:
                swing_highs.append((i, float(highs[i])))
        left_l = np.nanmin(lows[i - window : i])
        right_l = np.nanmin(lows[i + 1 : i + 1 + window])
        if lows[i] <= left_l and lows[i] <= right_l:
            if not swing_lows or i - swing_lows[-1][0] >= window or lows[i] < swing_lows[-1][1] - 1e-9:
                swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


def _rsi_swings(
    rsi: pd.Series, window: int = 5
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """RSI serisinin fraktal tepelerini/diplerini bulur (NaN güvenli).

    Dönüş: (rsi_peaks, rsi_troughs) — [(idx, değer), ...] (mutlak indeks).
    """
    vals = rsi.astype(float).values
    n = len(vals)
    peaks: list[tuple[int, float]] = []
    troughs: list[tuple[int, float]] = []
    for i in range(window, n - window):
        if not np.isfinite(vals[i]):
            continue
        left_h = vals[i - window : i]
        right_h = vals[i + 1 : i + 1 + window]
        left_h = left_h[np.isfinite(left_h)]
        right_h = right_h[np.isfinite(right_h)]
        if len(left_h) and len(right_h) and vals[i] >= left_h.max() and vals[i] >= right_h.max():
            if not peaks or i - peaks[-1][0] >= window or vals[i] > peaks[-1][1] + 1e-9:
                peaks.append((i, float(vals[i])))
        left_l = vals[i - window : i]
        right_l = vals[i + 1 : i + 1 + window]
        left_l = left_l[np.isfinite(left_l)]
        right_l = right_l[np.isfinite(right_l)]
        if len(left_l) and len(right_l) and vals[i] <= left_l.min() and vals[i] <= right_l.min():
            if not troughs or i - troughs[-1][0] >= window or vals[i] < troughs[-1][1] - 1e-9:
                troughs.append((i, float(vals[i])))
    return peaks, troughs


def _trendline(
    p1: tuple[int, float], p2: tuple[int, float]
) -> tuple[float, float] | None:
    """İki noktadan geçen doğrunun (m, b) katsayılarını döndürür."""
    x1, y1 = p1
    x2, y2 = p2
    if x2 <= x1:
        return None
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def _build_levels(direction: str, df: pd.DataFrame) -> dict[str, Any]:
    """Sinyal seviyelerini (giriş/stop/hedefler) yapısal pivotlardan üretir.

    LONG : stop son Swing_Low'un ALTINA (kayıp aşağıda, hedefler yukarıda).
    SHORT: stop son Swing_High'ın ÜSTÜNE (kayıp yukarıda, hedefler aşağıda).
    Hedefler R-çarpanı ile: T1=2R, T2=3.5R, T3=5R.
    """
    entry = float(df["close"].astype(float).iloc[-1])
    swing_highs, swing_lows = find_swing_pivots(df, window=5)
    if direction == "LONG":
        stop = float(swing_lows[-1][1]) if swing_lows else float(df["low"].astype(float).tail(20).min())
        if stop >= entry:  # geçersiz stop koruması
            stop = float(df["low"].astype(float).tail(20).min())
        risk = entry - stop
        if risk <= 0:
            risk = entry * 0.02
        t1 = entry + 2.0 * risk
        t2 = entry + 3.5 * risk
        t3 = entry + 5.0 * risk
    else:
        stop = float(swing_highs[-1][1]) if swing_highs else float(df["high"].astype(float).tail(20).max())
        if stop <= entry:
            stop = float(df["high"].astype(float).tail(20).max())
        risk = stop - entry
        if risk <= 0:
            risk = entry * 0.02
        t1 = entry - 2.0 * risk
        t2 = entry - 3.5 * risk
        t3 = entry - 5.0 * risk
    return {
        "entry": round(entry, 6),
        "stop": round(stop, 6),
        "t1": round(t1, 6),
        "t2": round(t2, 6),
        "t3": round(t3, 6),
        "base_rr": 2.0,
    }


def usdt_d_macro_filter(
    usdt_d_series: Any | None, slope_bars: int = 5
) -> dict[str, Any]:
    """USDT.D son `slope_bars` barlık doğrusal eğim (np.polyfit).

    Eğim > +1e-9 → REDDET (para dolara kaçıyor → kripto LONG yok).
    Eğim <= 0     → ONAY (düşüyor/yatay → kripto LONG serbest).
    Veri yok      → ONAY (bilinmeyen, bloklama yok — ancak düşük güven).
    """
    if usdt_d_series is None:
        return {"approved": True, "slope": 0.0, "reason": "USDT.D verisi yok (nötr onay)"}
    vals = pd.Series(usdt_d_series).astype(float).dropna()
    if len(vals) < slope_bars:
        return {"approved": True, "slope": 0.0, "reason": "USDT.D serisi kısa (nötr onay)"}
    vals = vals.tail(slope_bars).values
    x = np.arange(len(vals), dtype=float)
    slope = float(np.polyfit(x, vals, 1)[0])
    approved = slope <= 1e-9
    return {
        "approved": approved,
        "slope": slope,
        "reason": "USDT.D ONAY (düşüş/yatay)" if approved
        else f"USDT.D RED (eğim {slope:+.4f} yukarı — para dolara kaçıyor)",
    }


def relative_strength_score(
    asset_returns: float, btc_returns: float
) -> float:
    """RS_Score = Varlık_Getirisi − BTC_Getirisi (son 15 bar)."""
    return float(asset_returns - btc_returns)


def _relative_strength(
    btc_close: Any | None, asset_close: Any | None
) -> float | None:
    if btc_close is None or asset_close is None:
        return None
    try:
        btc_arr = pd.Series(btc_close).astype(float).dropna().tail(15)
        ast_arr = pd.Series(asset_close).astype(float).dropna().tail(15)
        if len(btc_arr) >= 2 and len(ast_arr) >= 2:
            btc_ret = (float(btc_arr.iloc[-1]) / float(btc_arr.iloc[0])) - 1.0
            ast_ret = (float(ast_arr.iloc[-1]) / float(ast_arr.iloc[0])) - 1.0
            return relative_strength_score(ast_ret, btc_ret)
    except Exception:  # noqa: BLE001
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  🟢 MOTOR — LONG (4 şart)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_long_signal(
    df: pd.DataFrame,
    rsi_period: int = 14,
    pivot_window: int = 5,
    lookback: int = 100,
    tolerance: float = 0.015,
) -> dict[str, Any]:
    """LONG kararı — 4 şartın HEPSİ sağlanmalı (KURAL 1 + 2).

    KURAL 1: Zirveden uzaklık sabit yüzde DEĞİL — `3 * ATR(14)` ile ölçeklenir.
    KURAL 2: Trend kırılımı %1.5 toleranslıdır (BREAK veya NEAR sayılır).

    Dönüş: {"signal", "direction", "reason", "dip_ok", "pos_div",
            "price_break", "rsi_break", "price_state", "rsi_state",
            "near", "last_high", "last_low", "price_tl", "current_price", "levels"}
    """
    res: dict[str, Any] = {
        "signal": False,
        "direction": "LONG",
        "reason": "",
        "dip_ok": False,
        "pos_div": False,
        "price_break": False,
        "rsi_break": False,
        "price_state": "NO",
        "rsi_state": "NO",
        "near": False,
        "last_high": None,
        "last_low": None,
        "price_tl": None,
        "current_price": None,
        "levels": None,
    }
    if df is None or len(df) < 60:
        res["reason"] = "Sinyal Yok: yetersiz veri"
        return res

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    n = len(df)
    current = float(close.iloc[-1])
    res["current_price"] = current

    rsi = compute_rsi(close, rsi_period)
    swing_highs, swing_lows = find_swing_pivots(df, window=pivot_window)

    res["last_high"] = swing_highs[-1][1] if swing_highs else None
    res["last_low"] = swing_lows[-1][1] if swing_lows else None

    # 1) Zirveden Uzaklık — DİNAMİK ATR (sabit %30 SİLİNDİ)
    hh100 = float(high.tail(lookback).max())
    atr_last = float(compute_atr(df, 14).iloc[-1])
    if not np.isfinite(atr_last) or atr_last <= 0:
        atr_last = hh100 * 0.05  # ATR yoksa volatilite vekili (güvenli fallback)
    dip_ok = current < (hh100 - 3.0 * atr_last)
    res["dip_ok"] = dip_ok

    # 2) RSI Pozitif Uyumsuzluk (fiyat Lower Low + RSI Higher Low)
    pos_div = False
    if len(swing_lows) >= 2:
        (i1, l1), (i2, l2) = swing_lows[-2], swing_lows[-1]
        price_ll = l2 < l1
        r1, r2 = rsi.iloc[i1], rsi.iloc[i2]
        if (not pd.isna(r1)) and (not pd.isna(r2)):
            rsi_hl = float(r2) > float(r1)
            pos_div = bool(price_ll and rsi_hl)
    res["pos_div"] = pos_div

    # 3) FİYAT Düşeni Kırma — TOLERANSLI (son 2 Swing_High → düşen çizgi)
    price_state = "NO"
    price_tl = None
    if len(swing_highs) >= 2:
        line = _trendline(swing_highs[-2], swing_highs[-1])
        if line is not None:
            m, b = line
            if m < 0:  # DÜŞEN trend çizgisi
                price_tl = m * (n - 1) + b
                price_state = _break_state(current, price_tl, tolerance, "up")
    if price_state == "NO" and len(swing_highs) >= 1:
        y_ref = swing_highs[-1][1]  # en son lokal tepe (yatay direnç)
        st = _break_state(current, y_ref, tolerance, "up")
        if st != "NO":
            price_state = st
            price_tl = y_ref
    res["price_break"] = price_state == "BREAK"
    res["price_state"] = price_state
    res["price_tl"] = price_tl

    # 4) RSI Düşeni Kırma — TOLERANSLI
    rsi_peaks, _ = _rsi_swings(rsi, pivot_window)
    rsi_state = "NO"
    if len(rsi_peaks) >= 2:
        line = _trendline(rsi_peaks[-2], rsi_peaks[-1])
        if line is not None:
            m, b = line
            if m < 0:
                y_now = m * (n - 1) + b
                rsi_state = _break_state(float(rsi.iloc[-1]), y_now, tolerance, "up")
    if rsi_state == "NO" and len(rsi_peaks) >= 1:
        st = _break_state(float(rsi.iloc[-1]), rsi_peaks[-1][1], tolerance, "up")
        if st != "NO":
            rsi_state = st
    res["rsi_break"] = rsi_state == "BREAK"
    res["rsi_state"] = rsi_state

    price_ok = price_state in ("BREAK", "NEAR")
    rsi_ok = rsi_state in ("BREAK", "NEAR")
    res["near"] = (price_state == "NEAR") or (rsi_state == "NEAR")
    res["signal"] = bool(dip_ok and pos_div and price_ok and rsi_ok)

    if res["signal"]:
        base = "Fiyat Düşeni Kırdı + RSI Pozitif Uyumsuzluk"
        res["reason"] = f"POTANSİYEL KIRILIM: {base}" if res["near"] else base
        res["levels"] = _build_levels("LONG", df)
    else:
        missing = []
        if not dip_ok:
            missing.append("zirveye yakın (3*ATR dip şartı yok)")
        if not pos_div:
            missing.append("RSI pozitif uyumsuzluk yok")
        if not price_ok:
            missing.append("fiyat düşen trend kırılımı yok")
        if not rsi_ok:
            missing.append("RSI düşen trend kırılımı yok")
        res["reason"] = "Sinyal Yok: " + "; ".join(missing)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
#  🔴 MOTOR — SHORT (4 şart)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_short_signal(
    df: pd.DataFrame,
    rsi_period: int = 14,
    pivot_window: int = 5,
    lookback: int = 100,
    tolerance: float = 0.015,
) -> dict[str, Any]:
    """SHORT kararı — 4 şartın HEPSİ sağlanmalı (KURAL 1 + 2)."""
    res: dict[str, Any] = {
        "signal": False,
        "direction": "SHORT",
        "reason": "",
        "peak_ok": False,
        "neg_div": False,
        "price_break": False,
        "rsi_break": False,
        "price_state": "NO",
        "rsi_state": "NO",
        "near": False,
        "last_high": None,
        "last_low": None,
        "price_tl": None,
        "current_price": None,
        "levels": None,
    }
    if df is None or len(df) < 60:
        res["reason"] = "Sinyal Yok: yetersiz veri"
        return res

    close = df["close"].astype(float)
    low = df["low"].astype(float)
    n = len(df)
    current = float(close.iloc[-1])
    res["current_price"] = current

    rsi = compute_rsi(close, rsi_period)
    swing_highs, swing_lows = find_swing_pivots(df, window=pivot_window)

    res["last_high"] = swing_highs[-1][1] if swing_highs else None
    res["last_low"] = swing_lows[-1][1] if swing_lows else None

    # 1) Dipten Uzaklık — DİNAMİK ATR (sabit %30 SİLİNDİ)
    ll100 = float(low.tail(lookback).min())
    atr_last = float(compute_atr(df, 14).iloc[-1])
    if not np.isfinite(atr_last) or atr_last <= 0:
        atr_last = ll100 * 0.05
    peak_ok = current > (ll100 + 3.0 * atr_last)
    res["peak_ok"] = peak_ok

    # 2) RSI Negatif Uyumsuzluk (fiyat Higher High + RSI Lower High)
    neg_div = False
    if len(swing_highs) >= 2:
        (i1, h1), (i2, h2) = swing_highs[-2], swing_highs[-1]
        price_hh = h2 > h1
        r1, r2 = rsi.iloc[i1], rsi.iloc[i2]
        if (not pd.isna(r1)) and (not pd.isna(r2)):
            rsi_lh = float(r2) < float(r1)
            neg_div = bool(price_hh and rsi_lh)
    res["neg_div"] = neg_div

    # 3) FİYAT Yükseleni Kırma — TOLERANSLI
    price_state = "NO"
    price_tl = None
    if len(swing_lows) >= 2:
        line = _trendline(swing_lows[-2], swing_lows[-1])
        if line is not None:
            m, b = line
            if m > 0:  # YÜKSELEN trend çizgisi
                price_tl = m * (n - 1) + b
                price_state = _break_state(current, price_tl, tolerance, "down")
    if price_state == "NO" and len(swing_lows) >= 1:
        y_ref = swing_lows[-1][1]  # en son lokal dip (yatay destek)
        st = _break_state(current, y_ref, tolerance, "down")
        if st != "NO":
            price_state = st
            price_tl = y_ref
    res["price_break"] = price_state == "BREAK"
    res["price_state"] = price_state
    res["price_tl"] = price_tl

    # 4) RSI Yükseleni Kırma — TOLERANSLI
    _, rsi_troughs = _rsi_swings(rsi, pivot_window)
    rsi_state = "NO"
    if len(rsi_troughs) >= 2:
        line = _trendline(rsi_troughs[-2], rsi_troughs[-1])
        if line is not None:
            m, b = line
            if m > 0:
                y_now = m * (n - 1) + b
                rsi_state = _break_state(float(rsi.iloc[-1]), y_now, tolerance, "down")
    if rsi_state == "NO" and len(rsi_troughs) >= 1:
        st = _break_state(float(rsi.iloc[-1]), rsi_troughs[-1][1], tolerance, "down")
        if st != "NO":
            rsi_state = st
    res["rsi_break"] = rsi_state == "BREAK"
    res["rsi_state"] = rsi_state

    price_ok = price_state in ("BREAK", "NEAR")
    rsi_ok = rsi_state in ("BREAK", "NEAR")
    res["near"] = (price_state == "NEAR") or (rsi_state == "NEAR")
    res["signal"] = bool(peak_ok and neg_div and price_ok and rsi_ok)

    if res["signal"]:
        base = "Fiyat Yükseleni Kırdı + RSI Negatif Uyumsuzluk"
        res["reason"] = f"POTANSİYEL KIRILIM: {base}" if res["near"] else base
        res["levels"] = _build_levels("SHORT", df)
    else:
        missing = []
        if not peak_ok:
            missing.append("dibe yakın (3*ATR zirve şartı yok)")
        if not neg_div:
            missing.append("RSI negatif uyumsuzluk yok")
        if not price_ok:
            missing.append("fiyat yükselen trend kırılımı yok")
        if not rsi_ok:
            missing.append("RSI yükselen trend kırılımı yok")
        res["reason"] = "Sinyal Yok: " + "; ".join(missing)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
#  BİRLEŞİK KARAR (KATI VETO)
# ═══════════════════════════════════════════════════════════════════════════════

def asymmetric_signal(
    df: pd.DataFrame,
    usdt_d_series: Any | None = None,
    btc_close: Any | None = None,
    asset_close: Any | None = None,
) -> dict[str, Any]:
    """Birleşik LONG/SHORT kararı — KATI VETO.

    Yalnızca LONG veya SHORT motorunun 4 şartı birden sağlanırsa sinyal üretir.
    İkisi birden tetiklenirse LONG önceliklidir (tek yön kuralı).
    Hiçbiri tetiklenmezse `signal=False` + `direction=None` + `reason="Sinyal Yok"`.

    Dönüş:
      {"direction", "signal", "reason", "long", "short",
       "macro_approved", "macro_reason", "rs_score", "levels"}
    """
    long_res = detect_long_signal(df)
    short_res = detect_short_signal(df)
    macro = usdt_d_macro_filter(usdt_d_series)

    direction: str | None = None
    reason = ""
    levels: dict[str, Any] | None = None
    near = False
    if long_res["signal"]:
        direction = "LONG"
        reason = long_res["reason"]
        levels = long_res["levels"]
        near = long_res["near"]
    elif short_res["signal"]:
        direction = "SHORT"
        reason = short_res["reason"]
        levels = short_res["levels"]
        near = short_res["near"]
    else:
        direction = None
        reason = "Sinyal Yok"
        levels = None

    return {
        "direction": direction,
        "signal": direction is not None,
        "reason": reason,
        "near": near,
        "long": long_res,
        "short": short_res,
        "macro_approved": bool(macro["approved"]),
        "macro_reason": macro["reason"],
        "rs_score": _relative_strength(btc_close, asset_close),
        "levels": levels,
    }
