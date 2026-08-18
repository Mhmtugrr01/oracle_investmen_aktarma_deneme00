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


def _significant_peaks(
    vals: np.ndarray,
    min_prominence: float,
    min_spacing: int,
) -> list[tuple[int, float]]:
    """Prominence tabanlı MAJÖR tepe/dip bulucu (saf NumPy, scipy YOK).

    Bir nokta ancak:
      - yerel maksimum ise, ve
      - prominence'ı (tepenin iki yanındaki su seviyesinin yüksek olanından
        yüksekliği) en az `min_prominence` ise,
      - aynı türden komşu önemli tepeyle arasında en az `min_spacing` bar varsa
    "MAJÖR" sayılır. Böylece mikro (birbirine 3-5 bar mesafedeki) tepelerden
    trend çizgisi çekilmez.

    Dönüş: [(idx, değer), ...] — büyüklük sırasında değil, zaman sırasında.
    """
    n = len(vals)
    peaks: list[tuple[int, float]] = []
    for i in range(1, n - 1):
        if not np.isfinite(vals[i]):
            continue
        if not (vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]):
            continue
        # Prominence: sağa ve sola yürü, KESİN daha yüksek noktaya kadar en derin
        # vadinin yüksek olanından tepe yüksekliğini çıkar (eşit yükseklik
        # boundary DEĞİLDİR — standart prominence `>` kullanır).
        left_base = vals[i]
        for j in range(i - 1, -1, -1):
            if vals[j] > vals[i]:
                break
            if vals[j] < left_base:
                left_base = vals[j]
        right_base = vals[i]
        for j in range(i + 1, n):
            if vals[j] > vals[i]:
                break
            if vals[j] < right_base:
                right_base = vals[j]
        base = max(left_base, right_base)
        if (vals[i] - base) >= min_prominence:
            peaks.append((i, float(vals[i])))
    # min_spacing: zaman sırasında birbirine çok yakın önemli tepelerde
    # daha belirgin olanı koru.
    filtered: list[tuple[int, float]] = []
    for p in peaks:
        if not filtered or p[0] - filtered[-1][0] >= min_spacing:
            filtered.append(p)
        elif p[1] > filtered[-1][1]:
            filtered[-1] = p
    return filtered


def find_significant_pivots(
    df: pd.DataFrame,
    prominence_mult: float = 1.5,
    min_spacing: int = 20,
    atr_period: int = 14,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """KURAL 1 — MAJÖR PİVOT (Significant Swing High/Low) tespiti.

    Sıradan (distance=3/5) küçük pencere KULLANILMAZ. Pivot ancak en az
    `prominence_mult * ATR(14)` kadarlık bir yükseliş/düşüş yaptıysa ve aynı
    türden önceki majör pivottan en az `min_spacing` bar uzaktaysa "majör"dir.
    Böylece trend çizgisi her TF'de kendi volatilitesine (ATR) göre ölçeklenir.

    Dönüş: (sig_highs, sig_lows) — [(idx, değer), ...] (zaman sırasında).
    """
    if df is None or len(df) < 30:
        return [], []
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    atr = compute_atr(df, atr_period)
    if atr is not None and len(atr) and np.isfinite(atr.iloc[-1]) and float(atr.iloc[-1]) > 0:
        atr_val = float(atr.iloc[-1])
    else:
        # ATR yoksa volatilite vekili: son 30 bar yüksek-düşük medyanı
        h30 = np.nanmedian(highs[-30:]) if len(highs) >= 30 else np.nanmax(highs)
        l30 = np.nanmedian(lows[-30:]) if len(lows) >= 30 else np.nanmin(lows)
        atr_val = max((h30 - l30), float(np.nanmax(highs) * 0.02))
    min_prom = atr_val * prominence_mult
    sig_highs = _significant_peaks(highs, min_prom, min_spacing)
    # Dipler: negatif alanda tepe arayıp geri çevir.
    sig_lows_neg = _significant_peaks(-lows, min_prom, min_spacing)
    sig_lows = [(i, -v) for i, v in sig_lows_neg]
    return sig_highs, sig_lows


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
    """Sinyal seviyelerini (giriş/stop/hedefler) MAJÖR pivotlardan üretir.

    LONG : stop son MAJÖR Swing_Low'un ALTINA (kayıp aşağıda, hedefler yukarıda).
    SHORT: stop son MAJÖR Swing_High'ın ÜSTÜNE (kayıp yukarıda, hedefler aşağıda).
    Hedefler R-çarpanı ile: T1=2R, T2=3.5R, T3=5R.
    """
    entry = float(df["close"].astype(float).iloc[-1])
    sig_highs, sig_lows = find_significant_pivots(df)
    if direction == "LONG":
        stop = float(sig_lows[-1][1]) if sig_lows else float(df["low"].astype(float).tail(20).min())
        if stop >= entry:  # geçersiz stop koruması
            stop = float(df["low"].astype(float).tail(20).min())
        risk = entry - stop
        if risk <= 0:
            risk = entry * 0.02
        t1 = entry + 2.0 * risk
        t2 = entry + 3.5 * risk
        t3 = entry + 5.0 * risk
    else:
        stop = float(sig_highs[-1][1]) if sig_highs else float(df["high"].astype(float).tail(20).max())
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
    prominence_mult: float = 1.5,
    min_spacing: int = 20,
) -> dict[str, Any]:
    """LONG kararı — 4 şartın HEPSİ sağlanmalı (KURAL 1 + 2 + 3 verisi).

    KURAL 1: Yalnızca MAJÖR pivotlar (ATR*1.5 prominence + min 20 bar aralık).
    KURAL 2: RSI, Fiyat'ın MAJÖR tepelerine AYNI İNDEKSTE hizalı okunur.
    KURAL 3: AI promptu için ATR / trend yaşı / kırılım gücü hesaplanır.

    Dönüş: {"signal", "direction", "reason", "dip_ok", "pos_div",
            "price_break", "rsi_break", "price_state", "rsi_state",
            "near", "last_high", "last_low", "price_tl", "current_price",
            "atr", "trend_age_bars", "body_atr_ratio", "volume_ratio", "levels"}
    """
    res: dict[str, Any] = {
        "signal": False, "direction": "LONG", "reason": "",
        "dip_ok": False, "pos_div": False,
        "price_break": False, "rsi_break": False,
        "price_state": "NO", "rsi_state": "NO", "near": False,
        "last_high": None, "last_low": None, "price_tl": None, "current_price": None,
        "atr": None, "trend_age_bars": None, "body_atr_ratio": None, "volume_ratio": None,
        "levels": None,
    }
    if df is None or len(df) < 60:
        res["reason"] = "Sinyal Yok: yetersiz veri"
        return res

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    n = len(df)
    current = float(close.iloc[-1])
    res["current_price"] = current

    rsi = compute_rsi(close, rsi_period)
    atr_series = compute_atr(df, 14)
    atr_last = float(atr_series.iloc[-1])
    if not np.isfinite(atr_last) or atr_last <= 0:
        atr_last = float(high.tail(lookback).max()) * 0.05
    res["atr"] = atr_last

    # KURAL 1 — MAJÖR pivotlar
    sig_highs, sig_lows = find_significant_pivots(
        df, prominence_mult=prominence_mult, min_spacing=min_spacing
    )
    res["last_high"] = sig_highs[-1][1] if sig_highs else None
    res["last_low"] = sig_lows[-1][1] if sig_lows else None
    res["trend_age_bars"] = (n - 1 - sig_highs[0][0]) if sig_highs else None

    # KURAL 3 — kırılım gücü verisi
    try:
        open_arr = df["open"].astype(float)
        body = float(close.iloc[-1]) - float(open_arr.iloc[-1])
        res["body_atr_ratio"] = (body / atr_last) if atr_last > 0 else None
        vol = df["volume"].astype(float) if "volume" in df.columns else None
        if vol is not None and len(vol) >= 20:
            vol_avg = float(vol.rolling(20).mean().iloc[-1])
            if np.isfinite(vol_avg) and vol_avg > 0:
                res["volume_ratio"] = float(vol.iloc[-1]) / vol_avg
    except Exception:  # noqa: BLE001
        pass

    # 1) Zirveden Uzaklık — DİNAMİK ATR
    hh100 = float(high.tail(lookback).max())
    dip_ok = current < (hh100 - 3.0 * atr_last)
    res["dip_ok"] = dip_ok

    # 2) RSI Pozitif Uyumsuzluk (son 2 MAJÖR dip + HİZALI RSI — KURAL 2)
    pos_div = False
    if len(sig_lows) >= 2:
        (i1, l1), (i2, l2) = sig_lows[-2], sig_lows[-1]
        price_ll = l2 < l1
        r1, r2 = rsi.iloc[i1], rsi.iloc[i2]
        if (not pd.isna(r1)) and (not pd.isna(r2)):
            rsi_hl = float(r2) > float(r1)
            pos_div = bool(price_ll and rsi_hl)
    res["pos_div"] = pos_div

    # 3) FİYAT Düşeni Kırma (son 2 MAJÖR tepe → düşen çizgi, TOLERANSLI)
    price_state = "NO"
    price_tl = None
    if len(sig_highs) >= 2:
        line = _trendline(sig_highs[-2], sig_highs[-1])
        if line is not None:
            m, b = line
            if m < 0:
                price_tl = m * (n - 1) + b
                price_state = _break_state(current, price_tl, tolerance, "up")
    if price_state == "NO" and len(sig_highs) >= 1:
        y_ref = sig_highs[-1][1]
        st = _break_state(current, y_ref, tolerance, "up")
        if st != "NO":
            price_state = st
            price_tl = y_ref
    res["price_break"] = price_state == "BREAK"
    res["price_state"] = price_state
    res["price_tl"] = price_tl

    # 4) RSI Düşeni Kırma — Fiyat'ın MAJÖR tepelerine HİZALI RSI (KURAL 2)
    rsi_state = "NO"
    if len(sig_highs) >= 2:
        rp1 = (sig_highs[-2][0], float(rsi.iloc[sig_highs[-2][0]]))
        rp2 = (sig_highs[-1][0], float(rsi.iloc[sig_highs[-1][0]]))
        if (not pd.isna(rp1[1])) and (not pd.isna(rp2[1])):
            line = _trendline(rp1, rp2)
            if line is not None:
                m, b = line
                if m < 0:
                    y_now = m * (n - 1) + b
                    rsi_state = _break_state(float(rsi.iloc[-1]), y_now, tolerance, "up")
    if rsi_state == "NO" and len(sig_highs) >= 1:
        r_last = float(rsi.iloc[sig_highs[-1][0]])
        if not pd.isna(r_last):
            st = _break_state(float(rsi.iloc[-1]), r_last, tolerance, "up")
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
    prominence_mult: float = 1.5,
    min_spacing: int = 20,
) -> dict[str, Any]:
    """SHORT kararı — 4 şartın HEPSİ sağlanmalı (KURAL 1 + 2 + 3 verisi)."""
    res: dict[str, Any] = {
        "signal": False, "direction": "SHORT", "reason": "",
        "peak_ok": False, "neg_div": False,
        "price_break": False, "rsi_break": False,
        "price_state": "NO", "rsi_state": "NO", "near": False,
        "last_high": None, "last_low": None, "price_tl": None, "current_price": None,
        "atr": None, "trend_age_bars": None, "body_atr_ratio": None, "volume_ratio": None,
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
    atr_series = compute_atr(df, 14)
    atr_last = float(atr_series.iloc[-1])
    if not np.isfinite(atr_last) or atr_last <= 0:
        atr_last = float(low.tail(lookback).min()) * 0.05
    res["atr"] = atr_last

    sig_highs, sig_lows = find_significant_pivots(
        df, prominence_mult=prominence_mult, min_spacing=min_spacing
    )
    res["last_high"] = sig_highs[-1][1] if sig_highs else None
    res["last_low"] = sig_lows[-1][1] if sig_lows else None
    res["trend_age_bars"] = (n - 1 - sig_lows[0][0]) if sig_lows else None

    try:
        open_arr = df["open"].astype(float)
        body = float(close.iloc[-1]) - float(open_arr.iloc[-1])
        res["body_atr_ratio"] = (body / atr_last) if atr_last > 0 else None
        vol = df["volume"].astype(float) if "volume" in df.columns else None
        if vol is not None and len(vol) >= 20:
            vol_avg = float(vol.rolling(20).mean().iloc[-1])
            if np.isfinite(vol_avg) and vol_avg > 0:
                res["volume_ratio"] = float(vol.iloc[-1]) / vol_avg
    except Exception:  # noqa: BLE001
        pass

    # 1) Dipten Uzaklık — DİNAMİK ATR
    ll100 = float(low.tail(lookback).min())
    peak_ok = current > (ll100 + 3.0 * atr_last)
    res["peak_ok"] = peak_ok

    # 2) RSI Negatif Uyumsuzluk (son 2 MAJÖR tepe + HİZALI RSI — KURAL 2)
    neg_div = False
    if len(sig_highs) >= 2:
        (i1, h1), (i2, h2) = sig_highs[-2], sig_highs[-1]
        price_hh = h2 > h1
        r1, r2 = rsi.iloc[i1], rsi.iloc[i2]
        if (not pd.isna(r1)) and (not pd.isna(r2)):
            rsi_lh = float(r2) < float(r1)
            neg_div = bool(price_hh and rsi_lh)
    res["neg_div"] = neg_div

    # 3) FİYAT Yükseleni Kırma (son 2 MAJÖR dip → yükselen çizgi, TOLERANSLI)
    price_state = "NO"
    price_tl = None
    if len(sig_lows) >= 2:
        line = _trendline(sig_lows[-2], sig_lows[-1])
        if line is not None:
            m, b = line
            if m > 0:
                price_tl = m * (n - 1) + b
                price_state = _break_state(current, price_tl, tolerance, "down")
    if price_state == "NO" and len(sig_lows) >= 1:
        y_ref = sig_lows[-1][1]
        st = _break_state(current, y_ref, tolerance, "down")
        if st != "NO":
            price_state = st
            price_tl = y_ref
    res["price_break"] = price_state == "BREAK"
    res["price_state"] = price_state
    res["price_tl"] = price_tl

    # 4) RSI Yükseleni Kırma — Fiyat'ın MAJÖR diplerine HİZALI RSI (KURAL 2)
    rsi_state = "NO"
    if len(sig_lows) >= 2:
        rt1 = (sig_lows[-2][0], float(rsi.iloc[sig_lows[-2][0]]))
        rt2 = (sig_lows[-1][0], float(rsi.iloc[sig_lows[-1][0]]))
        if (not pd.isna(rt1[1])) and (not pd.isna(rt2[1])):
            line = _trendline(rt1, rt2)
            if line is not None:
                m, b = line
                if m > 0:
                    y_now = m * (n - 1) + b
                    rsi_state = _break_state(float(rsi.iloc[-1]), y_now, tolerance, "down")
    if rsi_state == "NO" and len(sig_lows) >= 1:
        r_last = float(rsi.iloc[sig_lows[-1][0]])
        if not pd.isna(r_last):
            st = _break_state(float(rsi.iloc[-1]), r_last, tolerance, "down")
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
