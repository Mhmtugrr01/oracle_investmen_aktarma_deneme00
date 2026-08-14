"""
PROJECT OLYMPUS — core/asymmetric_engine.py
============================================
FAZ 2 — ASİMETRİK KANTİTATİF MOTOR (SAF Pandas/NumPy, ta-lib YOK)

Sadece LONG (AL) yönlü asimetrik fırsatları avlayan 3 motor:

  MOTOR 1 — Likidite Sweep + CHOCH (Piyasa Yapısı Kırılımı)
      Swing_Low altına iğne (Sweep) + 1-3 bar içinde Swing_High üzerinde
      GÖVDELİ kapanış (CHOCH) = kurumsal dönüş kanıtı.

  MOTOR 2 — Kutsal Kase (RSI Divergence + Trendline Breakout)
      Fiyat Lower Low yaparken RSI Higher Low (pozitif uyumsuzluk) +
      RSI'nın son iki tepesinden geçen DÜŞEN trend çizgisinin (y=mx+b)
      YUKARI kırılması = momentum patlaması.

  MOTOR 3 — Göreli Güç (RS_Beta) + USDT.D Makro Onayı
      USDT.D son 5 bar eğimi yukarıysa kripto LONG REDDEDİLİR (para dolara
      kaçıyor); düşüyor/yataysa ONAY. Aynı anda çok varlık sinyal verirse
      RS_Score = Varlık_Getirisi − BTC_Getirisi ile lider 1-2 varlık seçilir.

Tüm fonksiyonlar AĞSIZDIR (offline) — saf veri işleme, test edilebilir.
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
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


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


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR 1 — LİKİDİTE SWEEP + CHOCH (LONG)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_sweep_choch_long(
    df: pd.DataFrame,
    lookback: int = 12,
    pivot_window: int = 5,
    choch_window: int = 3,
) -> dict[str, Any]:
    """Motor 1 — LONG Sweep + CHOCH.

    Şartlar:
      1. Swing_Low: son `lookback` barda fraktal bir lokal dip.
      2. Sweep    : bir mum `Swing_Low` ALTINA iğne atar (Low < Swing_Low)
                    ama kapanışını Swing_Low ALTINDA yapmaz (Close > Swing_Low).
      3. CHOCH    : Sweep'ten sonraki 1-3 bar içinde Close, bir önceki
                    Swing_High ÜZERİNDE ve GÖVDELİ (Close > Open) olur.

    Dönüş: {"sweep", "choch", "signal", "reason", "swing_low", "swing_high"}
    """
    empty = {"sweep": False, "choch": False, "signal": False,
             "reason": "yetersiz veri", "swing_low": None, "swing_high": None}
    if df is None or len(df) < lookback + choch_window + 2:
        return empty

    closes = df["close"].astype(float).values
    opens = df["open"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)

    # Base penceresi: sweep(1 bar) + CHOCH(`choch_window` bar) bölgesi HARİÇ tutulur.
    # Swing_Low  = base'deki EN DÜŞÜK Low
    # Swing_High = base'deki EN YÜKSEK High
    window_end = max(1, n - (choch_window + 1))
    window_start = max(0, window_end - lookback)
    base = df.iloc[window_start:window_end]
    if len(base) < 10:
        return {**empty, "reason": "yetersiz base penceresi"}
    swing_low = float(base["low"].min())
    swing_high = float(base["high"].max())

    # Sweep: son `choch_window + 1` barda bir mum Swing_Low ALTINA iğne atar
    # ve kapanışını Swing_Low ALTINDA yapmaz (Close > Swing_Low).
    sweep_idx: int | None = None
    for i in range(window_end, n):
        if lows[i] < swing_low and closes[i] > swing_low:
            sweep_idx = i
            break
    swept = sweep_idx is not None

    # CHOCH: Sweep'ten sonraki 1-3 barda Close > Swing_High (GÖVDELİ, Close>Open)
    choch = False
    if swept and sweep_idx is not None:
        for j in range(sweep_idx + 1, min(n, sweep_idx + 1 + choch_window)):
            if closes[j] > swing_high and closes[j] > opens[j]:
                choch = True
                break

    signal = bool(swept and choch)
    reason = (
        "Sweep+CHOCH" if signal
        else ("sweep yok" if not swept else "CHOCH kırılımı yok")
    )
    return {
        "sweep": swept,
        "choch": choch,
        "signal": signal,
        "reason": reason,
        "swing_low": swing_low,
        "swing_high": swing_high,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR 2 — KUTSAL KASE: RSI DIVERGENCE + TRENDLINE BREAKOUT (LONG)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_rsi_divergence_trendline(
    df: pd.DataFrame,
    rsi_period: int = 14,
    pivot_window: int = 5,
    lookback: int = 40,
) -> dict[str, Any]:
    """Motor 2 — Pozitif uyumsuzluk + RSI düşen trend çizgisi kırılımı.

    Şartlar:
      1. Pozitif uyumsuzluk: son iki Swing_Low'da fiyat Lower Low yaparken
         aynı noktalardaki RSI Higher Low yapar.
      2. RSI Breakout: RSI'ın son iki TEPESİNDEN geçen düşen doğru
         (y = mx + b) hesaplanır; güncel RSI bu direnci YUKARI kırar.

    Dönüş: {"divergence", "breakout", "signal", "reason"}
    """
    empty = {"divergence": False, "breakout": False, "signal": False, "reason": "yetersiz veri"}
    if df is None or len(df) < lookback:
        return empty

    # RSI TAM seriden hesaplanır (Wilder warm-up'ı korunur), sonra pencereye hizalanır.
    # Truncated seride RSI hesaplamak RSI değerlerini bozar (yanlış tepe/dip).
    full_rsi = compute_rsi(df["close"].astype(float), rsi_period)
    tail_start = max(0, len(df) - lookback)
    tail = df.iloc[tail_start:].reset_index(drop=True)
    rsi = full_rsi.iloc[tail_start:].reset_index(drop=True)
    rsi_vals = rsi.values
    n = len(tail)
    if n < pivot_window * 2 + 4:
        return empty

    def _safe_max(arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        return float(arr.max()) if len(arr) else -np.inf

    # 1) Pozitif uyumsuzluk (fiyat dip pivotları + RSI orada)
    _, fl = find_swing_pivots(tail, window=pivot_window)
    divergence = False
    if len(fl) >= 2:
        prev_idx, prev_low = fl[-2]
        last_idx, last_low = fl[-1]
        price_ll = last_low < prev_low
        rsi_prev = rsi.iloc[prev_idx]
        rsi_last = rsi.iloc[last_idx]
        if not (pd.isna(rsi_prev) or pd.isna(rsi_last)):
            rsi_prev = float(rsi_prev)
            rsi_last = float(rsi_last)
            rsi_hl = rsi_last > rsi_prev
            divergence = bool(price_ll and rsi_hl)

    # 2) RSI trendline kırılımı (son iki RSI tepe noktasından geçen doğru)
    peaks: list[tuple[int, float]] = []
    for i in range(pivot_window, n - pivot_window):
        left = _safe_max(rsi_vals[i - pivot_window : i])
        right = _safe_max(rsi_vals[i + 1 : i + 1 + pivot_window])
        if np.isfinite(rsi_vals[i]) and rsi_vals[i] >= left and rsi_vals[i] >= right:
            if not peaks or i - peaks[-1][0] >= pivot_window or rsi_vals[i] > peaks[-1][1] + 1e-9:
                peaks.append((i, float(rsi_vals[i])))

    breakout = False
    if len(peaks) >= 2:
        (x1, y1), (x2, y2) = peaks[-2], peaks[-1]
        if x2 > x1:
            m = (y2 - y1) / (x2 - x1)
            if m < 0:  # DÜŞEN trend çizgisi — sadece bu anlamlı
                b = y1 - m * x1
                for cx in (n - 1, n - 2):  # güncel veya bir önceki bar
                    if 0 <= cx < n:
                        trendline_rsi = m * cx + b
                        if rsi_vals[cx] > trendline_rsi:
                            breakout = True
                            break

    signal = bool(divergence and breakout)
    reason = (
        "RSI Breakout" if signal
        else ("divergence yok" if not divergence else "RSI trendline kırılımı yok")
    )
    return {"divergence": divergence, "breakout": breakout, "signal": signal, "reason": reason}


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR 3 — GÖRELİ GÜÇ (RS_Beta) + USDT.D MAKRO ONAYI
# ═══════════════════════════════════════════════════════════════════════════════

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
    """RS_Score = Varlık_Getirisi − BTC_Getirisi (son 15 bar).

    Pozitif RS_Score = varlık BTC'den daha güçlü (BTC düşerken en az düşen).
    """
    return float(asset_returns - btc_returns)


# ═══════════════════════════════════════════════════════════════════════════════
#  BİRLEŞİK ASİMETRİK LONG SİNYALİ
# ═══════════════════════════════════════════════════════════════════════════════

def asymmetric_long_signal(
    df: pd.DataFrame,
    usdt_d_series: Any | None = None,
    btc_close: Any | None = None,
    asset_close: Any | None = None,
) -> dict[str, Any]:
    """Üç motorun birleşik LONG kararı.

    Sinyal ancak: Motor1 (Sweep+CHOCH) VE Motor2 (RSI Breakout) VE
    Motor3 (USDT.D onayı) aynı anda doğruysa üretilir.

    Dönüş:
      {"signal", "reason", "macro_approved", "macro_reason",
       "motors": {"sweep_choch":..., "rsi_div":...}, "rs_score"}
    """
    m1 = detect_sweep_choch_long(df)
    m2 = detect_rsi_divergence_trendline(df)
    m3 = usdt_d_macro_filter(usdt_d_series)

    reasons: list[str] = []
    if m1["signal"]:
        reasons.append("Sweep+CHOCH")
    if m2["signal"]:
        reasons.append("RSI Breakout")

    signal = bool(m1["signal"] and m2["signal"] and m3["approved"])
    reason = " & ".join(reasons) if reasons else "yapısal sinyal yok"

    rs_score = None
    if btc_close is not None and asset_close is not None:
        try:
            btc_arr = pd.Series(btc_close).astype(float).dropna().tail(15)
            ast_arr = pd.Series(asset_close).astype(float).dropna().tail(15)
            if len(btc_arr) >= 2 and len(ast_arr) >= 2:
                btc_ret = (float(btc_arr.iloc[-1]) / float(btc_arr.iloc[0])) - 1.0
                ast_ret = (float(ast_arr.iloc[-1]) / float(ast_arr.iloc[0])) - 1.0
                rs_score = relative_strength_score(ast_ret, btc_ret)
        except Exception:  # noqa: BLE001
            rs_score = None

    return {
        "signal": signal,
        "reason": reason,
        "macro_approved": m3["approved"],
        "macro_reason": m3["reason"],
        "motors": {"sweep_choch": m1, "rsi_div": m2},
        "rs_score": rs_score,
    }
