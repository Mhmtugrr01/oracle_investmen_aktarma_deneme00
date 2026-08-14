# -*- coding: utf-8 -*-
"""
FAZ 2 KANIT SCRIPTİ — Asimetrik Kantitatif Motor (Mock Data)
============================================================
Sahte OHLCV, tam olarak şunu simüle eder:
  1) Swing Low altına iğne (Sweep) + 1-3 bar içinde Swing High üzerinde GÖVDELİ
     kapanış (CHOCH)                    → Motor 1
  2) Fiyat Lower Low yaparken RSI Higher Low (pozitif uyumsuzluk) + RSI düşen
     trend çizgisini yukarı kırıyor     → Motor 2
  3) USDT.D düşüyor (makro onay)        → Motor 3

Beklenen çıktı:
  [SCANNER] FIRSAT ONAYLANDI: MOCK_COIN -> LONG_FIRSAT (Sebep: Sweep+CHOCH & RSI Breakout)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.asymmetric_engine import (
    asymmetric_long_signal,
    detect_sweep_choch_long,
    detect_rsi_divergence_trendline,
    usdt_d_macro_filter,
)


def build_mock_ohlcv() -> pd.DataFrame:
    """Sweep + CHOCH + RSI breakout grafiğini simüle eden sahte OHLCV.

    Son 3 barda: [sweep, choch1, choch2] — sweep Swing_Low altına iğne atar,
    sonraki 1-2 bar Swing_High üzerinde GÖVDELİ kapanış yapar (CHOCH).
    """
    n = 46
    close = np.empty(n)
    # 0-10 : yükseliş (95 → 107)
    close[0:11] = np.linspace(95.0, 107.0, 11)
    # 11-14: geri çekilme (107 → 102)
    close[11:15] = np.linspace(107.0, 102.0, 4)
    # 15-17: toparlanma (102 → 106)  → Swing High #1 (~107.5)
    close[15:18] = np.linspace(102.0, 106.0, 3)
    # 18-24: UZUN düşüş (106 → 99)   → Dip #1 bar 24 (RSI iyice düşer ~35)
    close[18:25] = np.linspace(106.0, 99.0, 7)
    # 25-27: toparlanma (99 → 103)   → Swing High #2 (~104)
    close[25:28] = np.linspace(99.0, 103.0, 3)
    # 28-30: hafif geri çekilme → kapanış 100 (Dip #2)
    close[28] = 101.0
    close[29] = 100.5
    close[30] = 100.0
    # 31-41: toparlanma (100 → 104) — base penceresi (lows 97'nin üstünde)
    close[31:42] = np.linspace(100.0, 104.0, 11)
    # 42   : SWEEP — low 95.5 (Dip#2=97 altına iğne), close 102 (geri döndü)
    close[42] = 102.0
    # 43-44: CHOCH — Swing High (~104) ÜZERİNDE gövdeyle kapanışlar
    close[43] = 104.8
    close[44] = 106.0
    # 45   : devam
    close[45] = 107.0

    open_ = np.empty(n)
    open_[0] = close[0]
    for i in range(1, n):
        open_[i] = close[i - 1]
    open_[42] = 100.0    # sweep barı (yeşil gövde)
    open_[43] = 101.5
    open_[44] = 103.5
    open_[45] = 105.5

    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    high[17] = 107.5   # Swing High #1
    # Dip #1 bar 24 net bir fraktal dip olsun (sağ/sol pencereler daha yüksek)
    low[24] = 98.5
    low[23] = 99.6
    low[25] = 99.2
    low[26] = 100.5
    low[27] = 102.0
    low[28] = 101.5
    low[29] = 99.8
    low[30] = 97.0     # Dip #2 (base içinde en düşük) — sağ penceresi 97 üstü
    low[42] = 95.5     # SWEEP: Dip#2 (97) ALTINA iğne
    high[44] = 107.0   # CHOCH tepe

    volume = np.full(n, 1_000_000.0)
    volume[42] = 2_500_000.0    # sweep + hacim
    volume[43:45] = 3_000_000.0  # CHOCH barları hacimli

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    return df


def build_btc_series() -> pd.Series:
    """BTC hafif geriliyor — MOCK_COIN göreli güçlü (RS_Score pozitif)."""
    return pd.Series(np.linspace(100.0, 98.5, 15))


def main() -> None:
    print("══════ FAZ 2 — ASİMETRİK MOTOR MOCK KANITI ══════\n")
    df = build_mock_ohlcv()
    btc = build_btc_series()

    m1 = detect_sweep_choch_long(df)
    m2 = detect_rsi_divergence_trendline(df)
    usdt_series = [8.20, 8.15, 8.10, 8.05, 8.00]  # USDT.D DÜŞÜYOR → onay
    m3 = usdt_d_macro_filter(usdt_series)

    print(f"  MOTOR 1 — Sweep+CHOCH : {m1['reason']} (sweep={m1['sweep']}, choch={m1['choch']}, swing_low={m1['swing_low']})")
    print(f"  MOTOR 2 — RSI Div+Brk : {m2['reason']} (divergence={m2['divergence']}, breakout={m2['breakout']})")
    print(f"  MOTOR 3 — USDT.D      : {m3['reason']} (onay={m3['approved']})")

    combined = asymmetric_long_signal(df, usdt_d_series=usdt_series, btc_close=btc, asset_close=df["close"])
    print(f"\n  BİRLEŞİK SİNYAL        : {combined['reason']} | onay={combined['signal']}")
    print(f"  RS_Score               : {combined['rs_score']}")

    if combined["signal"]:
        # İstenen tam log satırı
        print("\n[SCANNER] FIRSAT ONAYLANDI: MOCK_COIN -> LONG_FIRSAT (Sebep: Sweep+CHOCH & RSI Breakout)")
        print("\n✅ KANIT: Mock veri, asimetrik motoru TAM olarak tetikledi (Sweep+CHOCH & RSI Breakout & USDT.D onayı).")
    else:
        print("\n❌ Mock veri motorları tetiklemedi — ayarla.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
