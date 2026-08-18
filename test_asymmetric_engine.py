# -*- coding: utf-8 -*-
"""
FAZ 5 KANIT MODÜLÜ — Katı Veto Motoru (LONG & SHORT) Mock Grafikleri
=====================================================================
`build_mock_ohlcv()`        → 🟢 LONG kurulumu (4 şartı birden tetikler)
`build_mock_ohlcv_short()`  → 🔴 SHORT kurulumu (4 şartı birden tetikler)

Bu modül `test_scan_flow.py` tarafından import edilir (kalıcı modül, silme).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.asymmetric_engine import (
    asymmetric_signal,
    detect_long_signal,
    detect_short_signal,
    usdt_d_macro_filter,
)


def _ohlcv_from_close(close: np.ndarray) -> pd.DataFrame:
    """Kapanış serisinden open/high/low/volume üretir (pivotlar doğal oluşur)."""
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def build_mock_ohlcv() -> pd.DataFrame:
    """🟢 LONG kurulumu: zirveden ~%34 düşüş → pozitif RSI uyumsuzluğu →
    fiyat düşen trend çizgisini yukarı kırma + RSI düşen çizgi kırılımı.

    Segmanlar (n=120):
      0-55   : 100 → 200 (yükseliş, tepe 200 @ bar55)
      56-70  : 200 → 140 (sert düşüş → dip1, RSI çok düşer ~17)
      71-80  : 140 → 160 (toparlanma → daha düşük tepe)
      81-99  : dalgalı yatay düşüş (RSI yüksek kalır)
      100    : 128 (keskin lower-low dip2)
      101-119: 128 → 132 (ralli → düşen trendi yukarı kırar)
    """
    n = 120
    close = np.empty(n)
    close[0:56] = np.linspace(100.0, 200.0, 56)
    close[56:71] = np.linspace(200.0, 140.0, 15)
    close[71:81] = np.linspace(140.0, 160.0, 10)
    close[81:100] = [
        160, 158, 160.5, 157, 159.5, 156, 158.5, 155, 157.5, 154,
        156.5, 153, 155.5, 152, 154.5, 151, 153.5, 150, 152.5,
    ]
    close[100] = 128.0
    close[101:120] = np.linspace(128.0, 132.0, 19)
    return _ohlcv_from_close(close)


def build_mock_ohlcv_short() -> pd.DataFrame:
    """🔴 SHORT kurulumu: dipten ~%72 yükseliş → negatif RSI uyumsuzluğu →
    fiyat yükselen trend çizgisini aşağı kırma + RSI yükselen çizgi kırılımı.

    Segmanlar (n=120):
      0-55   : 200 → 100 (düşüş, dip 100 @ bar55)
      56-70  : 100 → 160 (sert ralli → tepe1, RSI çok yükselir ~82)
      71-80  : 160 → 140 (geri çekilme → daha yüksek dip)
      81-99  : dalgalı yatay yükseliş (RSI düşük kalır)
      100    : 172 (keskin higher-high tepe2)
      101-119: 172 → 168 (düşüş → yükselen trendi aşağı kırar)
    """
    n = 120
    close = np.empty(n)
    close[0:56] = np.linspace(200.0, 100.0, 56)
    close[56:71] = np.linspace(100.0, 160.0, 15)
    close[71:81] = np.linspace(160.0, 140.0, 10)
    close[81:100] = [
        140, 142, 139.5, 143, 140.5, 144, 141.5, 145, 142.5, 146,
        143.5, 147, 144.5, 148, 145.5, 149, 146.5, 150, 147.5,
    ]
    close[100] = 172.0
    close[101:120] = np.linspace(172.0, 168.0, 19)
    return _ohlcv_from_close(close)


def build_btc_series() -> pd.Series:
    """BTC hafif geriliyor — MOCK_COIN göreli güçlü (RS_Score pozitif)."""
    return pd.Series(np.linspace(100.0, 98.5, 15))


def main() -> None:
    print("══════ FAZ 5 — KATI VETO MOTORU MOCK KANITI (LONG & SHORT) ══════\n")
    usdt_down = [8.20, 8.15, 8.10, 8.05, 8.00]  # USDT.D DÜŞÜYOR → kripto onay

    df_long = build_mock_ohlcv()
    l = detect_long_signal(df_long)
    print(f"  🟢 LONG : signal={l['signal']} | {l['reason']}")
    print(f"         dip_ok={l['dip_ok']} pos_div={l['pos_div']} price_break={l['price_break']} rsi_break={l['rsi_break']}")
    if l["levels"]:
        print(f"         levels: entry={l['levels']['entry']} stop={l['levels']['stop']} t1={l['levels']['t1']}")

    df_short = build_mock_ohlcv_short()
    s = detect_short_signal(df_short)
    print(f"  🔴 SHORT: signal={s['signal']} | {s['reason']}")
    print(f"         peak_ok={s['peak_ok']} neg_div={s['neg_div']} price_break={s['price_break']} rsi_break={s['rsi_break']}")

    # Birleşik: LONG öncelikli — LONG grafiğinde LONG, SHORT grafiğinde SHORT dönmeli
    comb_l = asymmetric_signal(df_long, usdt_d_series=usdt_down, btc_close=build_btc_series(), asset_close=df_long["close"])
    comb_s = asymmetric_signal(df_short, usdt_d_series=usdt_down)
    print(f"\n  BİRLEŞİK LONG grafiği : {comb_l['direction']} → {comb_l['reason']}")
    print(f"  BİRLEŞİK SHORT grafiği: {comb_s['direction']} → {comb_s['reason']}")

    # Sert veto: sıradan trend (sinyal yok) motoru tetiklememeli
    flat = _ohlcv_from_close(np.linspace(100.0, 110.0, 120))
    comb_flat = asymmetric_signal(flat)
    print(f"  DÜZ TREND (kontrol)  : signal={comb_flat['signal']} direction={comb_flat['direction']}")

    ok = bool(l["signal"] and s["signal"] and comb_l["direction"] == "LONG"
              and comb_s["direction"] == "SHORT" and not comb_flat["signal"])
    if ok:
        print("\n[SCANNER] FIRSAT ONAYLANDI: MOCK_COIN -> LONG_FIRSAT (Sebep: Fiyat Düşeni Kırdı + RSI Pozitif Uyumsuzluk)")
        print("✅ KANIT: LONG + SHORT motorları ve katı veto (Sinyal Yok) doğru çalışıyor.")
    else:
        print("\n❌ Mock grafikler motoru tetiklemedi — ayarla.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
