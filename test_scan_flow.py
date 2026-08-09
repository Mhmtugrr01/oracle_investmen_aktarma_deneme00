"""
PROJECT OLYMPUS V2 — OFFLINE DOĞRULAMA TESTLERİ (Ağ gerektirmez, zaman sınırlıdır)
==============================================================================
Kapsam:
  1. Config katı doğrulama (yeni scan_schedule anahtarları)
  2. Rejim motoru: compute_regime + correlate_signal_with_regime (sentetik)
  3. MTF: aggregate_mtf_points + _compute_tf_indicators (sentetik OHLCV)
  4. Scanner: kinetic skor + başlangıç mesajı formatı
  5. ScanStore: SQLite roundtrip (geçici db)
  6. market_data LRU cache limiti

Koşum:  python test_scan_flow.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

_OK = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _OK, _FAIL
    if cond:
        _OK += 1
        print(f"  ✅ {name}")
    else:
        _FAIL += 1
        print(f"  ❌ {name} {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def test_config():
    print("[1] CONFIG katı doğrulama")
    import asyncio as _a
    from core.config import load_oracle_config

    async def _run():
        cfg = await load_oracle_config()
        s = cfg.scan_schedule
        return s

    s = _a.run(_run())
    check("overnight_start_hour==4", s.overnight_start_hour == 4, f"got {s.overnight_start_hour}")
    check("overnight_end_hour==9", s.overnight_end_hour == 9, f"got {s.overnight_end_hour}")
    check("delivery_hour==9", s.delivery_hour == 9, f"got {s.delivery_hour}")
    check("prefilter_concurrency>=1", s.prefilter_concurrency >= 1)
    check("deep_scan_max_assets>=1", s.deep_scan_max_assets >= 1)
    check("per_asset_timeout_sec>=30", s.per_asset_timeout_sec >= 30)
    check("scan_wallclock_timeout_min>=10", s.scan_wallclock_timeout_min >= 10)
    check("heartbeat_interval_min>=1", s.heartbeat_interval_min >= 1)
    check(
        "tf_window 6 tf",
        s.tf_window == ["5m", "15m", "1h", "4h", "1d", "1w"],
        f"got {s.tf_window}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. REJİM MOTORU
# ─────────────────────────────────────────────────────────────────────────────
def _make_snapshot(**overrides) -> "RegimeSnapshot":
    from core.regime_engine import DominanceMTF, RegimeSnapshot

    dom = DominanceMTF(
        btc_d={"5m": [55.0, 54.9], "15m": [55.0, 54.8], "1h": [55.0, 54.7],
               "4h": [55.0, 54.6], "1d": [55.0, 54.5], "1w": [55.0, 54.0]},
        btc_d_trend={"5m": "FALLING", "15m": "FALLING", "1h": "FALLING",
                     "4h": "FALLING", "1d": "FALLING", "1w": "FALLING"},
        usdt_d=5.1,
        usdt_d_trend="FALLING",
        confidence="high",
        source="test",
    )
    defaults = dict(
        captured_at="2025-01-01T04:00:00Z",
        dxy=103.5,
        dxy_change_5d=0.3,
        vix=14.0,
        spy_change_5d=1.2,
        us10y=4.1,
        us10y_delta_7d=0.05,
        gold_change_5d=0.8,
        btc_change_7d=3.5,
        usd_try=38.0,
        usdt_d=5.1,
        btc_d=55.0,
        total_market_cap=2.5e12,
        dominance=dom,
        econ_events_today=[],
        primary_trend="RISK_ON",
        intraday_timing="BULLISH",
        risk_appetite=1.10,
        entry_delay_hint="",
        exit_urgency="NORMAL",
        source_flags=[],
        warnings=[],
    )
    defaults.update(overrides)
    return RegimeSnapshot(**defaults)


def test_regime():
    print("[2] REJİM MOTORU (sentetik)")
    from core.regime_engine import compute_regime, correlate_signal_with_regime

    snap = _make_snapshot()
    computed = compute_regime(snap)
    check("primary_trend korunur", computed.primary_trend in ("RISK_ON", "MIXED", "RISK_OFF", "NEUTRAL"))
    check("risk_appetite aralıkta", 0.40 <= computed.risk_appetite <= 1.20, f"got {computed.risk_appetite}")

    # VIX yüksek → risk appetite düşmeli
    snap_vix = _make_snapshot(vix=40.0, primary_trend="RISK_OFF", intraday_timing="BEARISH")
    computed_vix = compute_regime(snap_vix)
    check("VIX>35 risk iştahı 0.5", computed_vix.risk_appetite <= 0.6, f"got {computed_vix.risk_appetite}")

    # LONG + rejim destekli → aligned
    corr = correlate_signal_with_regime(tf="15m", direction="LONG", regime=snap)
    check("LONG rejim destekli aligned", corr["aligned"] is True)
    check("validity_text dolu", bool(corr["validity_text"]))
    check("confidence 0.55-1.20", 0.55 <= corr["confidence_modifier"] <= 1.20)

    # SHORT + rejim ters → aligned=False
    corr_short = correlate_signal_with_regime(tf="1h", direction="SHORT", regime=snap)
    check("SHORT rejim ters aligned=False", corr_short["aligned"] is False)

    # rejim None → standart pencere
    corr_none = correlate_signal_with_regime(tf="4h", direction="LONG", regime=None)
    check("rejim yok standart pencere", corr_none["aligned"] is False and corr_none["validity_hours_min"] > 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MTF
# ─────────────────────────────────────────────────────────────────────────────
def _synth_ohlcv(n: int = 160, trend: float = 0.15, start_price: float = 100.0) -> pd.DataFrame:
    """Basit sentetik OHLCV serisi üretir (timestamp UTC)."""
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="1h")
    prices = start_price * np.cumprod(1 + np.random.default_rng(42).normal(trend / n, 0.004, n))
    df = pd.DataFrame(
        {
            "open": prices * 0.998,
            "high": prices * 1.006,
            "low": prices * 0.994,
            "close": prices,
            "volume": np.random.default_rng(7).uniform(1e5, 5e5, n),
        },
        index=idx,
    ).reset_index()
    df = df.rename(columns={"index": "timestamp"})
    return df


def test_mtf():
    print("[3] MTF AGREGASYON + GÖSTERGE (sentetik)")
    from agents.quant_engine import _compute_tf_indicators
    from core.multi_tf import aggregate_mtf_points, format_mtf_summary

    # Gösterge hesabı sentetik veride çalışmalı
    df = _synth_ohlcv()
    ind = _compute_tf_indicators(df)
    check("bias hesaplandı", ind["bias"] in ("BULLISH", "BEARISH", "NEUTRAL", "OVERSOLD", "OVERBOUGHT"), f"got {ind['bias']}")
    check("rsi sayısal", isinstance(ind["rsi"], float))
    check("price sayısal", ind["price"] > 0)

    # Saf agregasyon: çoğunluk BULL + 5m/15m BULL → NOW
    pts = {
        "5m": {"bias": "BULLISH"},
        "15m": {"bias": "ACCUMULATING"},
        "1h": {"bias": "BULLISH"},
        "4h": {"bias": "BULLISH"},
        "1d": {"bias": "NEUTRAL"},
        "1w": {"bias": "BULLISH"},
    }
    agg = aggregate_mtf_points("TEST/USDT", pts)
    check("MTF BULLISH yön", agg.signal_bias == "BULLISH", f"got {agg.signal_bias}")
    check("MTF giriş NOW", agg.entry_timing == "NOW", f"got {agg.entry_timing}")
    check("MTF aligned>=2", agg.aligned_count >= 2, f"got {agg.aligned_count}")
    check("MTF validity_text dolu", bool(agg.validity_text))

    # Ters senaryo: çoğunluk BEAR
    pts_bear = {tf: {"bias": "BEARISH" if tf != "1d" else "OVERBOUGHT"} for tf in ("5m", "15m", "1h", "4h", "1d", "1w")}
    agg_bear = aggregate_mtf_points("TEST/USDT", pts_bear)
    check("MTF BEARISH yön", agg_bear.signal_bias == "BEARISH", f"got {agg_bear.signal_bias}")
    check("MTF giriş AVOID", agg_bear.entry_timing == "AVOID", f"got {agg_bear.entry_timing}")

    # format çıktısı boş değil
    check("format_mtf_summary dolu", len(format_mtf_summary(agg, "TEST/USDT")) > 20)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCANNER
# ─────────────────────────────────────────────────────────────────────────────
def _synth_vshape(n_down: int = 24, n_up: int = 10, recover: float = 0.45) -> pd.DataFrame:
    """Deterministik V-şekli: önce düşüş, sonra küçük toparlanma (RSI 50-55 artan)."""
    down = np.linspace(100.0, 90.0, n_down)
    up = 90.0 + np.linspace(0.0, 10.0 * recover, n_up)
    closes = np.concatenate([down, up])
    n = len(closes)
    df = pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.004,
            "low": closes * 0.996,
            "close": closes,
            "volume": np.full(n, 3e5),
        }
    )
    df["timestamp"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="1h")
    return df


def test_scanner_offline():
    print("[4] SCANNER (kinetic + mesaj formatı)")
    import pandas_ta as ta
    from core.scanner import OracleScanner

    scanner = OracleScanner(lambda a: None, lambda t: None, {"scan_schedule": {}, "asset_universe": {}})

    # Pozitif: V-şekli toparlanma (RSI 50-55, artan) → kinetik > 0
    df_v = _synth_vshape()
    rsi_v = ta.rsi(df_v["close"], length=14).dropna()
    assert 50 <= float(rsi_v.iloc[-1]) < 55 and float(rsi_v.iloc[-1]) > float(rsi_v.iloc[-2])
    k = scanner._compute_olympus_kinetic(df_v)
    check("kinetic V-şekli > 0", k > 0.0, f"got {k}")

    # Negatif 1: düşen seri (RSI ~0) → 0
    df_bear = _synth_ohlcv(n=120, trend=-0.6, start_price=100.0)
    df_bear["close"] = np.linspace(100.0, 60.0, len(df_bear))
    k_bear = scanner._compute_olympus_kinetic(df_bear)
    check("kinetic düşen seri == 0", k_bear == 0.0, f"got {k_bear}")

    # Negatif 2: sabit seri (RSI NaN) → 0 (çökme yok)
    df_flat = _synth_ohlcv(n=80)
    df_flat["close"] = 100.0
    k_flat = scanner._compute_olympus_kinetic(df_flat)
    check("kinetic sabit seri == 0", k_flat == 0.0, f"got {k_flat}")

    # Start mesajı formatı (rejim yok)
    msg = scanner._build_start_message(21, None)
    check("start mesajı varlık sayısı", "21" in msg)
    check("start mesajı katman", "4 Katmanlı" in msg)

    # Regime'li start mesajı
    snap = _make_snapshot()
    msg2 = scanner._build_start_message(21, snap)
    check("start mesajı rejim", "RISK_ON" in msg2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCAN STORE
# ─────────────────────────────────────────────────────────────────────────────
def test_scan_store():
    print("[5] SCAN STORE (SQLite roundtrip)")
    from core.scan_store import ScanStore

    tmp = Path(tempfile.mkdtemp()) / "scan_store_test.db"
    store = ScanStore(db_path=tmp)

    async def _run():
        await store.start_run("scan_test_1", "test")
        await store.record_result(
            run_id="scan_test_1",
            symbol="BTC/USDT",
            signal="STRONG_BUY",
            composite=0.85,
            base_rr=3.5,
            t1=70000.0, t2=75000.0, t3=80000.0,
            stop_loss=62000.0,
            trade_type="STRONG_LONG_TERM_ENTRY",
            oracle_summary="test",
            tf_bias_json='{"4h": "BULLISH"}',
            regime_json='{"primary_trend": "RISK_ON"}',
            correlation_json='{"aligned": true}',
        )
        await store.finish_run("scan_test_1", "done", 3, 1)
        last = await store.get_last_run()
        results = await store.get_results("scan_test_1")
        return last, results

    last, results = asyncio.run(_run())
    check("run kaydedildi", last is not None and last.get("run_id") == "scan_test_1")
    check("sonuç kaydedildi", len(results) == 1 and results[0]["signal"] == "STRONG_BUY")
    check("composite doğru", abs(results[0]["composite"] - 0.85) < 1e-9)

    try:
        tmp.unlink()
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 6. MARKET DATA LRU
# ─────────────────────────────────────────────────────────────────────────────
def test_cache_lru():
    print("[6] MARKET DATA LRU cache")
    import tools.market_data as md

    md._CACHE_MAX_ENTRIES = 5
    for i in range(10):
        md._cache_set_df(f"testkey|{i}", pd.DataFrame({"close": [float(i)]}))
    check("cache limit 5", len(md._DATA_CACHE) <= 5, f"got {len(md._DATA_CACHE)}")

    # En eski anahtar atılmış olmalı (0..4)
    oldest = md._cache_get_df("testkey|0")
    newest = md._cache_get_df("testkey|9")
    check("en eski atıldı", oldest is None)
    check("en yeni duruyor", newest is not None and float(newest["close"].iloc[0]) == 9.0)

    # TTL mantığı bozulmadı
    md._DATA_CACHE.clear()


# ─────────────────────────────────────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n══════ PROJECT OLYMPUS V2 — OFFLINE DOĞRULAMA ══════\n")
    t0 = time.time()
    test_config()
    test_regime()
    test_mtf()
    test_scanner_offline()
    test_scan_store()
    test_cache_lru()
    dur = time.time() - t0
    print(f"\n══════ SONUÇ: {_OK} geçti, {_FAIL} başarısız ({dur:.1f}s) ══════")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
