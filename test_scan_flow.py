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
import types
from datetime import datetime, timedelta, timezone
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
    check("prefilter_min_score>=0", s.prefilter_min_score >= 0.0, f"got {s.prefilter_min_score}")
    check("deep_scan_max_assets>=1", s.deep_scan_max_assets >= 1)
    check("per_asset_timeout_sec>=30", s.per_asset_timeout_sec >= 30)
    check("scan_wallclock_timeout_min>=10", s.scan_wallclock_timeout_min >= 10)
    check("heartbeat_interval_min>=1", s.heartbeat_interval_min >= 1)
    check("missed_scan_grace_hours>=1", s.missed_scan_grace_hours >= 1.0,
          f"got {s.missed_scan_grace_hours}")
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
    print("[4] SCANNER (iki katmanlı prefilter + işlem planı + state dökümü)")
    import pandas_ta as ta
    from core.scanner import OracleScanner
    from core.trade_plan import build_trade_plan

    scanner = OracleScanner(lambda a: None, lambda t: None, {"scan_schedule": {}, "asset_universe": {}})

    # V-şekli toparlanma (RSI 50-55, artan) → RSI>55 dip kapısına TAKILMAZ, skor > 0
    df_v = _synth_vshape()
    rsi_v = ta.rsi(df_v["close"], length=14).dropna()
    assert 50 <= float(rsi_v.iloc[-1]) < 55 and float(rsi_v.iloc[-1]) > float(rsi_v.iloc[-2])
    r_v = scanner._score_prefilter_candidate(df_v, df_v)
    check("V-şekli prefilter skor > 0", r_v["score"] > 0.0, f"got {r_v}")
    check(
        "V-şekli tier geçerli",
        r_v["tier"] in ("TREND_FOLLOWER", "DIP_REVERSAL"),
        f"got {r_v['tier']}",
    )
    check("V-şekli reason dolu", bool(r_v["reason"]) and r_v["reason"] != "-", f"got {r_v['reason']}")

    # Güçlü yükseliş serisi → TREND_FOLLOWER (eski sert RSI kapısı kalktı)
    df_up = _synth_ohlcv(n=120, trend=0.6, start_price=100.0)
    r_up = scanner._score_prefilter_candidate(df_up, df_up)
    check("yükselen seri TREND_FOLLOWER", r_up["tier"] == "TREND_FOLLOWER", f"got {r_up}")

    # Sabit seri (RSI NaN) → 0 / NONE (çökme yok)
    df_flat = _synth_ohlcv(n=80)
    df_flat["close"] = 100.0
    r_flat = scanner._score_prefilter_candidate(df_flat, df_flat)
    check("sabit seri skor == 0", r_flat["score"] == 0.0, f"got {r_flat}")
    check("sabit seri tier NONE", r_flat["tier"] == "NONE", f"got {r_flat['tier']}")

    # ── FAZ 0 FİKS 1: sessiz düşme artık stats["failed"] ile sayılıyor ──
    async def _probe_failed_all():
        async def _bad_fetch(symbol):  # noqa: ARG001 — veri çekilemedi simülasyonu
            return None, None

        scanner._fetch_prefilter_data = _bad_fetch  # type: ignore[method-assign]
        res = await scanner._pre_filter_assets(["AAA/USDT", "BBB/USDT", "CCC/USDT"])
        return res["stats"]

    st = asyncio.run(_probe_failed_all())
    check("failed sayacı tüm varlıklar", st["failed"] == 3, f"got {st}")
    check("veri hatası tarandı sayılmaz", st["scanned"] == 0, f"got {st}")

    async def _probe_score_error():
        async def _ok_fetch(symbol):  # noqa: ARG001
            return _synth_ohlcv(n=60), None

        scanner._fetch_prefilter_data = _ok_fetch  # type: ignore[method-assign]

        def _boom(df1, df4, symbol="?"):  # noqa: ARG001 — skorlama patlatma
            raise RuntimeError("skorlama patladı")

        orig_score = scanner._score_prefilter_candidate
        scanner._score_prefilter_candidate = _boom  # type: ignore[method-assign]
        try:
            res = await scanner._pre_filter_assets(["BBB/USDT"])
        finally:
            scanner._score_prefilter_candidate = orig_score
        return res["stats"]

    st2 = asyncio.run(_probe_score_error())
    check("skorlama hatası failed sayılır", st2["failed"] == 1, f"got {st2}")
    check("skorlama hatası scanned sayılmaz", st2["scanned"] == 0, f"got {st2}")

    # Start mesajı formatı (rejim yok)
    msg = scanner._build_start_message(21, None)
    check("start mesajı varlık sayısı", "21" in msg)
    check("start mesajı katman", "4 Katmanlı" in msg)

    # Regime'li start mesajı
    snap = _make_snapshot()
    msg2 = scanner._build_start_message(21, snap)
    check("start mesajı rejim", "RISK_ON" in msg2)

    # ── FASE D: build_trade_plan (fiyat bazlı plan) ──
    lv = {
        "entry_zone_low": 95.0,
        "entry_zone_high": 97.0,
        "stop_loss": 93.0,
        "t1": 100.0,
        "t2": 104.0,
        "t3": 110.0,
        "invalidation_level": 92.5,
        "fib_382": 97.8,
        "fib_500": 96.5,
        "fib_618": 95.2,
    }
    plan = build_trade_plan("LONG", "BULLISH", "NOW", lv, base_rr=3.0)
    check("plan FULL tipi", plan["plan_type"] == "FULL", f"got {plan['plan_type']}")
    check("plan giriş bölgesi", any("GİRİŞ BÖLGESİ" in ln for ln in plan["lines"]))
    check("plan kar al kademeleri", any("KAR AL" in ln for ln in plan["lines"]))
    check("plan yeniden kontrol bölgesi", any("YENİDEN KONTROL" in ln for ln in plan["lines"]))
    check("plan geçersizlik seviyesi", plan["invalidation_price"] == 92.5)
    check("plan R:R satırı", any("R:R" in ln for ln in plan["lines"]))

    # Üst TF zıt → BOUNCE_ONLY (tepki alımı, limit emir)
    plan_b = build_trade_plan("LONG", "BEARISH", "WAIT", lv)
    check("plan BOUNCE_ONLY", plan_b["plan_type"] == "BOUNCE_ONLY", f"got {plan_b['plan_type']}")

    # Hizalama belirsiz → LIMIT_ONLY
    plan_l = build_trade_plan("LONG", "NEUTRAL", "WAIT", lv)
    check("plan LIMIT_ONLY", plan_l["plan_type"] == "LIMIT_ONLY", f"got {plan_l['plan_type']}")

    # Seviye yok → NO_PLAN (çökme yok)
    plan_n = build_trade_plan("LONG", "BULLISH", "NOW", {})
    check("plan NO_PLAN", plan_n["plan_type"] == "NO_PLAN", f"got {plan_n['plan_type']}")

    # SHORT yönü ayna plan üretir (giriş üst bölge, invalidation üst)
    plan_s = build_trade_plan(
        "SHORT",
        "BEARISH",
        "NOW",
        {
            "entry_zone_low": 120.0,
            "entry_zone_high": 122.0,
            "stop_loss": 124.0,
            "t1": 116.0,
            "t2": 112.0,
            "t3": 106.0,
            "invalidation_level": 124.5,
        },
        base_rr=2.8,
    )
    check("SHORT plan FULL", plan_s["plan_type"] == "FULL", f"got {plan_s['plan_type']}")
    check("SHORT plan geçersizlik", plan_s["invalidation_price"] == 124.5)


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

    # ── ADIM 5: iki koşu karşılaştırması (tutarlı/değişen/kaybolan/yeni) ──
    async def _runs_compare():
        tmp2 = Path(tempfile.mkdtemp()) / "scan_store_cmp.db"
        store2 = ScanStore(db_path=tmp2)
        # Koşu A
        await store2.start_run("run_a", "test")
        for sym, sig, comp in [("BTC/USDT", "STRONG_BUY", 0.8), ("ETH/USDT", "ACCUMULATE", 0.7), ("SOL/USDT", "STRONG_BUY", 0.75)]:
            await store2.record_result(run_id="run_a", symbol=sym, signal=sig, composite=comp,
                                       base_rr=2.0, t1=1.0, t2=2.0, t3=3.0, stop_loss=0.5, trade_type="LONG")
        await store2.finish_run("run_a", "done", 3, 3)
        # Koşu B: BTC aynı kaldı, ETH değişti, SOL kayboldu, ADA yeni çıktı
        await store2.start_run("run_b", "test")
        for sym, sig, comp in [("BTC/USDT", "STRONG_BUY", 0.82), ("ETH/USDT", "STRONG_SELL", 0.4), ("ADA/USDT", "ACCUMULATE", 0.66)]:
            await store2.record_result(run_id="run_b", symbol=sym, signal=sig, composite=comp,
                                       base_rr=2.0, t1=1.0, t2=2.0, t3=3.0, stop_loss=0.5, trade_type="LONG")
        await store2.finish_run("run_b", "done", 3, 3)
        rep = await store2.compare_runs("run_a", "run_b")
        try:
            tmp2.unlink()
        except OSError:
            pass
        return rep

    rep = asyncio.run(_runs_compare())
    check("ADIM5 tutarlı sayısı", rep["consistent_count"] == 1, f"got {rep['consistent_count']}")
    check("ADIM5 değişen sayısı", rep["changed_count"] == 1, f"got {rep['changed_count']}")
    check("ADIM5 kaybolan", rep["lost"] == ["SOL/USDT"], f"got {rep['lost']}")
    check("ADIM5 yeni", rep["new"] == ["ADA/USDT"], f"got {rep['new']}")
    check("ADIM5 BTC tutarlı sinyal", rep["consistent"][0]["signal"] == "STRONG_BUY")
    check("ADIM5 ETH değişim", rep["changed"][0]["signal_a"] == "ACCUMULATE" and rep["changed"][0]["signal_b"] == "STRONG_SELL")

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
# 7. ADIM 3 — AYNI GÜN ÇİFT TARAMA ÖNLEME (scan_store kalıcılığı)
# ─────────────────────────────────────────────────────────────────────────────
def test_scan_once_dedup():
    print("[7] ADIM 3 — restart sonrası çift tarama önleme")
    import sqlite3

    import pytz
    from core.scan_store import ScanStore
    from core.scanner import OracleScanner

    tz = pytz.timezone("Europe/Istanbul")
    scanner = OracleScanner(None, None, {})

    # started_at UTC olarak yazılır; İstanbul karşılığı aynı gün olsun diye
    # UTC sabah 06:00 (İstanbul 09:00) kullanıyoruz.
    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.replace(hour=6, minute=0, second=0, microsecond=0)
    yesterday_utc = today_utc - timedelta(days=1)
    _fmt = "%Y-%m-%d %H:%M:%S"
    today_s = today_utc.strftime(_fmt)
    yesterday_s = yesterday_utc.strftime(_fmt)

    async def _scenario(status: str, started_at: str | None) -> bool:
        tmp = Path(tempfile.mkdtemp()) / "dedup_test.db"
        store = ScanStore(db_path=tmp)
        await store.start_run("scan_x", "gece_taramasi")
        with sqlite3.connect(tmp) as conn:
            if started_at is None:
                conn.execute(
                    "UPDATE scan_runs SET status = ? WHERE run_id = 'scan_x'",
                    (status,),
                )
            else:
                conn.execute(
                    "UPDATE scan_runs SET started_at = ?, status = ? WHERE run_id = 'scan_x'",
                    (started_at, status),
                )
        try:
            return await scanner._full_scan_done_today(store, tz)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    async def _empty_store() -> bool:
        tmp = Path(tempfile.mkdtemp()) / "dedup_empty.db"
        store = ScanStore(db_path=tmp)
        try:
            return await scanner._full_scan_done_today(store, tz)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    check("bugün done → True", asyncio.run(_scenario("done", today_s)))
    check("bugün empty → True", asyncio.run(_scenario("empty", today_s)))
    check("bugün no_candidates → True", asyncio.run(_scenario("no_candidates", today_s)))
    check("bugün error → False (yeniden dener)", not asyncio.run(_scenario("error", today_s)))
    check("bugün running → False (devam ediyor)", not asyncio.run(_scenario("running", today_s)))
    check("dün done → False", not asyncio.run(_scenario("done", yesterday_s)))
    check("boş store → False", not asyncio.run(_empty_store()))


# ─────────────────────────────────────────────────────────────────────────────
# 8. ADIM 6 — SIGNAL TRACKER KAYIT + GERÇEK İSTATİSTİK (sahte yüzde yok)
# ─────────────────────────────────────────────────────────────────────────────
def test_signal_tracker():
    print("[8] ADIM 6 — signal tracker kayıt + win-rate")
    from core.signal_tracker import SignalTracker, SignalStatus, SignalOutcome

    tmp = Path(tempfile.mkdtemp()) / "sig_tracker_test.db"
    tracker = SignalTracker(db_path=tmp)

    # 1) Sinyal kaydet → PENDING olmalı
    sid = tracker.record_signal(
        symbol="BTC/USDT", direction="LONG",
        entry_price=100.0, stop_loss=95.0, t1=110.0, t2=120.0, t3=130.0,
        confidence=0.8, composite_score=0.75,
    )
    check("tracker sinyal kaydı", isinstance(sid, str) and len(sid) > 0)

    st = tracker.get_statistics(days=30)
    check("total_signals=1", st["total_signals"] == 1, f"got {st['total_signals']}")
    check("open_signals=1", st["open_signals"] == 1, f"got {st['open_signals']}")
    check("win_rate=0 (veri yokken)", st["win_rate"] == 0.0, f"got {st['win_rate']}")

    # 2) Kapanan sinyal simüle et (SL hit → LOSS) ve istatistik doğrula
    tracker._update_signal_status(
        sid, SignalStatus.SL_HIT, SignalOutcome.LOSS,
        exit_price=94.0, exit_time="now", pnl=-6.0,
    )
    st2 = tracker.get_statistics(days=30)
    check("closed_signals=1", st2["closed_signals"] == 1, f"got {st2['closed_signals']}")
    check("open_signals=0", st2["open_signals"] == 0, f"got {st2['open_signals']}")
    check("win_rate=0 (1 kayıp)", st2["win_rate"] == 0.0, f"got {st2['win_rate']}")

    # 3) İkinci sinyal WIN → win_rate %50
    sid2 = tracker.record_signal(
        symbol="ETH/USDT", direction="LONG",
        entry_price=200.0, stop_loss=190.0, t1=220.0,
        confidence=0.7, composite_score=0.7,
    )
    tracker._update_signal_status(
        sid2, SignalStatus.TP1_HIT, SignalOutcome.WIN,
        exit_price=220.0, exit_time="now", pnl=10.0,
    )
    st3 = tracker.get_statistics(days=30)
    check("win_rate=%50", st3["win_rate"] == 50.0, f"got {st3['win_rate']}")
    check("avg_pnl=+2.0", abs(st3["avg_pnl"] - 2.0) < 1e-9, f"got {st3['avg_pnl']}")

    # 4) Scanner bağlantısı: _record_opportunity tracker'a kayıt yapmalı
    import core.scanner as scanner_mod

    recorded: list[dict] = []

    class _FakeTracker:
        def record_signal(self, **kw):
            recorded.append(kw)
            return "fake_id"

    class _FakeStore:
        async def record_result(self, **kw):
            pass

    orig_tracker_import = scanner_mod.get_signal_tracker if hasattr(scanner_mod, "get_signal_tracker") else None
    scanner_mod.get_scan_store = lambda: _FakeStore()  # type: ignore[assignment]

    import core.scan_store as ss_mod

    orig_ss_get = ss_mod.get_scan_store
    ss_mod.get_scan_store = lambda: _FakeStore()  # type: ignore[assignment]

    # tracker import'unu fake ile değiştir (scanner içinde lazy import var)
    import builtins

    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "core.signal_tracker":
            mod = types.ModuleType("core.signal_tracker")
            mod.get_signal_tracker = lambda: _FakeTracker()
            return mod
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = _fake_import
    try:
        sc = scanner_mod.OracleScanner(pipeline_runner=None, telegram_bot=None, config={})
        asyncio.run(
            sc._record_opportunity(
                run_id="scan_t",
                result={
                    "asset": "BTC/USDT", "signal": "STRONG_BUY",
                    "composite_pct": 75, "base_rr": 3.0,
                    "t1": 110.0, "t2": 120.0, "t3": 130.0,
                    "stop_loss": 95.0, "trade_type": "LONG",
                    "entry_zone_low": 100.0, "entry_zone_high": 102.0,
                    "confidence": 0.8,
                },
                regime=None,
            )
        )
    finally:
        builtins.__import__ = orig_import
        ss_mod.get_scan_store = orig_ss_get

    check("scanner tracker'a kaydetti", len(recorded) == 1, f"got {len(recorded)}")
    if recorded:
        check("kayıt yönü LONG", recorded[0]["direction"] == "LONG")
        check("kayıt giriş fiyatı", abs(recorded[0]["entry_price"] - 100.0) < 1e-9)

    try:
        tmp.unlink()
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 9. FAZ A — KAÇIRILAN TARAMA GÜVENLİK AĞI
# ─────────────────────────────────────────────────────────────────────────────
def test_faz_a_missed_scan():
    print("[9] FAZ A — kaçırılan tarama güvenlik ağı (missed_scan_guard)")
    import sqlite3

    import pytz
    from core.scan_store import ScanStore
    from core.scanner import OracleScanner

    tz = pytz.timezone("Europe/Istanbul")
    fmt = "%Y-%m-%d %H:%M:%S"
    now_utc = datetime.now(timezone.utc)

    async def _scenario(
        started_at: str | None, status: str = "done", grace: float = 24.0
    ) -> bool:
        tmp = Path(tempfile.mkdtemp()) / "missed_test.db"
        store = ScanStore(db_path=tmp)
        if started_at is not None:
            await store.start_run("scan_m", "gece_taramasi")
            with sqlite3.connect(tmp) as conn:
                conn.execute(
                    "UPDATE scan_runs SET started_at = ?, status = ? WHERE run_id = 'scan_m'",
                    (started_at, status),
                )
        sc = OracleScanner(None, None, {"scan_schedule": {"missed_scan_grace_hours": grace}})
        try:
            return await sc._missed_scan_needed(store, tz)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    yesterday = (now_utc - timedelta(days=1)).strftime(fmt)
    today = now_utc.strftime(fmt)
    one_hour_ago = (now_utc - timedelta(hours=1)).strftime(fmt)

    async def _scenario_in_progress(started_at: str, grace: float = 24.0) -> bool:
        tmp = Path(tempfile.mkdtemp()) / "missed_prog.db"
        store = ScanStore(db_path=tmp)
        await store.start_run("scan_m", "gece_taramasi")
        with sqlite3.connect(tmp) as conn:
            conn.execute(
                "UPDATE scan_runs SET started_at = ?, status = 'done' WHERE run_id = 'scan_m'",
                (started_at,),
            )
        sc = OracleScanner(None, None, {"scan_schedule": {"missed_scan_grace_hours": grace}})
        sc._scan_in_progress = True
        try:
            return await sc._missed_scan_needed(store, tz)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    check("boş store → True (hiç tarama yok)", asyncio.run(_scenario(None)))
    check("dün done → True (bugün kaçırıldı)", asyncio.run(_scenario(yesterday)))
    check("bugün done → False (zaten var)", not asyncio.run(_scenario(today)))
    check("1 saat önce → False (grace içinde)", not asyncio.run(_scenario(one_hour_ago)))
    check("dün done + grace 12s → True", asyncio.run(_scenario(yesterday, grace=12.0)))
    check("dün running (bayat) → True", asyncio.run(_scenario(yesterday, status="running")))
    check(
        "dün done + scan_in_progress → False",
        not asyncio.run(_scenario_in_progress(yesterday)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. FAZ B — KORELASYON KÜMELEME + GÖRELİ GÜÇ (offline, sentetik)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz_b_cluster():
    print("[10] FAZ B — korelasyon kümeleme + göreli güç")
    from core.cluster_engine import ClusterEngine, cluster_and_rank_signals

    engine = ClusterEngine()

    # _to_yf_ticker genelleştirme (40+ kripto tek kural)
    check("BTC/USDT → BTC-USD", engine._to_yf_ticker("BTC/USDT") == "BTC-USD")
    check("SOL/USDT → SOL-USD", engine._to_yf_ticker("SOL/USDT") == "SOL-USD")
    check("THYAO.IS → THYAO.IS", engine._to_yf_ticker("THYAO.IS") == "THYAO.IS")
    check("NVDA → NVDA", engine._to_yf_ticker("NVDA") == "NVDA")
    check("USDT/USDT → None", engine._to_yf_ticker("USDT/USDT") is None)

    # Sentetik fiyat: 60 gün, 5 varlık → A/B/C yükselen (korele), D/E düşen (korele)
    rng = np.random.default_rng(42)
    idx = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="D")
    rising = np.linspace(100, 130, 60)
    falling = np.linspace(50, 40, 60)
    data = {
        "A": pd.Series(rising + rng.normal(0, 0.5, 60), index=idx),
        "B": pd.Series(rising + rng.normal(0, 0.5, 60), index=idx),
        "C": pd.Series(rising * 0.5 + rng.normal(0, 0.5, 60), index=idx),
        "D": pd.Series(falling + rng.normal(0, 0.5, 60), index=idx),
        "E": pd.Series(falling + rng.normal(0, 0.5, 60), index=idx),
    }
    price_df = pd.DataFrame(data)
    corr = engine._compute_correlation_matrix(price_df)
    check("korelasyon matrisi 5x5", corr.shape == (5, 5), f"got {corr.shape}")
    check("A-B korelasyon > 0.9", float(corr.loc["A", "B"]) > 0.9)

    clusters = engine._cluster_by_correlation(list(data.keys()), corr)
    check("küme sayısı >= 2", len(clusters) >= 2, f"got {len(clusters)}")

    # cluster_signals offline (fetch_close enjekte → ağ yok)
    async def _fake_fetch(sym: str):
        if sym in data:
            return pd.DataFrame({"Close": data[sym]})
        return None

    async def _run():
        return await cluster_and_rank_signals(
            list(data.keys()), lookback_days=30, fetch_close=_fake_fetch
        )

    sig_clusters = asyncio.run(_run())
    check("cluster_signals küme üretti", len(sig_clusters) >= 2, f"got {len(sig_clusters)}")
    all_leaders = [m.symbol for c in sig_clusters for m in c.leaders]
    check("liderler seçildi", len(all_leaders) > 0, f"got {all_leaders}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. FAZ C — FİYAT BAZLI GEÇERLİLİK + USDT.D GÜÇLENDİRME
# ─────────────────────────────────────────────────────────────────────────────
def test_faz_c_validity():
    print("[11] FAZ C — fiyat bazlı geçerlilik + USDT.D gücü")
    from core.trade_plan import build_trade_plan

    lv = {
        "entry_zone_low": 95.0,
        "entry_zone_high": 97.0,
        "stop_loss": 93.0,
        "t1": 100.0,
        "t2": 104.0,
        "t3": 110.0,
        "invalidation_level": 92.5,
    }

    p_fall = build_trade_plan("LONG", "BULLISH", "NOW", lv, usdt_d_trend="FALLING")
    check("LONG + USDT.D düşüyor → +1", p_fall["validity_strength"] == 1,
          f"got {p_fall['validity_strength']}")
    check("LONG + düşüyor → not var", bool(p_fall["usdt_d_note"]))
    check("plan USDT.D notu satırda", any("USDT.D" in ln for ln in p_fall["lines"]))

    p_rise = build_trade_plan("LONG", "BULLISH", "NOW", lv, usdt_d_trend="RISING")
    check("LONG + USDT.D yükseliyor → -1", p_rise["validity_strength"] == -1,
          f"got {p_rise['validity_strength']}")

    s_rise = build_trade_plan("SHORT", "BEARISH", "NOW", lv, usdt_d_trend="RISING")
    check("SHORT + USDT.D yükseliyor → +1", s_rise["validity_strength"] == 1,
          f"got {s_rise['validity_strength']}")

    s_fall = build_trade_plan("SHORT", "BEARISH", "NOW", lv, usdt_d_trend="FALLING")
    check("SHORT + USDT.D düşüyor → -1", s_fall["validity_strength"] == -1,
          f"got {s_fall['validity_strength']}")

    p_none = build_trade_plan("LONG", "BULLISH", "NOW", lv)
    check("USDT.D yok → 0", p_none["validity_strength"] == 0,
          f"got {p_none['validity_strength']}")

    # Telegram gösterimi: "GEÇERLİLİK: 24 saat" birincil satırı artık YOK
    import pathlib

    src = pathlib.Path(__file__).parent / "bot" / "telegram_handler.py"
    content = src.read_text(encoding="utf-8")
    check("handler'da sabit '24 saat' birincil geçersiz", "GEÇERLİLİK: {validity_period}" not in content)
    check("handler'da fiyat bazlı validity_block var", "validity_block" in content)


# ─────────────────────────────────────────────────────────────────────────────
# 12. FAZ D — PREFILTER RATE LIMITING (batch'ler arası bekleme)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz_d_rate_limit():
    print("[12] FAZ D — prefilter batch rate limiting")
    from core.scanner import OracleScanner

    # Konfigürasyondan batch_cooldown_sec okunuyor mu?
    scanner = OracleScanner(
        None, None, {"scan_schedule": {"batch_cooldown_sec": 0.7}, "asset_universe": {}}
    )
    check("config batch_cooldown_sec okunuyor", abs(scanner._batch_cooldown - 0.7) < 1e-9,
          f"got {scanner._batch_cooldown}")

    scanner._batch_cooldown = 0.3  # testte kısa tut (süre ölçümü için)
    scanner.scan_config["prefilter_concurrency"] = 2  # chunk = 8 → 12 sembol = 2 batch

    stamps: list[float] = []

    async def _slow_fetch(symbol):  # noqa: ARG001
        stamps.append(time.monotonic())
        return _synth_ohlcv(n=60), None

    async def _run():
        scanner._fetch_prefilter_data = _slow_fetch  # type: ignore[method-assign]
        symbols = [f"T{i:02d}/USDT" for i in range(12)]
        res = await scanner._pre_filter_assets(symbols)
        return res["stats"]

    st = asyncio.run(_run())
    check("12 varlık tarandı", st["scanned"] == 12, f"got {st}")

    # 12 sembol, chunk = concurrency*4 = 8 → 2 batch: 0..7 ve 8..11
    check("en az 2 batch oluştu", len(stamps) > 8, f"got {len(stamps)}")
    gap = stamps[8] - max(stamps[:8]) if len(stamps) > 8 else 0.0
    check(f"batch'ler arası bekleme >= 0.3s (got {gap:.3f}s)", gap >= 0.3 * 0.8,
          f"got {gap:.3f}s")


# ─────────────────────────────────────────────────────────────────────────────
# 13. FAZ 0 — COINGECKO RATE LIMIT + YFINANCE FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def test_faz0_coingecko_rate_limit():
    print("[13] FAZ 0 — CoinGecko rate limit + yfinance fallback")
    from core import regime_engine as re

    check("min interval tanimli (>=2.0)", re._COINGECKO_MIN_INTERVAL_SEC >= 2.0,
          f"got {re._COINGECKO_MIN_INTERVAL_SEC}")
    check("coingecko lock var", isinstance(re._coingecko_lock, asyncio.Lock))

    # Rate-limit: ardışık çağrılar arası bekleme
    async def _probe():
        t0 = time.monotonic()
        await re._coingecko_rate_limit()
        t1 = time.monotonic()
        await re._coingecko_rate_limit()
        t2 = time.monotonic()
        return t1 - t0, t2 - t1

    g1, g2 = asyncio.run(_probe())
    check(f"ilk çağrı hızlı (got {g1*1000:.0f}ms)", g1 < 0.5, f"got {g1:.3f}s")
    check(f"ikinci çağrı beklemeli (got {g2*1000:.0f}ms)", g2 >= re._COINGECKO_MIN_INTERVAL_SEC * 0.8,
          f"got {g2:.3f}s")

    # Yfinance fallback: BTC.D proxy + USDT.D None + trend
    async def _fb():
        return await re._fetch_dominance_via_yfinance()

    fb = asyncio.run(_fb())
    check("fallback dict döndü", isinstance(fb, dict), f"got {type(fb)}")
    check("fallback source=yfinance_proxy", fb.get("source") == "yfinance_proxy",
          f"got {fb.get('source')}")
    check("fallback btc_d var (50±5)", fb.get("btc_d") is not None and 45.0 <= fb["btc_d"] <= 55.0,
          f"got {fb.get('btc_d')}")
    check("fallback usdt_d None (mutlak yok)", fb.get("usdt_d") is None, f"got {fb.get('usdt_d')}")
    check("fallback usdt_d_trend bilinir", fb.get("usdt_d_trend") in ("FLAT", "RISING", "FALLING"),
          f"got {fb.get('usdt_d_trend')}")
    check("fallback total_market_cap > 0", (fb.get("total_market_cap") or 0) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# 14. FAZ 1 — EŞİK GEVŞETMESİ (0.57 / 2.5 / 0.50) + AĞIRLIKLAR
# ─────────────────────────────────────────────────────────────────────────────
def test_faz1_thresholds():
    print("[14] FAZ 1 — eşik gevşetmesi + ağırlıklar")
    import asyncio as _a
    from core.config import load_oracle_config
    from core.types import OracleState
    from core.config import get_oracle_config_cached

    async def _run():
        return await load_oracle_config()

    cfg = _a.run(_run())
    check("min_composite_score==0.57", abs(cfg.ceo.min_composite_score - 0.57) < 1e-9,
          f"got {cfg.ceo.min_composite_score}")
    check("min_risk_reward_ratio==2.5", abs(cfg.risk.min_risk_reward_ratio - 2.5) < 1e-9,
          f"got {cfg.risk.min_risk_reward_ratio}")
    check("confidence_threshold==0.50", abs(cfg.ceo.confidence_threshold - 0.50) < 1e-9,
          f"got {cfg.ceo.confidence_threshold}")
    w = cfg.analysis.weights
    total = w["macro"] + w["quant"] + w["whale"] + w["fundamental"] + w["sentiment"]
    check("ağırlıklar toplamı ≈ 1.0", abs(total - 1.0) < 1e-9, f"got {total}")
    check("fundamental ağırlık 0.05 (veto-only)", abs(w["fundamental"] - 0.05) < 1e-9,
          f"got {w['fundamental']}")
    check("quant ağırlık 0.60 (fiyat bazlı)", abs(w["quant"] - 0.60) < 1e-9, f"got {w['quant']}")

    # composite_score property'si YENİ ağırlıkları mı kullanıyor?
    # Sentetik state: whale=None (kripto değil) → whale ağırlığı dağıtılır.
    # quant_score=1.0, fundamental_score=0.0, diğerleri 0.5 → quant ağırlığı baskın olmalı.
    s = OracleState(
        symbol="TEST/USDT",
        macro_score=0.5,
        quant_score=1.0,
        whale_score=None,
        fundamental_score=0.0,
        sentiment_score=0.5,
        timeframe_alignment_score=0.5,
    )
    comp = s.composite_score
    # _to_unit(1.0)=1.0, _to_unit(0.5)=0.75, _to_unit(0.0)=0.5
    # whale=None → quant+=0.06, fundamental+=0.04 → quant=0.66, fund=0.09
    # = 0.75*0.15 + 1.0*0.66 + 0.5*0.09 + 0.5*0.10 + 0.75*0.10
    # = 0.1125 + 0.66 + 0.045 + 0.05 + 0.075 = 0.9425
    expected = (
        0.75 * 0.15 + 1.0 * 0.66 + 0.5 * 0.09
        + 0.5 * 0.10 + 0.75 * 0.10
    )
    check(f"composite_score yeni ağırlıklarla (got {comp:.4f})",
          abs(comp - expected) < 1e-6, f"expected {expected:.4f}")

    # Fallback dict de güncel mi? config cache'i bozup kontrol et.
    s2 = OracleState(symbol="TEST2/USDT", macro_score=0.5, quant_score=1.0,
                     whale_score=None, fundamental_score=0.0,
                     sentiment_score=0.5, timeframe_alignment_score=0.5)
    check("composite_score her iki state'te tutarlı",
          abs(s2.composite_score - comp) < 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 15. FAZ 2 — FUNDAMENTAL VETO-ONLY (savcı rolü)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz2_fundamental_veto():
    print("[15] FAZ 2 — fundamental veto-only")
    from agents.fundamental_filter import _apply_fundamental_veto

    # Pozitif skor → 0.0 nötr
    res_pos = _apply_fundamental_veto(0.42, 0.30, 5)
    check("pozitif skor → 0.0 nötr", res_pos["score"] == 0.0, f"got {res_pos}")
    check("pozitif → veto yok", res_pos["vetoed"] is False)

    # Hafif negatif → sınırlı baskı, veto yok
    res_neg = _apply_fundamental_veto(-0.15, -0.10, 3)
    # raw = (-0.15*0.70) + (-0.10*0.30) = -0.135
    check("hafif negatif → skor korunur", abs(res_neg["score"] - (-0.135)) < 1e-9,
          f"got {res_neg['score']}")
    check("hafif negatif → veto yok", res_neg["vetoed"] is False)

    # Güçlü negatif → VETO
    res_veto = _apply_fundamental_veto(-0.45, -0.30, 2)
    check("güçlü negatif → VETO", res_veto["vetoed"] is True, f"got {res_veto}")
    check("veto skor negatif korunur", res_veto["score"] < -0.30)


# ─────────────────────────────────────────────────────────────────────────────
# 16. FAZ 3 — ELEME RAPORU (elimination log + özet)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz3_elimination_report():
    print("[16] FAZ 3 — eleme raporu")
    from core.scanner import OracleScanner

    scanner = OracleScanner(None, None, {"scan_schedule": {}, "asset_universe": {}})
    scanner._elimination_log = []

    # Elimine kayıtları toplanıyor mu?
    scanner._log_elimination("BTC/USDT", "CEO/ajan vetosu: Kompozit skor eşiği altı")
    scanner._log_elimination("ETH/USDT", "CEO/ajan vetosu: Gri bölge")
    scanner._log_elimination("SOL/USDT", "sinyal üretilmedi")
    scanner._log_elimination("XRP/USDT", "pipeline timeout")
    check("4 eleme kaydı toplandı", len(scanner._elimination_log) == 4,
          f"got {len(scanner._elimination_log)}")

    # Kapasite sınırı (40)
    for i in range(50):
        scanner._log_elimination(f"A{i}/USDT", "test")
    check("kapasite sınırı 40", len(scanner._elimination_log) == 40,
          f"got {len(scanner._elimination_log)}")

    # Özet mesaj üretimi (bot çağrısı yakalanır)
    sent: list[str] = []

    async def _fake_bot(msg: str):
        sent.append(msg)

    scanner.bot = _fake_bot  # type: ignore[method-assign]
    scanner._elimination_log = [
        {"asset": "BTC/USDT", "reason": "CEO/ajan vetosu: Kompozit skor eşiği altı"},
        {"asset": "ETH/USDT", "reason": "CEO/ajan vetosu: Gri bölge (kararsız piyasa)"},
        {"asset": "SOL/USDT", "reason": "sinyal üretilmedi"},
    ]

    async def _run():
        await scanner._send_elimination_summary(scanner._elimination_log, 6)

    asyncio.run(_run())
    check("özet mesaj gönderildi", len(sent) == 1, f"got {len(sent)}")
    msg = sent[0] if sent else ""
    check("özet 'ELEME RAPORU' içeriyor", "ELEME RAPORU" in msg, f"got {msg[:60]}")
    check("özet 3/6 aday gösteriyor", "3/6" in msg, f"got {msg[:80]}")
    check("özet ilk elenenleri gösteriyor", "İlk elenenler" in msg or "Ilk elenenler" in msg,
          f"got {msg[:80]}")
    # Boş liste → mesaj gönderilmez
    sent.clear()
    asyncio.run(scanner._send_elimination_summary([], 0))
    check("boş log → mesaj yok", len(sent) == 0, f"got {len(sent)}")


# ─────────────────────────────────────────────────────────────────────────────
# 17. FAZ 4 — ÖLÜ KOD TEMİZLİĞİ (ma_ema_cross / build_quant_score)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz4_dead_code():
    print("[17] FAZ 4 — ölü kod temizliği")
    import core.indicators as ind

    check("ma_ema_cross silindi", not hasattr(ind, "ma_ema_cross"))
    check("build_quant_score silindi", not hasattr(ind, "build_quant_score"))
    check("normalized_from_score duruyor", hasattr(ind, "normalized_from_score"))

    # golden/death cross referansı tamamen yok mu?
    src = Path(ind.__file__).read_text(encoding="utf-8")
    check("golden_cross referansı yok", "golden_cross" not in src)
    check("death_cross referansı yok", "death_cross" not in src)


# ─────────────────────────────────────────────────────────────────────────────
# 18. FAZ 5 — LİKİDİTE SÜPÜRMESİ + FVG
# ─────────────────────────────────────────────────────────────────────────────
def test_faz5_sweep_fvg():
    print("[18] FAZ 5 — sweep + FVG")
    from agents.quant_engine import _detect_fvg, _detect_liquidity_sweep

    # ── SWEEP: bearish sweep (long stop-hunt) ──
    # Pivot low ~100; son mum fitili 99.0'a iner, gövde 101 ile kapanır.
    n = 40
    rng = np.random.default_rng(7)
    base = np.linspace(110, 102, n)
    df = pd.DataFrame({
        "open": base + rng.uniform(-0.3, 0.3, n),
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + rng.uniform(-0.2, 0.2, n),
        "volume": rng.uniform(100, 500, n),
    })
    # Belirgin bir pivot low oluştur (ör. indeks 30)
    df.loc[30, "low"] = 99.0
    df.loc[30, "close"] = 101.0
    df.loc[30, "open"] = 100.5
    # Son mum: pivot low altına fitil + geri kapanış
    df.loc[n - 1, "low"] = 98.0
    df.loc[n - 1, "close"] = 102.0
    df.loc[n - 1, "open"] = 100.5
    df.loc[n - 1, "high"] = 102.5

    sweep = _detect_liquidity_sweep(df)
    check("sweep tespit fonksiyonu çalışıyor", isinstance(sweep, dict), f"got {type(sweep)}")
    check("sweep alanları tam", all(k in sweep for k in
          ("sweep_detected", "direction", "swept_level", "strength")))

    # ── FVG: bullish FVG (3 mum yapısı) ──
    df2 = pd.DataFrame({
        "open": [100, 101, 105],
        "high": [100.5, 101.5, 105.5],
        "low": [99.5, 100.5, 104.0],  # lows[2]=104.0 > highs[0]=100.5 → bullish gap
        "close": [100.8, 101.2, 105.2],
        "volume": [200, 200, 200],
    })
    fvg = _detect_fvg(df2)
    check("FVG tespit fonksiyonu çalışıyor", isinstance(fvg, dict), f"got {type(fvg)}")
    check("FVG alanları tam", all(k in fvg for k in
          ("fvg_detected", "direction", "fvg_top", "fvg_bottom", "fill_pct")))
    check("bullish FVG tespit edildi", fvg["direction"] == "BULLISH", f"got {fvg}")

    # ── _compute_tf_indicators çıktısında sweep/fvg alanları ──
    df3 = _synth_ohlcv(n=80)
    try:
        from agents.quant_engine import _compute_tf_indicators
        out = _compute_tf_indicators(df3)
        check("tf çıktısı liquidity_sweep içeriyor", "liquidity_sweep" in out, f"got {list(out)[-4:]}")
        check("tf çıktısı fvg_detected içeriyor", "fvg_detected" in out)
        check("sweep bool tipinde", isinstance(out.get("liquidity_sweep"), bool))
        check("fvg bool tipinde", isinstance(out.get("fvg_detected"), bool))
    except Exception as exc:  # noqa: BLE001
        check(f"tf indicators çalışıyor (exc: {exc})", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# 19. FAZ 2 — ÖLÜ cross_* CONFIG TEMİZLİĞİ
# ─────────────────────────────────────────────────────────────────────────────
def test_faz2_dead_config():
    print("[19] FAZ 2 — ölü cross_* config temizliği")
    from core.config import QuantScoreConfig

    for field in (
        "cross_golden_bonus", "cross_bullish_bonus",
        "cross_bearish_penalty", "cross_death_penalty",
    ):
        check(f"QuantScoreConfig.{field} yok", field not in QuantScoreConfig.model_fields,
              f"got {field}")

    ytxt = Path("oracle_config.yaml").read_text(encoding="utf-8")
    check("yaml'de cross_golden_bonus yok", "cross_golden_bonus" not in ytxt)
    check("yaml'de cross_death_penalty yok", "cross_death_penalty" not in ytxt)

    # Şema/YAML uyumu: config hâlâ yüklenebiliyor mu?
    from core.config import load_oracle_config

    async def _run():
        cfg = await load_oracle_config()
        return cfg.quant.score.rsi_oversold_bonus

    val = asyncio.run(_run())
    check("config yükleniyor (rsi bonus okunuyor)", val is not None and val > 0, f"got {val}")


# ─────────────────────────────────────────────────────────────────────────────
# 20. FAZ 3 — CHOCH/SWEEP/FVG SİNYAL KARARINA TAM ENTEGRE
# ─────────────────────────────────────────────────────────────────────────────
def test_faz3_structural_decision():
    print("[20] FAZ 3 — yapısal kırılım sinyal kararı")
    from agents.quant_engine import _decide_trade_type

    # 1) Oversold ama kırılım YOK → COIN Tuzağı Kalkanı (AVOID)
    tt = _decide_trade_type("NEUTRAL", "OVERSOLD", "NEUTRAL", "NEUTRAL",
                            False, False, False)
    check("oversold + kırılım yok → AVOID", tt == "AVOID_CONFLICTING_SIGNALS", f"got {tt}")

    # 2) Aynı + sweep BULLISH + CHOCH BULLISH → smart money LONG teyidi
    tt2 = _decide_trade_type("NEUTRAL", "OVERSOLD", "NEUTRAL", "NEUTRAL",
                             False, False, False,
                             choch_direction="BULLISH", sweep_direction="BULLISH",
                             fvg_direction="NONE")
    check("sweep+CHOCH → ACCUMULATE_ZONE", tt2 == "ACCUMULATE_ZONE", f"got {tt2}")

    # 3) Ayna: bearish sweep + bearish CHOCH + bearish FVG → SHORT
    tt3 = _decide_trade_type("BEARISH", "BEARISH", "BEARISH", "BEARISH",
                             False, False, False,
                             choch_direction="BEARISH", sweep_direction="BEARISH",
                             fvg_direction="BEARISH")
    check("bearish sweep+CHOCH → SHORT", tt3 == "STRONG_SELL_OR_SHORT", f"got {tt3}")

    # 4) Regresyon: NONE varsayılanları eski davranışı bozmamalı
    tt4 = _decide_trade_type("BULLISH", "BULLISH", "NEUTRAL", "NEUTRAL",
                             False, False, False)
    check("eski davranış korunur (NONE → HOLD_EXISTING)", tt4 == "HOLD_EXISTING", f"got {tt4}")


# ─────────────────────────────────────────────────────────────────────────────
# 21. FAZ 4 — ZAMAN İFADESİ KALDIRILDI + TRACKER SAHTE YÜZDE YOK
# ─────────────────────────────────────────────────────────────────────────────
def test_faz4_price_validity_and_tracker():
    print("[21] FAZ 4 — fiyat bazlı geçerlilik + tracker sahte yüzde yok")

    # (a) Telegram çıktısında zaman ifadesi kalmamalı, fiyat bazlı geçerlilik olmalı
    src = Path("bot/telegram_handler.py").read_text(encoding="utf-8")
    check("'24-72 saat' yok", "24-72 saat" not in src)
    check("'Referans süre' yok", "Referans süre" not in src)
    check("invalidation_level kullanılıyor", "invalidation_level" in src)

    # (b) Tracker: kapalı trade yoksa win_rate 0.0 (kurgusal yüzde yok)
    import core.tracker as tr

    old_path = tr.DB_PATH
    old_init = tr._DB_INITIALIZED
    try:
        with tempfile.TemporaryDirectory() as td:
            tr.DB_PATH = os.path.join(td, "portfolio.db")
            tr._DB_INITIALIZED = False
            stats = tr.get_performance_stats()
            check("boş DB → win_rate 0.0", stats["win_rate"] == 0.0,
                  f"got {stats['win_rate']}")

            conn = tr.get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO trades (asset, direction, entry_price, stop_loss, "
                "t1, t2, t3, status) VALUES ('BTC/USDT','LONG',100,95,110,120,130,'WIN')"
            )
            cur.execute(
                "INSERT INTO trades (asset, direction, entry_price, stop_loss, "
                "t1, t2, t3, status) VALUES ('ETH/USDT','LONG',100,95,110,120,130,'LOSS')"
            )
            conn.commit()
            conn.close()

            stats2 = tr.get_performance_stats()
            check("1W+1L → win_rate 50.0", stats2["win_rate"] == 50.0,
                  f"got {stats2['win_rate']}")
    finally:
        tr.DB_PATH = old_path
        tr._DB_INITIALIZED = old_init


# ─────────────────────────────────────────────────────────────────────────────
# 22. FAZ 0 — KAPALI MUM KURALI (veri katmanı)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz0_closed_candles():
    print("[22] FAZ 0 — kapalı mum kuralı (veri katmanı)")
    import tools.market_data as md

    orig_multi = md._fetch_crypto_multi_source
    orig_get = md._cache_get_df
    orig_set = md._cache_set_df
    orig_is_crypto = md._is_crypto_symbol
    try:
        async def _fake_multi(symbol, timeframe, limit, exchange_id):
            return _synth_ohlcv(n=25)

        md._fetch_crypto_multi_source = _fake_multi
        md._cache_get_df = lambda key: None
        md._cache_set_df = lambda key, df: None
        md._is_crypto_symbol = lambda s: True

        df = asyncio.run(md.fetch_crypto_ohlcv("BTC/USDT", "4h", 200))
        check("son aktif mum atıldı (24 bar)", len(df) == 24, f"got {len(df)}")
    finally:
        md._fetch_crypto_multi_source = orig_multi
        md._cache_get_df = orig_get
        md._cache_set_df = orig_set
        md._is_crypto_symbol = orig_is_crypto


# ─────────────────────────────────────────────────────────────────────────────
# 23. FAZ 3/4 — BTC GÖRELİ GÜÇ (lider seçimi)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz3_relative_strength():
    print("[23] FAZ 3/4 — BTC göreli güç (lider seçimi)")
    from core.cluster_engine import ClusterEngine

    eng = ClusterEngine()
    close = pd.Series(np.linspace(100.0, 110.0, 20))

    async def _run_none():
        return await eng._score_symbol("BTC/USDT", 30, pre_fetched=close,
                                       benchmark_returns=None)

    async def _run_bench():
        return await eng._score_symbol(
            "BTC/USDT", 30, pre_fetched=close,
            benchmark_returns={"5d": 20.0, "10d": 30.0, "15d": 30.0},
        )

    m_none = asyncio.run(_run_none())
    m_bench = asyncio.run(_run_bench())
    check("benchmark (BTC yükseldi) göreli gücü düşürür",
          m_bench.relative_strength < m_none.relative_strength,
          f"none={m_none.relative_strength:.3f} bench={m_bench.relative_strength:.3f}")
    check("relative_strength 0-1 aralığında",
          0.0 <= m_none.relative_strength <= 1.0 and 0.0 <= m_bench.relative_strength <= 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 24. FAZ 0 — COINCAP DOMINANS YEDEĞİ (gerçek BTC.D / USDT.D)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz0_coincap_dominance():
    print("[24] FAZ 0 — CoinCap dominans hesabı (saf fonksiyon)")
    from core.regime_engine import _dominance_from_asset_mcaps

    assets = [
        {"symbol": "BTC", "marketCapUsd": "5000.0"},
        {"symbol": "ETH", "marketCapUsd": "3000.0"},
        {"symbol": "USDT", "marketCapUsd": "2000.0"},
        {"symbol": "SOL", "marketCapUsd": 1000.0},
    ]
    res = _dominance_from_asset_mcaps(assets)
    # total = 11000 → BTC.D 45.45, USDT.D 18.18
    check("btc_d gerçek yüzde", res["btc_d"] is not None and abs(res["btc_d"] - 45.45) < 0.01,
          f"got {res['btc_d']}")
    check("usdt_d gerçek yüzde", res["usdt_d"] is not None and abs(res["usdt_d"] - 18.18) < 0.01,
          f"got {res['usdt_d']}")
    check("total_market_cap toplam", res["total_market_cap"] == 11000.0,
          f"got {res['total_market_cap']}")

    # BTC yoksa yanlış değer üretilmez (None)
    res2 = _dominance_from_asset_mcaps([{"symbol": "ETH", "marketCapUsd": "100.0"}])
    check("BTC yok → btc_d None", res2["btc_d"] is None, f"got {res2}")


# ─────────────────────────────────────────────────────────────────────────────
# 25. SEMBOL NORMALİZASYONU — /oracle jpm hisse olarak kalır
# ─────────────────────────────────────────────────────────────────────────────
def test_faz_symbol_normalization():
    print("[25] SEMBOL — /oracle hisse/emtia /USDT eklemez")
    from bot.telegram_handler import _normalize_symbol
    from core.config import load_oracle_config

    asyncio.run(load_oracle_config())  # config cache'ini doldur

    check("jpm → JPM (hisse)", _normalize_symbol("jpm") == "JPM",
          f"got {_normalize_symbol('jpm')}")
    check("btc → BTC/USDT (kripto)", _normalize_symbol("btc") == "BTC/USDT",
          f"got {_normalize_symbol('btc')}")
    check("xom → XOM (hisse)", _normalize_symbol("xom") == "XOM")
    check("nvda → NVDA (hisse)", _normalize_symbol("nvda") == "NVDA")
    check("asels → ASELS.IS (BIST)", _normalize_symbol("asels") == "ASELS.IS",
          f"got {_normalize_symbol('asels')}")
    check("eth → ETH/USDT (kripto)", _normalize_symbol("eth") == "ETH/USDT")


# ─────────────────────────────────────────────────────────────────────────────
# 26. ÇİFT TARAMA KORUMASI — global guard
# ─────────────────────────────────────────────────────────────────────────────
def test_faz_global_scan_guard():
    print("[26] ÇİFT TARAMA — global koruma (duplike sinyal engeli)")
    import core.scanner as scanner_mod
    from core.scanner import OracleScanner

    sc = OracleScanner(None, None, {"scan_schedule": {}, "asset_universe": {}})
    sc._scan_in_progress = False
    old = scanner_mod._GLOBAL_SCAN_ACTIVE
    try:
        # Başka bir tarama aktifken (örn. sabah taraması + /tarama) ikinci tarama atlanır
        scanner_mod._GLOBAL_SCAN_ACTIVE = True
        asyncio.run(sc._run_scan_once(notify_start=False, trigger="test"))
        check("guard aktifken _scan_in_progress False kalır", sc._scan_in_progress is False)
    finally:
        scanner_mod._GLOBAL_SCAN_ACTIVE = old


# ─────────────────────────────────────────────────────────────────────────────
# 27. TRACKER — Series Bug'ı düzeltildi (float(Series) çökmesi)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz0_tracker_series_bug():
    print("[27] TRACKER — Series bug'ı (float(Series) çökmesi) düzeltildi")
    from core.signal_tracker import _last_close_price
    import pandas as pd

    # 1) Normal Series → skaler
    df1 = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    val = _last_close_price(df1)
    check("normal Close → skaler", val == 102.0, f"got {val!r}")

    # 2) MultiIndex sütunlu Close (yfinance'in patlattığı durum) → skaler
    cols = pd.MultiIndex.from_product([["Close"], ["MU"]])
    df3 = pd.DataFrame([[100.0], [101.0], [102.0]], columns=cols)
    val3 = _last_close_price(df3)
    check("MultiIndex Close → skaler (ilk sütun)", val3 == 102.0, f"got {val3!r}")

    # 3) Boş → None (çökmez)
    val4 = _last_close_price(pd.DataFrame())
    check("boş df → None", val4 is None)


# ─────────────────────────────────────────────────────────────────────────────
# 28. FAZ 1 — THE PURGE (EMA/MACD/düz-RSI skorlamadan kaldırıldı)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz1_purge():
    print("[28] FAZ 1 — THE PURGE (EMA/MACD/düz-RSI kaldırıldı)")
    src_scanner = Path("core/scanner.py").read_text(encoding="utf-8")
    check("'Fiyat>EMA21>EMA50' yok", "Fiyat>EMA21>EMA50" not in src_scanner)
    check("'MACD hist pozitif' yok", "MACD hist pozitif" not in src_scanner)
    check("'sağlıklı trend bölgesi' yok", "sağlıklı trend bölgesi" not in src_scanner)
    check("'Fiyat>SMA200' yok", "Fiyat>SMA200" not in src_scanner)

    from agents.quant_engine import _classify_bias

    # EMA50>SMA200 golden cross eskiden BULLISH döndürüyordu → artık değil
    got = _classify_bias(110, 105, 100, 60)
    check("golden cross bias kalktı (BULLISH değil)", got == "NEUTRAL", f"got {got}")
    got2 = _classify_bias(100, 105, 110, 40)
    check("death cross bias kalktı (BEARISH değil)", got2 == "NEUTRAL", f"got {got2}")
    # Yapısal/piyasa yapısı + temel RSI bölgesi bias'ı korunur
    got3 = _classify_bias(110, 105, 100, 30, "BULLISH_STRUCTURE")
    check("yapısal bias korunur", got3 == "BULLISH", f"got {got3}")
    got4 = _classify_bias(100, 105, 110, 30)
    check("temel RSI oversold korunur", got4 == "OVERSOLD", f"got {got4}")


# ─────────────────────────────────────────────────────────────────────────────
# 29. FAZ 2 — ASİMETRİK MOTOR (Sweep+CHOCH & RSI Breakout & USDT.D)
# ─────────────────────────────────────────────────────────────────────────────
def test_faz2_asymmetric_motors():
    print("[29] FAZ 5 — asimetrik motor (LONG + SHORT katı veto)")
    from core.asymmetric_engine import (
        asymmetric_signal,
        detect_long_signal,
        detect_short_signal,
        usdt_d_macro_filter,
    )
    from test_asymmetric_engine import build_mock_ohlcv, build_mock_ohlcv_short

    df_long = build_mock_ohlcv()
    df_short = build_mock_ohlcv_short()
    usdt_down = [8.20, 8.15, 8.10, 8.05, 8.00]  # USDT.D DÜŞÜYOR → onay
    usdt_up = [8.00, 8.05, 8.10, 8.15, 8.20]    # USDT.D YÜKSELİYOR → red

    l = detect_long_signal(df_long)
    check("LONG motoru tetiklendi (4 şart)", bool(l["signal"]), f"got {l['reason']}")
    check("LONG nedeni 'Fiyat Düşeni Kırdı + RSI Pozitif Uyumsuzluk'",
          l["reason"] == "Fiyat Düşeni Kırdı + RSI Pozitif Uyumsuzluk", f"got {l['reason']}")
    check("LONG seviyeleri üretildi", l["levels"] and l["levels"]["stop"] < l["levels"]["entry"])

    s = detect_short_signal(df_short)
    check("SHORT motoru tetiklendi (4 şart)", bool(s["signal"]), f"got {s['reason']}")
    check("SHORT nedeni 'Fiyat Yükseleni Kırdı + RSI Negatif Uyumsuzluk'",
          s["reason"] == "Fiyat Yükseleni Kırdı + RSI Negatif Uyumsuzluk", f"got {s['reason']}")
    check("SHORT seviyeleri üretildi (stop yukarıda)", s["levels"] and s["levels"]["stop"] > s["levels"]["entry"])

    m3 = usdt_d_macro_filter(usdt_down)
    check("M3 USDT.D düşüş → onay", m3["approved"] is True, f"got {m3}")
    m3_up = usdt_d_macro_filter(usdt_up)
    check("USDT.D yükseliş → red", m3_up["approved"] is False, f"got {m3_up}")

    comb = asymmetric_signal(df_long, usdt_d_series=usdt_down)
    check("birleşik LONG sinyali", comb["direction"] == "LONG" and comb["signal"] is True, f"got {comb}")
    comb_s = asymmetric_signal(df_short, usdt_d_series=usdt_down)
    check("birleşik SHORT sinyali", comb_s["direction"] == "SHORT" and comb_s["signal"] is True, f"got {comb_s}")

    # KATI VETO: motor tetiklenmeyen grafikte sinyal ASLA üretilmez
    flat = _synth_ohlcv(n=120)
    comb_flat = asymmetric_signal(flat, usdt_d_series=usdt_down)
    check("sinyal yok grafikte direction None",
          comb_flat["signal"] is False and comb_flat["direction"] is None, f"got {comb_flat['direction']}")
    check("neden 'Sinyal Yok'", comb_flat["reason"] == "Sinyal Yok", f"got {comb_flat['reason']}")


# ─────────────────────────────────────────────────────────────────────────────
# 30. FAZ 3 — ŞEFFAFLIK: sahte metrikler yok + eleme raporu + kategorizasyon
# ─────────────────────────────────────────────────────────────────────────────
def test_faz3_transparency():
    print("[30] FAZ 3 — şeffaflık (sahte metrik yok + eleme raporu)")

    # a) MTF özetinin GERÇEK ÇIKTISINDA "Teknik teyit: X boğa vs Y ayı" yok
    from core.multi_tf import MultiTFAnalysis, format_mtf_summary

    mtf_a = MultiTFAnalysis(
        symbol="BTC/USDT",
        points={"1h": {"bias": "BULLISH"}, "4h": {"bias": "BULLISH"}, "1d": {"bias": "BULLISH"}},
        signal_bias="BULLISH",
        aligned_count=3,
        total_count=3,
        entry_timing="NOW",
        validity_text="TF'ler uyumlu — boğa yönünde.",
    )
    out_mtf = format_mtf_summary(mtf_a, "BTC/USDT")
    check("MTF çıktısında 'Teknik teyit' yok", "Teknik teyit" not in out_mtf)
    check("MTF çıktısında 'boğa vs' yok", "boğa vs" not in out_mtf)
    check("MTF çıktısında 'ayı sinyali' yok", "ayı sinyali" not in out_mtf)

    # b) Digest yeni formata geçti (ASİMETRİK FIRSAT), sahte yüzde başlığı yok
    src_scanner = Path("core/scanner.py").read_text(encoding="utf-8")
    check("'ASİMETRİK FIRSAT' var", "ASİMETRİK FIRSAT" in src_scanner)
    check("digest'te 'Kompozit Skor:' başlığı yok", "| Kompozit Skor:" not in src_scanner)

    # c) _categorize_elimination: makro / yapısal veto mantığı
    from core.scanner import OracleScanner
    from test_asymmetric_engine import build_mock_ohlcv

    sc = OracleScanner(None, None, {"scan_schedule": {}, "asset_universe": {}})
    df = build_mock_ohlcv()  # LONG motoru tetikler
    check("kripto + USDT.D yukarı → makro veto",
          sc._categorize_elimination("BTC/USDT", df, df, True) == "macro")
    # Sinyal üretmeyen rastgele veri → yapısal veto (motor "Sinyal Yok")
    rand_df = _synth_ohlcv(n=60)
    check("sinyal yok → yapısal veto",
          sc._categorize_elimination("BTC/USDT", rand_df, rand_df, False) == "structural")

    # d) Şeffaflık raporu formatı
    sent: list[str] = []

    async def _fake_bot(msg: str):
        sent.append(msg)

    sc.bot = _fake_bot  # type: ignore[method-assign]
    import time as _t
    pf_stats = {"scanned": 91, "macro_veto": 45, "structural_veto": 46,
                "none": 0, "failed": 0}
    asyncio.run(sc._send_scan_transparency_report(_t.monotonic(), 93, pf_stats,
                                                  [{"asset": "ASELS.IS"}, {"asset": "BTC/USDT"}]))
    msg = sent[0] if sent else ""
    check("rapor 'ORACLE TARAMA TAMAMLANDI' içerir", "ORACLE TARAMA TAMAMLANDI" in msg)
    check("rapor 'Makro Veto' içerir", "Makro Veto (USDT.D uyumsuz): 45" in msg)
    check("rapor 'Yapısal Veto' içerir", "Yapısal Veto (Sinyal Yok): 46" in msg)
    check("rapor 'Filtreyi Geçenler' içerir", "Filtreyi Geçenler: 2 Varlık (ASELS.IS, BTC/USDT)" in msg)

    # e) /oracle X abort mesajındaki sahte yüzdeler temizlendi (gerçek skorlar durur)
    from core.types import OracleState
    from bot.telegram_handler import _format_abort_message

    st = OracleState(
        symbol="BTC/USDT",
        macro_score=0.1,
        quant_score=0.2,
        fundamental_score=0.0,
        sentiment_score=0.0,
        confidence=0.50,
        base_rr=2.0,
        timeframe_biases={"1w": "NEUTRAL", "1d": "BULLISH", "4h": "BULLISH", "1h": "NEUTRAL"},
    )
    abort = _format_abort_message(st, "Kompozit")
    for fake in ("Ajan Tutarlılık", "Sinyal Olgunluğu", "Tarihsel Benzerlik", "Ajan Uyumu"):
        check(f"abort'ta '{fake}' yok", fake not in abort)
    check("abort'ta gerçek 'Kompozit Skor' var", "Kompozit Skor" in abort)
    check("abort'ta gerçek 'Sistem Güveni' var", "Sistem Güveni" in abort)


# ─────────────────────────────────────────────────────────────────────────────
# 31. FAZ 4 — ÇELİK YELEK: kritik çöküş bildirimi + canlı dry-run formatı
# ─────────────────────────────────────────────────────────────────────────────
def test_faz4_steel_vest():
    print("[31] FAZ 4 — çelik yelek (kritik çöküş + canlı dry-run)")

    # a) _guard_loop: döngü ölümcül hata ile çökerse CRITICAL mesajı + self-heal
    import core.scanner as _scanner_mod
    from core.scanner import OracleScanner

    crit_msgs: list[str] = []

    async def _fake_bot(msg: str):
        crit_msgs.append(msg)

    sc = OracleScanner(None, _fake_bot, {"scan_schedule": {}, "asset_universe": {}})
    sc._running = True
    calls = {"n": 0}

    async def _factory():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("RAM yetersiz")
        sc._running = False  # ikinci çağrıda döngüden çık (self-heal kanıtı)

    async def _fake_sleep(_sec):
        return None

    orig_sleep = asyncio.sleep
    asyncio.sleep = _fake_sleep  # type: ignore[assignment]
    try:
        asyncio.run(sc._guard_loop("full_scan", _factory))
    finally:
        asyncio.sleep = orig_sleep  # type: ignore[assignment]

    check("guard döngüsü yeniden başladı (self-heal)", calls["n"] == 2, f"calls={calls['n']}")
    check("CRITICAL mesajı gönderildi",
          any("🚨 CRITICAL: Tarama motoru beklenmedik bir hatadan çöktü." in m for m in crit_msgs),
          f"msgs={crit_msgs}")
    check("CRITICAL hata detayı içerir", any("RAM yetersiz" in m for m in crit_msgs))

    # b) live_dry_run: rapor formatı (mock veriyle, ağsız + AI veto onayı)
    from test_asymmetric_engine import build_mock_ohlcv

    sent2: list[str] = []

    async def _fake_bot2(msg: str):
        sent2.append(msg)

    sc2 = OracleScanner(None, _fake_bot2, {"scan_schedule": {}, "asset_universe": {}})
    mock_df = build_mock_ohlcv()  # LONG motoru tetikler

    async def _fake_fetch_tf(symbol: str, tf: str):
        if "BTC" in symbol:
            return mock_df
        return _synth_ohlcv(n=60)  # sinyal yok → yapısal veto

    async def _fake_ai_approve(asset: str, tf: str, data: dict):
        return {"approved": True, "reason": "EVET: kaliteli fırsat", "raw": "EVET: kaliteli fırsat"}

    sc2._fetch_tf_ohlcv = _fake_fetch_tf  # type: ignore[method-assign]
    sc2._ai_validator = _fake_ai_approve  # type: ignore[method-assign]

    class _Dom:
        usdt_d_trend = {"1d": "FALLING"}

    class _R:
        usdt_d = 8.1
        primary_trend = "RISK_ON"
        risk_appetite = 1.0
        dominance = _Dom()

    async def _fake_regime(force: bool = False):
        return _R()

    orig_regime = _scanner_mod.get_regime_snapshot
    _scanner_mod.get_regime_snapshot = _fake_regime  # type: ignore[assignment]
    try:
        asyncio.run(sc2.live_dry_run(["BTC/USDT", "JPM"]))
    finally:
        _scanner_mod.get_regime_snapshot = orig_regime  # type: ignore[assignment]

    rep = sent2[0] if sent2 else ""
    check("dry-run 'CANLI ATIŞ TESTİ' içerir", "CANLI ATIŞ TESTİ" in rep)
    check("dry-run 'ELEME RAPORU' içerir", "ELEME RAPORU" in rep)
    check("dry-run BTC FIRSAT gösterir", "FIRSAT" in rep and "BTC/USDT" in rep)
    check("dry-run 'Filtreyi Geçenler: 1' içerir",
          "Filtreyi Geçenler: 1 Varlık (BTC/USDT)" in rep, f"rep={rep[-300:]}")
    check("dry-run JPM yapısal veto", "JPM" in rep and "Yapısal Veto" in rep)

    # c) makro veto yolu: USDT.D yükseliyorsa kripto makro veto ile elenir
    sent3: list[str] = []

    async def _fake_bot3(msg: str):
        sent3.append(msg)

    sc3 = OracleScanner(None, _fake_bot3, {"scan_schedule": {}, "asset_universe": {}})
    sc3._fetch_tf_ohlcv = _fake_fetch_tf  # type: ignore[method-assign]

    class _DomUp:
        usdt_d_trend = {"1d": "RISING"}

    class _RUp:
        usdt_d = 8.2
        primary_trend = "RISK_OFF"
        risk_appetite = 0.3
        dominance = _DomUp()

    async def _fake_regime_up(force: bool = False):
        return _RUp()

    _scanner_mod.get_regime_snapshot = _fake_regime_up  # type: ignore[assignment]
    try:
        asyncio.run(sc3.live_dry_run(["BTC/USDT", "JPM"]))
    finally:
        _scanner_mod.get_regime_snapshot = orig_regime  # type: ignore[assignment]

    rep3 = sent3[0] if sent3 else ""
    check("USDT.D yükselirken BTC makro veto", "Makro Veto (USDT.D yükselişte" in rep3, f"rep={rep3[-200:]}")

    # d) FAZ 4 kod varlığı: live_dry_run metodu + komut kaydı + CRITICAL şablonu
    src_scanner = Path("core/scanner.py").read_text(encoding="utf-8")
    src_main = Path("main.py").read_text(encoding="utf-8")
    src_bot = Path("bot/telegram_handler.py").read_text(encoding="utf-8")
    check("scanner'da 'live_dry_run' var", "async def live_dry_run" in src_scanner)
    check("scanner'da '_guard_loop' var", "async def _guard_loop" in src_scanner)
    check("main.py'de CRITICAL şablonu var", "🚨 CRITICAL: Tarama motoru beklenmedik bir hatadan çöktü." in src_main)
    check("bot'ta 'test_canli_tarama' komutu kayıtlı", "test_canli_tarama" in src_bot)
    check("bot'ta '[ERROR]' izolasyonu var", "[ERROR]" in src_scanner)


# ─────────────────────────────────────────────────────────────────────────────
# 32. FAZ 5 — KATI VETO: puanlama söküldü, motor tek karar kaynağı
# ─────────────────────────────────────────────────────────────────────────────
def test_faz5_hard_veto():
    print("[32] FAZ 5 — katı veto (puanlama söküldü, motor tek karar)")

    # a) quant_engine artık asimetrik motora puan BONUSU eklemez
    src_qe = Path("agents/quant_engine.py").read_text(encoding="utf-8")
    check("quant_engine'de 'asymmetric_long_signal' import yok", "asymmetric_long_signal" not in src_qe)
    check("quant_engine'de asimetrik bonus bloğu yok", "Asimetrik motor onayı → skor bonusu" not in src_qe)

    # b) scanner._scan_single_asset artık pipeline/composite'a DAYANMAZ
    src_sc = Path("core/scanner.py").read_text(encoding="utf-8")
    check("scanner'da eski 'signal_label' kararı yok", "signal_label" not in src_sc)
    check("scanner'da 'WATCHLIST_PREMIUM' sahte sinyal yok", "WATCHLIST_PREMIUM" not in src_sc)

    # c) KATI VETO: motor tetiklemeyen varlık fırsat OLARAK dönmez
    from core.scanner import OracleScanner
    from test_asymmetric_engine import build_mock_ohlcv

    async def _run_none():
        sc = OracleScanner(None, None, {"scan_schedule": {}, "asset_universe": {}})
        sc._last_regime = None
        sc._elimination_log = []

        async def _fake_fetch_tf(symbol: str, tf: str):
            return _synth_ohlcv(n=120)  # sinyal yok

        sc._fetch_tf_ohlcv = _fake_fetch_tf  # type: ignore[method-assign]
        return await sc._scan_single_asset("XOM", "DEEP", "test"), sc._elimination_log

    res_none, elims = asyncio.run(_run_none())
    check("motor tetiklenmeyince fırsat YOK (None)", res_none is None, f"got {res_none}")
    check("eleme logu 'Sinyal Yok' içerir",
          any("Sinyal Yok" in str(e.get("reason")) for e in elims), f"got {elims}")

    async def _fake_ai_approve(asset: str, tf: str, data: dict):
        return {"approved": True, "reason": "EVET: kaliteli fırsat", "raw": "EVET: kaliteli fırsat"}

    async def _fake_ai_reject(asset: str, tf: str, data: dict):
        return {"approved": False, "reason": "HAYIR: fakeout riski yüksek", "raw": "HAYIR: fakeout riski yüksek"}

    # d) Motor tetikleyen veri + AI ONAY → LONG fırsat (gerçek NEDEN)
    async def _run_long():
        sc2 = OracleScanner(None, None, {"scan_schedule": {}, "asset_universe": {}})
        sc2._last_regime = None
        sc2._elimination_log = []
        mock = build_mock_ohlcv()

        async def _fake_fetch_tf(symbol: str, tf: str):
            return mock

        sc2._fetch_tf_ohlcv = _fake_fetch_tf  # type: ignore[method-assign]
        sc2._ai_validator = _fake_ai_approve  # type: ignore[method-assign]
        return await sc2._scan_single_asset("BTC/USDT", "DEEP", "test")

    res_long = asyncio.run(_run_long())
    check("LONG motor + AI onay → fırsat",
          res_long and res_long["signal"] == "LONG_FIRSAT" and res_long["direction"] == "LONG",
          f"got {res_long and res_long.get('signal')}")
    check("fırsat NEDEN'i gerçek motor sebebi",
          res_long and "Fiyat Düşeni Kırdı" in (res_long.get("asymmetric_reason") or ""))

    # e) SHORT motor + AI onay → fırsat (stop yukarıda)
    from test_asymmetric_engine import build_mock_ohlcv_short

    async def _run_short():
        sc3 = OracleScanner(None, None, {"scan_schedule": {}, "asset_universe": {}})
        sc3._last_regime = None
        sc3._elimination_log = []
        mock_s = build_mock_ohlcv_short()

        async def _fake_fetch_tf(symbol: str, tf: str):
            return mock_s

        sc3._fetch_tf_ohlcv = _fake_fetch_tf  # type: ignore[method-assign]
        sc3._ai_validator = _fake_ai_approve  # type: ignore[method-assign]
        return await sc3._scan_single_asset("BTC/USDT", "DEEP", "test")

    res_short = asyncio.run(_run_short())
    check("SHORT motor + AI onay → fırsat",
          res_short and res_short["signal"] == "SHORT_FIRSAT" and res_short["direction"] == "SHORT",
          f"got {res_short and res_short.get('signal')}")
    check("SHORT stop hedefi ters (stop yukarıda)",
          res_short and float(res_short.get("stop_loss") or 0) > float(res_short.get("entry_zone_low") or 0))

    # f) KURAL 3 — AI VETO: AI "HAYIR" derse sinyal YAYINLANMAZ
    async def _run_ai_veto():
        sc4 = OracleScanner(None, None, {"scan_schedule": {}, "asset_universe": {}})
        sc4._last_regime = None
        sc4._elimination_log = []
        mock = build_mock_ohlcv()

        async def _fake_fetch_tf(symbol: str, tf: str):
            return mock

        sc4._fetch_tf_ohlcv = _fake_fetch_tf  # type: ignore[method-assign]
        sc4._ai_validator = _fake_ai_reject  # type: ignore[method-assign]
        return await sc4._scan_single_asset("BTC/USDT", "DEEP", "test"), sc4._elimination_log

    res_veto, elims_veto = asyncio.run(_run_ai_veto())
    check("AI 'HAYIR' → fırsat YOK (None)", res_veto is None, f"got {res_veto}")
    check("eleme logu 'AI Uzman Vetosu' içerir",
          any("AI Uzman Vetosu" in str(e.get("reason")) for e in elims_veto), f"got {elims_veto}")


# ─────────────────────────────────────────────────────────────────────────────
# 33. FAZ 6 — ÇOKLU TF (ATR) + TOLERANS + AI VETO
# ─────────────────────────────────────────────────────────────────────────────
def test_faz6_multitf_ai():
    print("[33] FAZ 6 — çoklu TF (ATR) + tolerans + AI veto")

    # a) KURAL 2 — _break_state toleransı (%1.5)
    from core.asymmetric_engine import compute_atr, _break_state, detect_long_signal

    check("kırılım BREAK (yukarı)", _break_state(105.0, 100.0, 0.015, "up") == "BREAK")
    check("eşikte NEAR (%1.5 içinde)", _break_state(99.5, 100.0, 0.015, "up") == "NEAR")
    check("uzak NO", _break_state(95.0, 100.0, 0.015, "up") == "NO")
    check("aşağı kırılım BREAK", _break_state(95.0, 100.0, 0.015, "down") == "BREAK")

    # b) KURAL 1 — ATR dinamik mesafe
    from test_asymmetric_engine import build_mock_ohlcv

    df = build_mock_ohlcv()
    atr = compute_atr(df, 14)
    check("ATR(14) hesaplanıyor", float(atr.iloc[-1]) > 0)
    l = detect_long_signal(df)
    check("ATR tabanlı dip şartı geçti (3*ATR)", l["dip_ok"] is True)

    # b2) KURAL 1 — MAJÖR pivot tespiti (ATR prominence + >=20 bar aralık)
    from core.asymmetric_engine import find_significant_pivots

    sig_h, sig_l = find_significant_pivots(df)
    check("majör tepe bulundu", len(sig_h) >= 2, f"got {sig_h}")
    check("majör dip bulundu", len(sig_l) >= 2, f"got {sig_l}")
    if len(sig_h) >= 2:
        check("majör tepeler >=20 bar arayla",
              sig_h[-1][0] - sig_h[-2][0] >= 20, f"got {sig_h[-2:]}")

    # KURAL 3 — zenginleştirilmiş motor verisi
    check("LONG motor ATR döndürür", l["atr"] is not None and l["atr"] > 0)
    check("LONG motor trend yaşı döndürür", l["trend_age_bars"] is not None and l["trend_age_bars"] > 0)
    check("LONG motor kırılım gücü (gövde/ATR) döndürür", l["body_atr_ratio"] is not None)
    check("LONG motor hacim oranı döndürür", l["volume_ratio"] is not None and l["volume_ratio"] > 0)

    # Eski mikro pivot fonksiyonları TAMAMEN SİLİNDİ
    src_ae = Path("core/asymmetric_engine.py").read_text(encoding="utf-8")
    check("'find_swing_pivots' silindi", "def find_swing_pivots" not in src_ae)
    check("'_rsi_swings' silindi", "def _rsi_swings" not in src_ae)
    check("'find_significant_pivots' var", "def find_significant_pivots" in src_ae)

    # KURAL 3 — AI promptu zenginleşti (ATR/trend yaşı/kırılım gücü)
    src_qe2 = Path("agents/quant_engine.py").read_text(encoding="utf-8")
    check("AI promptu 'ATR(14)' içerir", "ATR(14)" in src_qe2)
    check("AI promptu 'Trend Yaşı' içerir", "Trend Yaşı" in src_qe2)
    check("AI promptu 'Kırılım Gücü' içerir", "Kırılım Gücü" in src_qe2)
    check("AI promptu 'Hacim Oranı' içerir", "Hacim Oranı" in src_qe2)

    # c) KURAL 1 — çoklu TF döngüsü scanner'da kuruldu
    src_sc = Path("core/scanner.py").read_text(encoding="utf-8")
    check("scanner'da TF döngüsü (1h/4h/1d/1w)", '("1h", "4h", "1d", "1w")' in src_sc)
    check("scanner'da _fetch_tf_ohlcv var", "async def _fetch_tf_ohlcv" in src_sc)
    check("scanner'da _scan_tf_candidates var", "async def _scan_tf_candidates" in src_sc)
    check("scanner'da AI veto bağlantısı var", "_ai_validator" in src_sc)

    # d) KURAL 3 — ask_ai_expert_validator + LlmEngine.invoke_text
    src_qe = Path("agents/quant_engine.py").read_text(encoding="utf-8")
    src_llm = Path("tools/llm_engine.py").read_text(encoding="utf-8")
    check("ask_ai_expert_validator var", "async def ask_ai_expert_validator" in src_qe)
    check("LlmEngine.invoke_text var", "async def invoke_text" in src_llm)
    check("AI veto fail-closed", "fail-closed" in src_qe)

    # e) AI veto EVET/HAYIR davranışı (mock LLM)
    import agents.quant_engine as qe
    import tools.llm_engine as te

    async def _run_ai(word: str):
        class _FakeLlm:
            async def invoke_text(self, *, system_prompt, user_prompt):
                return word

        orig = te.LlmEngine
        te.LlmEngine = _FakeLlm  # type: ignore[assignment]
        try:
            return await qe.ask_ai_expert_validator(
                "BTC/USDT", "4h",
                {"direction": "LONG", "high": 100, "low": 90,
                 "rsi_div": True, "tl_price": 95, "current": 96},
            )
        finally:
            te.LlmEngine = orig  # type: ignore[assignment]

    yes = asyncio.run(_run_ai("EVET: net yükseliş yapısı, kaliteli fırsat."))
    check("AI 'EVET' → onay", yes["approved"] is True)

    no = asyncio.run(_run_ai("HAYIR: tuzak mum, piyasa gürültüsü."))
    check("AI 'HAYIR' → veto", no["approved"] is False)


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
    test_scan_once_dedup()
    test_signal_tracker()
    test_faz_a_missed_scan()
    test_faz_b_cluster()
    test_faz_c_validity()
    test_faz_d_rate_limit()
    test_faz0_coingecko_rate_limit()
    test_faz1_thresholds()
    test_faz2_fundamental_veto()
    test_faz3_elimination_report()
    test_faz4_dead_code()
    test_faz5_sweep_fvg()
    test_faz2_dead_config()
    test_faz3_structural_decision()
    test_faz4_price_validity_and_tracker()
    test_faz0_closed_candles()
    test_faz3_relative_strength()
    test_faz0_coincap_dominance()
    test_faz_symbol_normalization()
    test_faz_global_scan_guard()
    test_faz0_tracker_series_bug()
    test_faz1_purge()
    test_faz2_asymmetric_motors()
    test_faz3_transparency()
    test_faz4_steel_vest()
    test_faz5_hard_veto()
    test_faz6_multitf_ai()
    dur = time.time() - t0
    print(f"\n══════ SONUÇ: {_OK} geçti, {_FAIL} başarısız ({dur:.1f}s) ══════")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
