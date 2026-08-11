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

    # ── FASE C: composite_score @property → model_dump'a enjekte ediliyor ──
    class _FakeState:
        def __init__(self):
            self._d = {"signal_label": "STRONG_BUY", "t1": 110.0, "t2": 115.0, "t3": 122.0}

        def model_dump(self):
            return dict(self._d)

        @property
        def composite_score(self) -> float:
            return 0.77

        @property
        def is_halted(self) -> bool:
            return False

    sd = scanner._pipeline_state_to_dict(_FakeState())
    check(
        "state dökümü composite_score enjekte",
        abs(float(sd.get("composite_score", 0.0)) - 0.77) < 1e-9,
        f"got {sd.get('composite_score')}",
    )
    check("state dökümü is_halted False", sd.get("is_halted") is False)

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
    dur = time.time() - t0
    print(f"\n══════ SONUÇ: {_OK} geçti, {_FAIL} başarısız ({dur:.1f}s) ══════")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
