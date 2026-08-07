"""DÜĞÜM 1 — Makro Likidite Ajanı (Macro Sentinel) - Carry-Trade Reformed v2.0."""

import asyncio
import os
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Eksik Kütüphane Mühürlendi!

from loguru import logger
from core.asset_classifier import AssetType, classify_asset, is_crypto, is_gold, is_bist
from core.console import BLUE, CYAN, agent_print, error_print
from core.types import AgentNode, OracleState, PipelineStatus
from core.indicators import normalized_from_score
from tools.market_data import (
    build_ssl_context,
    fetch_macro_bundle,
    fetch_stock_macro_data,
    pct_change_over,
)

# ── Ekonomik Takvim Önbelleği ─────────────────────────────────────────────────
import datetime as _dt
_ECON_CALENDAR_CACHE: dict = {"data": [], "fetched_at": None}
_ECON_CACHE_TTL_HOURS = 6


async def _fetch_economic_calendar() -> list[dict]:
    """
    ForexFactory JSON feed üzerinden bu haftanın yüksek etkili ekonomik takvimini çeker.
    Ücretsiz, API key gerektirmez. Fallback: boş liste (sinyal engeli tetiklenmez).
    """
    global _ECON_CALENDAR_CACHE
    now = _dt.datetime.utcnow()
    cached_at = _ECON_CALENDAR_CACHE.get("fetched_at")
    if cached_at and (now - cached_at).total_seconds() < _ECON_CACHE_TTL_HOURS * 3600:
        return _ECON_CALENDAR_CACHE["data"]

    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    _ECON_CALENDAR_CACHE["data"] = data if isinstance(data, list) else []
                    _ECON_CALENDAR_CACHE["fetched_at"] = now
                    return _ECON_CALENDAR_CACHE["data"]
    except Exception as e:
        logger.warning(f"[MACRO] Ekonomik takvim çekilemedi: {e}")
    return []


def _check_high_impact_events_today(events: list[dict]) -> list[str]:
    """Bugün açıklanacak yüksek etkili (Impact: High) olayları listeler."""
    today_str = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    high_impact = []
    for ev in events:
        # ForexFactory formatı: {"date": "Jul 14, 2026", "impact": "High", "title": "CPI m/m", ...}
        ev_date_raw = str(ev.get("date", ""))
        ev_impact = str(ev.get("impact", "")).lower()
        ev_title = str(ev.get("title", ""))
        if "high" not in ev_impact:
            continue
        try:
            ev_dt = _dt.datetime.strptime(ev_date_raw, "%b %d, %Y")
            if ev_dt.strftime("%Y-%m-%d") == today_str:
                high_impact.append(ev_title)
        except ValueError:
            pass
    return high_impact


async def _fetch_coingecko_global() -> dict:
    url = "https://api.coingecko.com/api/v3/global"
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        connector = aiohttp.TCPConnector(ssl=build_ssl_context(True))
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"CoinGecko global HTTP {resp.status}")
                data = await resp.json()
    except aiohttp.ClientConnectorCertificateError:
        logger.warning("CoinGecko SSL doğrulama başarısız, doğrulamasız fallback aktif.")
        connector = aiohttp.TCPConnector(ssl=build_ssl_context(False))
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"CoinGecko global HTTP {resp.status}")
                data = await resp.json()
    return data.get("data", {})


def _get_dominance_via_yfinance_sync() -> dict:
    """
    CoinGecko alternatifi: yfinance üzerinden BTC dominans proxy hesabı.
    USDT dominansı yfinance ile güvenilir şekilde üretilemediğinden None döner.
    """
    tickers = yf.download(
        ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"],
        period="1d",
        interval="1h",
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    close = tickers.get("Close")
    if close is None or close.empty:
        raise ValueError("yfinance close verisi boş")

    latest = close.ffill().iloc[-1]
    btc_price = float(latest.get("BTC-USD"))
    eth_price = float(latest.get("ETH-USD"))
    bnb_price = float(latest.get("BNB-USD"))
    sol_price = float(latest.get("SOL-USD"))

    if any(price <= 0 for price in (btc_price, eth_price, bnb_price, sol_price)):
        raise ValueError("yfinance fiyat verisi geçersiz")

    btc_cap = btc_price * 19_700_000
    eth_cap = eth_price * 120_000_000
    bnb_cap = bnb_price * 145_000_000
    sol_cap = sol_price * 465_000_000

    top4_cap = btc_cap + eth_cap + bnb_cap + sol_cap
    total_cap_estimate = top4_cap * 1.35
    if total_cap_estimate <= 0:
        raise ValueError("toplam cap tahmini oluşturulamadı")

    btc_dominance = (btc_cap / total_cap_estimate) * 100.0
    return {
        "btc_dominance": round(btc_dominance, 2),
        "usdt_dominance": None,
        "total_market_cap": float(total_cap_estimate),
        "source": "yfinance_proxy",
        "warning": "CoinGecko erişilemedi — BTC.D yfinance proxy ile hesaplandı, USDT.D mevcut değil",
    }


def _trend_label(delta_pct_7d: float) -> str:
    if delta_pct_7d > 0.2:
        return "RISING"
    if delta_pct_7d < -0.2:
        return "FALLING"
    return "FLAT"


def _compute_macro_score_0_100(
    dxy_chg: float,
    vix_chg: float,
    spy_chg: float,
    vix_level: float,
) -> tuple[float, list[str]]:
    """
    Makro skor (0-100).
    DXY/VIX yukselis = risk-off (dusuk skor), dusus = risk-on (yuksek skor).
    """
    score = 50.0
    notes: list[str] = []

    # DXY: guclu dolar kripto/risk varlik baskisi
    if dxy_chg > 1.0:
        score -= min(20.0, dxy_chg * 4)
        notes.append(f"DXY +{dxy_chg:.2f}% (risk-off)")
    elif dxy_chg < -1.0:
        score += min(20.0, abs(dxy_chg) * 4)
        notes.append(f"DXY {dxy_chg:.2f}% (dolar zayif)")

    # VIX: korku endeksi
    if vix_chg > 10.0:
        score -= min(18.0, vix_chg * 0.8)
        notes.append(f"VIX +{vix_chg:.2f}% (korku artisi)")
    elif vix_chg < -10.0:
        score += min(18.0, abs(vix_chg) * 0.8)
        notes.append(f"VIX {vix_chg:.2f}% (korku azalisi)")

    if vix_level > 30:
        score -= 10
        notes.append(f"VIX seviye {vix_level:.1f} (yuksek)")
    elif vix_level < 18:
        score += 6
        notes.append(f"VIX seviye {vix_level:.1f} (dusuk)")

    # SPY: genel risk iştahı proxy
    if spy_chg > 2.0:
        score += min(12.0, spy_chg * 2)
        notes.append(f"SPY +{spy_chg:.2f}% (risk-on)")
    elif spy_chg < -2.0:
        score -= min(12.0, abs(spy_chg) * 2)
        notes.append(f"SPY {spy_chg:.2f}% (risk-off)")

    return round(max(0.0, min(100.0, score)), 2), notes


def _compute_usdt_dominance_trend(usdt_d_now: float | None, usdt_d_prev: float | None) -> str:
    """Gerçek USDT dominance trendi: iki farklı zaman dilimindeki değerden hesaplanır."""
    if usdt_d_now is None or usdt_d_prev is None:
        return "UNKNOWN"
    delta = usdt_d_now - usdt_d_prev
    return _trend_label(delta)


def _compute_cross_score_crypto(
    usdt_d: float | None, usdt_d_trend: str,
    btc_d: float | None, btc_change_7d: float,
    total_market_cap: float | None, is_altcoin: bool,
    dxy_delta_7d: float, us10y: float, us10y_trend: str,
    vix_level: float, jpy_risk: bool,
    warnings: list[str], critical_dominance_outage: bool,
) -> tuple[float, float]:
    """Kripto varlıklarına özel cross-asset skoru ve confidence modifier."""
    confidence_modifier = 1.0

    if usdt_d_trend == "RISING" and usdt_d is not None and usdt_d > 5.5:
        warnings.append("USDT.D YÜKSELİYOR — Kripto'dan çıkış var, sinyal ağırlığı düşürüldü")
        confidence_modifier *= 0.70
    elif usdt_d_trend == "FALLING" and usdt_d is not None:
        warnings.append("📉 USDT.D DÜŞÜYOR — Nakit kriptoya geçiş yapıyor")

    btc_d_trend = _trend_label(btc_change_7d)
    if btc_d_trend == "RISING" and is_altcoin:
        warnings.append("BTC.D YÜKSELİYOR — Altcoin sezonu değil, dikkat")
        confidence_modifier *= 0.85
    elif btc_d_trend == "FALLING" and is_altcoin:
        warnings.append("🚀 BTC.D DÜŞÜŞ — Altcoin sezonu işaretleri")

    if dxy_delta_7d > 1.0:
        warnings.append("DXY GÜÇLENİYOR — Kripto baskı altında")
        confidence_modifier *= 0.82

    if us10y > 4.5 and us10y_trend == "RISING":
        warnings.append("US10Y YÜKSELİYOR — Reel faiz baskısı kripto için negatif")
        confidence_modifier *= 0.87

    if vix_level > 25:
        warnings.append(f"VIX YÜKSEK ({vix_level:.1f}) — Risk-off, kripto baskısı")
        confidence_modifier *= 0.80 if vix_level <= 35 else 0.55

    if jpy_risk:
        warnings.append("⚠️ JPY GÜÇLİ — Carry unwind = kripto satış baskısı")
        confidence_modifier *= 0.88

    risk_penalty = (1.0 - confidence_modifier) * 100.0
    reversal_bonus = 12.0 if usdt_d_trend == "FALLING" and (usdt_d or 0.0) > 4.0 else 0.0
    base = 70.0 - risk_penalty + reversal_bonus
    base += 2.0 if (total_market_cap or 0.0) > 0 else -2.0
    if usdt_d is None:
        warnings.append("USDT.D verisi mevcut değil (CoinGecko erişilemedi) — bu analiz eksik")
        base *= 0.80
    if critical_dominance_outage:
        base = 25.0
        warnings.append("Cross-asset analizi devre dışı — diğer ajanlar çalışmaya devam ediyor")
    return round(max(0.0, min(100.0, base)), 2), confidence_modifier


def _compute_cross_score_gold(
    dxy_delta_7d: float, us10y: float, us10y_trend: str,
    vix_level: float, jpy_risk: bool, gold_change_7d: float,
    warnings: list[str],
) -> tuple[float, float]:
    """Altına özel cross-asset skoru. USDT.D/BTC.D burada hiç yer almaz."""
    confidence_modifier = 1.0
    base = 55.0

    # DXY: altınla ters korelasyon (%40 ağırlık)
    if dxy_delta_7d > 1.5:
        base -= min(16.0, dxy_delta_7d * 5)
        confidence_modifier *= 0.80
        warnings.append(f"DXY GÜÇLENİYOR ({dxy_delta_7d:+.2f}%) — Altın baskı altında")
    elif dxy_delta_7d < -1.5:
        base += min(16.0, abs(dxy_delta_7d) * 5)
        warnings.append(f"DXY ZAYIFLADI ({dxy_delta_7d:+.2f}%) — Altın için pozitif")

    # US10Y reel faiz: altınla ters korelasyon (%35 ağırlık)
    if us10y > 4.5 and us10y_trend == "RISING":
        base -= 12.0
        confidence_modifier *= 0.85
        warnings.append(f"US10Y YÜKSEK ({us10y:.2f}%) — Reel faiz altın üzerinde baskı")
    elif us10y < 3.5 or us10y_trend == "FALLING":
        base += 8.0
        warnings.append(f"US10Y DÜŞÜŞ ({us10y:.2f}%) — Düşen reel faiz altın için pozitif")

    # VIX: yüksek VIX = safe-haven akışı = altın için BULLISH (%15 ağırlık)
    if vix_level > 30:
        base += 10.0
        warnings.append(f"🟡 VIX YÜKSEK ({vix_level:.1f}) — Altın safe-haven talebi artabilir")
    elif vix_level > 20:
        base += 4.0

    # JPY güçlenme: safe-haven yapısı altın ile aynı yönde
    if jpy_risk:
        base += 5.0
        warnings.append("🇯🇵 JPY GÜÇLİ — Safe-haven akışı altını destekleyebilir")

    if gold_change_7d > 2.0:
        base += 5.0
    elif gold_change_7d < -2.0:
        base -= 5.0

    return round(max(0.0, min(100.0, base)), 2), confidence_modifier


def _compute_cross_score_us_stock(
    dxy_delta_7d: float, us10y: float, us10y_trend: str,
    vix_level: float, spy_chg: float, jpy_risk: bool,
    warnings: list[str],
) -> tuple[float, float]:
    """ABD hisse senetlerine özel cross-asset skoru."""
    confidence_modifier = 1.0
    base = 65.0

    if vix_level > 25:
        base -= min(20.0, (vix_level - 25) * 1.2)
        confidence_modifier *= 0.80 if vix_level <= 35 else 0.55
        warnings.append(f"VIX YÜKSEK ({vix_level:.1f}) — Hisse senetleri baskı altında")

    if dxy_delta_7d > 1.0:
        base -= min(10.0, dxy_delta_7d * 3)
        warnings.append(f"DXY GÜÇLENİYOR — Uluslararası gelirlere olumsuz etkisi olabilir")

    if us10y > 5.0 and us10y_trend == "RISING":
        base -= 10.0
        confidence_modifier *= 0.88
        warnings.append(f"US10Y > 5% ({us10y:.2f}%) — Hisse değerlemeleri baskı altında")
    elif us10y < 4.0 and us10y_trend == "FALLING":
        base += 6.0
        warnings.append(f"US10Y DÜŞÜŞ ({us10y:.2f}%) — Hisse senetleri için uygun ortam")

    if spy_chg > 2.0:
        base += min(8.0, spy_chg * 1.5)
    elif spy_chg < -2.0:
        base -= min(10.0, abs(spy_chg) * 1.5)
        warnings.append(f"SPY ZAYIF ({spy_chg:+.2f}%) — Genel piyasa baskısı")

    if jpy_risk:
        base -= 5.0
        confidence_modifier *= 0.92
        warnings.append("⚠️ JPY GÜÇLİ — Carry unwind hisse satışına yol açabilir")

    return round(max(0.0, min(100.0, base)), 2), confidence_modifier


def _compute_cross_score_bist(
    usdtry_chg: float | None, dxy_delta_7d: float,
    vix_level: float, warnings: list[str],
) -> tuple[float, float]:
    """BIST hisseleri için Türkiye makro odaklı cross-asset skoru."""
    confidence_modifier = 1.0
    base = 55.0

    if usdtry_chg is not None:
        if usdtry_chg > 3.0:
            base -= min(25.0, usdtry_chg * 4)
            confidence_modifier *= 0.65
            warnings.append(f"USDTRY SERT YÜKSELİŞ ({usdtry_chg:+.2f}%) — TL baskı altında, BIST olumsuz")
        elif usdtry_chg > 1.5:
            base -= 10.0
            confidence_modifier *= 0.82
            warnings.append(f"USDTRY YUKARI ({usdtry_chg:+.2f}%) — Kur baskısı izleniyor")
        elif usdtry_chg < -1.0:
            base += 8.0
            warnings.append(f"USDTRY AŞAĞI ({usdtry_chg:+.2f}%) — TL güçlenme, BIST pozitif")
    else:
        warnings.append("USDTRY verisi alınamadı — BIST makro analizi eksik")
        base -= 5.0

    if vix_level > 25:
        base -= min(15.0, (vix_level - 25) * 1.0)
        confidence_modifier *= 0.85
        warnings.append(f"VIX YÜKSEK ({vix_level:.1f}) — Gelişmekte olan piyasalardan kanali olumsuz")

    if dxy_delta_7d > 1.5:
        base -= 8.0
        warnings.append(f"DXY GÜÇLİ — Gelişmekte olan piyasaları baskılıyor")

    return round(max(0.0, min(100.0, base)), 2), confidence_modifier


async def run_macro_sentinel(state: OracleState) -> OracleState:
    cycle = state.retry_count + 1
    asset_type = classify_asset(state.symbol)
    agent_print(
        "MACRO_SENTINEL",
        f"Devrede -> {state.symbol} [{asset_type.value}] | Döngü #{cycle}",
        CYAN,
    )

    try:
        is_crypto_flag = is_crypto(state.symbol)
        bundle = await fetch_macro_bundle()
        dxy_ext = await fetch_stock_macro_data("DXY", period="1mo", interval="1d")
        us10y_df = await fetch_stock_macro_data("^TNX", period="1mo", interval="1d")
        gold_df = await fetch_stock_macro_data("GC=F", period="1mo", interval="1d")

        warnings: list[str] = []

        # Dominance verileri: yalnızca kripto için çekilir ve kullanılır
        btc_d: float | None = None
        usdt_d: float | None = None
        usdt_d_prev: float | None = None
        total_market_cap: float | None = None
        critical_dominance_outage = False

        if is_crypto_flag:
            try:
                cg_global = await _fetch_coingecko_global()
                usdt_d = float(cg_global.get("market_cap_percentage", {}).get("usdt", 0.0))
                btc_d = float(cg_global.get("market_cap_percentage", {}).get("btc", 0.0))
                total_market_cap = float(cg_global.get("total_market_cap", {}).get("usd", 0.0))
                # USDT.D trend için önceki değer proxy (CoinGecko tarihsel API key gerektirir)
                usdt_d_prev = usdt_d * 0.97 if usdt_d else None
            except Exception as exc:
                logger.warning(f"CoinGecko başarısız: {exc}. yfinance proxy devreye giriyor.")
                try:
                    dominance_data = await asyncio.to_thread(_get_dominance_via_yfinance_sync)
                    btc_d = dominance_data.get("btc_dominance")
                    usdt_d = dominance_data.get("usdt_dominance")
                    total_market_cap = dominance_data.get("total_market_cap")
                    if dominance_data.get("warning"):
                        warnings.append(str(dominance_data["warning"]))
                except Exception as exc2:
                    logger.error(f"yfinance proxy da başarısız: {exc2}")
                    critical_dominance_outage = True
                    warnings.append("KRİTİK: BTC.D ve USDT.D verileri alınamadı (CoinGecko + yfinance başarısız)")

        dxy_df = bundle["DXY"]
        vix_df = bundle["VIX"]
        spy_df = bundle["SPY"]

        try:
            calendar_data = await _fetch_economic_calendar()
            econ_events_today = _check_high_impact_events_today(calendar_data)
            if econ_events_today:
                event_str = " | ".join(econ_events_today[:3])
                warnings.append(
                    f"[EKONOMİK TAKVİM] YÜKSEK ETKİLİ VERİ GÜNÜ: {event_str} "
                    "— Bugün büyük pozisyon almaktan kaçının!"
                )
                agent_print("MACRO_SENTINEL", f"⚠️ Ekonomik Takvim: {event_str}", BLUE)
        except Exception as ec_exc:
            logger.warning(f"[MACRO] Ekonomik takvim işleme hatası: {ec_exc}")

        dxy_chg = pct_change_over(dxy_df, bars=5)
        vix_chg = pct_change_over(vix_df, bars=5)
        spy_chg = pct_change_over(spy_df, bars=5)
        vix_level = float(vix_df["close"].iloc[-1])
        dxy_price = float(dxy_df["close"].iloc[-1])
        us10y = float(us10y_df["close"].iloc[-1])
        us10y_delta_7d = pct_change_over(us10y_df, bars=5)
        dxy_delta_7d = pct_change_over(dxy_ext, bars=5)
        gold_change_7d = pct_change_over(gold_df, bars=5)
        dxy_trend = _trend_label(dxy_delta_7d)
        us10y_trend = _trend_label(us10y_delta_7d)

        jpy_risk = False
        usdjpy_df = bundle.get("USDJPY")
        if usdjpy_df is not None:
            jpy_change_7d = pct_change_over(usdjpy_df, bars=5)
            if jpy_change_7d < -1.50:
                jpy_risk = True

        usdtry_chg: float | None = None
        if is_bist(state.symbol):
            try:
                usdtry_df = await fetch_stock_macro_data("TRY=X", period="1mo", interval="1d")
                usdtry_chg = pct_change_over(usdtry_df, bars=5)
            except Exception as e:
                logger.warning(f"[BIST MACRO] USDTRY çekilemedi: {e}")

        btc_change_7d = 0.0
        if is_crypto_flag:
            try:
                btc_df = await fetch_stock_macro_data("BTC-USD", period="1mo", interval="1d")
                btc_change_7d = pct_change_over(btc_df, bars=5)
            except Exception:
                pass

        is_altcoin = is_crypto_flag and not state.symbol.upper().startswith("BTC/")
        usdt_d_trend = _compute_usdt_dominance_trend(usdt_d, usdt_d_prev)

        if is_crypto_flag:
            cross_asset_score, confidence_modifier = _compute_cross_score_crypto(
                usdt_d=usdt_d, usdt_d_trend=usdt_d_trend,
                btc_d=btc_d, btc_change_7d=btc_change_7d,
                total_market_cap=total_market_cap, is_altcoin=is_altcoin,
                dxy_delta_7d=dxy_delta_7d, us10y=us10y, us10y_trend=us10y_trend,
                vix_level=vix_level, jpy_risk=jpy_risk,
                warnings=warnings, critical_dominance_outage=critical_dominance_outage,
            )
        elif is_gold(state.symbol):
            cross_asset_score, confidence_modifier = _compute_cross_score_gold(
                dxy_delta_7d=dxy_delta_7d, us10y=us10y, us10y_trend=us10y_trend,
                vix_level=vix_level, jpy_risk=jpy_risk,
                gold_change_7d=gold_change_7d, warnings=warnings,
            )
        elif is_bist(state.symbol):
            cross_asset_score, confidence_modifier = _compute_cross_score_bist(
                usdtry_chg=usdtry_chg, dxy_delta_7d=dxy_delta_7d,
                vix_level=vix_level, warnings=warnings,
            )
        else:  # ABD hisseleri ve endeksler
            cross_asset_score, confidence_modifier = _compute_cross_score_us_stock(
                dxy_delta_7d=dxy_delta_7d, us10y=us10y, us10y_trend=us10y_trend,
                vix_level=vix_level, spy_chg=spy_chg, jpy_risk=jpy_risk,
                warnings=warnings,
            )

        score_0_100, notes = _compute_macro_score_0_100(dxy_chg, vix_chg, spy_chg, vix_level)
        macro_score = normalized_from_score(score_0_100)

        agent_print("MACRO_SENTINEL", f"DXY={dxy_price:.2f} ({dxy_chg:+.2f}%) | VIX={vix_level:.2f} | US10Y={us10y:.2f}%", BLUE)
        agent_print("MACRO_SENTINEL", f"Makro Skor={score_0_100:.1f}/100 -> {macro_score:+.3f} | Cross={cross_asset_score:.1f} | ConfMod={confidence_modifier:.2f}", BLUE)
        for note in notes:
            agent_print("MACRO_SENTINEL", note, CYAN)
        for warning in warnings:
            agent_print("MACRO_SENTINEL", warning, CYAN)

        btc_d_text = "NA" if btc_d is None else f"{btc_d:.2f}"
        usdt_d_text = "NA" if usdt_d is None else f"{usdt_d:.2f}"
        return state.model_copy(
            update={
                "current_node": AgentNode.MACRO_SENTINEL,
                "status": PipelineStatus.RUNNING,
                "macro_score": macro_score,
                "cross_asset_score": round(cross_asset_score, 2),
                "cross_asset_warnings": warnings,
                "messages": [
                    f"[MACRO_SENTINEL] DXY={dxy_price:.2f} VIX={vix_level:.2f} "
                    f"score={score_0_100:.1f} norm={macro_score:+.3f} "
                    f"btc_d={btc_d_text} usdt_d={usdt_d_text} dxy_trend={dxy_trend} "
                    f"cross_asset_score={cross_asset_score:.1f}",
                    f"[ASSET_TYPE] {asset_type.value} | conf_mod={confidence_modifier:.2f}",
                ],
            }
        )

    except Exception as exc:
        msg = f"Makro Hata Olustu: {exc}"
        error_print(msg)
        return state.model_copy(
            update={
                "current_node": AgentNode.MACRO_SENTINEL,
                "status": PipelineStatus.RUNNING,
                "fatal_error": msg,
                "messages": [f"[MACRO_SENTINEL] ERROR {msg}"],
            }
        )

        warnings: list[str] = []
        critical_dominance_outage = False
        try:
            cg_global = await _fetch_coingecko_global()
            dominance_data = {
                "btc_dominance": float(cg_global.get("market_cap_percentage", {}).get("btc", 0.0)),
                "usdt_dominance": float(cg_global.get("market_cap_percentage", {}).get("usdt", 0.0)),
                "total_market_cap": float(cg_global.get("total_market_cap", {}).get("usd", 0.0)),
                "source": "coingecko",
                "warning": None,
            }
        except Exception as exc:
            logger.warning(f"CoinGecko başarısız: {exc}. yfinance proxy devreye giriyor.")
            try:
                dominance_data = await asyncio.to_thread(_get_dominance_via_yfinance_sync)
            except Exception as exc2:
                logger.error(f"yfinance proxy da başarısız: {exc2}")
                dominance_data = {
                    "btc_dominance": None,
                    "usdt_dominance": None,
                    "total_market_cap": None,
                    "source": "unavailable",
                    "warning": (
                        "KRİTİK: BTC.D ve USDT.D verileri alınamadı "
                        "(CoinGecko + yfinance başarısız)"
                    ),
                }
                critical_dominance_outage = True

        btc_d = dominance_data.get("btc_dominance")
        usdt_d = dominance_data.get("usdt_dominance")
        total_market_cap = dominance_data.get("total_market_cap")
        if dominance_data.get("warning"):
            warning_text = str(dominance_data["warning"])
            if is_crypto_symbol or "USDT.D" not in warning_text:
                warnings.append(warning_text)

        dxy_df = bundle["DXY"]
        vix_df = bundle["VIX"]
        spy_df = bundle["SPY"]

        # ── 📅 EKONOMİK TAKVİM (Forex Factory — ücretsiz) ────────────────────
        econ_events_today: list[str] = []
        try:
            calendar_data = await _fetch_economic_calendar()
            econ_events_today = _check_high_impact_events_today(calendar_data)
            if econ_events_today:
                event_str = " | ".join(econ_events_today[:3])
                warnings.append(
                    f"[EKONOMİK TAKVİM] YÜKSEK ETKİLİ VERİ GÜNÜ: {event_str} "
                    "— Bugün büyük pozisyon almaktan kaçının!"
                )
                agent_print("MACRO_SENTINEL", f"⚠️ Ekonomik Takvim: {event_str}", BLUE)
        except Exception as ec_exc:
            logger.warning(f"[MACRO] Ekonomik takvim işleme hatası: {ec_exc}")

        # ── 🇯🇵 JAPON YENİ CARRY-TRADE RISK MONITOR (R06) ──
        usdjpy_df = bundle.get("USDJPY")
        if usdjpy_df is not None:
            # JPY=X düşüşü = Yen'in ABD dolarına karşı değer kazanması (Carry Unwind Tehlikesi)
            jpy_change_7d = pct_change_over(usdjpy_df, bars=5)
            if jpy_change_7d < -1.50:
                warnings.append(f"⚠️ JAPON YENİ GÜÇLENİYOR (Carry-Trade Unwind Riski) -> USD/JPY: {jpy_change_7d:+.2f}%")

        dxy_chg = pct_change_over(dxy_df, bars=5)
        vix_chg = pct_change_over(vix_df, bars=5)
        spy_chg = pct_change_over(spy_df, bars=5)
        vix_level = float(vix_df["close"].iloc[-1])
        dxy_price = float(dxy_df["close"].iloc[-1])

        us10y = float(us10y_df["close"].iloc[-1])
        us10y_delta_7d = pct_change_over(us10y_df, bars=5)
        dxy_delta_7d = pct_change_over(dxy_ext, bars=5)
        btc_change_7d = pct_change_over(btc_df, bars=5)
        total2_change_7d = pct_change_over(total2_df, bars=5)
        total3_change_7d = pct_change_over(total3_df, bars=5)
        gold_change_7d = pct_change_over(gold_df, bars=5)

        btc_d_trend = _trend_label(btc_change_7d)
        
        # ── USDT.D ÇOKLU ZAMAN DİLİMİ SÜZGECİ (Multi-Timeframe USDT.D Tracker) ──
        usdt_d_trend = "UNKNOWN"
        if usdt_d is not None:
            usdt_d_trend = _trend_label(usdt_d - 7.0) if usdt_d > 7.0 else "FALLING"
            
        # ── 🇯🇵 USDT.D AYNALAMA KALKANI: ÇOK BOYUTLU REVERSAL ANALİZİ (R03) ──
        # USDT-USD grafiğinin fiyat ve RSI momentumunu saniyede hesaplar
        usdt_reversal_detected = False
        try:
            usdt_hist = await fetch_stock_macro_data("USDT-USD", period="1mo", interval="1d")
            if not usdt_hist.empty:
                usdt_close = usdt_hist["close"]
                # pandas_ta ile RSI hesapla
                usdt_rsi_s = ta.rsi(usdt_close, length=14)
                if usdt_rsi_s is not None and not usdt_rsi_s.empty:
                    u_price_now = float(usdt_close.iloc[-1])
                    u_price_prev = float(usdt_close.iloc[-14])
                    u_rsi_now = float(usdt_rsi_s.iloc[-1])
                    u_rsi_prev = float(usdt_rsi_s.iloc[-14])
                    
                    # 1. Negatif Uyumsuzluk (Zirveden para çıkışı teyidi)
                    if u_price_now > u_price_prev and u_rsi_now < u_rsi_prev:
                        usdt_reversal_detected = True
                        warnings.append("📉 USDT.D AYNALAMA: Negatif Uyumsuzluk Teyit Edildi — Akıllı Para Nakitten Kriptoya Geçiyor!")
                        
                    # 2. RSI Direnç Sarkması (RSI 70/75 Zirve Rejection)
                    if u_rsi_prev > 68.0 and u_rsi_now < 68.0:
                        usdt_reversal_detected = True
                        warnings.append("📉 USDT.D AYNALAMA: RSI Direnci Aşağı Kırıldı — Para nakitte kalmıyor!")
        except Exception as e:
            logger.warning(f"[USDT.D] Derin aynalama analizi atlandı: {e}")
            
        dxy_trend = _trend_label(dxy_delta_7d)
        us10y_trend = _trend_label(us10y_delta_7d)

        confidence_modifier = 1.0
        asset_is_altcoin = not state.symbol.upper().startswith("BTC/")

        if usdt_d_trend == "RISING" and usdt_d is not None and usdt_d > 5.5:
            warnings.append("USDT.D YÜKSELİYOR — Kripto'dan çıkış var, sinyal ağırlığı düşürüldü")
            confidence_modifier *= 0.70

        if btc_d_trend == "RISING" and asset_is_altcoin:
            warnings.append("BTC.D YÜKSELİYOR — Altcoin sezonu değil, dikkat")
            confidence_modifier *= 0.85

        if dxy_trend == "RISING" and dxy_delta_7d > 1.0:
            warnings.append("DXY GÜÇLENİYOR — Risk varlıkları baskı altında")
            confidence_modifier *= 0.80

        if us10y > 4.5 and us10y_trend == "RISING":
            warnings.append("US10Y YÜKSEK — Reel faiz baskısı BTC/Altın için negatif")
            confidence_modifier *= 0.85

        if vix_level > 25:
            warnings.append(f"VIX YÜKSEK ({vix_level:.1f}) — Piyasa paniği, yeni pozisyon riski artmış")
            if vix_level > 35:
                confidence_modifier *= 0.50

        total2_vs_btc = total2_change_7d - btc_change_7d
        if total2_vs_btc > 5.0:
            warnings.append("🚀 ALTCOIN SEZONU HAREKETLENİYOR — Altcoin performansı BTC'yi geçiyor!")
        elif total2_vs_btc < -5.0:
            warnings.append("⚠️ BTC DOMİNANS ARTIŞI — Para BTC'ye akıyor, altcoin riski yüksek")

        risk_penalty = (1.0 - confidence_modifier) * 100.0
        
        # USDT.D Aynalama bonusunu (Reversal) çapraz skora dahil et!
        reversal_bonus = 15.0 if usdt_reversal_detected else 0.0
        
        cross_asset_score = max(
            0.0,
            min(
                100.0,
                70.0
                - risk_penalty
                + reversal_bonus
                + (2.0 if (total_market_cap or 0.0) > 0 else -2.0)
                + (2.0 if gold_change_7d > 0 else 0.0)
                + (2.0 if total3_change_7d > 0 else 0.0),
            ),
        )
        if usdt_d is None and is_crypto_symbol:
            warnings.append("USDT.D verisi mevcut değil (CoinGecko erişilemedi) — bu analiz eksik")
            cross_asset_score = max(20.0, cross_asset_score * 0.80)

        if critical_dominance_outage:
            cross_asset_score = 25.0
            warnings.append("Cross-asset analizi devre dışı — diğer ajanlar çalışmaya devam ediyor")

        score_0_100, notes = _compute_macro_score_0_100(
            dxy_chg, vix_chg, spy_chg, vix_level
        )
        macro_score = normalized_from_score(score_0_100)

        agent_print(
            "MACRO_SENTINEL",
            f"DXY={dxy_price:.2f} ({dxy_chg:+.2f}%) | VIX={vix_level:.2f} ({vix_chg:+.2f}%)",
            BLUE,
        )
        agent_print(
            "MACRO_SENTINEL",
            f"SPY 5g degisim={spy_chg:+.2f}% | Makro Skor={score_0_100:.1f}/100 -> {macro_score:+.3f}",
            BLUE,
        )
        for note in notes:
            agent_print("MACRO_SENTINEL", note, CYAN)
        for warning in warnings:
            agent_print("MACRO_SENTINEL", warning, CYAN)

        btc_d_text = "NA" if btc_d is None else f"{btc_d:.2f}"
        usdt_d_text = "NA" if usdt_d is None else f"{usdt_d:.2f}"
        return state.model_copy(
            update={
                "current_node": AgentNode.MACRO_SENTINEL,
                "status": PipelineStatus.RUNNING,
                "macro_score": macro_score,
                "cross_asset_score": round(cross_asset_score, 2),
                "cross_asset_warnings": warnings,
                "messages": [
                    f"[MACRO_SENTINEL] DXY={dxy_price:.2f} VIX={vix_level:.2f} "
                    f"score={score_0_100:.1f} norm={macro_score:+.3f} "
                    f"btc_d={btc_d_text} usdt_d={usdt_d_text} dxy_trend={dxy_trend} "
                    f"cross_asset_score={cross_asset_score:.1f}",
                    f"[USDT_D_MODIFIER] {confidence_modifier}" # Dinamik Güç Çarpanı CEO şatılına yükleniyor!
                ],
            }
        )

    except Exception as exc:
        msg = f"Makro Hata Olustu: {exc}"
        error_print(msg)
        return state.model_copy(
            update={
                "current_node": AgentNode.MACRO_SENTINEL,
                "status": PipelineStatus.RUNNING,
                "fatal_error": msg,
                "messages": [f"[MACRO_SENTINEL] ERROR {msg}"],
            }
        )