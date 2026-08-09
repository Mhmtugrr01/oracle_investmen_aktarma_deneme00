"""
KATMAN 0 — REGIME ENGINE (Rejim Motoru)
========================================
Tarama başına BİR KEZ global piyasa rejimini çeker, hesaplar ve cache'ler.

Sorumluluklar:
  1. Global verileri tek çekim toplar (DXY, VIX, SPY, US10Y, GC=F, BTC-USD,
     TRY=X, USDT.D, BTC.D, total mcap, ekonomik takvim).
  2. BTC.D çoklu zaman dilimi (5m/15m/1h/4h/1d/1w) serisini CoinGecko'nun
     ÜCRETSİZ market_cap_chart uçlarından üretir:
        BTC.D(t) = BTC market_cap(t) / total_market_cap(t) * 100
  3. `compute_regime()` ile rejim çıktısını üretir:
     primary_trend / intraday_timing / risk_appetite / entry_delay_hint / exit_urgency.
  4. `correlate_signal_with_regime()` ile sinyalin rejime göre geçerlilik
     penceresini hesaplar (örn: "15m LONG → 1h/4h BTC.D düşerken 2-6 saat geçerli").

USDT.D gerçeği: Stablecoin mcap tarihçesi ücretsiz API'de YOK. Bu yüzden:
  - USDT.D güncel değer CoinGecko /global'den,
  - günlük örnekler scan_store (SQLite) üzerinden birikir,
  - intraday timing proxy'si olarak BTC.D intraday momentumu kullanılır.
Sinyal güven etiketi (confidence) ile sunulur.

Singleton: `RegimeSnapshotProvider` tarama kapsamında BİR KEZ çeker;
macro_sentinel ve diğer ajanlar `get_regime_snapshot()` ile cache'ten okur.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from tools.market_data import (
    build_ssl_context,
    fetch_macro_bundle,
    fetch_stock_macro_data,
    pct_change_over,
)

logger = logging.getLogger(__name__)

COINGECKO_API = "https://api.coingecko.com/api/v3"

# ── Çoklu zaman dilimi listesi (proje standardı) ─────────────────────────────
MTF_TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d", "1w")

# ── Ekonomik takvim önbelleği (makro_sentinel ile aynı kaynak) ────────────────
_ECON_CACHE: dict[str, Any] = {"data": [], "fetched_at": None}
_ECON_CACHE_TTL_HOURS = 6


# ═══════════════════════════════════════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DominanceMTF:
    """BTC.D / USDT.D çoklu zaman dilimi serisi (değerler % cinsinden)."""

    btc_d: dict[str, list[float]] = field(default_factory=dict)
    btc_d_trend: dict[str, str] = field(default_factory=dict)
    usdt_d: dict[str, list[float]] = field(default_factory=dict)
    usdt_d_trend: dict[str, str] = field(default_factory=dict)
    confidence: str = "low"  # high | medium | low
    source: str = "unavailable"


@dataclass
class RegimeSnapshot:
    """Tek taramada kullanılan global rejim özeti."""

    captured_at: float = 0.0
    # Global seviyeler / momentum
    dxy: float | None = None
    dxy_change_5d: float | None = None
    vix: float | None = None
    spy_change_5d: float | None = None
    us10y: float | None = None
    us10y_delta_7d: float | None = None
    gold_change_5d: float | None = None
    btc_change_7d: float | None = None
    usd_try: float | None = None
    # Dominans / likidite
    usdt_d: float | None = None
    btc_d: float | None = None
    total_market_cap: float | None = None
    dominance: DominanceMTF = field(default_factory=DominanceMTF)
    # Ekonomik takvim
    econ_events_today: list[str] = field(default_factory=list)
    # Hesaplanmış rejim
    primary_trend: str = "NEUTRAL"  # RISK_ON | RISK_OFF | MIXED | NEUTRAL
    intraday_timing: str = "CHOPPY"  # BULLISH | BEARISH | CHOPPY
    risk_appetite: float = 1.0  # 0.4 .. 1.2 (çarpımsal güven düzeltmesi)
    entry_delay_hint: str = ""
    exit_urgency: str = "NORMAL"
    # Veri kaynak durumu (hangi uçlar başarılı/hangi fallback devrede)
    source_flags: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  COINGECKO YARDIMCILARI (SSL fallback + retry)
# ═══════════════════════════════════════════════════════════════════════════════

async def _coingecko_get_json(path: str, timeout_sec: int = 15) -> Any | None:
    """CoinGecko GET — sertifika fallback'li, 1 retry'li. Hata => None."""
    url = f"{COINGECKO_API}/{path}"
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    attempts = [(True, False), (False, False), (True, True)]  # (verify, retry)
    for verify, _retry in attempts:
        try:
            connector = aiohttp.TCPConnector(ssl=build_ssl_context(verify))
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"CoinGecko {path} HTTP {resp.status}")
                    return await resp.json()
        except aiohttp.ClientConnectorCertificateError:
            logger.warning(f"[REGIME] CoinGecko SSL hatası, doğrulamasız deneniyor: {path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[REGIME] CoinGecko {path} başarısız: {exc}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  BTC.D ÇOKLU ZAMAN DİLİMİ SERİSİ (Ücretsiz CoinGecko uçları)
# ═══════════════════════════════════════════════════════════════════════════════

def _resample_last(points: list[tuple[float, float]], rule_min: int) -> list[float]:
    """[(ts_s, value)] listesini rule_min dakikalık kovalara indirir (son değer)."""
    buckets: dict[int, float] = {}
    for ts, val in points:
        bucket = int(ts // (rule_min * 60))
        buckets[bucket] = val
    return [buckets[k] for k in sorted(buckets)]


def _ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1.0)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1.0 - k))
    return ema


def _trend_from_ema(values: list[float], fast: int = 2, slow: int = 5) -> str:
    """RISING | FALLING | FLAT — EMA(fast) vs EMA(slow)."""
    if len(values) < slow + 1:
        return "FLAT"
    ema_fast = _ema_series(values, fast)[-1]
    ema_slow = _ema_series(values, slow)[-1]
    diff = ema_fast - ema_slow
    eps = max(abs(max(values)) * 0.0001, 0.005)
    if diff > eps:
        return "RISING"
    if diff < -eps:
        return "FALLING"
    return "FLAT"


async def _fetch_btc_d_mtf() -> DominanceMTF:
    """
    BTC.D = BTC mcap / total mcap  (CoinGecko ücretsiz uçları)
      - days=1  => 5dk çözünürlük (5m/15m/1h/4h türetilir)
      - days=30 => günlük çözünürlük (1d/1w türetilir)
    """
    mtf = DominanceMTF(confidence="low", source="coingecko")
    try:
        total_5m_raw = await _coingecko_get_json("global/market_cap_chart?days=1")
        btc_5m_raw = await _coingecko_get_json("coins/bitcoin/market_chart?vs_currency=usd&days=1")
        if isinstance(total_5m_raw, dict) and isinstance(btc_5m_raw, dict):
            total_pts = total_5m_raw.get("market_cap") or []
            btc_pts = btc_5m_raw.get("market_caps") or []
            total_map = {int(ts / 1000): float(v) for ts, v in total_pts}
            btc_map = {int(ts / 1000): float(v) for ts, v in btc_pts}
            shared_ts = sorted(set(total_map) & set(btc_map))
            if len(shared_ts) > 50:
                points = [
                    (float(ts), btc_map[ts] / total_map[ts] * 100.0 if total_map[ts] else 0.0)
                    for ts in shared_ts
                ]
                for tf, rule in (("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240)):
                    series = _resample_last(points, rule)
                    if len(series) > 6:
                        mtf.btc_d[tf] = series
                        mtf.btc_d_trend[tf] = _trend_from_ema(series, fast=2, slow=5)
                mtf.confidence = "high"
        # Günlük (1d) + haftalık (1w) trend için 30 günlük seri
        total_30d = await _coingecko_get_json("global/market_cap_chart?days=30")
        btc_30d = await _coingecko_get_json("coins/bitcoin/market_chart?vs_currency=usd&days=30")
        if isinstance(total_30d, dict) and isinstance(btc_30d, dict):
            total_map = {int(ts / 1000): float(v) for ts, v in (total_30d.get("market_cap") or [])}
            btc_map = {int(ts / 1000): float(v) for ts, v in (btc_30d.get("market_caps") or [])}
            shared_ts = sorted(set(total_map) & set(btc_map))
            if len(shared_ts) > 10:
                points = [
                    (float(ts), btc_map[ts] / total_map[ts] * 100.0 if total_map[ts] else 0.0)
                    for ts in shared_ts
                ]
                daily = _resample_last(points, 24 * 60)
                if len(daily) > 6:
                    mtf.btc_d["1d"] = daily
                    mtf.btc_d_trend["1d"] = _trend_from_ema(daily, fast=2, slow=5)
                    mtf.btc_d_trend["1w"] = _trend_from_ema(daily, fast=2, slow=5) if len(daily) >= 12 else "FLAT"
                if mtf.confidence == "low":
                    mtf.confidence = "medium"
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[REGIME] BTC.D MTF hesabı başarısız: {exc}")
    return mtf


# ═══════════════════════════════════════════════════════════════════════════════
#  EKONOMİK TAKVİM (hafif, makro_sentinel ile aynı kaynak)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_econ_calendar_light() -> list[dict]:
    global _ECON_CACHE
    now = _dt.datetime.utcnow()
    cached_at = _ECON_CACHE.get("fetched_at")
    if cached_at and (now - cached_at).total_seconds() < _ECON_CACHE_TTL_HOURS * 3600:
        return _ECON_CACHE["data"]
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    _ECON_CACHE["data"] = data if isinstance(data, list) else []
                    _ECON_CACHE["fetched_at"] = now
                    return _ECON_CACHE["data"]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[REGIME] Ekonomik takvim çekilemedi: {exc}")
    return []


def _high_impact_today(events: list[dict]) -> list[str]:
    """ForexFactory formatında bugünkü YÜKSEK etkili olayları döndürür."""
    today_str = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    out: list[str] = []
    for ev in events:
        if "high" not in str(ev.get("impact", "")).lower():
            continue
        try:
            ev_dt = _dt.datetime.strptime(str(ev.get("date", "")), "%b %d, %Y")
            if ev_dt.strftime("%Y-%m-%d") == today_str:
                out.append(str(ev.get("title", "")))
        except ValueError:
            continue
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  REJİM HESAPLAMA
# ═══════════════════════════════════════════════════════════════════════════════

def _trend_label(delta_pct: float | None, threshold: float = 0.2) -> str:
    """RISING | FALLING | FLAT — makro_sentinel ile uyumlu eşikler."""
    if delta_pct is None:
        return "UNKNOWN"
    if delta_pct > threshold:
        return "RISING"
    if delta_pct < -threshold:
        return "FALLING"
    return "FLAT"


def compute_regime(snapshot: RegimeSnapshot) -> RegimeSnapshot:
    """
    Toplanan ham veriden rejim çıktılarını hesaplar (mutasyon + dönüş).
    """
    dxy_trend = _trend_label(snapshot.dxy_change_5d, 0.4)
    spy_trend = _trend_label(snapshot.spy_change_5d, 0.5)
    vix = snapshot.vix
    us10y_trend = _trend_label(snapshot.us10y_delta_7d, 0.2)
    us10y = snapshot.us10y

    # ── 1. PRIMARY TREND (günlük/büyük resim) ────────────────────────────────
    risk_points = 0
    total_points = 0
    if snapshot.spy_change_5d is not None:
        total_points += 1
        if snapshot.spy_change_5d > 0:
            risk_points += 1
    if snapshot.dxy_change_5d is not None:
        total_points += 1
        if snapshot.dxy_change_5d < 0:
            risk_points += 1
    if vix is not None:
        total_points += 1
        if vix < 20:
            risk_points += 1
    if us10y is not None and us10y_trend != "UNKNOWN":
        total_points += 1
        if us10y < 4.2:
            risk_points += 1

    btc_d_trend_d = snapshot.dominance.btc_d_trend.get("1d", "FLAT")
    if btc_d_trend_d != "UNKNOWN":
        total_points += 1
        if btc_d_trend_d != "RISING":
            risk_points += 1

    if total_points == 0:
        snapshot.primary_trend = "NEUTRAL"
    else:
        ratio = risk_points / total_points
        if ratio >= 0.75:
            snapshot.primary_trend = "RISK_ON"
        elif ratio <= 0.35:
            snapshot.primary_trend = "RISK_OFF"
        else:
            snapshot.primary_trend = "MIXED"

    # ── 2. INTRADAY TIMING (5m-4h BTC.D momentumu → altcoin para akışı) ─────
    btc_5m = snapshot.dominance.btc_d_trend.get("5m", "FLAT")
    btc_15m = snapshot.dominance.btc_d_trend.get("15m", "FLAT")
    btc_1h = snapshot.dominance.btc_d_trend.get("1h", "FLAT")
    btc_4h = snapshot.dominance.btc_d_trend.get("4h", "FLAT")

    # BTC.D düşüyor = para altcoinlere akıyor (risk iştahı intraday yükseliyor)
    timing_signal = 0
    timing_total = 0
    for trend in (btc_5m, btc_15m, btc_1h, btc_4h):
        if trend in ("RISING", "FALLING"):
            timing_total += 1
            if trend == "FALLING":
                timing_signal += 1
    if timing_total >= 2:
        if timing_signal / timing_total >= 0.6:
            snapshot.intraday_timing = "BULLISH"
        elif timing_signal / timing_total <= 0.3:
            snapshot.intraday_timing = "BEARISH"
        else:
            snapshot.intraday_timing = "CHOPPY"
    else:
        snapshot.intraday_timing = "CHOPPY"

    # ── 3. RISK APPETITE (çarpımsal güven düzeltmesi) ────────────────────────
    appetite = 1.0
    if vix is not None:
        if vix > 35:
            appetite *= 0.50
        elif vix > 25:
            appetite *= 0.80
        elif vix < 16:
            appetite *= 1.08
    if snapshot.dxy_change_5d is not None and snapshot.dxy_change_5d > 1.0:
        appetite *= 0.85
    if snapshot.usdt_d is not None and snapshot.usdt_d > 5.5 and _trend_label(snapshot.usdt_d or 0.0, 0.0) != "FALLING":
        appetite *= 0.85
    if us10y is not None and us10y > 4.5 and us10y_trend == "RISING":
        appetite *= 0.90
    if snapshot.spy_change_5d is not None and snapshot.spy_change_5d > 1.5:
        appetite *= 1.10
    if btc_d_trend_d == "FALLING" and snapshot.intraday_timing == "BULLISH":
        appetite *= 1.05
    snapshot.risk_appetite = round(max(0.40, min(1.20, appetite)), 3)

    # ── 4. GİRİŞ GECİKMESİ / ÇIKIŞ ACİLİYETİ ────────────────────────────────
    if snapshot.intraday_timing == "BULLISH":
        snapshot.entry_delay_hint = "Mevcut momentum rejimi lehine — girişte acele etmeye gerek yok, 5-15m teyit yeterli."
    elif snapshot.intraday_timing == "CHOPPY":
        snapshot.entry_delay_hint = "Rejim net değil — 1-4 saat arası bekleyip BTC.D yönünü teyit et."
    else:
        snapshot.entry_delay_hint = "BTC.D yükseliyor (para BTC'ye akıyor) — girişi ertele, BTC.D 4h dönüşünü bekle."

    if snapshot.primary_trend == "RISK_OFF" and (vix is not None and vix > 25):
        snapshot.exit_urgency = "HIZLANDIR — risk-off rejimi, kâr alma/zarar kesme öncelikli."
    elif snapshot.primary_trend == "RISK_ON":
        snapshot.exit_urgency = "NORMAL — rejim risk-on, trend takibi serbest."
    else:
        snapshot.exit_urgency = "NORMAL"

    if snapshot.econ_events_today:
        snapshot.warnings.append(
            "[REJİM] Yüksek etkili veri günü: " + " | ".join(snapshot.econ_events_today[:3])
        )
    return snapshot


# ═══════════════════════════════════════════════════════════════════════════════
#  SİNYAL ↔ REJİM KORELASYONU (Geçerlilik Penceresi)
# ═══════════════════════════════════════════════════════════════════════════════

_TF_VALIDITY_HOURS: dict[str, tuple[float, float]] = {
    "5m": (0.5, 2.0),
    "15m": (2.0, 6.0),
    "1h": (6.0, 24.0),
    "4h": (24.0, 72.0),
    "1d": (72.0, 240.0),
    "1w": (168.0, 1008.0),
}


def correlate_signal_with_regime(
    tf: str,
    direction: str,
    regime: RegimeSnapshot | None,
) -> dict[str, Any]:
    """
    Bir sinyalin rejimle korelasyonunu ve geçerlilik penceresini döndürür.

    Örnek: tf="15m", direction="LONG" iken regime.intraday_timing="BULLISH"
    ve primary_trend="RISK_ON" ise -> uzun pencere (2-6 saat), yüksek güven.
    Ters rejimde pencere kısalır + invalidation notu eklenir.
    """
    tf = tf if tf in _TF_VALIDITY_HOURS else "1h"
    direction = direction.upper()
    lo_h, hi_h = _TF_VALIDITY_HOURS[tf]

    result: dict[str, Any] = {
        "tf": tf,
        "direction": direction,
        "aligned": True,
        "confidence_modifier": 1.0,
        "validity_hours_min": lo_h,
        "validity_hours_max": hi_h,
        "validity_text": "",
        "invalidation_text": "",
    }

    if regime is None:
        result["aligned"] = False
        result["confidence_modifier"] = 0.90
        result["validity_text"] = f"{tf} sinyali: rejim verisi yok, standart pencere geçerli ({lo_h:g}-{hi_h:g} saat)."
        result["invalidation_text"] = "Rejim verisi alınamadığından sinyal teyidi için 2. zaman dilimini bekle."
        return result

    timing = regime.intraday_timing
    primary = regime.primary_trend

    if direction in ("LONG", "BUY", "ACCUMULATE", "STRONG_BUY"):
        # Altcoin para akışı + risk-on => LONG destekli
        if timing == "BULLISH" and primary in ("RISK_ON", "MIXED"):
            result["aligned"] = True
            result["confidence_modifier"] = min(1.15, 1.0 + (regime.risk_appetite - 1.0) * 0.5)
            result["validity_text"] = (
                f"{tf} LONG: rejim destekli (BTC.D {_fmt_btc_trend(regime, '1h')}, "
                f"genel {primary}). Pencere {lo_h:g}-{hi_h:g} saat; "
                f"{hi_h:g} saat üzeri günlük direnç bölgelerinde kar al."
            )
            result["invalidation_text"] = "BTC.D 4h yukarı dönerse veya DXY güçlenirse sinyal geçersiz."
        elif timing == "BEARISH" or primary == "RISK_OFF":
            result["aligned"] = False
            result["confidence_modifier"] = max(0.60, regime.risk_appetite)
            result["validity_text"] = (
                f"{tf} LONG: rejimle TERS (BTC.D {_fmt_btc_trend(regime, '4h')}, {primary}). "
                f"Pencere daraltıldı ({lo_h:g}-{min(hi_h, lo_h * 2):g} saat) — sadece kısa vadeli tepki."
            )
            result["invalidation_text"] = "BTC.D dönüşü veya VIX sıçraması sinyali anında iptal eder."
        else:
            result["aligned"] = True
            result["confidence_modifier"] = regime.risk_appetite
            result["validity_text"] = (
                f"{tf} LONG: rejim nötr (timing={timing}, trend={primary}). "
                f"Pencere {lo_h:g}-{hi_h:g} saat; yüksek zaman diliminde teyit iste."
            )
            result["invalidation_text"] = "BTC.D 4h yukarı dönüşü veya DXY > 1 günlük güçlenme geçersiz kılar."
    else:
        # SHORT / SELL / REDUCE
        if timing == "BEARISH" and primary in ("RISK_OFF", "MIXED"):
            result["aligned"] = True
            result["confidence_modifier"] = min(1.15, 1.0 + (1.0 - regime.risk_appetite) * 0.4)
            result["validity_text"] = (
                f"{tf} SHORT: rejim destekli (BTC.D {_fmt_btc_trend(regime, '1h')} yükseliyor, {primary}). "
                f"Pencere {lo_h:g}-{hi_h:g} saat."
            )
            result["invalidation_text"] = "BTC.D 4h aşağı dönerse veya VIX hızla düşerse sinyal geçersiz."
        elif timing == "BULLISH" or primary == "RISK_ON":
            result["aligned"] = False
            result["confidence_modifier"] = max(0.60, 1.0 - (regime.risk_appetite - 0.8))
            result["validity_text"] = (
                f"{tf} SHORT: rejimle TERS (BTC.D {_fmt_btc_trend(regime, '4h')}, {primary}). "
                f"Pencere daraltıldı ({lo_h:g}-{min(hi_h, lo_h * 2):g} saat)."
            )
            result["invalidation_text"] = "Risk-on devam ederse sinyal iptal olur."
        else:
            result["aligned"] = True
            result["confidence_modifier"] = regime.risk_appetite
            result["validity_text"] = (
                f"{tf} SHORT: rejim nötr. Pencere {lo_h:g}-{hi_h:g} saat."
            )
            result["invalidation_text"] = "BTC.D 4h aşağı dönüşü veya DXY zayıflaması geçersiz kılar."

    result["confidence_modifier"] = round(max(0.55, min(1.20, result["confidence_modifier"])), 3)
    return result


def _fmt_btc_trend(regime: RegimeSnapshot, tf: str) -> str:
    trend = regime.dominance.btc_d_trend.get(tf, "FLAT")
    return {"RISING": "yükseliyor", "FALLING": "düşüyor"}.get(trend, "yatay")


# ═══════════════════════════════════════════════════════════════════════════════
#  PROVIDER (Singleton + TTL)
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeSnapshotProvider:
    """Tarama kapsamında global rejimi BİR KEZ çeker, TTL boyunca servis eder."""

    def __init__(self, ttl_sec: int = 1800) -> None:
        self._ttl_sec = ttl_sec
        self._snapshot: RegimeSnapshot | None = None
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    def get(self) -> RegimeSnapshot | None:
        if self._snapshot is None:
            return None
        if time.time() - self._fetched_at > self._ttl_sec:
            return None
        return self._snapshot

    def clear(self) -> None:
        self._snapshot = None
        self._fetched_at = 0.0

    async def fetch_and_cache(self, force: bool = False) -> RegimeSnapshot:
        """Global verileri tek çekim toplar, rejimi hesaplar, cache'ler."""
        async with self._lock:
            if not force and self.get() is not None:
                return self._snapshot  # type: ignore[return-value]

            snap = RegimeSnapshot(captured_at=time.time())

            # ── Yatırım varlıkları (yfinance, paylaşılan TTL cache) ──────────
            try:
                bundle = await fetch_macro_bundle()  # DXY + VIX + SPY (paralel)
                dxy_df = bundle["DXY"]
                vix_df = bundle["VIX"]
                spy_df = bundle["SPY"]
                snap.dxy = float(dxy_df["close"].iloc[-1])
                snap.dxy_change_5d = pct_change_over(dxy_df, bars=5)
                snap.vix = float(vix_df["close"].iloc[-1])
                snap.spy_change_5d = pct_change_over(spy_df, bars=5)
                snap.source_flags["yf_bundle"] = "ok"
            except Exception as exc:  # noqa: BLE001
                snap.source_flags["yf_bundle"] = f"fail:{exc}"
                logger.warning(f"[REGIME] Makro bundle başarısız: {exc}")

            extra_tickers: dict[str, tuple[str, str]] = {
                "us10y": ("^TNX", "1mo"),
                "gold": ("GC=F", "1mo"),
                "btc": ("BTC-USD", "1mo"),
                "try": ("TRY=X", "1mo"),
            }
            extra_results = await asyncio.gather(
                *[
                    fetch_stock_macro_data(ticker, period=period, interval="1d")
                    for ticker, period in extra_tickers.values()
                ],
                return_exceptions=True,
            )
            for (key, _), result in zip(extra_tickers.items(), extra_results):
                if isinstance(result, Exception):
                    snap.source_flags[f"yf_{key}"] = f"fail:{result}"
                    continue
                df: Any = result
                try:
                    if key == "us10y":
                        snap.us10y = float(df["close"].iloc[-1])
                        snap.us10y_delta_7d = pct_change_over(df, bars=5)
                    elif key == "gold":
                        snap.gold_change_5d = pct_change_over(df, bars=5)
                    elif key == "btc":
                        snap.btc_change_7d = pct_change_over(df, bars=5)
                    elif key == "try":
                        snap.usd_try = float(df["close"].iloc[-1])
                    snap.source_flags[f"yf_{key}"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    snap.source_flags[f"yf_{key}"] = f"fail:{exc}"

            # ── Dominans / likidite (CoinGecko) ──────────────────────────────
            cg_global = await _coingecko_get_json("global")
            if isinstance(cg_global, dict):
                data = cg_global.get("data", cg_global)
                snap.usdt_d = float(data.get("market_cap_percentage", {}).get("usdt") or 0.0) or None
                snap.btc_d = float(data.get("market_cap_percentage", {}).get("btc") or 0.0) or None
                snap.total_market_cap = float(data.get("total_market_cap", {}).get("usd") or 0.0) or None
                snap.source_flags["coingecko_global"] = "ok"
            else:
                snap.source_flags["coingecko_global"] = "fail"

            # ── BTC.D çoklu zaman dilimi ─────────────────────────────────────
            snap.dominance = await _fetch_btc_d_mtf()
            if snap.btc_d is not None and snap.dominance.confidence == "low":
                snap.dominance.btc_d["1d"] = [snap.btc_d]
                snap.dominance.btc_d_trend["1d"] = "FLAT"
                snap.dominance.confidence = "medium"

            # ── USDT.D proxy notu (tarihçe ücretsiz API'de yok) ──────────────
            if snap.usdt_d is not None:
                snap.dominance.usdt_d["1d"] = [snap.usdt_d]
                snap.dominance.usdt_d_trend["1d"] = "FLAT"
                snap.dominance.source = "coingecko+proxy"

            # ── Ekonomik takvim ──────────────────────────────────────────────
            try:
                events = await _fetch_econ_calendar_light()
                snap.econ_events_today = _high_impact_today(events)
                snap.source_flags["econ_calendar"] = "ok"
            except Exception as exc:  # noqa: BLE001
                snap.source_flags["econ_calendar"] = f"fail:{exc}"

            # ── Rejimi hesapla ───────────────────────────────────────────────
            compute_regime(snap)

            # ── Kalıcı tarihçe (scan_store varsa; yoksa sessizce atla) ──────
            try:
                from core.scan_store import record_regime_snapshot

                await record_regime_snapshot(snap)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[REGIME] scan_store kaydı atlandı: {exc}")

            self._snapshot = snap
            self._fetched_at = time.time()
            return snap


# Modül seviyesi singleton
_PROVIDER: RegimeSnapshotProvider | None = None


def get_provider() -> RegimeSnapshotProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = RegimeSnapshotProvider()
    return _PROVIDER


async def get_regime_snapshot(force: bool = False) -> RegimeSnapshot | None:
    """
    Cache'ten rejim anlık görüntüsünü döndürür; yoksa/force ise çeker.
    Tarama başında bir kez `force=True` ile çağrılır, sonrası okuma amaçlıdır.
    """
    provider = get_provider()
    existing = provider.get()
    if existing is not None and not force:
        return existing
    try:
        return await provider.fetch_and_cache(force=force)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[REGIME] Rejim çekilemedi: {exc}")
        return None
