"""
PROJECT OLYMPUS V2 — core/scanner.py (4 KATMANLI PİPELINE)
═══════════════════════════════════════════════════════════
KATMAN 0 — REJİM MOTORU  : Makro rejim tarama başında BİR KEZ çekilir (512MB dostu)
KATMAN 1 — PREFILTER     : Sınırlı eşzamanlılıkla hızlı momentum süzgeci (önbellekli veri)
KATMAN 2 — DERİN PİPELINE: Aday başına timeout + heartbeat + duvar saati bütçesi + MTF
KATMAN 3 — TESLİMAT      : Rejim korelasyonlu Digest v2, gece 04:00→09:00 penceresi

Gece taraması 04:00'te başlar, tarama tamamlanınca (en geç 09:00) otomatik teslim edilir.
/tarama komutu da aynı korumalı akışı on-demand tetikler (handler asla bloke olmaz).
"""

from __future__ import annotations

import asyncio
import gc
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from loguru import logger

from core.asset_classifier import is_crypto
from core.multi_tf import (
    _resample_4h_from_1h,
    analyze_multi_tf,
    format_mtf_summary,
)
from core.regime_engine import RegimeSnapshot, correlate_signal_with_regime, get_regime_snapshot
from core.scan_store import get_scan_store
from core.trade_plan import build_trade_plan
from tools.market_data import fetch_crypto_ohlcv, fetch_stock_macro_data


# ── GLOBAL TARAMA KİLİDİ ──────────────────────────────────────────────────────
# Tüm OracleScanner örnekleri (sabah döngüsü + /tarama komutu) aynı anda tarama
# yapmamalı. Örnek başına `_scan_in_progress` yalnızca AYNI örneği korur; /tarama
# her seferinde YENİ bir OracleScanner oluşturduğu için iki tarama aynı anda
# koşuyordu → duplike sinyal/digest/tracker kaydı. Bu global bayrak bunu engeller.
_GLOBAL_SCAN_ACTIVE = False


class OracleScanner:
    def __init__(self, pipeline_runner, telegram_bot, config: dict):
        """
        pipeline_runner: Mevcut LangGraph pipeline'ını çalıştıran fonksiyon
        telegram_bot: Telegram mesaj gönderme fonksiyonu
        config: oracle_config.yaml içeriği
        """
        self.pipeline = pipeline_runner
        self.bot = telegram_bot
        self.config = config
        self.asset_universe = config.get("asset_universe", {})
        self.scan_config = config.get("scan_schedule", {})

        self._watchlist: dict = {}
        self._alert_cooldowns: dict[str, float] = {}  # Anti-Spam Sistemi
        self._last_full_scan: Optional[datetime] = None
        self._running = False

        # V2: tarama koruması + ilerleme takibi
        self._scan_in_progress = False
        self._scan_progress: dict[str, int] = {"scanned": 0, "total": 0, "found": 0}
        self._last_regime: Optional[RegimeSnapshot] = None

        # Batch-fetching configuration to avoid rate limits and full-loop failures
        self._batch_size: int = int(self.scan_config.get("batch_size", 40))
        self._batch_cooldown: float = float(self.scan_config.get("batch_cooldown_sec", 1.7))
        # Concurrency limit for per-batch parallel tasks
        self._concurrency_limit: int = int(self.scan_config.get("concurrency_limit", 8))

        # FAZ B: korelasyon kümeleme sonuçları (her tarama koşusunda yenilenir)
        self._cluster_rank_map: dict[str, int] = {}
        self._cluster_theme_map: dict[str, str] = {}

    async def start(self):
        """Tarayıcıyı başlat — üç paralel döngü çalıştır."""
        self._running = True
        await asyncio.gather(
            self._full_scan_loop(),
            self._watchlist_monitor_loop(),
            self._daily_briefing_loop(),
            self._signal_tracker_loop(),
        )

    async def _signal_tracker_loop(self):
        """
        ADIM 6 — Açık sinyallerin durumunu periyodik günceller (win/loss takibi).
        Kayıtlı sinyallerin TP/SL seviyelerine ulaşıp ulaşmadığını kontrol eder;
        böylece /stats gerçek kapanan sinyallerden win-rate üretir (sahte veri yok).
        """
        await asyncio.sleep(60)  # başlangıç stabilizasyonu
        while self._running:
            try:
                from core.signal_tracker import get_signal_tracker

                updated = await get_signal_tracker().update_open_signals()
                if updated:
                    logger.info(f"[SCANNER] Tracker: {updated} sinyal durumu güncellendi.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[SCANNER] Tracker döngüsü hatası: {exc}")
            await asyncio.sleep(1800)  # 30 dakikada bir

    async def stop(self):
        self._running = False

    def _get_all_assets(self) -> list[str]:
        result: list[str] = []
        for assets in self.asset_universe.values():
            result.extend(assets)
        return list(set(result)) # Çift varlıkları temizle

    def _pipeline_state_to_dict(self, state) -> dict:
        if state is None:
            return {}
        if hasattr(state, "model_dump"):
            d = state.model_dump()
            # ⚠️ KRİTİK DÜZELTME: composite_score bir @property'dir ve Pydantic v2
            # model_dump() property alanlarını İÇERMEZ. Bu yüzden scanner hep 0.0
            # okuyordu → "Puan Onay: 0%" + WATCHLIST_PREMIUM (composite>=0.35) hiç
            # tetiklenmiyordu. Değeri state objesinden açıkça enjekte ediyoruz.
            try:
                d["composite_score"] = float(state.composite_score)
            except Exception:
                d["composite_score"] = 0.0
            try:
                d["is_halted"] = bool(state.is_halted)
            except Exception:
                d["is_halted"] = False
            return d
        if isinstance(state, dict):
            return state
        return dict(state)

    def _ticker_for_watchlist(self, asset: str) -> Optional[str]:
        token = asset.upper()
        if "/" in token:
            # Genel kripto eşlemesi: BTC/USDT → BTC-USD (40+ sembolü tek kuralda çözer)
            base = token.split("/")[0]
            if base in ("USDT", "USDC", "BUSD"):
                return None
            return f"{base}-USD"
        return token

    # =========================================================================
    # ── ⚔️ İKİ KATMANLI PREFILTER SKORLAMA (FASE A) ──
    #    Katman 1 "TREND TAKİPÇİSİ": yükselen trenddeki varlıklar (momentum)
    #    Katman 2 "DİP DÖNÜŞÜ"     : aşırı satım + dönüş sinyalleri (divergence/
    #                                hook/CHoCH/trendline kırılımı)
    #    Sert "RSI<55 VE artıyor" kapısı KALDIRILDI — RSI>78 sadece ceza alır.
    # =========================================================================
    @staticmethod
    def _pick_ohlcv_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
        c = "Close" if "Close" in df.columns else "close"
        h = "High" if "High" in df.columns else "high"
        l = "Low" if "Low" in df.columns else "low"
        v = "Volume" if "Volume" in df.columns else "volume"
        return c, h, l, v

    def _score_tier_trend(self, df: pd.DataFrame, symbol: str = "?") -> tuple[float, list[str]]:
        """TREND TAKİPÇİSİ katmanı: 0-100 puan + neden listesi."""
        if df is None or len(df) < 30:
            return 0.0, []
        try:
            c, h, l, v = self._pick_ohlcv_columns(df)
            close = df[c].astype(float)
            score = 0.0
            reasons: list[str] = []

            rsi_s = ta.rsi(close, length=14)
            if rsi_s is None or rsi_s.dropna().empty:
                return 0.0, []
            rsi = float(rsi_s.iloc[-1])
            prev_rsi = float(rsi_s.iloc[-2])
            price = float(close.iloc[-1])

            # RSI bölgesi: 50-70 sağlıklı trend; sınır bölgeler; >78 aşırı alım cezası
            if 50.0 <= rsi <= 70.0:
                score += 20.0
                reasons.append(f"RSI {rsi:.0f} sağlıklı trend bölgesi")
            elif 45.0 <= rsi < 50.0 or 70.0 < rsi <= 78.0:
                score += 10.0
                reasons.append(f"RSI {rsi:.0f} sınır bölge")
            if rsi > 78.0:
                score -= 15.0
                reasons.append(f"RSI {rsi:.0f} > 78 — aşırı alım cezası")
            if rsi > prev_rsi:
                score += 8.0
                reasons.append("RSI yükseliyor")

            # EMA yapısı (boğa düzeni)
            ema21 = ta.ema(close, length=21)
            ema50 = ta.ema(close, length=50)
            if (
                ema21 is not None and ema50 is not None
                and not ema21.dropna().empty and not ema50.dropna().empty
            ):
                e21 = float(ema21.iloc[-1])
                e50 = float(ema50.iloc[-1])
                if price > e21 > e50:
                    score += 20.0
                    reasons.append("Fiyat>EMA21>EMA50 (boğa düzeni)")
                elif price > e50:
                    score += 8.0

            # Uzun vade yapısı
            sma200 = ta.sma(close, length=200)
            if sma200 is not None and not sma200.dropna().empty:
                if price > float(sma200.iloc[-1]):
                    score += 10.0
                    reasons.append("Fiyat>SMA200 (uzun vade yukarı)")
            else:
                score += 5.0

            # MACD momentum
            macd_df = ta.macd(close, fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                hist_col = [col for col in macd_df.columns if col.lower().startswith("macdh")]
                hist_col = hist_col or [macd_df.columns[-1]]
                hist_now = float(macd_df[hist_col[0]].iloc[-1])
                hist_prev = float(macd_df[hist_col[0]].iloc[-2])
                if hist_now > 0 and hist_now > hist_prev:
                    score += 15.0
                    reasons.append("MACD hist pozitif ve büyüyor")
                elif hist_now > 0:
                    score += 8.0
                else:
                    score -= 5.0

            # VWAP (sıralı DatetimeIndex gerektirir — copy ile garanti altına al)
            try:
                _vw = pd.DataFrame(
                    {
                        "high": df[h].astype(float).values,
                        "low": df[l].astype(float).values,
                        "close": close.values,
                        "volume": df[v].astype(float).values,
                    },
                    index=pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(df), freq="h"),
                )
                vwap_s = ta.vwap(high=_vw["high"], low=_vw["low"], close=_vw["close"], volume=_vw["volume"])
                if vwap_s is not None and not vwap_s.dropna().empty and price > float(vwap_s.iloc[-1]):
                    score += 10.0
                    reasons.append("Fiyat VWAP üstünde")
            except Exception:
                pass

            # OBV (hacim akışı)
            obv = ta.obv(close, df[v].astype(float))
            if obv is not None and not obv.dropna().empty:
                if float(obv.iloc[-1]) >= float(obv.iloc[-5]):
                    score += 8.0
                    reasons.append("OBV yukarı (kurumsal akış teyidi)")

            # Hacim patlaması
            vol_mean = float(df[v].tail(20).mean())
            if vol_mean > 0 and float(df[v].iloc[-1]) > vol_mean * 1.2:
                score += 5.0
                reasons.append("Hacim 1.2x+")

            # 5 barlık momentum
            if len(close) >= 6 and price > float(close.iloc[-6]):
                score += 4.0

            return max(0.0, min(100.0, score)), reasons[:7]
        except Exception as exc:
            logger.warning(f"[SCANNER] {symbol} TREND katmanı skorlama HATASI: {exc}")
            return 0.0, []

    def _score_tier_dip(self, df: pd.DataFrame, symbol: str = "?") -> tuple[float, list[str]]:
        """DİP DÖNÜŞÜ katmanı: aşırı satım + dönüş teyitleri (0-100 puan)."""
        if df is None or len(df) < 30:
            return 0.0, []
        try:
            c, h, l, v = self._pick_ohlcv_columns(df)
            close = df[c].astype(float)
            score = 0.0
            reasons: list[str] = []

            rsi_s = ta.rsi(close, length=14)
            if rsi_s is None or rsi_s.dropna().empty:
                return 0.0, []
            rsi = float(rsi_s.iloc[-1])
            prev_rsi = float(rsi_s.iloc[-2])

            # RSI > 55 ise dip bölgesi değil → bu katman boş döner
            if rsi > 55.0:
                return 0.0, []

            if rsi < 30.0:
                score += 20.0
                reasons.append(f"RSI {rsi:.0f} aşırı satım (<30)")
            elif rsi < 45.0:
                score += 15.0
                reasons.append(f"RSI {rsi:.0f} satım bölgesi (<45)")
            if rsi > prev_rsi:
                score += 10.0
                reasons.append("RSI dönüşe başladı")

            # RSI hook (30 geri alımı)
            if prev_rsi < 30.0 <= rsi:
                score += 20.0
                reasons.append("RSI hook (30 geri alımı)")

            # ── PIVOT DIVERGENCE (gerçek salınım tabanlı) ──
            from agents.quant_engine import _detect_pivot_divergence

            div = _detect_pivot_divergence(df)
            if div.get("divergence") in ("POSITIVE_DIVERGENCE", "HIDDEN_BULLISH_DIVERGENCE"):
                score += 25.0
                strength = div.get("strength", "WEAK")
                reasons.append(f"{strength} boğa divergence (fiyat dip vs RSI dip)")
            elif div.get("divergence") in ("NEGATIVE_DIVERGENCE", "HIDDEN_BEARISH_DIVERGENCE"):
                score -= 15.0
                reasons.append("ayı divergence mevcut")

            # CHoCH + RSI trendline break
            from agents.quant_engine import _detect_choch, _detect_rsi_trendline_break

            choch = _detect_choch(df, lookback=20)
            if choch.get("choch_detected") and choch.get("direction") == "BULLISH":
                score += 15.0
                reasons.append("CHoCH BULLISH (yapısal kırılım)")
            rt = _detect_rsi_trendline_break(df, rsi_period=14)
            if rt.get("trendline_break") and rt.get("break_direction") == "BULLISH":
                score += 15.0
                reasons.append("RSI trendline kırılımı (momentum)")

            # Fiyat trendline kırılımı (düşeni kırma)
            from agents.quant_engine import _detect_price_breakout, _detect_rsi_breakout, _detect_rsi_hook

            if _detect_price_breakout(df):
                score += 10.0
                reasons.append("Düşen trend çizgisi kırıldı (hacim teyitli)")
            if _detect_rsi_breakout(df):
                score += 10.0
                reasons.append("RSI düşen trend kırıldı")
            if _detect_rsi_hook(df):
                score += 15.0
                reasons.append("RSI hook (çukurdan kalkış)")

            # Bollinger sıkışması (volatilite kompresyonu → kırılım beklentisi)
            bb = ta.bbands(close, length=20, std=2.0)
            if bb is not None and not bb.empty:
                bbl = [col for col in bb.columns if col.lower().startswith("bbl")]
                bbu = [col for col in bb.columns if col.lower().startswith("bbu")]
                bbm = [col for col in bb.columns if col.lower().startswith("bbm")]
                if bbl and bbu and bbm:
                    curr_w = (float(bb[bbu[0]].iloc[-1]) - float(bb[bbl[0]].iloc[-1])) / max(
                        float(bb[bbm[0]].iloc[-1]), 1e-9
                    )
                    prev_w = (float(bb[bbu[0]].iloc[-10]) - float(bb[bbl[0]].iloc[-10])) / max(
                        float(bb[bbm[0]].iloc[-10]), 1e-9
                    )
                    if curr_w < prev_w * 0.90:
                        score += 10.0
                        reasons.append("Bollinger sıkışması (squeeze)")

            # Kurumsal iz: CLV + hacim
            high_now = float(df[h].iloc[-1])
            low_now = float(df[l].iloc[-1])
            range_val = high_now - low_now
            if range_val <= 0:
                range_val = 1e-4
            clv = ((float(close.iloc[-1]) - low_now) - (high_now - float(close.iloc[-1]))) / range_val
            vol_mean = float(df[v].tail(20).mean())
            vol_ratio = float(df[v].iloc[-1] / vol_mean) if vol_mean > 0 else 1.0
            if clv > 0.4 and vol_ratio > 1.2:
                score += 10.0
                reasons.append("CLV+ hacim (kurumsal iz)")

            return max(0.0, min(100.0, score)), reasons[:7]
        except Exception as exc:
            logger.warning(f"[SCANNER] {symbol} DİP katmanı skorlama HATASI: {exc}")
            return 0.0, []

    def _score_prefilter_candidate(
        self, df_1d: pd.DataFrame, df_4h: Optional[pd.DataFrame], symbol: str = "?"
    ) -> dict:
        """
        İki katmanı 1d + 4h verisiyle birleştirir.
        Dönüş: {score, tier, reason} — tier: TREND_FOLLOWER | DIP_REVERSAL | NONE
        """
        t1d, r1d = self._score_tier_trend(df_1d, symbol)
        t4h, r4h = self._score_tier_trend(df_4h, symbol) if df_4h is not None else (0.0, [])
        d1d, rd1d = self._score_tier_dip(df_1d, symbol)
        d4h, rd4h = self._score_tier_dip(df_4h, symbol) if df_4h is not None else (0.0, [])

        trend = round(0.65 * t1d + 0.35 * t4h, 1)
        dip = round(0.65 * d1d + 0.35 * d4h, 1)

        if trend >= dip and trend > 0:
            tier = "TREND_FOLLOWER"
            score = trend
            reasons = r1d + [f"4h: {x}" for x in r4h]
        elif dip > 0:
            tier = "DIP_REVERSAL"
            score = dip
            reasons = rd1d + [f"4h: {x}" for x in rd4h]
        else:
            tier = "NONE"
            score = 0.0
            reasons = []

        return {"score": float(score), "tier": tier, "reason": "; ".join(reasons[:7]) or "-"}

    async def _fetch_prefilter_data(
        self, symbol: str
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        KATMAN 1 veri kaynağı (v2): 1d + 4h çeker (önbellekli, RAM dostu).
        4h: kripto → CCXT 4h; hisse → 1h çekip 4h'a agrega eder.
        """
        try:
            if is_crypto(symbol):
                df_1d = await fetch_crypto_ohlcv(symbol, timeframe="1d", limit=120)
                df_4h = await fetch_crypto_ohlcv(symbol, timeframe="4h", limit=120)
            else:
                df_1d = await fetch_stock_macro_data(symbol, period="6mo", interval="1d")
                df_4h = None
                try:
                    df_1h = await fetch_stock_macro_data(symbol, period="1mo", interval="1h")
                    if df_1h is not None and not df_1h.empty:
                        df_4h = _resample_4h_from_1h(df_1h)
                        del df_1h
                except Exception:
                    df_4h = None
            return df_1d, df_4h
        except Exception as exc:
            logger.debug(f"[SCANNER] {symbol} prefilter verisi çekilemedi: {exc}")
            return None, None

    async def _pre_filter_assets(self, target_evren: list[str]) -> dict:
        """
        KATMAN 1 — PREFILTER (İKİ KATMANLI): trend takipçisi + dip dönüşü.
        Eşzamanlılık `prefilter_concurrency` (varsayılan 3) ile sınırlıdır;
        512MB RAM'de çok sayıda eşzamanlı indirme belleği patlatmaz.

        Dönüş: {
          "candidates": [{"symbol", "score", "tier", "reason"}, ...] (top-N),
          "stats": {"scanned": int, "trend": int, "dip": int, "none": int, "failed": int}
        }
        """
        logger.info("[SCANNER] KATMAN 1 — PREFILTER başladı (2 katmanlı: TREND + DİP)...")
        concurrency = max(1, int(self.scan_config.get("prefilter_concurrency", 3)))
        max_candidates = max(1, int(self.scan_config.get("deep_scan_max_assets", 6)))
        min_score = float(self.scan_config.get("prefilter_min_score", 30.0))
        sem = asyncio.Semaphore(concurrency)
        candidates: list[dict] = []
        stats = {"scanned": 0, "trend": 0, "dip": 0, "none": 0, "failed": 0}

        async def _probe(symbol: str) -> None:
            async with sem:
                try:
                    df_1d, df_4h = await self._fetch_prefilter_data(symbol)
                except Exception as exc:
                    # Sessiz düşme yok: her hata sayılır ve sembol loglanır.
                    stats["failed"] += 1
                    logger.warning(f"[SCANNER] {symbol} prefilter veri çekme HATASI: {exc}")
                    return
                if df_1d is None or df_1d.empty or len(df_1d) < 30:
                    stats["failed"] += 1
                    n_bars = 0 if df_1d is None else len(df_1d)
                    logger.warning(
                        f"[SCANNER] {symbol} prefilter atlandı: 1D veri yetersiz/boş "
                        f"(bar sayısı: {n_bars}, gereken: >=30)"
                    )
                    return
                try:
                    res = self._score_prefilter_candidate(df_1d, df_4h, symbol)
                    stats["scanned"] += 1
                    if res["score"] >= min_score and res["tier"] != "NONE":
                        candidates.append({"symbol": symbol, **res})
                        if res["tier"] == "TREND_FOLLOWER":
                            stats["trend"] += 1
                        else:
                            stats["dip"] += 1
                    else:
                        stats["none"] += 1
                except Exception as exc:
                    # Skorlama hatası da sessizce yutulmaz: sembol loglanır.
                    stats["failed"] += 1
                    logger.warning(f"[SCANNER] {symbol} prefilter skorlama HATASI: {exc}")
                finally:
                    del df_1d
                    if df_4h is not None:
                        del df_4h  # RAM disiplini

        chunk = concurrency * 4
        for i in range(0, len(target_evren), chunk):
            batch = target_evren[i : i + chunk]
            await asyncio.gather(*[_probe(sym) for sym in batch], return_exceptions=True)
            if i + chunk < len(target_evren):  # API rate-limit nezaketi: batch'ler arası bekleme
                await asyncio.sleep(self._batch_cooldown)

        sorted_cands = sorted(candidates, key=lambda x: x["score"], reverse=True)[:max_candidates]
        logger.info(
            f"[SCANNER] KATMAN 1 tamam: {stats['scanned']} varlık tarandı → "
            f"{len(sorted_cands)} aday ({stats['trend']} TREND + {stats['dip']} DİP) | "
            f"{stats['failed']} varlık veri hatası (çekilemedi/yetersiz) — derin analize geçiyor."
        )
        return {"candidates": sorted_cands, "stats": stats}

    # =========================================================================
    # ── ANA TARAMA METODU (4 KATMANLI PİPELINE) ──
    # =========================================================================
    def _build_start_message(self, count: int, regime: Optional[RegimeSnapshot]) -> str:
        lines = ["🔍 ORACLE TARAMA BAŞLADI (4 Katmanlı Pipeline)"]
        lines.append(f"📊 {count} varlık taranacak — heartbeat ile takip edilecek.")
        if regime is not None:
            try:
                emoji = {"RISK_ON": "🟢", "MIXED": "🟡", "RISK_OFF": "🔴", "NEUTRAL": "⚪"}.get(
                    regime.primary_trend, "⚪"
                )
                lines.append(
                    f"\n🌐 KATMAN 0 — REJİM: {emoji} {regime.primary_trend}"
                    f" | Timing: {regime.intraday_timing}"
                    f" | Risk İştahı: {regime.risk_appetite:.2f}"
                )
                lines.append(
                    f"   USDT.D: {regime.usdt_d:.2f}% | BTC.D: {regime.btc_d:.2f}%"
                    f" | DXY: {regime.dxy:.1f} | VIX: {regime.vix:.1f}"
                )
            except Exception:
                pass
        return "\n".join(lines)

    async def _heartbeat_loop(self, interval_min: int) -> None:
        """Tarama devam ederken periyodik ilerleme bildirimi (Telegram'ı asla bloke etmez)."""
        interval_sec = max(30, int(interval_min) * 60)
        while True:
            await asyncio.sleep(interval_sec)
            prog = self._scan_progress
            try:
                await self.bot(
                    "⏳ ORACLE taraması devam ediyor...\n"
                    f"   ✅ Derin analiz: {prog['scanned']}/{prog['total']}"
                    f" | 🔥 Fırsat: {prog['found']}"
                )
            except Exception as exc:
                logger.warning(f"[SCANNER] Heartbeat gönderilemedi: {exc}")

    async def _record_opportunity(self, run_id: str, result: dict, regime: Optional[RegimeSnapshot]) -> None:
        """Fırsatı rejimle korele eder ve scan_store'a yazar (kalıcılık + özet desteği)."""
        direction = (
            "LONG"
            if result.get("signal") in ("STRONG_BUY", "ACCUMULATE", "LONG_FIRSAT")
            else "SHORT"
        )
        corr: dict = {}
        if regime is not None:
            try:
                corr = correlate_signal_with_regime(tf="4h", direction=direction, regime=regime)
                result["_correlation"] = corr
            except Exception as exc:
                logger.debug(f"[SCANNER] Rejim korelasyonu hatası: {exc}")
        try:
            await get_scan_store().record_result(
                run_id=run_id,
                symbol=str(result.get("asset", "")),
                signal=str(result.get("signal", "")),
                composite=float(result.get("composite_pct", 0)) / 100.0,
                base_rr=float(result.get("base_rr") or 0.0),
                t1=float(result.get("t1") or 0.0),
                t2=float(result.get("t2") or 0.0),
                t3=float(result.get("t3") or 0.0),
                stop_loss=float(result.get("stop_loss") or 0.0),
                trade_type=str(result.get("trade_type") or ""),
                oracle_summary=str(result.get("oracle_summary") or ""),
                tf_bias_json=json.dumps(
                    result.get("timeframe_biases", {}), ensure_ascii=False, default=str
                ),
                regime_json=json.dumps(
                    {
                        "primary_trend": getattr(regime, "primary_trend", None) if regime else None,
                        "intraday_timing": getattr(regime, "intraday_timing", None) if regime else None,
                        "risk_appetite": getattr(regime, "risk_appetite", None) if regime else None,
                    },
                    default=str,
                ),
                correlation_json=json.dumps(corr, default=str),
            )
        except Exception as exc:
            logger.debug(f"[SCANNER] Sonuç kaydı hatası: {exc}")

        # ── ADIM 6: SIGNAL TRACKER — gerçek win/loss verisi toplamak ────────
        # (Tarama tarafı üretilen her sinyal kaydedilir; /stats gerçek veriden
        #  win-rate hesaplar. Hiçbir yerde sahte yüzde yoktur.)
        try:
            from core.signal_tracker import get_signal_tracker

            tracker = get_signal_tracker()
            entry = (
                result.get("entry_zone_low")
                or result.get("entry_zone_high")
                or result.get("t1")
            )
            if entry:
                await asyncio.to_thread(
                    tracker.record_signal,
                    symbol=str(result.get("asset", "")),
                    direction=direction,
                    entry_price=float(entry),
                    stop_loss=float(result.get("stop_loss") or 0.0),
                    t1=float(result.get("t1") or 0.0),
                    t2=float(result["t2"]) if result.get("t2") else None,
                    t3=float(result["t3"]) if result.get("t3") else None,
                    confidence=float(result.get("confidence") or 0.0),
                    composite_score=float(result.get("composite_pct", 0)) / 100.0,
                    cluster_leader_rank=result.get("cluster_leader_rank"),
                )
                logger.info(
                    f"[SCANNER] Tracker'a kaydedildi: {result.get('asset')} {direction} "
                    f"@{float(entry):.4g}"
                )
        except Exception as exc:
            logger.debug(f"[SCANNER] Tracker kaydı hatası: {exc}")

    async def _run_scan_once(self, notify_start: bool = True, trigger: str = "otomatik") -> None:
        """
        4 katmanlı tam tarama:
          Katman 0: Rejim BİR KEZ çekilir (tüm varlıklar paylaşır)
          Katman 1: Prefilter (sınırlı eşzamanlılık, önbellekli veri)
          Katman 2: Derin pipeline sıralı + varlık başına timeout + heartbeat + gc
          Katman 3: Digest v2 teslimatı (rejim korelasyonu + MTF + geçerlilik penceresi)
        """
        global _GLOBAL_SCAN_ACTIVE
        if _GLOBAL_SCAN_ACTIVE or self._scan_in_progress:
            logger.info("[SCANNER] Tarama zaten aktif — global çift tetikleme koruması (guard).")
            return
        _GLOBAL_SCAN_ACTIVE = True
        self._scan_in_progress = True
        run_id = f"scan_{int(time.time())}"
        # FAZ B: önceki koşunun küme haritalarını sıfırla (bayat veri sızması olmasın)
        self._cluster_rank_map = {}
        self._cluster_theme_map = {}
        # FAZ 3: bu koşudaki eleme nedenleri (her aday için)
        self._elimination_log: list[dict] = []
        store = get_scan_store()
        started = time.monotonic()
        try:
            await store.start_run(run_id, trigger)
            all_assets = self._get_all_assets()

            # ── KATMAN 0: REJİM MOTORU (tüm tarama için tek çekiş) ──────────
            regime: Optional[RegimeSnapshot] = None
            try:
                regime = await get_regime_snapshot(force=True)
                self._last_regime = regime
            except Exception as exc:
                logger.warning(f"[SCANNER] Rejim motoru hatası (tarama sorunsuz devam): {exc}")

            if notify_start:
                try:
                    await self.bot(self._build_start_message(len(all_assets), regime))
                except Exception as exc:
                    logger.warning(f"[SCANNER] Başlangıç bildirimi gönderilemedi: {exc}")

            # ── KATMAN 1: PREFILTER (İKİ KATMANLI) ─────────────────────────
            prefilter = await self._pre_filter_assets(all_assets)
            candidates = prefilter["candidates"]
            pf_stats = prefilter["stats"]
            if not candidates:
                logger.info("[SCANNER] Bu turda süzgeci geçen aday yok — tarama tamam.")
                try:
                    await self.bot(
                        f"🔎 KATMAN 1 tamam: {pf_stats['scanned']} varlık tarandı, "
                        f"aday yok ({pf_stats['trend']} TREND + {pf_stats['dip']} DİP "
                        f"eşiği geçemedi"
                        + (f", {pf_stats['failed']} varlık veri hatası" if pf_stats["failed"] else "")
                        + ")."
                    )
                except Exception:
                    pass
                await store.finish_run(run_id, "no_candidates", pf_stats["scanned"], 0)
                return

            try:
                await self.bot(
                    f"🔎 KATMAN 1 — PREFILTER: {pf_stats['scanned']} varlık tarandı → "
                    f"{len(candidates)} aday ({pf_stats['trend']} TREND + "
                    f"{pf_stats['dip']} DİP"
                    + (f", {pf_stats['failed']} varlık veri hatası" if pf_stats["failed"] else "")
                    + ") → derin analiz başlıyor."
                )
            except Exception as exc:
                logger.warning(f"[SCANNER] Prefilter bildirimi gönderilemedi: {exc}")

            # ── FAZ B: KORELASYON KÜMELEME (adayları tema bazında grupla) ──
            # Aynı temadan çok sayıda sinyal üretilirse bunlar tek tema olarak
            # ele alınır ve göreli güç lideri sıralanır. market_data cache'i
            # kullanılır (Faz 2.2 ilkesi: gereksiz tekrar indirme yok).
            try:
                if len(candidates) >= 2:
                    from core.cluster_engine import cluster_and_rank_signals

                    async def _cluster_close(sym: str):
                        if is_crypto(sym):
                            return await fetch_crypto_ohlcv(sym, timeframe="1d", limit=60)
                        return await fetch_stock_macro_data(sym, period="6mo", interval="1d")

                    clusters = await cluster_and_rank_signals(
                        [c["symbol"] for c in candidates],
                        lookback_days=30,
                        fetch_close=_cluster_close,
                    )
                    for cl in clusters:
                        ranked = sorted(cl.members, key=lambda m: m.alpha_score, reverse=True)
                        for i, m in enumerate(ranked, start=1):
                            self._cluster_rank_map[m.symbol] = i
                            self._cluster_theme_map[m.symbol] = cl.theme_description
                    if clusters:
                        leaders = [cl.leaders[0].symbol for cl in clusters if cl.leaders]
                        logger.info(
                            f"[SCANNER] FAZ B: {len(candidates)} aday → {len(clusters)} küme | "
                            f"liderler: {leaders}"
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[SCANNER] FAZ B kümeleme atlandı (tarama devam): {exc}")

            # ── KATMAN 2: DERİN PİPELINE ────────────────────────────────────
            budget_min = max(10, int(self.scan_config.get("scan_wallclock_timeout_min", 240)))
            deadline = time.monotonic() + budget_min * 60
            per_asset_timeout = max(
                30, int(self.scan_config.get("per_asset_timeout_sec", 300))
            )
            heartbeat_min = max(1, int(self.scan_config.get("heartbeat_interval_min", 10)))
            self._scan_progress = {"scanned": 0, "total": len(candidates), "found": 0}

            heartbeat_task = asyncio.create_task(self._heartbeat_loop(heartbeat_min))
            opportunities: list[dict] = []
            try:
                for cand in candidates:
                    asset = cand["symbol"]
                    if time.monotonic() >= deadline:
                        logger.warning("[SCANNER] Duvar saati bütçesi doldu — kısmi teslimata geçiliyor.")
                        try:
                            await self.bot(
                                "⏳ Tarama süre bütçesi doldu; "
                                f"{len(opportunities)} fırsat ile teslim ediliyor."
                            )
                        except Exception:
                            pass
                        break

                    try:
                        result = await asyncio.wait_for(
                            self._scan_single_asset(
                                asset, cand["tier"], cand.get("reason")
                            ),
                            timeout=per_asset_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"[SCANNER] {asset} pipeline timeout ({per_asset_timeout}s) — atlandı."
                        )
                        self._log_elimination(asset, f"pipeline timeout ({per_asset_timeout}s)")
                        continue
                    except Exception as exc:
                        logger.error(f"[SCANNER FAIL-SAFE] {asset} pipeline hatası: {exc}")
                        self._log_elimination(asset, f"pipeline hatası: {str(exc)[:80]}")
                        continue
                    finally:
                        self._scan_progress["scanned"] += 1
                        gc.collect()  # Render 512MB RAM disiplini

                    if result and result.get("signal") not in ("AVOID", "WATCH", None):
                        # Varlık özelinde MTF analizi (5m/15m/1h/4h/1d/1w)
                        try:
                            mtf = await asyncio.wait_for(
                                analyze_multi_tf(asset, max_concurrency=2),
                                timeout=min(per_asset_timeout, 180),
                            )
                            if mtf is not None:
                                result["mtf_summary"] = format_mtf_summary(mtf, asset)
                                result["mtf_bias"] = mtf.signal_bias
                                result["mtf_entry_timing"] = mtf.entry_timing
                                del mtf
                        except Exception as exc:
                            logger.debug(f"[SCANNER] {asset} MTF analizi atlandı: {exc}")

                        opportunities.append(result)
                        self._scan_progress["found"] = len(opportunities)
                        logger.info(
                            f"[SCANNER] FIRSAT ONAYLANDI: {result.get('asset')} → "
                            f"{result.get('signal')} (skor: {result.get('composite_pct', 0)}%)"
                        )
                        await self._record_opportunity(run_id, result, regime)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── KATMAN 3: TESLİMAT (Digest v2) ──────────────────────────────
            await self._send_opportunity_digest(opportunities, regime)
            await self._send_elimination_summary(self._elimination_log, len(candidates))
            status = "done" if opportunities else "empty"
            await store.finish_run(run_id, status, len(candidates), len(opportunities))
            self._last_full_scan = datetime.now(timezone.utc)
            logger.info(
                f"[SCANNER] Tarama tamamlandı ({int(time.monotonic() - started)}s, "
                f"{len(opportunities)} fırsat)."
            )
        except Exception as exc:
            logger.error(f"[SCANNER] Tarama beklenmeyen hata: {exc}")
            try:
                await store.finish_run(run_id, "error", 0, 0)
            except Exception:
                pass
        finally:
            self._scan_in_progress = False
            _GLOBAL_SCAN_ACTIVE = False

    async def _full_scan_done_today(self, store, tz) -> bool:
        """
        ADIM 3 — Render restart güvenlik ağı.
        Bugün (İstanbul) TAMAMLANMIŞ bir tarama koşusu scan_store'da var mı?
        _last_full_scan sadece bellek içindedir; restart'ta kaybolur. Bu kontrol
        aynı gün içinde ikinci kez tam tarama başlatılmasını önler.
        """
        try:
            last = await store.get_last_run()
            if not last:
                return False
            status = str(last.get("status", ""))
            # Tamamlanmış sayılan durumlar: başarılı / adaysız / fırsatsız
            if status not in ("done", "empty", "no_candidates"):
                return False  # hâlâ çalışıyor veya hata → yeniden deneme serbest
            started = last.get("started_at")
            if not started:
                return False
            try:
                started_dt = datetime.fromisoformat(str(started))
                if started_dt.tzinfo is None:
                    started_dt = started_dt.replace(tzinfo=timezone.utc)
                started_ist = started_dt.astimezone(tz)
            except ValueError:
                return False
            return started_ist.date() == datetime.now(tz).date()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SCANNER] bugün tarama var mı kontrolü başarısız: {exc}")
            return False

    async def _missed_scan_needed(self, store, tz) -> bool:
        """
        FAZ A — KAÇIRILAN TARAMA GÜVENLİK AĞI.
        Render restart'ta 04:00-09:00 (İstanbul) gece penceresi kaçırıldıysa ve
        bugün henüz tamamlanmış tarama kaydı YOKSA, son tarama `missed_scan_grace_hours`
        (varsayılan 24s) öncesinden eskiyse → True döner.
        Böylece pencere dışında da kaçırılan günün taraması telafi edilir;
        sistem bugünü boş geçirip taramayı tamamen atlamaz.
        """
        if self._scan_in_progress:
            return False
        try:
            if await self._full_scan_done_today(store, tz):
                return False  # bugün zaten tamamlanmış tarama var
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SCANNER] FAZ A bugün tarama kontrolü başarısız: {exc}")
            return False
        grace_h = float(self.scan_config.get("missed_scan_grace_hours", 24))
        try:
            last_run = await store.get_last_run()
            if not last_run:
                return True  # hiç tarama yok → güvenlik ağı tetiklesin
            started = last_run.get("started_at")
            if not started:
                return True
            last_dt = datetime.fromisoformat(str(started))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600.0
            return hours_since > grace_h
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SCANNER] FAZ A son tarama zamanı okunamadı: {exc}")
            return False

    async def _full_scan_loop(self):
        """
        Gece penceresi (varsayılan 04:00-09:00 İstanbul) içinde tam tarama yapar.
        Pencere dışında bir sonraki pencere başlangıcına kadar uyur.
        Hata olursa pencere içinde 10 dk sonra otomatik yeniden dener.
        """
        import pytz

        tz = pytz.timezone("Europe/Istanbul")
        overnight_start = int(self.scan_config.get("overnight_start_hour", 4))
        overnight_end = int(self.scan_config.get("overnight_end_hour", 9))

        await asyncio.sleep(45)  # başlangıç stabilizasyonu
        while self._running:
            try:
                now = datetime.now(tz)
                in_window = overnight_start <= now.hour < overnight_end
                if in_window:
                    last = self._last_full_scan
                    needs_scan = self._scan_in_progress is False and (
                        last is None or (now - last.astimezone(tz)).total_seconds() > 30 * 60
                    )
                    if needs_scan:
                        # ADIM 3: bellek kaybolsa bile scan_store aynı günü hatırlar
                        if await self._full_scan_done_today(get_scan_store(), tz):
                            logger.info(
                                "[SCANNER] Bugün zaten tamamlanmış tarama kaydı var (scan_store) "
                                "— restart sonrası çift tarama önlendi."
                            )
                            needs_scan = False
                    if needs_scan:
                        logger.info(
                            f"[SCANNER] Gece penceresi aktif ({overnight_start:02d}:00-{overnight_end:02d}:00) "
                            f"— tam tarama başlatılıyor."
                        )
                        await self._run_scan_once(notify_start=True, trigger="gece_taramasi")
                        continue
                else:
                    # ── FAZ A: KAÇIRILAN TARAMA GÜVENLİK AĞI ──
                    # Render restart'ta 04:00-09:00 penceresi kaçırıldıysa ve bugün
                    # henüz tamamlanmış tarama yoksa, hemen telafi taraması başlat.
                    if await self._missed_scan_needed(get_scan_store(), tz):
                        logger.info(
                            "[SCANNER] FAZ A güvenlik ağı: bugün tarama yapılmamış ve son "
                            "tarama çok eski — kaçırılan tarama telafi ediliyor."
                        )
                        await self._run_scan_once(notify_start=True, trigger="kacirilan_telafi")
                        continue
                    logger.info(
                        f"[SCANNER] Gece penceresi kapalı — sonraki tarama "
                        f"{overnight_start:02d}:00'te planlandı."
                    )

                next_start = now.replace(hour=overnight_start, minute=0, second=0, microsecond=0)
                if now >= next_start:
                    next_start += timedelta(days=1)
                await asyncio.sleep((next_start - now).total_seconds())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"[SCANNER] Tam tarama döngüsü hatası: {exc}")
                await asyncio.sleep(600)  # pencere içinde 10 dk sonra tekrar dene

    async def _scan_single_asset(
        self, asset: str, category: str, selection_reason: Optional[str] = None
    ) -> Optional[dict]:
        def _log_elimination(reason: str) -> None:
            """FAZ 3: eleme nedenini koşu loguna ekle (kapasite sınırlı)."""
            self._log_elimination(asset, reason)

        try:
            state = await self.pipeline(asset)
            state_data = self._pipeline_state_to_dict(state)
            if not state_data:
                _log_elimination("pipeline state boş döndü")
                return None

            signal = state_data.get("signal_label") or state_data.get("signal")
            composite = float(state_data.get("composite_score", 0.0))
            base_rr = state_data.get("base_rr")
            # EGER ISLEM ORACLE/CEO TARAFINDAN IPTAL (ABORT) EDILDIYSE ASLA LISTEYE YAZMA!
            status_str = str(state_data.get("status", "")).upper()
            if "ABORT" in status_str or "FAIL" in status_str or state_data.get("fatal_error"):
                fe = str(state_data.get("fatal_error") or "").strip()
                _log_elimination(
                    f"CEO/ajan vetosu: {fe[:90]}" if fe else f"durum: {status_str}"
                )
                return None

            if signal in ["STRONG_BUY", "ACCUMULATE", "STRONG_SELL", "SHORT", "REDUCE", "LONG_FIRSAT", "SHORT_FIRSAT"]:
                return {
                    "asset": asset,
                    "category": category,
                    "signal": signal,
                    "composite_pct": int(abs(composite) * 100),
                    "base_rr": base_rr,
                    "t1": state_data.get("t1"),
                    "t2": state_data.get("t2"),
                    "t3": state_data.get("t3"),
                    "stop_loss": state_data.get("stop_loss"),
                    "entry_zone_low": state_data.get("entry_zone_low"),
                    "entry_zone_high": state_data.get("entry_zone_high"),
                    "invalidation_level": state_data.get("invalidation_level"),
                    "confidence": state_data.get("confidence"),
                    "trade_type": state_data.get("trade_type"),
                    "timeframe_biases": state_data.get("timeframe_biases", {}),
                    "pattern_outcome_bias": state_data.get("pattern_outcome_bias"),
                    "oracle_summary": state_data.get("oracle_summary", ""),
                    "cross_asset_warnings": state_data.get("cross_asset_warnings", []),
                    "historical_similarity_score": state_data.get("historical_similarity_score"),
                    "selection_reason": selection_reason or "",
                    "cluster_leader_rank": self._cluster_rank_map.get(asset),
                    "cluster_theme": self._cluster_theme_map.get(asset),
                    "scanned_at": datetime.now(timezone.utc).isoformat(),
                }

            # YENİ: WATCHLIST_PREMIUM koşulu
            entry_low = state_data.get("entry_zone_low")
            hist_score = state_data.get("historical_similarity_score", 0)
            hist_bias = state_data.get("pattern_outcome_bias", "")
            tf_biases = state_data.get("timeframe_biases", {})
            oversold_tfs = sum(1 for b in tf_biases.values() if b == "OVERSOLD")

            if (oversold_tfs >= 2 
                and hist_score >= 70 
                and "BULLISH" in hist_bias 
                and composite >= 0.35
                and entry_low):
                return {
                    "asset": asset,
                    "category": category,
                    "signal": "WATCHLIST_PREMIUM",
                    "composite_pct": int(abs(composite) * 100),
                    "oracle_summary": (
                        f"OVERSOLD {oversold_tfs}/4 zaman diliminde. "
                        f"Tarihsel benzerlik {int(hist_score)}/100 → {hist_bias}. "
                        f"Makro henüz risk-off — limit emir bölgesi: {entry_low:.4f} altına"
                    ),
                    "timeframe_biases": tf_biases,
                    "pattern_outcome_bias": hist_bias,
                    "historical_similarity_score": hist_score,
                    "selection_reason": selection_reason or "",
                    "cluster_leader_rank": self._cluster_rank_map.get(asset),
                    "cluster_theme": self._cluster_theme_map.get(asset),
                    "scanned_at": datetime.now(timezone.utc).isoformat(),
                }

            stop = state_data.get("stop_loss")
            resistance = state_data.get("t1")
            if entry_low and stop:
                self._watchlist[asset] = {
                    "support": entry_low,
                    "stop": stop,
                    "resistance": resistance,
                    "last_price": None,
                    "category": category,
                }

            # FAZ 3: sinyal üretilmedi — eleme nedenini açıkla
            if signal:
                _log_elimination(f"sinyal eşiği: {signal} (kompozit {composite*100:.0f}%)")
            else:
                _log_elimination(f"sinyal üretilmedi (kompozit {composite*100:.0f}%)")

            return None
        except Exception as exc:
            logger.warning(f"[SCANNER] {asset} pipeline hatası: {exc}")
            _log_elimination(f"beklenmeyen hata: {str(exc)[:80]}")
            return None

    def _log_elimination(self, asset: str, reason: str) -> None:
        """FAZ 3 — Eleme nedenini koşu loguna ekle (kapasite sınırlı)."""
        try:
            log = getattr(self, "_elimination_log", None)
            if log is None:
                return
            if len(log) >= 40:
                return
            log.append({"asset": asset, "reason": reason})
        except Exception:
            pass

    async def _watchlist_monitor_loop(self):
        interval_min = self.scan_config.get("watchlist_check_interval_min", 15)
        interval_sec = interval_min * 60

        while self._running:
            await asyncio.sleep(interval_sec)

            if not self._watchlist:
                continue

            try:
                for asset, levels in list(self._watchlist.items()):
                    try:
                        ticker_symbol = self._ticker_for_watchlist(asset)
                        if not ticker_symbol:
                            logger.debug(f"[WATCHLIST] {asset} için yfinance sembolü yok, atlandı")
                            continue
                        ticker = yf.Ticker(ticker_symbol)
                        hist = ticker.history(period="1d", interval="15m")
                        if hist.empty:
                            continue

                        current_price = float(hist["Close"].iloc[-1])
                        support = levels.get("support")
                        resistance = levels.get("resistance")

                        if support and current_price <= support * 1.03:
                            await self._send_watchlist_alert(
                                asset=asset,
                                current_price=current_price,
                                level=support,
                                level_type="DESTEK/GİRİŞ BÖLGESİ",
                                direction="yaklaşıyor ⬇️",
                            )
                            del self._watchlist[asset]
                        elif resistance and current_price >= resistance * 0.97:
                            await self._send_watchlist_alert(
                                asset=asset,
                                current_price=current_price,
                                level=resistance,
                                level_type="HEDEF-1/DİRENÇ",
                                direction="yaklaşıyor ⬆️",
                            )

                    except Exception as exc:
                        logger.warning(f"[WATCHLIST] {asset} fiyat kontrolü hatası: {exc}")
                        continue

            except Exception as exc:
                logger.error(f"[WATCHLIST] Döngü hatası: {exc}")

    async def _daily_briefing_loop(self):
        import pytz

        tz = pytz.timezone("Europe/Istanbul")
        target_hour = self.config.get("scan_schedule", {}).get("daily_briefing_hour", 8)

        while self._running:
            now = datetime.now(tz)
            next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)

            wait_sec = (next_run - now).total_seconds()
            logger.info(
                f"[BRIEFING] Sonraki brifing: {next_run.strftime('%Y-%m-%d %H:%M')} ({int(wait_sec/3600)}s sonra)"
            )
            await asyncio.sleep(wait_sec)
            await self._send_daily_briefing()

    async def _send_opportunity_digest(self, opportunities: list, regime: Optional[RegimeSnapshot] = None):
        """KATMAN 3 — TESLİMAT: Rejim özeti + MTF matrisi + geçerlilik penceresi (Digest v2)."""
        if not opportunities:
            return

        lines = ["🛡️ 𝗢𝗟𝗬𝗠𝗣𝗨𝗦 𝗢𝗥𝗔𝗖𝗟𝗘 — SABAH TESLİMAT RAPORU\n"]

        if regime is not None:
            try:
                emoji = {"RISK_ON": "🟢", "MIXED": "🟡", "RISK_OFF": "🔴", "NEUTRAL": "⚪"}.get(
                    regime.primary_trend, "⚪"
                )
                lines.append(
                    f"🌐 REJİM: {emoji} {regime.primary_trend}"
                    f" | Timing: {regime.intraday_timing}"
                    f" | Risk İştahı: {regime.risk_appetite:.2f}"
                )
                lines.append(
                    f"   USDT.D: {regime.usdt_d:.2f}% | BTC.D: {regime.btc_d:.2f}%"
                    f" | DXY: {regime.dxy:.1f} | VIX: {regime.vix:.1f}"
                )
                if getattr(regime, "warnings", None):
                    lines.append(f"   ⚠️ {'; '.join(regime.warnings[:3])}")
                lines.append("")
            except Exception:
                pass

        lines.append(f"📊 {len(opportunities)} fırsat tespit edildi:\n")

        signal_emojis = {
            "STRONG_BUY": "🟢🟢",
            "ACCUMULATE": "🟢",
            "STRONG_SELL": "🔴🔴",
            "SHORT": "🔴",
            "REDUCE": "🟠",
            "LONG_FIRSAT": "🟢",
            "SHORT_FIRSAT": "🔴",
            "WATCHLIST_PREMIUM": "👁️",
        }

        for opp in sorted(opportunities, key=lambda x: x.get("composite_pct", 0), reverse=True):
            emoji = signal_emojis.get(opp.get("signal", ""), "⚪")
            rr = f"R:R 1:{opp['base_rr']:.1f}" if opp.get("base_rr") else ""
            lines.append(
                f"🔥 {opp.get('asset')} — {opp.get('signal')} | Kompozit Skor: "
                f"{opp.get('composite_pct')}% | {rr}"
            )
            # 🔎 NEDEN: bu varlık adaya nasıl seçildi? (FASE B — şeffaf gerekçe)
            reason = opp.get("selection_reason")
            if reason:
                lines.append(f"   🔎 NEDEN: {reason}")
            if opp.get("mtf_summary"):
                lines.append(f"   {opp['mtf_summary']}")
            # 📗 FİYAT BAZLI İŞLEM PLANI (FASE D — Digest v3)
            try:
                direction = (
                    "LONG"
                    if opp.get("signal") in ("STRONG_BUY", "ACCUMULATE", "LONG_FIRSAT")
                    else "SHORT"
                )
                plan = build_trade_plan(
                    direction=direction,
                    mtf_bias=opp.get("mtf_bias"),
                    entry_timing=opp.get("mtf_entry_timing"),
                    levels={
                        "entry_zone_low": opp.get("entry_zone_low"),
                        "entry_zone_high": opp.get("entry_zone_high"),
                        "stop_loss": opp.get("stop_loss"),
                        "t1": opp.get("t1"),
                        "t2": opp.get("t2"),
                        "t3": opp.get("t3"),
                        "invalidation_level": opp.get("invalidation_level"),
                        "fib_382": opp.get("fib_382"),
                        "fib_500": opp.get("fib_500"),
                        "fib_618": opp.get("fib_618"),
                    },
                    price=opp.get("last_price"),
                    base_rr=opp.get("base_rr"),
                    usdt_d_trend=(
                        getattr(getattr(regime, "dominance", None), "usdt_d_trend", {}).get("1d")
                        if regime is not None else None
                    ),
                )
                if plan["plan_type"] != "NO_PLAN":
                    lines.append(f"   {plan['header']}")
                    for pl in plan["lines"]:
                        lines.append(f"   {pl}")
            except Exception as exc:
                logger.warning(f"[DIGEST] İşlem planı üretilemedi ({opp.get('asset')}): {exc}")
            # FAZ B: küme teması + göreli güç lider rozeti
            theme = opp.get("cluster_theme")
            rank = opp.get("cluster_leader_rank")
            if theme:
                leader = "⭐ Küme Lideri" if rank == 1 else (f"Küme #{rank}" if rank else "")
                lines.append(f"   🧩 {theme}" + (f" — {leader}" if leader else ""))
            corr = opp.get("_correlation", {})
            if corr.get("validity_text"):
                marker = "✅" if corr.get("aligned") else "⚠️"
                lines.append(f"   {marker} {corr['validity_text']}")
            if corr.get("invalidation_text"):
                lines.append(f"   🚫 {corr['invalidation_text']}")

        lines.append(
            f"\n⏱ Rapor: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
            " | Detay: /oracle <symbol>"
        )

        await self.bot("\n".join(lines))

    async def _send_elimination_summary(self, eliminations: list, candidates_total: int):
        """FAZ 3 — ELEME RAPORU: adayların neden elendiğini şeffaf biçimde özetler.

        Amaç: "Neden 0 fırsat?" sorusuna net cevap. Her derin analize giren adayın
        hangi eşikte elendiği (CEO veto, kompozit, R:R, sinyal yok, timeout vb.)
        Telegram'da kısa ve okunabilir bir özet olarak paylaşılır.
        """
        if not eliminations:
            return

        # Nedenlere göre grupla (aynı nedeni tek satırda topla)
        from collections import Counter

        counter = Counter()
        by_asset: dict[str, str] = {}
        for item in eliminations:
            asset = str(item.get("asset", "?"))
            reason = str(item.get("reason", "")).strip()
            # Veto mesajlarını kategorize et
            upper = reason.upper()
            if "VETO" in upper:
                cat = "🛑 Fundamental / CEO veto"
            elif "GRİ" in upper or "GREY" in upper or "GRI" in upper:
                cat = "⚪ Gri bölge (kararsız piyasa)"
            elif "KOMPOZİT" in upper or "KOMPOZIT" in upper or "PUAN" in upper:
                cat = "📉 Kompozit skor eşiği altı"
            elif "R:R" in upper or "RISK/ÖDÜL" in upper or "RR" in upper:
                cat = "📐 R:R eşiği altı"
            elif "GÜVEN" in upper or "CONFIDENCE" in upper or "GUVEN" in upper:
                cat = "🔒 Güven eşiği altı"
            elif "TUTARSIZ" in upper or "VARIANCE" in upper or "SAPMA" in upper:
                cat = "🔄 Ajan tutarsızlığı (variance)"
            elif "TIMEOUT" in upper:
                cat = "⏱ Pipeline timeout"
            elif "SİNYAL" in upper or "SINYAL" in upper:
                cat = "📡 Sinyal eşiği geçilemedi"
            elif "HATA" in upper or "FAIL" in upper:
                cat = "⚠️ Pipeline hatası"
            else:
                cat = "❓ Diğer eleme"
            counter[cat] += 1
            if asset not in by_asset:
                by_asset[asset] = reason

        # En sık tekrarlanan 3 eleme nedenini ilk 3'e koy
        lines = [f"🧾 ELEME RAPORU — {len(eliminations)}/{candidates_total} aday elendi\n"]
        top = counter.most_common(3)
        for cat, count in top:
            lines.append(f"{cat}: {count}")
        if len(counter) > 3:
            for cat, count in counter.most_common()[3:]:
                lines.append(f"  • {cat}: {count}")
        lines.append("")

        # Örnek: ilk 3 farklı varlığın spesifik nedenini göster
        lines.append("🔍 İlk elenenler:")
        for asset, reason in list(by_asset.items())[:3]:
            lines.append(f"   • {asset}: {reason[:100]}")
        if len(by_asset) > 3:
            lines.append(f"   … ve {len(by_asset) - 3} varlık daha")

        await self.bot("\n".join(lines))

    async def _send_watchlist_alert(self, asset, current_price, level, level_type, direction):
        # 4 SAATLIK (14400 SANİYE) ANTI-SPAM ENGELLEYİCİ
        cooldown_key = f"{asset}_{level_type}"
        current_time = time.time()
        last_alert_time = self._alert_cooldowns.get(cooldown_key, 0)
        
        if current_time - last_alert_time < 14400:
            logger.debug(f"[ANTI-SPAM] {asset} için {level_type} uyarısı bloke edildi (Cooldown aktif).")
            return
            
        # Alarm geçiş izni alındıysa zamanı güncelle
        self._alert_cooldowns[cooldown_key] = current_time
        
        msg = f"⚡ PUSULANDI! LIKİDASYON YAKINLAŞTI! \n\n📌 VARLIK: {asset}\n📍 {level_type} {direction}\n💰 Aktif Fiat: {current_price:.4f}\n🎯 Sızma Eşiti: {level:.4f}\n📏 Marj TPay: {abs(current_price - level) / level * 100:.1f}%\n"
        await self.bot(msg)

    async def _send_daily_briefing(self):
        import pytz

        tz = pytz.timezone("Europe/Istanbul")
        msg_lines = [
            "🌅 OLYMPUS ORACLE — GÜNLÜK BRİFİNG",
            f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n",
            "Sabah taraması başlatılıyor, sonuçlar kısa sürede gelecek...",
        ]
        await self.bot("\n".join(msg_lines))

        # Çift tarama koruması: bugün zaten gece taraması yapıldıysa tekrar tetikleme
        last = self._last_full_scan
        if last is not None and last.astimezone(tz).date() == datetime.now(tz).date():
            logger.info("[BRIEFING] Bugünkü gece taraması tamamlandı, çift tarama atlandı.")
            return
        await self._run_scan_once()