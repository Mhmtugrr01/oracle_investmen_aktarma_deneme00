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
from core.multi_tf import analyze_multi_tf, format_mtf_summary
from core.regime_engine import RegimeSnapshot, correlate_signal_with_regime, get_regime_snapshot
from core.scan_store import get_scan_store
from tools.market_data import fetch_crypto_ohlcv, fetch_stock_macro_data


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

    async def start(self):
        """Tarayıcıyı başlat — üç paralel döngü çalıştır."""
        self._running = True
        await asyncio.gather(
            self._full_scan_loop(),
            self._watchlist_monitor_loop(),
            self._daily_briefing_loop(),
        )

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
            return state.model_dump()
        if isinstance(state, dict):
            return state
        return dict(state)

    def _ticker_for_watchlist(self, asset: str) -> Optional[str]:
        token = asset.upper()
        if "/" in token:
            crypto_yf_map = {
                "BTC/USDT": "BTC-USD",
                "ETH/USDT": "ETH-USD",
                "INJ/USDT": "INJ-USD",
                "RNDR/USDT": "RNDR-USD",
                "FET/USDT": "FET-USD",
            }
            return crypto_yf_map.get(token)
        return token

    # =========================================================================
    # ── ⚔️ OLYMPUS KINETIC SCORING: "PERAKENDE (RETAIL) TUZAGI IMHACISI" ──
    # =========================================================================
    def _compute_olympus_kinetic(self, df: pd.DataFrame) -> float:
        """Momentum Kinetiği, Kurumsal İz, Divergence ve Sıkışmayı süzgeçler."""
        try:
            if df is None or len(df) < 25: 
                return 0.0
            
            c_close = "Close" if "Close" in df.columns else "close"
            c_high = "High" if "High" in df.columns else "high"
            c_low = "Low" if "Low" in df.columns else "low"
            c_vol = "Volume" if "Volume" in df.columns else "volume"

            close_s = df[c_close]
            high_s = df[c_high]
            low_s = df[c_low]
            vol_s = df[c_vol]

            # 1. RASYONEL BIÇAK ENGELLEYİCİSİ (RSI-HOOK KALKANI | %35 ETKİ)
            rsi = ta.rsi(close_s, length=14)
            if rsi is None or rsi.dropna().empty: return 0.0
            
            curr_rsi = float(rsi.iloc[-1])
            prev_rsi = float(rsi.iloc[-2])
            
            # RSI 55 altında + yön aldıysa fırsat penceresi açık
            # Eski 40 eşiği 500 varlıkta hiçbir fırsat bırakmıyordu
            if curr_rsi >= 55.0: return 0.0
            if curr_rsi <= prev_rsi: return 0.0
            
            momentum_score = 35.0  

            # 2. HİSSİYATSIZ HÜKÜMRAN (Kurumsal İz ve Mum Gerçekliği - CLV | %25 ETKİ)
            range_val = high_s.iloc[-1] - low_s.iloc[-1]
            if range_val == 0: range_val = 0.0001
            clv = ((close_s.iloc[-1] - low_s.iloc[-1]) - (high_s.iloc[-1] - close_s.iloc[-1])) / range_val
            
            vol_mean = float(vol_s.tail(20).mean())
            vol_ratio = float(vol_s.iloc[-1] / vol_mean) if vol_mean > 0 else 1.0
            
            smart_money_score = 0.0
            if clv > 0.4 and vol_ratio > 1.2:
                smart_money_score = 25.0
            elif clv > 0.0:
                smart_money_score = 10.0

            # 3. YANILSAMAYI SÖK (Bullish Divergence Koruması | %25 ETKİ)
            divergence_score = 0.0
            min_c = float(close_s.iloc[-6:-1].min())
            min_r = float(rsi.iloc[-6:-1].min())
            if close_s.iloc[-1] <= min_c * 1.01 and curr_rsi > min_r:
                divergence_score = 25.0

            # 4. RUBBER-BAND SIKISMASINA SİNYAL GEÇİŞİ (VOLATILITY COMPRESSION | %15)
            squeeze_score = 0.0
            bb = ta.bbands(close_s, length=20, std=2.0)
            if bb is not None and not bb.empty:
                bbl = float(bb[[c for c in bb.columns if "BBL" in c][0]].iloc[-1])
                bbu = float(bb[[c for c in bb.columns if "BBU" in c][0]].iloc[-1])
                sma = float(bb[[c for c in bb.columns if "BBM" in c][0]].iloc[-1])
                width_curr = (bbu - bbl) / sma if sma != 0 else 0
                width_avg = np.mean([
                    (float(bb[[c for c in bb.columns if "BBU" in c][0]].iloc[-i-1]) - float(bb[[c for c in bb.columns if "BBL" in c][0]].iloc[-i-1])) / float(bb[[c for c in bb.columns if "BBM" in c][0]].iloc[-i-1] or 1)
                    for i in range(10)
                ])
                if width_curr < width_avg * 0.90:
                    squeeze_score = 15.0

            total_kinetic_power = momentum_score + smart_money_score + divergence_score + squeeze_score

            # ── 🛡️ ATR SHIELD ENTEGRASYONU (R08 Mükemmelleştirme) ──
            # Günlük bar boyutu, 14 günlük ATR'nin %50'sinin altındaysa sahte hacim cezası uygular.
            atr_series = ta.atr(high_s, low_s, close_s, length=14)
            if atr_series is not None and not atr_series.dropna().empty:
                curr_atr = float(atr_series.iloc[-1])
                curr_range = float(high_s.iloc[-1] - low_s.iloc[-1])
                if curr_range < curr_atr * 0.5:
                    total_kinetic_power -= 20.0
                    logger.debug(f"[ATR SHIELD] Düşük oynaklık saptandı, ceza uygulandı: -20")

            return max(0.0, total_kinetic_power)

        except Exception as e:
            return 0.0

    async def _fetch_single_asset_data(self, symbol: str) -> tuple[str, Optional[pd.DataFrame]]:
        """
        KATMAN 1 veri kaynağı — ÖNBELBEKLİ market_data fonksiyonlarını kullanır.
        (Ham yf.download yerine: hem rate-limit dostu hem RAM dostu)
        """
        try:
            if is_crypto(symbol):
                df = await fetch_crypto_ohlcv(symbol, timeframe="1d", limit=120)
            else:
                df = await fetch_stock_macro_data(symbol, period="6mo", interval="1d")
            return symbol, df
        except Exception as exc:
            logger.debug(f"[SCANNER] {symbol} veri çekilemedi: {exc}")
            return symbol, None

    async def _pre_filter_assets(self, target_evren: list[str]) -> list[str]:
        """
        KATMAN 1 — PREFILTER: OLYMPUS KINETIC süzgeci.
        Eşzamanlılık `prefilter_concurrency` (varsayılan 3) ile sınırlandırılır;
        böylece 512MB RAM'de çok sayıda eşzamanlı indirme belleği patlatmaz.
        """
        logger.info("[SCANNER] KATMAN 1 — PREFILTER başladı (OLYMPUS KINETIC)...")
        concurrency = max(1, int(self.scan_config.get("prefilter_concurrency", 3)))
        max_candidates = max(1, int(self.scan_config.get("deep_scan_max_assets", 6)))
        sem = asyncio.Semaphore(concurrency)
        candidates: list[tuple[str, float]] = []

        async def _probe(symbol: str) -> None:
            async with sem:
                _, df = await self._fetch_single_asset_data(symbol)
                if df is None or df.empty or len(df) < 25:
                    return
                try:
                    k_score = self._compute_olympus_kinetic(df)
                    if k_score > 0.0:
                        candidates.append((symbol, k_score))
                finally:
                    del df  # RAM disiplini

        chunk = concurrency * 4
        for i in range(0, len(target_evren), chunk):
            batch = target_evren[i : i + chunk]
            await asyncio.gather(*[_probe(sym) for sym in batch], return_exceptions=True)

        sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)[:max_candidates]
        logger.info(
            f"[SCANNER] KATMAN 1 tamam: {len(candidates)} adaydan "
            f"{len(sorted_cands)} derin analize geçiyor."
        )
        return [sym for sym, _ in sorted_cands]

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

    async def _run_scan_once(self, notify_start: bool = True, trigger: str = "otomatik") -> None:
        """
        4 katmanlı tam tarama:
          Katman 0: Rejim BİR KEZ çekilir (tüm varlıklar paylaşır)
          Katman 1: Prefilter (sınırlı eşzamanlılık, önbellekli veri)
          Katman 2: Derin pipeline sıralı + varlık başına timeout + heartbeat + gc
          Katman 3: Digest v2 teslimatı (rejim korelasyonu + MTF + geçerlilik penceresi)
        """
        if self._scan_in_progress:
            logger.info("[SCANNER] Tarama zaten aktif — çift tetikleme koruması (guard).")
            return
        self._scan_in_progress = True
        run_id = f"scan_{int(time.time())}"
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

            # ── KATMAN 1: PREFILTER ─────────────────────────────────────────
            hot = await self._pre_filter_assets(all_assets)
            if not hot:
                logger.info("[SCANNER] Bu turda süzgeci geçen aday yok — tarama tamam.")
                await store.finish_run(run_id, "no_candidates", 0, 0)
                return

            # ── KATMAN 2: DERİN PİPELINE ────────────────────────────────────
            budget_min = max(10, int(self.scan_config.get("scan_wallclock_timeout_min", 240)))
            deadline = time.monotonic() + budget_min * 60
            per_asset_timeout = max(
                30, int(self.scan_config.get("per_asset_timeout_sec", 300))
            )
            heartbeat_min = max(1, int(self.scan_config.get("heartbeat_interval_min", 10)))
            self._scan_progress = {"scanned": 0, "total": len(hot), "found": 0}

            heartbeat_task = asyncio.create_task(self._heartbeat_loop(heartbeat_min))
            opportunities: list[dict] = []
            try:
                for asset in hot:
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
                            self._scan_single_asset(asset, "KINETIC_ALPHA"),
                            timeout=per_asset_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"[SCANNER] {asset} pipeline timeout ({per_asset_timeout}s) — atlandı."
                        )
                        continue
                    except Exception as exc:
                        logger.error(f"[SCANNER FAIL-SAFE] {asset} pipeline hatası: {exc}")
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
            status = "done" if opportunities else "empty"
            await store.finish_run(run_id, status, len(hot), len(opportunities))
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
                        logger.info(
                            f"[SCANNER] Gece penceresi aktif ({overnight_start:02d}:00-{overnight_end:02d}:00) "
                            f"— tam tarama başlatılıyor."
                        )
                        await self._run_scan_once(notify_start=True, trigger="gece_taramasi")
                        continue
                else:
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

    async def _scan_single_asset(self, asset: str, category: str) -> Optional[dict]:
        try:
            state = await self.pipeline(asset)
            state_data = self._pipeline_state_to_dict(state)
            if not state_data:
                return None

            signal = state_data.get("signal_label") or state_data.get("signal")
            composite = float(state_data.get("composite_score", 0.0))
            base_rr = state_data.get("base_rr")
            # EGER ISLEM ORACLE/CEO TARAFINDAN IPTAL (ABORT) EDILDIYSE ASLA LISTEYE YAZMA!
            status_str = str(state_data.get("status", "")).upper()
            if "ABORT" in status_str or "FAIL" in status_str or state_data.get("fatal_error"):
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
                    "trade_type": state_data.get("trade_type"),
                    "timeframe_biases": state_data.get("timeframe_biases", {}),
                    "pattern_outcome_bias": state_data.get("pattern_outcome_bias"),
                    "oracle_summary": state_data.get("oracle_summary", ""),
                    "cross_asset_warnings": state_data.get("cross_asset_warnings", []),
                    "historical_similarity_score": state_data.get("historical_similarity_score"),
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

            return None
        except Exception as exc:
            logger.warning(f"[SCANNER] {asset} pipeline hatası: {exc}")
            return None

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
                f"🔥 {opp.get('asset')} — {opp.get('signal')} | Puan Onay: "
                f"{opp.get('composite_pct')}% | {rr}"
            )
            if opp.get("mtf_summary"):
                lines.append(f"   {opp['mtf_summary']}")
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