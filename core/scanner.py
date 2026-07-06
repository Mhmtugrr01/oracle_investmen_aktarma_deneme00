"""
Olympus Oracle — Otomatik Çoklu Varlık Tarayıcı
Her 4 saatte bir tüm varlık evrenini tarar.
Her 15 dakikada bir izleme listesindeki varlıkların
kritik seviyelere yakınlığını kontrol eder.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from loguru import logger


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
        # Batch-fetching configuration to avoid rate limits and full-loop failures
        self._batch_size: int = int(self.scan_config.get("batch_size", 50))
        self._batch_cooldown: float = float(self.scan_config.get("batch_cooldown_sec", 1.5))
        # Concurrency limit for per-batch parallel tasks
        self._concurrency_limit: int = int(self.scan_config.get("concurrency_limit", 8))

    async def start(self):
        """Tarayıcıyı başlat — iki paralel döngü çalıştır."""
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
        return result

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
                "FET/USDT": "FET-USD",
            }
            return crypto_yf_map.get(token)
        return token

    async def _run_scan_once(self):
        logger.info(f"[SCANNER] Tam tarama başlatılıyor — {len(self._get_all_assets())} varlık")
        opportunities: list[dict] = []

        all_categories = list(self.asset_universe.items())
        sem = asyncio.Semaphore(self._concurrency_limit)

        async def _safe_run(asset: str, category: str):
            async with sem:
                try:
                    return await self._scan_single_asset(asset, category)
                except Exception as e:
                    logger.warning(f"[SCANNER] {asset} pipeline hatası (görev düz): {e}")
                    return None

        for category, assets in all_categories:
            # chunk assets into batches to avoid rate limits and single-point failures
            for i in range(0, len(assets), self._batch_size):
                batch = assets[i : i + self._batch_size]
                tasks = [asyncio.create_task(_safe_run(a, category)) for a in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    try:
                        if isinstance(res, Exception):
                            logger.warning(f"[SCANNER] Batch task exception: {res}")
                            continue
                        result = res
                        if result and result.get("signal") not in ["AVOID", "WATCH", None]:
                            opportunities.append(result)
                            logger.info(
                                f"[SCANNER] FIRSAT: {result.get('asset')} → {result.get('signal')} "
                                f"(skor: {result.get('composite_pct', 0)}%)"
                            )
                    except Exception as exc:
                        logger.warning(f"[SCANNER] Batch result işlenirken hata: {exc}")
                        continue

                # polite cooldown between batches to reduce rate-limit and remote errors
                await asyncio.sleep(self._batch_cooldown)

        if opportunities:
            await self._send_opportunity_digest(opportunities)
        else:
            logger.info("[SCANNER] Bu turda işlem kaliteli fırsat bulunamadı.")

        self._last_full_scan = datetime.now(timezone.utc)

    async def _full_scan_loop(self):
        interval_hours = self.scan_config.get("full_scan_interval_hours", 4)
        interval_sec = interval_hours * 3600
        await asyncio.sleep(60)
        while self._running:
            try:
                # Kullanıcı işlemleri çökmesin diye Tarama görevi Event Loop içinden korumaya alınıyor!
                await asyncio.wait_for(self._run_scan_once(), timeout=900) 
            except asyncio.TimeoutError:
                logger.error("[SCANNER] Tarama süresi 15 dakikayı geçti, döngü zorla atlandı.")
            except Exception as e:
                logger.error(f"[SCANNER] Tam tarama hata aldı: {e}")
            await asyncio.sleep(interval_sec)

        while self._running:
            await self._run_scan_once()
            await asyncio.sleep(interval_sec)

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
                import yfinance as yf

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

    async def _send_opportunity_digest(self, opportunities: list):
        if not opportunities:
            return

        lines = ["🔍 OLYMPUS ORACLE — TARAMA ÖZETI\n"]
        lines.append(f"📊 {len(opportunities)} fırsat tespit edildi:\n")

        signal_emojis = {
            "STRONG_BUY": "🟢🟢",
            "ACCUMULATE": "🟢",
            "STRONG_SELL": "🔴🔴",
            "SHORT": "🔴",
            "REDUCE": "🟠",
            "LONG_FIRSAT": "🟢",
            "SHORT_FIRSAT": "🔴",
        }

        for opp in sorted(opportunities, key=lambda x: x.get("composite_pct", 0), reverse=True):
            emoji = signal_emojis.get(opp["signal"], "⚪")
            rr = f"R:R 1:{opp['base_rr']:.1f}" if opp.get("base_rr") else ""
            lines.append(
                f"{emoji} {opp['asset']} ({opp['category'].upper()}) — "
                f"{opp['signal']} | Skor: {opp['composite_pct']}% | {rr}"
            )

        lines.append("\n💡 Detay için: /oracle [sembol]")
        lines.append(f"🕐 Tarama: {datetime.now().strftime('%H:%M')}")

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
        
        msg = (
            f"⚡ OLYMPUS ORACLE — SEVİYE ALARMI\n\n"
            f"📌 VARLIK: {asset}\n"
            f"📍 {level_type} {direction}\n"
            f"💰 Mevcut Fiyat: {current_price:.4f}\n"
            f"🎯 Kritik Seviye: {level:.4f}\n"
            f"📏 Mesafe: {abs(current_price - level) / level * 100:.1f}%\n\n"
            f"🔍 Detay analiz için: /oracle {asset.split('/')[0]}"
        )
        await self.bot(msg)

    async def _send_daily_briefing(self):
        msg_lines = [
            "🌅 OLYMPUS ORACLE — GÜNLÜK BRİFİNG",
            f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n",
            "Sabah taraması başlatılıyor, sonuçlar kısa sürede gelecek...",
        ]
        await self.bot("\n".join(msg_lines))
        await self._run_scan_once()
