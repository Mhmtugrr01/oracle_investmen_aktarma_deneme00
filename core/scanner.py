"""
PROJECT OLYMPUS — core/scanner.py (The R08 Final-Verdict Edition - True Async & ATR Shield)
Her 4 saatte bir tüm varlık evrenini tarar.
Her 15 dakikada bir izleme listesindeki varlıkların
kritik seviyelere yakınlığını kontrol eder.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
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
            
            if curr_rsi >= 40.0: return 0.0 
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
        """yfinance indirmesini izole ve thread-safe asenkron olarak çalıştırır."""
        ticker = symbol.replace("/USDT", "-USD").replace("/USD", "-USD") if "/" in symbol else symbol
        try:
            df = await asyncio.to_thread(
                yf.download, ticker, period="60d", interval="1d", progress=False, auto_adjust=True
            )
            return symbol, df
        except Exception as e:
            logger.warning(f"[SCANNER] {symbol} veri indirme başarısız: {e}")
            return symbol, None

    async def _pre_filter_assets(self, target_evren: list[str]) -> list[str]:
        """Geniş evreni asenkron paralel havuzlarla hızlı ve güvenli süzgeçten geçirir."""
        logger.info("[SCANNER] TRUE CONCURRENT QUANT MATRIX PROCESSING STARTED...")
        candidates = []
        
        for i in range(0, len(target_evren), self._batch_size):
            batch = target_evren[i : i + self._batch_size]
            
            # ── 🚀 CO-ROUTINE GATHERING (Gerçek Paralel Akış) ──
            # Batch içindeki tüm varlık indirme işlemleri aynı anda asenkron olarak tetiklenir
            tasks = [self._fetch_single_asset_data(sym) for sym in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in results:
                if isinstance(res, Exception) or res is None:
                    continue
                symbol, df = res
                if df is None or df.empty:
                    continue
                
                k_score = self._compute_olympus_kinetic(df)
                if k_score > 0.0:
                    raw_sym = symbol.replace("-USD", "/USDT")
                    candidates.append((raw_sym, k_score))
                    
            logger.info(f"[SCANNER] Concurrency Batch {(i//self._batch_size)+1} processed successfully.")
            await asyncio.sleep(self._batch_cooldown)

        sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
        return [ass for ass, _ in sorted_cands]

    # =========================================================================
    # ── ANA TARAMA METODU ──
    # =========================================================================
    async def _run_scan_once(self, notify_start: bool = True):
        all_assets = self._get_all_assets()
        logger.info(f"[SCANNER] Tam tarama OLYMPUS KINETIC PROTOKOLÜYLE Başlatıldı — İzlenen Toplam Derinliği: {len(all_assets)} Varlık")
        if notify_start:
            try:
                await self.bot(
                    "🔍 ORACLE TARAMA BAŞLADI\n"
                    f"{len(all_assets)} varlık analiz ediliyor... (~15 dk)\n"
                    "Sonuçlar hazır olduğunda otomatik bildirim gelecek."
                )
            except Exception as exc:
                logger.warning(f"[SCANNER] Tarama başlangıç bildirimi gönderilemedi: {exc}")

        # ── 🛡️ CO-ROUTINE GATHERING SÜZGECİ (Filtreleme) ──
        hot_5 = await self._pre_filter_assets(all_assets)

        if not hot_5:
            logger.info("[SCANNER] Bütün Evren Taranmıştır Ancak Kuant Süzgecine Liyakatle Geçen Asimetrik Pusu Bulunamamıştır! (İptal)")
            return

        logger.info(f"🔥 Süzgeç İnfazını Geçip MİMAR AI Zırhına (The Oracle) Alınan Sıcak Adaylar: {hot_5}")

        opportunities: list[dict] = []
        for asset in hot_5:
            try:
                # Sadece süzgeci geçen sıcak 5 varlığı bizim ağır LangGraph pipeline'ına fırlatır!
                result = await self._scan_single_asset(asset, "KINETIC_ALPHA")
                if result and result.get("signal") not in ["AVOID", "WATCH", None]:
                    opportunities.append(result)
                    logger.info(
                        f"[SCANNER] FIRSAT ONAYLANDI: {result.get('asset')} → {result.get('signal')} "
                        f"(skor: {result.get('composite_pct', 0)}%)"
                    )
            except Exception as e:
                logger.error(f"[SCANNER FAIL-SAFE] {asset} pipeline hatası: {e}")
                continue
                
            await asyncio.sleep(2.0)

        if opportunities:
            await self._send_opportunity_digest(opportunities)
        else:
            logger.info("[SCANNER] Bu turda işlem kaliteli fırsat bulunamadı.")

        self._last_full_scan = datetime.now(timezone.utc)

    async def _full_scan_loop(self):
        interval_hours = self.scan_config.get("full_scan_interval_hours", 12) # Günde 2 Kez Tam Tarama için 12 Saat
        interval_sec = interval_hours * 3600
        await asyncio.sleep(60)
        while self._running:
            try:
                # Kullanıcı işlemleri çökmesin diye Tarama görevi Event Loop içinden korumaya alınıyor!
                await asyncio.wait_for(self._run_scan_once(), timeout=1600) 
            except asyncio.TimeoutError:
                logger.error("[SCANNER] Tarama süresi 25 dakikayı geçti, döngü zorla atlandı.")
            except Exception as e:
                logger.error(f"[SCANNER] Tam tarama hata aldı: {e}")
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

        lines = ["\n🛡️ 𝗢𝗟𝗬𝗠𝗣𝗨𝗦 𝗢𝗥𝗔𝗖𝗟𝗘 (KİNETİK SÜZGEÇ) NİHAİ KONTROL ODASI\n"]
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
                f"🔥 {opp['asset']} — {opp['signal']} | Puan Onay: {opp['composite_pct']}% | {rr}"
            )

        lines.append(f"⏱ Olympus Fişleniş Tarihi: {datetime.now().strftime('%H:%M')} | /oracle <symbol> Komutu İşleme Hazırdır.")

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
        msg_lines = [
            "🌅 OLYMPUS ORACLE — GÜNLÜK BRİFİNG",
            f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n",
            "Sabah taraması başlatılıyor, sonuçlar kısa sürede gelecek...",
        ]
        await self.bot("\n".join(msg_lines))
        await self._run_scan_once()