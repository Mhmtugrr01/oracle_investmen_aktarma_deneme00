"""
ORACLE ZAMANLAYICI — Proaktif Piyasa Alarmları
- Saatlik piyasa taraması (yalnızca kritik sinyal varsa alarm)
- Sabah 08:00 günlük brifing
- 4 saatlik cooldown — aynı varlık için tekrar alarm göndermez
"""
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

_scheduler: AsyncIOScheduler | None = None
_alert_callbacks: dict[str, callable] = {}
_alert_cooldown: dict[str, float] = {}
_COOLDOWN_SECONDS = 4 * 3600  # 4 saat


def register_alert_callback(user_id: str, callback):
    _alert_callbacks[user_id] = callback
    logger.info(f"[SCHEDULER] Alert registered: {user_id}")


def unregister_alert_callback(user_id: str):
    _alert_callbacks.pop(user_id, None)


def _is_on_cooldown(key: str) -> bool:
    import time
    last = _alert_cooldown.get(key, 0)
    return (time.time() - last) < _COOLDOWN_SECONDS


def _set_cooldown(key: str):
    import time
    _alert_cooldown[key] = time.time()


async def _broadcast_alert(message: str, cooldown_key: str | None = None):
    """Kayıtlı tüm kullanıcılara alert gönderir. Cooldown varsa atlar."""
    if cooldown_key:
        if _is_on_cooldown(cooldown_key):
            logger.debug(f"[SCHEDULER] Cooldown aktif, atlandı: {cooldown_key}")
            return
        _set_cooldown(cooldown_key)

    for user_id, callback in list(_alert_callbacks.items()):
        try:
            await callback(user_id, message)
        except Exception as e:
            logger.error(f"[SCHEDULER] Alert failed for {user_id}: {e}")


async def _hourly_quant_scan():
    """Saatlik otomatik tarama — sadece kritik sinyalde alarm."""
    logger.info("[SCHEDULER] ⏰ Saatlik tarama başlıyor...")
    if not _alert_callbacks:
        logger.debug("[SCHEDULER] Kayıtlı kullanıcı yok, tarama atlandı")
        return

    try:
        from agents.hft_quant import run_scheduled_scan

        async def send_fn(msg):
            cooldown_key = f"quant_alert_{datetime.now().strftime('%Y%m%d_%H')}"
            await _broadcast_alert(msg, cooldown_key=cooldown_key)

        result = await run_scheduled_scan(send_alert_fn=send_fn)
        logger.info(f"[SCHEDULER] Tarama: {result[:60]}")
    except Exception as e:
        logger.error(f"[SCHEDULER] Saatlik tarama hatası: {e}")


async def _morning_briefing():
    """Sabah 08:00 özet raporu."""
    logger.info("[SCHEDULER] 🌅 Sabah brifing başlıyor...")
    if not _alert_callbacks:
        return

    try:
        from agents.hft_quant import _fetch_macro_data, _fetch_coingecko_data, _fetch_fear_greed

        macro, cg, fg = await asyncio.gather(
            _fetch_macro_data(),
            _fetch_coingecko_data(),
            _fetch_fear_greed(),
            return_exceptions=True,
        )

        if isinstance(macro, Exception):
            macro = {}
        if isinstance(cg, Exception):
            cg = {}
        if isinstance(fg, Exception):
            fg = {}

        vix = macro.get("VIX", {}).get("price", 0)
        dxy = macro.get("DXY", {}).get("price", 0)
        btc = macro.get("BTC", {}).get("price", 0)
        btc_chg = macro.get("BTC", {}).get("change_pct", 0)
        sp500 = macro.get("SP500", {}).get("price", 0)
        sp500_chg = macro.get("SP500", {}).get("change_pct", 0)
        gold = macro.get("GOLD", {}).get("price", 0)

        btc_dom = cg.get("btc_dominance", macro.get("BTC_DOM_APPROX", {}).get("price", 0))
        total_mc = cg.get("total_market_cap_b", 0)

        fg_value = fg.get("value", 0)
        fg_class = fg.get("classification", "Nötr")
        fg_icon = "😨" if fg_value < 25 else ("😱" if fg_value < 40 else ("😐" if fg_value < 60 else ("😊" if fg_value < 75 else "🤑")))

        now = datetime.now().strftime("%Y-%m-%d")
        msg = f"""🌅 *ORACLE SABAH BRİFİNGİ — {now}*
━━━━━━━━━━━━━━━━━━━━━━
🌍 *MAKRO TABLO*
  😨 VIX: {vix:.1f} {'⚠️ Yüksek' if vix > 25 else '✅ Normal'}
  💵 DXY: {dxy:.2f}
  📈 S&P500: {sp500:,.0f} ({sp500_chg:+.1f}%)
  🏅 Altın: ${gold:,.0f}

🔵 *KRİPTO*
  ₿ BTC: ${btc:,.0f} ({btc_chg:+.1f}%)
  🔵 BTC Dom: %{btc_dom:.1f}
  💰 Total Kripto: ${total_mc:.0f}B

{fg_icon} *Korku & Açgözlülük: {fg_value}/100 ({fg_class})*
━━━━━━━━━━━━━━━━━━━━━━
⚡ /quant BTC ETH AAPL — tam analiz"""

        await _broadcast_alert(msg, cooldown_key=f"morning_{now}")
    except Exception as e:
        logger.error(f"[SCHEDULER] Sabah brifing hatası: {e}")


def start_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = AsyncIOScheduler(timezone="Europe/Istanbul")
    _scheduler.add_job(_hourly_quant_scan, "interval", hours=1, id="hourly_quant", replace_existing=True)
    _scheduler.add_job(_morning_briefing, "cron", hour=8, minute=0, id="morning_briefing", replace_existing=True)
    _scheduler.start()
    logger.success("[SCHEDULER] ✅ Aktif — saatlik tarama + 08:00 brifing")


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
