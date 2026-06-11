"""
ORACLE ZAMANLAYICI — Proaktif Piyasa Alarmları ve Otomatik Tarama
- Her saat piyasa taraması
- Kritik sinyal (güçlü toplama/dağıtım) tespit edilince Telegram alarmı
- Kullanıcı sormadan sistem kendiliğinden harekete geçer
"""
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger


_scheduler: AsyncIOScheduler | None = None
_alert_callbacks: dict[str, callable] = {}


def register_alert_callback(user_id: str, callback):
    """Telegram bot tarafından çağrılır — kullanıcı bazlı alert kaydı."""
    _alert_callbacks[user_id] = callback
    logger.info(f"[SCHEDULER] Alert callback registered for user {user_id}")


def unregister_alert_callback(user_id: str):
    _alert_callbacks.pop(user_id, None)


async def _broadcast_alert(message: str):
    """Kayıtlı tüm kullanıcılara alert gönderir."""
    for user_id, callback in list(_alert_callbacks.items()):
        try:
            await callback(user_id, message)
        except Exception as e:
            logger.error(f"[SCHEDULER] Alert failed for {user_id}: {e}")


async def _hourly_quant_scan():
    """Saatlik otomatik piyasa taraması."""
    logger.info("[SCHEDULER] ⏰ Hourly quant scan starting...")
    try:
        from agents.hft_quant import run_scheduled_scan

        async def send_fn(msg):
            await _broadcast_alert(msg)

        result = await run_scheduled_scan(send_alert_fn=send_fn)
        logger.info(f"[SCHEDULER] Scan result: {result[:80]}")
    except Exception as e:
        logger.error(f"[SCHEDULER] Hourly scan failed: {e}")


async def _morning_briefing():
    """Sabah 08:00 özet raporu."""
    logger.info("[SCHEDULER] 🌅 Morning briefing...")
    try:
        from agents.hft_quant import _fetch_macro_data

        macro = await _fetch_macro_data()
        now = datetime.now().strftime("%Y-%m-%d")

        vix = macro.get("VIX", {}).get("price", 0)
        dxy = macro.get("DXY", {}).get("price", 0)
        btc = macro.get("BTC", {}).get("price", 0)
        btc_chg = macro.get("BTC", {}).get("change_pct", 0)
        sp500 = macro.get("SP500", {}).get("price", 0)
        sp500_chg = macro.get("SP500", {}).get("change_pct", 0)
        gold = macro.get("GOLD", {}).get("price", 0)
        btc_dom = macro.get("BTC_DOM_APPROX", {}).get("price", 0)

        msg = f"""🌅 *ORACLE SABAH BRİFİNGİ — {now}*
━━━━━━━━━━━━━━━━━━━━━━
🌍 *MAKRO TABLO*
  😨 VIX: {vix:.1f} {'⚠️ Yüksek' if vix > 25 else '✅ Normal'}
  💵 DXY: {dxy:.2f}
  ₿ BTC: ${btc:,.0f} ({btc_chg:+.1f}%)
  🔵 BTC Dom ~: %{btc_dom:.1f}
  📈 S&P500: {sp500:,.0f} ({sp500_chg:+.1f}%)
  🏅 Altın: ${gold:,.0f}
━━━━━━━━━━━━━━━━━━━━━━
⚡ Tam analiz için: /quant BTC ETH AAPL"""

        await _broadcast_alert(msg)
    except Exception as e:
        logger.error(f"[SCHEDULER] Morning briefing failed: {e}")


def start_scheduler():
    """Zamanlayıcıyı başlatır."""
    global _scheduler

    if _scheduler and _scheduler.running:
        logger.info("[SCHEDULER] Already running")
        return

    _scheduler = AsyncIOScheduler(timezone="Europe/Istanbul")

    _scheduler.add_job(
        _hourly_quant_scan,
        "interval",
        hours=1,
        id="hourly_quant",
        replace_existing=True,
    )

    _scheduler.add_job(
        _morning_briefing,
        "cron",
        hour=8,
        minute=0,
        id="morning_briefing",
        replace_existing=True,
    )

    _scheduler.start()
    logger.success("[SCHEDULER] ✅ Scheduler started — hourly scans + morning briefings active")


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        logger.info("[SCHEDULER] Stopped")
