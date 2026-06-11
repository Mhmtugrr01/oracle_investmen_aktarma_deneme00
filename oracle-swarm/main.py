import asyncio
import signal
import sys
from loguru import logger
from core.config import settings
from core.memory import init_tables
from core.scheduler import start_scheduler, stop_scheduler


async def main():
    logger.info("🚀 Oracle Master-Swarm V4.0 başlatılıyor...")
    logger.info("☁️ Ortam: Cloud (Replit)")
    logger.info(f"🤖 Telegram Bot Token: {'✅ Set' if settings.telegram_bot_token else '❌ Eksik'}")
    logger.info(f"🗄️ Supabase URL: {'✅ Set' if settings.supabase_url else '❌ Eksik'}")
    logger.info(f"🧠 OpenAI Key: {'✅ Set' if settings.openai_api_key else '❌ Eksik'}")

    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN ortam değişkeni eksik!")
        sys.exit(1)

    try:
        await init_tables()
        logger.success("✅ Supabase tabloları kontrol edildi")
    except Exception as e:
        logger.warning(f"⚠️ Supabase init: {e} — Devam ediliyor")

    from bot_handler.bot import build_application

    app = build_application()

    await app.initialize()
    await app.start()
    await app.updater.start_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
    )

    logger.success("✅ Oracle Swarm sistemi hazır. Telegram bekleniyor...")
    logger.info("📱 Botunuza /start mesajı gönderin")
    logger.success("🟢 Telegram polling aktif")

    start_scheduler()
    logger.success("⏰ Zamanlayıcı aktif — saatlik piyasa taraması + sabah brifing")

    stop_event = asyncio.Event()

    def _stop(signum, frame):
        logger.info("🔴 Durdurma sinyali alındı")
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    await stop_event.wait()

    logger.info("🔄 Sistemi kapatıyorum...")
    stop_scheduler()
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
    logger.info("👋 Oracle Swarm kapatıldı")


if __name__ == "__main__":
    asyncio.run(main())
