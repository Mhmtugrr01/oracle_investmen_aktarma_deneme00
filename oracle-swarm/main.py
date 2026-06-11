import asyncio
import signal
import sys
from loguru import logger
from core.config import settings
from core.memory import init_tables


async def main():
    logger.info("🚀 Oracle Master-Swarm V4.0 başlatılıyor...")
    logger.info(f"☁️ Ortam: Cloud (Replit)")
    logger.info(f"🤖 Telegram Bot Token: {'✅ Set' if settings.telegram_bot_token else '❌ Eksik'}")
    logger.info(f"🗄️ Supabase URL: {'✅ Set' if settings.supabase_url else '❌ Eksik'}")

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

    logger.success("✅ Oracle Swarm sistemi hazır. Telegram bekleniyor...")
    logger.info("📱 Botunuza /start mesajı gönderin")

    await app.initialize()
    await app.start()
    await app.updater.start_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
    )

    logger.success("🟢 Telegram polling aktif")

    stop_event = asyncio.Event()

    def _stop(signum, frame):
        logger.info("🔴 Durdurma sinyali alındı")
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    await stop_event.wait()

    logger.info("🔄 Sistemi kapatıyorum...")
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
    logger.info("👋 Oracle Swarm kapatıldı")


if __name__ == "__main__":
    asyncio.run(main())
