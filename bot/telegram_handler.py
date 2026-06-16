"""
PROJECT OLYMPUS — Telegram Handler (FAZ 1 dummy echo).
FAZ 2'de LangGraph pipeline tetikleyicisine dönüştürülecek.
"""

from __future__ import annotations

import os

from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


class TelegramHandler:
    def __init__(self, token: str) -> None:
        self._token = token
        self._app: Application | None = None

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message:
            return
        await update.effective_message.reply_text(
            "👑 PROJECT OLYMPUS — The Oracle\n"
            "Otonom fon yöneticisi aktif.\n"
            "Komut: /analyze BTC/USDT"
        )

    async def _cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message:
            return
        symbol = context.args[0] if context.args else "BTC/USDT"
        await update.effective_message.reply_text(
            f"🔮 [{symbol}] analiz kuyruğuna alındı — FAZ 2'de pipeline bağlanacak."
        )

    async def _echo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message or not update.effective_message.text:
            return
        await update.effective_message.reply_text(
            f"Echo: {update.effective_message.text}"
        )

    def build(self) -> Application:
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("analyze", self._cmd_analyze))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._echo))
        self._app = app
        return app

    async def start(self) -> None:
        if self._app is None:
            self.build()
        assert self._app is not None
        logger.info("Telegram bot polling başlatılıyor...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self._app is None:
            return
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram bot durduruldu.")


def create_handler() -> TelegramHandler:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN ortam değişkeni tanımlı değil.")
    return TelegramHandler(token=token)
