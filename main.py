from __future__ import annotations

# SSL — kurumsal ağ sertifika enjeksiyonu (from __future__ sonrası, diğer importlardan önce)
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

"""PROJECT OLYMPUS — Production entrypoint (Telegram long-polling)."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from bot.telegram_handler import create_handler
from core.config import load_oracle_config
from core.console import system_print
from core.scanner import OracleScanner


async def bootstrap() -> None:
    load_dotenv()
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
    config = await load_oracle_config()
    handler = create_handler()
    await handler.start()

    allowed_raw = os.getenv("ALLOWED_USER_ID", "").strip()
    allowed_chat_id = int(allowed_raw) if allowed_raw else None

    async def send_telegram_message(message: str) -> None:
        if allowed_chat_id is None:
            logger.warning("Scanner mesajı için ALLOWED_USER_ID tanımlı değil.")
            return
        assert handler._app is not None
        await handler._app.bot.send_message(chat_id=allowed_chat_id, text=message)

    async def run_pipeline(symbol: str):
        return await handler._run_pipeline(
            symbol=symbol,
            user_id="scanner",
            chat_id=allowed_chat_id or 0,
            query=f"/scan {symbol}",
        )

    scanner = OracleScanner(
        pipeline_runner=run_pipeline,
        telegram_bot=send_telegram_message,
        config=config.model_dump(),
    )
    asyncio.create_task(scanner.start())

    system_print("Telegram long-polling aktif. Bot calisiyor.")
    await asyncio.Event().wait()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        asyncio.run(bootstrap())
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu.")
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Pipeline hatası: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
