from __future__ import annotations

# SSL — kurumsal ağ sertifika enjeksiyonu (from __future__ sonrası, diğer importlardan önce)
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

"""
PROJECT OLYMPUS — Production Entrypoint with FastAPI Port-Bypass (R05_MASTER)
Orijinal tarayıcı (Scanner) kapsamını %100 koruyan ve 7/24 ücretsiz yaşatan ana motor.
"""

import asyncio
import os
import sys
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from loguru import logger

from bot.telegram_handler import create_handler
from core.config import load_oracle_config
from core.console import system_print
from core.scanner import OracleScanner

# FastAPI Uygulaması (Render'ın Port kontrolünü geçip 7/24 ücretsiz yaşatmak için)
app = FastAPI()

# Global referanslar
handler = None
scanner = None


async def bootstrap_bg() -> None:
    """Asıl otomatik tarayıcı ve bot döngüsünü arka planda asenkron çalıştıran asıl motor."""
    try:
        load_dotenv()
        logger.remove()
        logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
        
        config = await load_oracle_config()
        global handler, scanner
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
        # Orijinal tarayıcı döngülerini (4 saatlik tam tarama, 15 dk seviye izleme) başlatır
        asyncio.create_task(scanner.start())

        system_print("Telegram long-polling ve otomatik tarayıcı aktif. Bot çalışıyor.")
    except Exception as exc:
        logger.exception(f"Arka plan bootstrap hatası: {exc}")


@app.on_event("startup")
async def startup_event():
    """FastAPI sunucusu ayağa kalktığında asıl bot/tarayıcı motorunu arka planda başlatır."""
    asyncio.create_task(bootstrap_bg())


@app.get("/")
def read_root():
    """Render'ın "Ben yaşıyorum" kontrolüne (Health Check) verilen yanıt."""
    return {"status": "active", "service": "Olympus Oracle R05_MASTER"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
            
    # Render portu dinamik atar (PORT env), yoksa 8000 portunu kullanır
    port = int(os.getenv("PORT", 8000))
    logger.info(f"[SYSTEM] Web sunucusu {port} portu üzerinden başlatılıyor...")
    
    try:
        # Uvicorn'u başlatır, bu blok bloke edicidir ve Render portunu dinler
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu.")
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Pipeline web sunucusu hatası: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
