import os
from pydantic_settings import BaseSettings
from loguru import logger


class Settings(BaseSettings):
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    openai_base_url: str = os.getenv("AI_INTEGRATIONS_OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_api_key: str = os.getenv("AI_INTEGRATIONS_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    # Groq — ücretsiz LLM (console.groq.com)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = "llama-3.3-70b-versatile"

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    database_url: str = os.getenv("DATABASE_URL", "")

    # Groq varsa Groq kullan, yoksa OpenAI
    @property
    def llm_model(self) -> str:
        return self.groq_model if self.groq_api_key else "gpt-4o-mini"

    @property
    def llm_model_heavy(self) -> str:
        return self.groq_model if self.groq_api_key else "gpt-4o"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    colorize=True,
)
