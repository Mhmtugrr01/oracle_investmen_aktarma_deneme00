from celery import Celery
from core.config import settings
from loguru import logger

celery_app = Celery(
    "oracle_swarm",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["tasks.task_queue"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Istanbul",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
)

logger.info("Celery app configured")
