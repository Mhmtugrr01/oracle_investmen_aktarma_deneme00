import json
from datetime import datetime
from supabase import create_client, Client
from core.config import settings
from loguru import logger


def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)


async def save_task(
    user_id: str,
    user_input: str,
    expanded_prompt: str,
    agent: str,
    result: str,
    status: str = "completed",
) -> str | None:
    try:
        sb = get_supabase()
        data = {
            "user_id": user_id,
            "user_input": user_input,
            "expanded_prompt": expanded_prompt,
            "agent": agent,
            "result": result,
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
        }
        res = sb.table("oracle_tasks").insert(data).execute()
        task_id = res.data[0]["id"] if res.data else None
        logger.info(f"Task saved: {task_id}")
        return task_id
    except Exception as e:
        logger.error(f"Memory save failed: {e}")
        return None


async def get_recent_tasks(user_id: str, limit: int = 10) -> list[dict]:
    try:
        sb = get_supabase()
        res = (
            sb.table("oracle_tasks")
            .select("*")
            .eq("user_id", str(user_id))
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error(f"Memory fetch failed: {e}")
        return []


async def init_tables():
    try:
        sb = get_supabase()
        sb.table("oracle_tasks").select("id").limit(1).execute()
        logger.info("Supabase tables OK")
    except Exception as e:
        logger.warning(f"Table check: {e} — tablo henüz yok, devam ediliyor")
        logger.info("Supabase'de tabloları kurmak için oracle-swarm/db/schema.sql dosyasını çalıştırın")
