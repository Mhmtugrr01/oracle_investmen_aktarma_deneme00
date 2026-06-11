import json
from datetime import datetime
from supabase import create_client, Client
from core.config import settings
from loguru import logger


def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)


_memory_fallback: list[dict] = []


async def save_task(
    user_id: str,
    user_input: str,
    expanded_prompt: str,
    agent: str,
    result: str,
    status: str = "completed",
) -> str | None:
    import uuid
    task_id = str(uuid.uuid4())
    record = {
        "id": task_id,
        "user_id": user_id,
        "user_input": user_input,
        "expanded_prompt": expanded_prompt,
        "agent": agent,
        "result": result[:500],
        "status": status,
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        sb = get_supabase()
        res = sb.table("oracle_tasks").insert(record).execute()
        saved_id = res.data[0]["id"] if res.data else task_id
        logger.info(f"Task saved to Supabase: {saved_id}")
        return saved_id
    except Exception as e:
        logger.warning(f"Supabase save failed (using local fallback): {e}")
        _memory_fallback.append(record)
        if len(_memory_fallback) > 50:
            _memory_fallback.pop(0)
        return task_id


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
        logger.warning(f"Supabase fetch failed (using local fallback): {e}")
        user_tasks = [t for t in _memory_fallback if t.get("user_id") == str(user_id)]
        return list(reversed(user_tasks))[-limit:]


async def init_tables():
    try:
        sb = get_supabase()
        sb.table("oracle_tasks").select("id").limit(1).execute()
        logger.info("Supabase tables OK")
    except Exception as e:
        logger.warning(f"Table check: {e} — tablo henüz yok, devam ediliyor")
        logger.info("Supabase'de tabloları kurmak için oracle-swarm/db/schema.sql dosyasını çalıştırın")
