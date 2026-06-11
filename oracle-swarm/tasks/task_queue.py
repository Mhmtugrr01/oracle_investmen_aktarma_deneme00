import asyncio
from tasks.celery_app import celery_app
from loguru import logger


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def run_oracle_task(self, user_id: str, user_input: str):
    logger.info(f"[CELERY] Starting oracle task for user {user_id}")
    try:
        from core.graph import oracle_graph, OracleState

        initial_state: OracleState = {
            "user_id": user_id,
            "user_input": user_input,
            "expanded_prompt": "",
            "agent": "",
            "result": "",
            "status": "pending",
            "task_id": None,
            "messages": [],
        }

        final_state = run_async(oracle_graph.ainvoke(initial_state))
        logger.success(f"[CELERY] Task completed: {final_state.get('task_id')}")
        return {
            "task_id": final_state.get("task_id"),
            "agent": final_state.get("agent"),
            "result": final_state.get("result", "")[:500],
            "status": "completed",
        }
    except Exception as e:
        logger.error(f"[CELERY] Task failed: {e}")
        raise self.retry(exc=e)


@celery_app.task
def scheduled_quant_scan():
    """Saatlik otomatik piyasa taraması."""
    logger.info("[CELERY SCHEDULED] Running quant scan")
    try:
        from agents.hft_quant import run_quant_agent
        result = run_async(run_quant_agent("BTC ETH AAPL rutin tarama"))
        logger.info(f"[CELERY SCHEDULED] Quant scan done: {result[:100]}")
        return result
    except Exception as e:
        logger.error(f"[CELERY SCHEDULED] Quant scan failed: {e}")
        return str(e)
