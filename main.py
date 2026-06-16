"""
PROJECT OLYMPUS — The Oracle
FAZ 2: LangGraph Brain Orchestration test ateşleyicisi.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger

from core.console import BOLD, CYAN, RESET, system_print
from core.graph import compile_oracle_graph
from core.types import OracleState, PipelineStatus


async def run_pipeline_simulation() -> OracleState:
    system_print("=" * 60, CYAN)
    system_print("PROJECT OLYMPUS - LangGraph Pipeline Simulasyonu", CYAN)
    system_print("=" * 60, CYAN)

    query = "Kullanici $FET icin tarama istiyor"
    initial_state = OracleState(
        query=query,
        symbol="FET/USDT",
        user_id="demo_user_001",
        chat_id=999001,
    )

    system_print(f"Kullanici girdisi: '{query}'", CYAN)
    system_print(f"Hedef sembol: {initial_state.symbol}", CYAN)
    system_print(f"Session: {initial_state.session_id}", CYAN)
    print(f"\n{BOLD}{CYAN}>> LangGraph compile & ainvoke basliyor...{RESET}\n", flush=True)

    graph = compile_oracle_graph()
    raw_result = await graph.ainvoke(initial_state)

    if isinstance(raw_result, OracleState):
        final_state = raw_result
    else:
        final_state = OracleState.model_validate(raw_result)

    print(f"\n{BOLD}{CYAN}>> Pipeline tamamlandi.{RESET}\n", flush=True)
    system_print(f"Durum       : {final_state.status.value}", CYAN)
    system_print(f"Retry       : {final_state.retry_count}/3", CYAN)
    system_print(f"CEO Onay    : {final_state.ceo_approved}", CYAN)
    system_print(f"Red Team    : {final_state.red_team_verdict or '-'}", CYAN)
    system_print(f"Alpha       : {final_state.alpha_signal or '-'}", CYAN)
    system_print(f"R:R         : {final_state.risk_reward_ratio or '-'}", CYAN)
    system_print(f"Fatal Error : {final_state.fatal_error or '-'}", CYAN)
    system_print(f"Mesaj sayisi: {len(final_state.messages)}", CYAN)
    system_print("=" * 60, CYAN)

    return final_state


async def bootstrap() -> None:
    load_dotenv()
    logger.remove()
    await run_pipeline_simulation()


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
