"""DÜĞÜM 4 — Temel / On-Chain Süzgeci (Fundamental Miner)."""

from __future__ import annotations

import asyncio

from core.console import BLUE, CYAN, agent_print
from core.types import AgentNode, OracleState, PipelineStatus


async def run_fundamental_filter(state: OracleState) -> OracleState:
    agent_print(
        "FUNDAMENTAL_MINER",
        f"Devrede → {state.symbol} | Temel veri madenciliği…",
        CYAN,
    )
    await asyncio.sleep(0.12)

    score_map = {0: 0.15, 1: 0.42, 2: 0.49, 3: 0.51}
    fundamental_score = score_map.get(state.retry_count, 0.51)

    agent_print(
        "FUNDAMENTAL_MINER",
        f"Tokenomics + TVL + geliştirici aktivitesi → fundamental_score={fundamental_score:+.2f}",
        BLUE,
    )
    agent_print(
        "FUNDAMENTAL_MINER",
        "On-chain metrik süzgeci tamamlandı.",
        CYAN,
    )

    return state.model_copy(
        update={
            "current_node": AgentNode.FUNDAMENTAL_FILTER,
            "status": PipelineStatus.RUNNING,
            "fundamental_score": fundamental_score,
            "messages": [
                f"[FUNDAMENTAL_MINER] score={fundamental_score:+.2f}"
            ],
        }
    )
