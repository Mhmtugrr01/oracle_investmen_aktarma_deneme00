"""DÜĞÜM 3 — Sembiyotik Balina Avcısı (Symbiotic Hunter)."""

from __future__ import annotations

import asyncio

from core.console import BLUE, MAGENTA, agent_print
from core.types import AgentNode, OracleState, PipelineStatus


async def run_whale_hunter(state: OracleState) -> OracleState:
    agent_print(
        "SYMBIOTIC_HUNTER",
        f"Devrede → {state.symbol} | Balina radarı aktif…",
        MAGENTA,
    )
    await asyncio.sleep(0.12)

    score_map = {0: 0.88, 1: 0.44, 2: 0.48, 3: 0.50}
    whale_score = score_map.get(state.retry_count, 0.50)

    agent_print(
        "SYMBIOTIC_HUNTER",
        f"On-chain büyük cüzdan hareketi → whale_score={whale_score:+.2f}",
        BLUE,
    )
    agent_print(
        "SYMBIOTIC_HUNTER",
        "Borsa giriş/çıkış akışı ve likidite duvarları izlendi.",
        MAGENTA,
    )

    return state.model_copy(
        update={
            "current_node": AgentNode.WHALE_HUNTER,
            "status": PipelineStatus.RUNNING,
            "whale_score": whale_score,
            "messages": [
                f"[WHALE_HUNTER] score={whale_score:+.2f}"
            ],
        }
    )
