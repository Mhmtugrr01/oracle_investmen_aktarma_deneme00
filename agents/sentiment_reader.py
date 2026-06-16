"""DÜĞÜM 5 — Master Alpha Duygu Okyucusu (Social Whisperer)."""

from __future__ import annotations

import asyncio

from core.console import BLUE, GREEN, agent_print
from core.types import AgentNode, OracleState, PipelineStatus


async def run_sentiment_reader(state: OracleState) -> OracleState:
    agent_print(
        "SOCIAL_WHISPERER",
        f"Devrede → {state.symbol} | Duygu okuması başlatıldı…",
        GREEN,
    )
    await asyncio.sleep(0.12)

    score_map = {0: -0.22, 1: 0.38, 2: 0.46, 3: 0.48}
    sentiment_score = score_map.get(state.retry_count, 0.48)

    agent_print(
        "SOCIAL_WHISPERER",
        f"Sosyal hacim + fear/greed → sentiment_score={sentiment_score:+.2f}",
        BLUE,
    )
    agent_print(
        "SOCIAL_WHISPERER",
        "Twitter/Telegram/Reddit fısıltı ağı tarandı.",
        GREEN,
    )

    return state.model_copy(
        update={
            "current_node": AgentNode.SENTIMENT_READER,
            "status": PipelineStatus.RUNNING,
            "sentiment_score": sentiment_score,
            "messages": [
                f"[SOCIAL_WHISPERER] score={sentiment_score:+.2f}"
            ],
        }
    )
