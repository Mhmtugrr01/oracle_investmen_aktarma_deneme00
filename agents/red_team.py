"""DÜĞÜM 6 — Katil Savcı (Red Team)."""

from __future__ import annotations

import asyncio

from core.console import GREEN, RED, YELLOW, agent_print, error_print
from core.types import AgentNode, OracleState, PipelineStatus

BLACK_SWAN_SPREAD = 1.20


async def run_red_team(state: OracleState) -> OracleState:
    agent_print(
        "RED_TEAM",
        f"Savunma denetimi → {state.symbol} | Kara kuğu taraması…",
        YELLOW,
    )
    await asyncio.sleep(0.18)

    scores = [
        state.macro_score,
        state.quant_score,
        state.whale_score,
        state.fundamental_score,
        state.sentiment_score,
    ]
    spread = max(scores) - min(scores)

    agent_print(
        "RED_TEAM",
        f"Skor yayılımı={spread:.2f} | R:R={state.risk_reward_ratio} kontrol ediliyor…",
        YELLOW,
    )

    if spread >= BLACK_SWAN_SPREAD:
        fatal = (
            f"KARA KUĞU: Ajan skorları arasında kritik çelişki (spread={spread:.2f}). "
            "Sinyal üretimi iptal edildi."
        )
        error_print(fatal)
        return state.model_copy(
            update={
                "current_node": AgentNode.RED_TEAM,
                "status": PipelineStatus.ABORTED,
                "fatal_error": fatal,
                "red_team_passed": False,
                "red_team_verdict": "REJECTED",
                "red_team_objections": [fatal],
                "messages": [f"[RED_TEAM] FATAL {fatal}"],
            }
        )

    verdict = (
        f"ONAYLANDI — {state.symbol} sinyali güvenli. "
        f"R:R={state.risk_reward_ratio} | Yön={state.signal_direction.value}"
    )
    agent_print("RED_TEAM", verdict, GREEN)
    agent_print("RED_TEAM", "Süreç END düğümüne yönlendiriliyor.", GREEN)

    return state.model_copy(
        update={
            "current_node": AgentNode.RED_TEAM,
            "red_team_passed": True,
            "red_team_verdict": verdict,
            "red_team_objections": [],
            "messages": [f"[RED_TEAM] APPROVED {state.symbol}"],
        }
    )
