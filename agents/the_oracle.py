"""DÜĞÜM 0 — The Oracle (CEO Yönetici)."""

from __future__ import annotations

import asyncio

from core.console import CYAN, GREEN, YELLOW, agent_print, warn_print
from core.types import AgentNode, OracleState, PipelineStatus, SignalDirection

MIN_RISK_REWARD = 0.85
MAX_SCORE_SPREAD = 0.55


async def run_the_oracle(state: OracleState) -> OracleState:
    agent_print(
        "THE_ORACLE",
        f"CEO denetimi → {state.symbol} | Tüm ajan raporları birleştiriliyor…",
        CYAN,
    )
    await asyncio.sleep(0.2)

    scores = [
        state.macro_score,
        state.quant_score,
        state.whale_score,
        state.fundamental_score,
        state.sentiment_score,
    ]
    spread = max(scores) - min(scores)
    composite = state.composite_score
    risk_reward = round(abs(composite) * 1.35 + 0.55, 2)

    agent_print(
        "THE_ORACLE",
        f"Skor yayılımı={spread:.2f} | Kompozit={composite:+.2f} | R:R={risk_reward}",
        CYAN,
    )

    inconsistent = spread > MAX_SCORE_SPREAD
    low_rr = risk_reward < MIN_RISK_REWARD

    if inconsistent or low_rr:
        reason_parts = []
        if inconsistent:
            reason_parts.append(f"ajan skorları tutarsız (spread={spread:.2f})")
        if low_rr:
            reason_parts.append(f"R:R yetersiz ({risk_reward} < {MIN_RISK_REWARD})")
        reason = " + ".join(reason_parts)

        new_retry = state.retry_count + 1
        warn_print(
            f"CEO RED → {reason} | Rötuş #{new_retry}/3"
        )
        agent_print(
            "THE_ORACLE",
            "Koşullu edge: analiz döngüsü başa sarılıyor…",
            YELLOW,
        )

        return state.model_copy(
            update={
                "current_node": AgentNode.THE_ORACLE,
                "status": PipelineStatus.RUNNING,
                "retry_count": new_retry,
                "ceo_approved": False,
                "ceo_revision_reason": reason,
                "risk_reward_ratio": risk_reward,
                "confidence": abs(composite),
                "messages": [f"[THE_ORACLE] RED retry={new_retry} reason={reason}"],
            }
        )

    direction = (
        SignalDirection.LONG
        if composite > 0.15
        else SignalDirection.SHORT
        if composite < -0.15
        else SignalDirection.NEUTRAL
    )
    alpha = (
        f"{direction.value.upper()} {state.symbol} | R:R={risk_reward} "
        f"| Güven={abs(composite):.0%}"
    )

    agent_print("THE_ORACLE", "CEO ONAY → Red Team'e sevk ediliyor.", GREEN)
    agent_print("THE_ORACLE", f"Alpha taslağı: {alpha}", GREEN)

    return state.model_copy(
        update={
            "current_node": AgentNode.THE_ORACLE,
            "status": PipelineStatus.RUNNING,
            "ceo_approved": True,
            "ceo_revision_reason": None,
            "signal_direction": direction,
            "alpha_signal": alpha,
            "risk_reward_ratio": risk_reward,
            "confidence": min(abs(composite) + 0.25, 1.0),
            "entry_price": 1.42,
            "stop_loss": 1.28,
            "take_profit": 1.74,
            "messages": [f"[THE_ORACLE] APPROVED rr={risk_reward}"],
        }
    )
