"""DÜĞÜM 6 — Katil Savcı (Red Team)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.config import load_oracle_config
from core.console import GREEN, RED, YELLOW, agent_print, error_print
from core.types import AgentNode, OracleState, PipelineStatus
from tools.llm_engine import LlmEngine


class RedTeamVerdict(BaseModel):
    fatal_error: bool
    confidence: float = Field(ge=0.0, le=1.0)
    justification_notes: str


_LLM = LlmEngine()


def _build_red_team_prompt(state: OracleState, consensus_variance: float) -> str:
    return (
        "Aşağıdaki sinyal raporunu acımasız kurumsal fon bakış açısıyla incele ve "
        "reddetmek için en kritik kusurları ara.\n\n"
        f"Sembol: {state.symbol}\n"
        f"Signal Direction: {state.signal_direction.value}\n"
        f"Composite Confidence: {state.confidence:.4f}\n"
        f"Base RR: {state.base_rr}\n"
        f"Consensus Variance: {consensus_variance:.4f}\n"
        f"Entry: {state.entry_price}\n"
        f"Stop: {state.stop_loss}\n"
        f"TP: {state.take_profit}\n"
        f"Macro Score: {state.macro_score:+.4f}\n"
        f"Quant Score: {state.quant_score:+.4f}\n"
        f"Whale Score: {state.whale_score:+.4f}\n"
        f"Fundamental Score: {state.fundamental_score:+.4f}\n"
        f"Sentiment Score: {state.sentiment_score:+.4f}\n"
    )


async def run_red_team(state: OracleState) -> OracleState:
    agent_print(
        "RED_TEAM",
        f"Savunma denetimi → {state.symbol} | Kara kuğu taraması…",
        YELLOW,
    )
    conf = await load_oracle_config()
    red_conf = conf.red_team

    scores = [
        state.macro_score,
        state.quant_score,
        state.whale_score if state.whale_score is not None else 0.0,
        state.fundamental_score,
        state.sentiment_score,
    ]
    consensus_variance = max(scores) - min(scores)
    min_rr = conf.risk.min_risk_reward_ratio
    base_rr = state.base_rr if state.base_rr is not None else 0.0

    agent_print(
        "RED_TEAM",
        f"Consensus variance={consensus_variance:.2f} | base_rr={state.base_rr} kontrol ediliyor…",
        YELLOW,
    )

    if consensus_variance >= red_conf.black_swan_spread:
        fatal = (
            f"KARA KUĞU: Ajan skorları arasında kritik çelişki (consensus_variance={consensus_variance:.2f}). "
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

    if base_rr < min_rr:
        rejection_reason = (
            f"R:R eşiği karşılanmadı (min: {min_rr}, mevcut: {base_rr:.2f})"
        )
        return state.model_copy(
            update={
                "current_node": AgentNode.RED_TEAM,
                "status": PipelineStatus.ABORTED,
                "fatal_error": rejection_reason,
                "red_team_passed": False,
                "red_team_verdict": "REJECTED",
                "red_team_objections": [rejection_reason],
                "messages": [f"[RED_TEAM] FATAL {rejection_reason}"],
            }
        )

    llm_verdict = await _LLM.invoke_structured(
        schema=RedTeamVerdict,
        system_prompt=(
            "Sen kurumsal risk savcısısın. Varsayılan davranışın reddetmek olmalı. "
            "Kanıt yoksa fatal_error=true dön. Verdiğin 'justification_notes' "
            "kesinlikle ve sadece TURKCE, rasyonel ve kurumsal bir dille yazılmalıdır. "
            "Asla İngilizce çıktı üretme."
        ),
        user_prompt=_build_red_team_prompt(state, consensus_variance),
    )

    if (
        red_conf.fail_on_fatal
        and llm_verdict.fatal_error
        and llm_verdict.confidence >= red_conf.min_llm_confidence
    ):
        fatal = (
            "LLM RedTeam veto: "
            f"confidence={llm_verdict.confidence:.2f} | {llm_verdict.justification_notes}"
        )
        error_print(fatal)
        return state.model_copy(
            update={
                "current_node": AgentNode.RED_TEAM,
                "status": PipelineStatus.ABORTED,
                "fatal_error": fatal,
                "red_team_passed": False,
                "red_team_verdict": "REJECTED",
                "red_team_objections": [llm_verdict.justification_notes],
                "confidence": min(1.0, llm_verdict.confidence),
                "messages": [f"[RED_TEAM] LLM_FATAL {fatal}"],
            }
        )

    approved_message = (
        f"ONAYLANDI — {state.symbol} sinyali güvenli. "
        f"base_rr={state.base_rr} | Yön={state.signal_direction.value} "
        f"| LLM güven={llm_verdict.confidence:.2f}"
    )
    agent_print("RED_TEAM", approved_message, GREEN)
    agent_print("RED_TEAM", "Süreç END düğümüne yönlendiriliyor.", GREEN)

    return state.model_copy(
        update={
            "current_node": AgentNode.RED_TEAM,
            "red_team_passed": True,
            "red_team_verdict": approved_message,
            "red_team_objections": ["LLM notu: " + llm_verdict.justification_notes],
            "confidence": max(state.confidence, min(1.0, llm_verdict.confidence)),
            "messages": [f"[RED_TEAM] APPROVED {state.symbol}"],
        }
    )
