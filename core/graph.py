"""
PROJECT OLYMPUS — LangGraph Brain Orchestration
Saf langgraph StateGraph + Pydantic OracleState.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agents.fundamental_filter import run_fundamental_filter
from agents.macro_sentinel import run_macro_sentinel
from agents.quant_engine import run_quant_engine
from agents.red_team import run_red_team
from agents.sentiment_reader import run_sentiment_reader
from agents.the_oracle import run_the_oracle
from agents.whale_hunter import run_whale_hunter
from core.config import get_oracle_config_cached
from core.console import system_print
from core.types import OracleState, PipelineStatus, SignalDirection

RouteAfterCeo = Literal["revise", "red_team", "end_failed", "end_aborted", "end_low_score"]
RouteAfterRedTeam = Literal["end_aborted", "end_completed"]
RouteAfterMacro = Literal["next", "end_aborted"]
RouteAfterQuant = Literal["next", "end_aborted"]
RouteAfterWhale = Literal["next", "end_aborted"]
RouteAfterFundamental = Literal["next", "end_aborted"]
RouteAfterSentiment = Literal["next", "end_aborted"]


def _max_ceo_retries() -> int:
    return get_oracle_config_cached().ceo.max_retries


def _diff_state(before: OracleState, after: OracleState) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for field in OracleState.model_fields:
        new_val = getattr(after, field)
        old_val = getattr(before, field)
        if field == "messages":
            added = [m for m in new_val if m not in old_val]
            if added:
                patch["messages"] = added
        elif new_val != old_val:
            patch[field] = new_val
    return patch


async def _wrap(agent_fn, state: OracleState) -> dict[str, Any]:
    result = await agent_fn(state)
    return _diff_state(state, result)


async def macro_sentinel_node(state: OracleState) -> dict[str, Any]:
    return await _wrap(run_macro_sentinel, state)


async def quant_engine_node(state: OracleState) -> dict[str, Any]:
    return await _wrap(run_quant_engine, state)


async def whale_hunter_node(state: OracleState) -> dict[str, Any]:
    return await _wrap(run_whale_hunter, state)


async def fundamental_filter_node(state: OracleState) -> dict[str, Any]:
    return await _wrap(run_fundamental_filter, state)


async def sentiment_reader_node(state: OracleState) -> dict[str, Any]:
    return await _wrap(run_sentiment_reader, state)


async def the_oracle_node(state: OracleState) -> dict[str, Any]:
    return await _wrap(run_the_oracle, state)


async def red_team_node(state: OracleState) -> dict[str, Any]:
    return await _wrap(run_red_team, state)


def route_after_macro(state: OracleState) -> RouteAfterMacro:
    return "end_aborted" if state.fatal_error else "next"


def route_after_quant(state: OracleState) -> RouteAfterQuant:
    return "end_aborted" if state.fatal_error else "next"


def route_after_whale(state: OracleState) -> RouteAfterWhale:
    return "end_aborted" if state.fatal_error else "next"


def route_after_fundamental(state: OracleState) -> RouteAfterFundamental:
    return "end_aborted" if state.fatal_error else "next"


def route_after_sentiment(state: OracleState) -> RouteAfterSentiment:
    return "end_aborted" if state.fatal_error else "next"


async def end_failed_node(state: OracleState) -> dict[str, Any]:
    max_retries = _max_ceo_retries()
    system_print(
        f"Pipeline iptal — CEO rötuş limiti ({max_retries}) aşıldı.",
        "\033[91m",
    )
    failed = state.mark_failed(
        f"CEO denetimi {max_retries} denemede tutarlılık sağlanamadı."
    )
    # Terminal fail durumunda sinyal alanlarını da sıfırla.
    failed = failed.model_copy(
        update={
            "signal_label": None,
            "signal_direction": SignalDirection.NO_TRADE,
            "ceo_approved": False,
        }
    )
    return _diff_state(state, failed)


async def end_completed_node(state: OracleState) -> dict[str, Any]:
    system_print("Pipeline ONAYLANDI — Sinyal üretildi.", "\033[92m")
    completed = state.mark_completed()
    return _diff_state(state, completed)


async def end_aborted_node(state: OracleState) -> dict[str, Any]:
    system_print(f"Pipeline İPTAL — {state.fatal_error}", "\033[91m")
    aborted = state.model_copy(
        update={"status": PipelineStatus.ABORTED, "current_node": state.current_node}
    )
    return _diff_state(state, aborted)


async def end_low_score_node(state: OracleState) -> dict[str, Any]:
    reason = "KOMPOZİT SKOR YETERSİZ — İŞLEM YAPILMADI"
    system_print(reason, "\033[93m")
    aborted = state.model_copy(
        update={
            "status": PipelineStatus.ABORTED,
            "fatal_error": reason,
            "signal_label": None,
            "signal_direction": SignalDirection.NO_TRADE,
            "ceo_approved": False,
            "messages": [f"[SYSTEM] {reason}"],
        }
    )
    return _diff_state(state, aborted)


def route_after_ceo(state: OracleState) -> RouteAfterCeo:
    conf = get_oracle_config_cached()
    max_retries = _max_ceo_retries()
    if state.fatal_error:
        return "end_aborted"
    if state.composite_score < conf.ceo.min_composite_score or state.confidence < conf.ceo.confidence_threshold:
        return "end_low_score"
    if state.ceo_approved:
        return "red_team"
    if state.retry_count >= max_retries:
        return "end_failed"
    return "revise"


def route_after_red_team(state: OracleState) -> RouteAfterRedTeam:
    if state.fatal_error:
        return "end_aborted"
    return "end_completed"


def build_oracle_graph() -> StateGraph:
    graph = StateGraph(OracleState)

    graph.add_node("macro_sentinel", macro_sentinel_node)
    graph.add_node("quant_engine", quant_engine_node)
    graph.add_node("whale_hunter", whale_hunter_node)
    graph.add_node("fundamental_filter", fundamental_filter_node)
    graph.add_node("sentiment_reader", sentiment_reader_node)
    graph.add_node("the_oracle", the_oracle_node)
    graph.add_node("red_team", red_team_node)
    graph.add_node("end_failed", end_failed_node)
    graph.add_node("end_completed", end_completed_node)
    graph.add_node("end_aborted", end_aborted_node)
    graph.add_node("end_low_score", end_low_score_node)

    graph.add_edge(START, "macro_sentinel")

    graph.add_conditional_edges(
        "macro_sentinel",
        route_after_macro,
        {
            "next": "quant_engine",
            "end_aborted": "end_aborted",
        },
    )

    graph.add_conditional_edges(
        "quant_engine",
        route_after_quant,
        {
            "next": "whale_hunter",
            "end_aborted": "end_aborted",
        },
    )

    graph.add_conditional_edges(
        "whale_hunter",
        route_after_whale,
        {
            "next": "fundamental_filter",
            "end_aborted": "end_aborted",
        },
    )

    graph.add_conditional_edges(
        "fundamental_filter",
        route_after_fundamental,
        {
            "next": "sentiment_reader",
            "end_aborted": "end_aborted",
        },
    )

    graph.add_conditional_edges(
        "sentiment_reader",
        route_after_sentiment,
        {
            "next": "the_oracle",
            "end_aborted": "end_aborted",
        },
    )

    graph.add_conditional_edges(
        "the_oracle",
        route_after_ceo,
        {
            "revise": "macro_sentinel",
            "red_team": "red_team",
            "end_failed": "end_failed",
            "end_aborted": "end_aborted",
            "end_low_score": "end_low_score",
        },
    )

    graph.add_conditional_edges(
        "red_team",
        route_after_red_team,
        {
            "end_aborted": "end_aborted",
            "end_completed": "end_completed",
        },
    )

    graph.add_edge("end_failed", END)
    graph.add_edge("end_completed", END)
    graph.add_edge("end_aborted", END)
    graph.add_edge("end_low_score", END)

    return graph


def compile_oracle_graph():
    return build_oracle_graph().compile()
