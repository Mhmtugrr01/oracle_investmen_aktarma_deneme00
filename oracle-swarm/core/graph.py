from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from core.llm import extrapolate_prompt, detect_intent
from core.memory import save_task
from loguru import logger
import operator


class OracleState(TypedDict):
    user_id: str
    user_input: str
    expanded_prompt: str
    agent: str
    result: str
    status: str
    task_id: str | None
    messages: Annotated[list[str], operator.add]


async def extrapolation_node(state: OracleState) -> OracleState:
    logger.info(f"[CEO ROUTER] Extrapolating: {state['user_input'][:50]}")
    expanded = await extrapolate_prompt(state["user_input"])
    agent = await detect_intent(expanded)
    logger.info(f"[CEO ROUTER] → Routing to: {agent}")
    return {
        **state,
        "expanded_prompt": expanded,
        "agent": agent,
        "messages": [f"CEO Router: Genişletildi ve {agent} ajanına yönlendirildi."],
    }


def route_to_agent(state: OracleState) -> str:
    mapping = {
        "SWE": "swe_agent",
        "QUANT": "quant_agent",
        "MARKETING": "marketing_agent",
        "EDGE": "edge_agent",
        "CEO": "ceo_report_node",
    }
    return mapping.get(state["agent"], "ceo_report_node")


async def ceo_report_node(state: OracleState) -> OracleState:
    from core.llm import llm_call
    result = await llm_call(
        messages=[{"role": "user", "content": state["expanded_prompt"]}],
        system="Sen Oracle CEO'sun. Stratejik analiz ve yönetim kararları veriyorsun. Türkçe yanıt ver.",
        max_tokens=4096,
    )
    return {**state, "result": result, "status": "completed"}


async def swe_node(state: OracleState) -> OracleState:
    from agents.swe_engineer import run_swe_agent
    result = await run_swe_agent(state["expanded_prompt"])
    return {**state, "result": result, "status": "completed"}


async def quant_node(state: OracleState) -> OracleState:
    from agents.hft_quant import run_quant_agent
    result = await run_quant_agent(state["expanded_prompt"])
    return {**state, "result": result, "status": "completed"}


async def marketing_node(state: OracleState) -> OracleState:
    from agents.marketing import run_marketing_agent
    result = await run_marketing_agent(state["expanded_prompt"])
    return {**state, "result": result, "status": "completed"}


async def edge_node(state: OracleState) -> OracleState:
    from agents.edge_daemon import run_edge_agent
    result = await run_edge_agent(state["expanded_prompt"])
    return {**state, "result": result, "status": "completed"}


async def save_node(state: OracleState) -> OracleState:
    task_id = await save_task(
        user_id=state["user_id"],
        user_input=state["user_input"],
        expanded_prompt=state["expanded_prompt"],
        agent=state["agent"],
        result=state["result"],
        status=state["status"],
    )
    return {**state, "task_id": task_id}


def build_graph():
    graph = StateGraph(OracleState)

    graph.add_node("extrapolation", extrapolation_node)
    graph.add_node("ceo_report_node", ceo_report_node)
    graph.add_node("swe_agent", swe_node)
    graph.add_node("quant_agent", quant_node)
    graph.add_node("marketing_agent", marketing_node)
    graph.add_node("edge_agent", edge_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("extrapolation")

    graph.add_conditional_edges(
        "extrapolation",
        route_to_agent,
        {
            "swe_agent": "swe_agent",
            "quant_agent": "quant_agent",
            "marketing_agent": "marketing_agent",
            "edge_agent": "edge_agent",
            "ceo_report_node": "ceo_report_node",
        },
    )

    for node in ["ceo_report_node", "swe_agent", "quant_agent", "marketing_agent", "edge_agent"]:
        graph.add_edge(node, "save")

    graph.add_edge("save", END)

    return graph.compile()


oracle_graph = build_graph()
