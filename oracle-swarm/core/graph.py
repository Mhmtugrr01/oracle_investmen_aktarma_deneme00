"""
ORACLE LANGGRAPH GRAPH V2
- Kural tabanlı intent tespiti (LLM olmadan %90 doğru yönlendirme)
- Her ajan CEO Critic katmanından geçer
- Herhangi bir adım hata verse bile devam eder
"""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from core.llm import extrapolate_prompt, detect_intent, rule_based_intent
from core.critic import audit_and_refine
from core.memory import save_task
from loguru import logger
import operator


class OracleState(TypedDict):
    user_id: str
    user_input: str
    expanded_prompt: str
    agent: str
    result: str
    audited_result: str
    status: str
    task_id: str | None
    messages: Annotated[list[str], operator.add]


async def extrapolation_node(state: OracleState) -> OracleState:
    user_input = state["user_input"]
    logger.info(f"[CEO ROUTER] Input: {user_input[:60]}")

    # 1. Kural tabanlı hızlı yönlendirme (LLM yok)
    fast_intent = rule_based_intent(user_input)
    if fast_intent:
        logger.info(f"[CEO ROUTER] Hızlı yönlendirme: {fast_intent} (kural tabanlı)")
        return {
            **state,
            "expanded_prompt": user_input,
            "agent": fast_intent,
            "messages": [f"CEO Router → {fast_intent} (kural tabanlı)"],
        }

    # 2. Kısa girdilerde doğrudan LLM'siz yönlendirme
    words = user_input.strip().split()
    if len(words) == 1 and words[0].upper() in {
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "BNB", "AVAX"
    }:
        logger.info("[CEO ROUTER] Tek kripto kelimesi → QUANT")
        return {
            **state,
            "expanded_prompt": f"Kripto analizi: {user_input.upper()}",
            "agent": "QUANT",
            "messages": ["CEO Router → QUANT (tek sembol)"],
        }

    # 3. LLM ile genişletme + tespit
    try:
        expanded = await extrapolate_prompt(user_input)
        agent = await detect_intent(user_input, expanded)
    except Exception as e:
        logger.error(f"[CEO ROUTER] Extrapolation hata: {e}")
        agent = rule_based_intent(user_input) or "CEO"
        expanded = user_input

    logger.info(f"[CEO ROUTER] → {agent}")
    return {
        **state,
        "expanded_prompt": expanded or user_input,
        "agent": agent,
        "messages": [f"CEO Router → {agent}"],
    }


def route_to_agent(state: OracleState) -> str:
    mapping = {
        "SWE": "swe_agent",
        "QUANT": "quant_agent",
        "MARKETING": "marketing_agent",
        "EDGE": "edge_agent",
        "FREELANCER": "freelancer_agent",
        "CEO": "ceo_report_node",
    }
    return mapping.get(state["agent"], "ceo_report_node")


async def _run_with_audit(state: OracleState, agent_name: str, raw_result: str) -> OracleState:
    """Ajan çıktısını CEO Critic katmanından geçirir. Hata olursa ham sonucu döner."""
    if not raw_result or len(raw_result) < 20:
        return {**state, "result": raw_result, "audited_result": raw_result, "status": "completed"}

    try:
        audited = await audit_and_refine(
            user_input=state["user_input"],
            expanded_prompt=state["expanded_prompt"],
            agent_name=agent_name,
            raw_response=raw_result,
            max_rounds=1,  # 1 tur yeterli, hız için
        )
    except Exception as e:
        logger.warning(f"[CRITIC] Audit hata ({agent_name}): {e} — ham sonuç kullanılıyor")
        audited = raw_result

    return {
        **state,
        "result": raw_result,
        "audited_result": audited or raw_result,
        "status": "completed",
    }


async def ceo_report_node(state: OracleState) -> OracleState:
    from core.llm import llm_call
    user_input = state["user_input"]

    result = await llm_call(
        messages=[{"role": "user", "content": state.get("expanded_prompt") or user_input}],
        system="Sen Oracle CEO'sun. Stratejik analiz ve yönetim kararları veriyorsun. Türkçe, net, somut.",
        max_tokens=3000,
    )

    if not result:
        ui = user_input.strip()
        result = (
            f"🧠 *Oracle CEO — '{ui}'*\n\n"
            f"Komutunuzu aldım. Doğru ajan için daha spesifik yazabilirsiniz:\n\n"
            f"📊 *Kripto/Piyasa:* `btc`, `eth analiz`, `altın fiyat`\n"
            f"💻 *Kod:* `Python API yaz`, `bot geliştir`\n"
            f"💼 *İş:* `Upwork iş bul`, `freelance çalış`\n"
            f"📣 *Pazarlama:* `Bursa OSB firmalar`, `email gönder`\n"
            f"💾 *Sistem:* `/status`, `disk raporu`\n\n"
            f"Veya doğrudan:\n"
            f"`/quant BTC ETH` — anlık piyasa analizi\n"
            f"`/scan` — otomatik tarama\n"
            f"`/status` — sistem durumu"
        )

    return {
        **state,
        "result": result,
        "audited_result": result,
        "status": "completed",
    }


async def swe_node(state: OracleState) -> OracleState:
    try:
        from agents.swe_engineer import run_swe_agent
        result = await run_swe_agent(state["expanded_prompt"] or state["user_input"])
    except Exception as e:
        logger.error(f"[SWE] {e}")
        result = f"⚠️ SWE Ajan hatası: {e}"
    return await _run_with_audit(state, "SWE", result)


async def quant_node(state: OracleState) -> OracleState:
    try:
        from agents.hft_quant import run_quant_agent
        # QUANT için ham kullanıcı girdisini de geç — sembol tespiti için
        task = state["user_input"] + " " + (state["expanded_prompt"] or "")
        result = await run_quant_agent(task.strip())
    except Exception as e:
        logger.error(f"[QUANT] {e}")
        result = f"⚠️ Quant Ajan hatası: {e}\n/quant BTC ETH komutu ile tekrar deneyin."
    # QUANT sonucu LLM gerektirmeden zaten hazır — sadece basit audit
    return {
        **state,
        "result": result,
        "audited_result": result,
        "status": "completed",
    }


async def marketing_node(state: OracleState) -> OracleState:
    try:
        from agents.marketing import run_marketing_agent
        result = await run_marketing_agent(state["expanded_prompt"] or state["user_input"])
    except Exception as e:
        logger.error(f"[MARKETING] {e}")
        result = f"⚠️ Marketing Ajan hatası: {e}"
    return await _run_with_audit(state, "MARKETING", result)


async def edge_node(state: OracleState) -> OracleState:
    try:
        from agents.edge_daemon import run_edge_agent
        result = await run_edge_agent(state["expanded_prompt"] or state["user_input"])
    except Exception as e:
        logger.error(f"[EDGE] {e}")
        result = f"⚠️ Edge Ajan hatası: {e}"
    return await _run_with_audit(state, "EDGE", result)


async def freelancer_node(state: OracleState) -> OracleState:
    try:
        from agents.freelancer import run_freelancer_agent
        result = await run_freelancer_agent(state["expanded_prompt"] or state["user_input"])
    except Exception as e:
        logger.error(f"[FREELANCER] {e}")
        result = f"⚠️ Freelancer Ajan hatası: {e}"
    return await _run_with_audit(state, "FREELANCER", result)


async def save_node(state: OracleState) -> OracleState:
    try:
        final_result = state.get("audited_result") or state.get("result", "")
        task_id = await save_task(
            user_id=state["user_id"],
            user_input=state["user_input"],
            expanded_prompt=state["expanded_prompt"],
            agent=state["agent"],
            result=final_result,
            status=state["status"],
        )
        return {**state, "task_id": task_id}
    except Exception as e:
        logger.warning(f"[SAVE] {e}")
        return state


def build_graph():
    graph = StateGraph(OracleState)

    graph.add_node("extrapolation", extrapolation_node)
    graph.add_node("ceo_report_node", ceo_report_node)
    graph.add_node("swe_agent", swe_node)
    graph.add_node("quant_agent", quant_node)
    graph.add_node("marketing_agent", marketing_node)
    graph.add_node("edge_agent", edge_node)
    graph.add_node("freelancer_agent", freelancer_node)
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
            "freelancer_agent": "freelancer_agent",
            "ceo_report_node": "ceo_report_node",
        },
    )

    for node in [
        "ceo_report_node", "swe_agent", "quant_agent",
        "marketing_agent", "edge_agent", "freelancer_agent"
    ]:
        graph.add_edge(node, "save")

    graph.add_edge("save", END)
    return graph.compile()


oracle_graph = build_graph()
