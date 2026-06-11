from core.llm import extrapolate_prompt, detect_intent, llm_call
from core.config import settings
from loguru import logger


async def run_ceo_router(user_input: str) -> dict:
    """
    CEO Router: Kısa kullanıcı girdisini alır, genişletir, yönlendirir.
    Returns dict with expanded_prompt and agent.
    """
    logger.info(f"[CEO ROUTER] Input: {user_input[:80]}")

    expanded = await extrapolate_prompt(user_input)
    agent = await detect_intent(expanded)

    logger.info(f"[CEO ROUTER] Agent: {agent}")
    return {
        "expanded_prompt": expanded,
        "agent": agent,
    }


async def generate_executive_report(context: str) -> str:
    """Yönetici özet raporu üretir."""
    report = await llm_call(
        messages=[{"role": "user", "content": context}],
        system="""Sen Oracle CEO'sun. 
Verilen bilgileri analiz edip kısa yönetici özet raporu yaz.
Format:
📊 YÖNETİCİ ÖZETİ
──────────────────
🎯 Hedef: ...
✅ Tamamlanan: ...
⚠️ Riskler: ...
📋 Sonraki Adımlar: ...

Türkçe, net, 300 kelime max.""",
        max_tokens=2048,
    )
    return report
