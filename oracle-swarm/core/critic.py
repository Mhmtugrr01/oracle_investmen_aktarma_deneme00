"""
ORACLE CRITIC ENGINE — CEO Denetim Katmanı
Alt ajan çıktısını denetler, gerekirse düzeltir.
LLM yoksa ham sonucu doğrudan döndürür — asla bloklamaz.
"""
from core.llm import llm_call
from loguru import logger


async def audit_and_refine(
    user_input: str,
    expanded_prompt: str,
    agent_name: str,
    raw_response: str,
    max_rounds: int = 1,
) -> str:
    """
    CEO Denetim Döngüsü.
    Kısa veya LLM'siz durumlarda ham sonucu döner — güvenli.
    """
    if not raw_response or len(raw_response) < 30:
        return raw_response

    # Çok uzun yanıtları direkt onayla — LLM maliyeti vs. faydayı dengele
    if len(raw_response) > 3000:
        logger.debug(f"[CRITIC] Uzun yanıt ({len(raw_response)}c) — doğrudan onaylandı")
        return raw_response

    logger.info(f"[CRITIC] Denetleniyor: {agent_name} ({len(raw_response)} chars)")

    try:
        audit_result = await _audit_response(
            user_input=user_input,
            agent_name=agent_name,
            response=raw_response,
        )

        if audit_result["verdict"] == "APPROVED":
            logger.debug(f"[CRITIC] ✅ Onaylandı (güven: {audit_result['confidence']})")
            return raw_response

        logger.info(f"[CRITIC] Revizyon gerekiyor: {audit_result['issues'][:80]}")

        if max_rounds >= 1:
            revised = await _revise_response(
                original_request=user_input,
                previous_response=raw_response,
                critic_notes=audit_result["issues"],
                agent_name=agent_name,
            )
            return revised if revised and len(revised) > 20 else raw_response

    except Exception as e:
        logger.warning(f"[CRITIC] Denetim hatası ({agent_name}): {e} — ham sonuç kullanılıyor")

    return raw_response


async def _audit_response(user_input: str, agent_name: str, response: str) -> dict:
    audit_prompt = f"""Kullanıcı istedi: {user_input[:150]}

{agent_name} ajanı şunu üretti:
{response[:1000]}

Tek satırda değerlendir:
VERDICT: APPROVED veya NEEDS_REVISION
ISSUES: (sadece NEEDS_REVISION ise, eksikler)
CONFIDENCE: (1-100)"""

    result = await llm_call(
        messages=[{"role": "user", "content": audit_prompt}],
        system="Kalite denetçisisin. Kısa değerlendir.",
        temperature=0.1,
        max_tokens=150,
    )

    if not result:
        return {"verdict": "APPROVED", "issues": "", "confidence": 75}

    verdict = "APPROVED"
    issues = ""
    confidence = 75

    for line in result.split("\n"):
        line = line.strip()
        if line.startswith("VERDICT:"):
            if "NEEDS" in line.upper():
                verdict = "NEEDS_REVISION"
        elif line.startswith("ISSUES:"):
            issues = line.replace("ISSUES:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = int("".join(filter(str.isdigit, line.split(":")[-1])))
            except Exception:
                pass

    return {"verdict": verdict, "issues": issues, "confidence": confidence}


async def _revise_response(
    original_request: str,
    previous_response: str,
    critic_notes: str,
    agent_name: str,
) -> str:
    revise_prompt = f"""İstek: {original_request[:150]}

Önceki yanıt:
{previous_response[:1500]}

Eksikler: {critic_notes}

Eksikleri gidererek daha iyi yanıt üret. Türkçe."""

    result = await llm_call(
        messages=[{"role": "user", "content": revise_prompt}],
        system=f"Sen {agent_name} ajanısın. Denetçi geri bildirimini dikkate al.",
        max_tokens=4000,
        temperature=0.6,
    )

    return result or previous_response
