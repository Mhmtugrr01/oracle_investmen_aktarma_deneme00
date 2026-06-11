"""
ORACLE CRITIC ENGINE — Çok Katmanlı AI Denetim Sistemi
CEO, alt ajan çıktısını denetler, eksikleri tamamlar, nihai sonucu üretir.
"""
from core.llm import llm_call
from core.config import settings
from loguru import logger


async def audit_and_refine(
    user_input: str,
    expanded_prompt: str,
    agent_name: str,
    raw_response: str,
    max_rounds: int = 2,
) -> str:
    """
    CEO Denetim Döngüsü:
    1. Alt ajan çıktısını denetler
    2. Eksik/yanıltıcı/hatalı yön varsa düzeltmek için alt ajana geri gönderir
    3. max_rounds sonra nihai çıktıyı tamamlayıp döndürür
    """
    logger.info(f"[CRITIC] Auditing {agent_name} output ({len(raw_response)} chars)")

    current_response = raw_response

    for round_num in range(1, max_rounds + 1):
        audit_result = await _audit_response(
            user_input=user_input,
            expanded_prompt=expanded_prompt,
            agent_name=agent_name,
            response=current_response,
        )

        if audit_result["verdict"] == "APPROVED":
            logger.success(f"[CRITIC] Round {round_num}: APPROVED")
            break

        logger.warning(f"[CRITIC] Round {round_num}: NEEDS_REVISION — {audit_result['issues'][:100]}")

        if round_num < max_rounds:
            current_response = await _revise_response(
                original_prompt=expanded_prompt,
                previous_response=current_response,
                critic_notes=audit_result["issues"],
                agent_name=agent_name,
            )
        else:
            current_response = await _finalize_response(
                user_input=user_input,
                previous_response=current_response,
                critic_notes=audit_result["issues"],
            )

    return current_response


async def _audit_response(
    user_input: str,
    expanded_prompt: str,
    agent_name: str,
    response: str,
) -> dict:
    """CEO denetçisi olarak çıktıyı değerlendirir."""

    audit_prompt = f"""
Sen Oracle CEO Denetçisisin. Bir alt ajan ({agent_name}) şu çıktıyı üretti:

KULLANICI İSTEĞİ: {user_input[:200]}

ALT AJAN ÇIKTISI:
{response[:2000]}

Bu çıktıyı şu kriterlere göre denetle:
1. Eksik bilgi var mı?
2. Yanıltıcı veya hatalı ifade var mı?
3. Kullanıcının asıl isteğini tam karşılıyor mu?
4. Somut/uygulanabilir mi yoksa belirsiz mi?

Yanıtın SADECE şu formatta olsun:
VERDICT: APPROVED veya NEEDS_REVISION
ISSUES: [Eğer NEEDS_REVISION ise eksiklikleri listele. APPROVED ise boş bırak.]
CONFIDENCE: [1-100 arası puan]
"""

    result = await llm_call(
        messages=[{"role": "user", "content": audit_prompt}],
        system="Sen titiz bir kalite denetçisisin. Kısa ve net değerlendir.",
        temperature=0.1,
        max_tokens=500,
    )

    verdict = "APPROVED"
    issues = ""
    confidence = 80

    for line in result.split("\n"):
        if line.startswith("VERDICT:"):
            v = line.replace("VERDICT:", "").strip()
            if "NEEDS" in v.upper():
                verdict = "NEEDS_REVISION"
        elif line.startswith("ISSUES:"):
            issues = line.replace("ISSUES:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = int(line.replace("CONFIDENCE:", "").strip())
            except Exception:
                confidence = 75

    return {"verdict": verdict, "issues": issues, "confidence": confidence}


async def _revise_response(
    original_prompt: str,
    previous_response: str,
    critic_notes: str,
    agent_name: str,
) -> str:
    """Alt ajanı revizyon için tekrar çalıştırır."""

    revise_prompt = f"""
Aşağıdaki yanıtı denetçi geri bildirimine göre geliştirilmiş şekilde yeniden yaz:

ORIJINAL GÖREV: {original_prompt[:500]}

ÖNCEKİ YANIT:
{previous_response[:1500]}

DENETÇİ NOTLARI (bunları mutlaka düzelt):
{critic_notes}

Daha eksiksiz, daha doğru ve daha uygulanabilir bir yanıt üret. Türkçe.
"""

    return await llm_call(
        messages=[{"role": "user", "content": revise_prompt}],
        system=f"Sen {agent_name} ajanısın. Denetçi geri bildirimini dikkate alarak daha iyi bir yanıt üret.",
        model=settings.llm_model_heavy,
        max_tokens=6000,
    )


async def _finalize_response(
    user_input: str,
    previous_response: str,
    critic_notes: str,
) -> str:
    """CEO son turu kendisi tamamlar."""

    final_prompt = f"""
Kullanıcı istedi: {user_input[:200]}

Mevcut yanıt:
{previous_response[:2000]}

Eksik kalan noktalar:
{critic_notes}

Bu eksiklikleri kendin tamamla ve nihai, eksiksiz, kullanıcıya hazır yanıtı üret.
Yanıtın başına "✅ CEO Onaylı — " ekle.
"""

    return await llm_call(
        messages=[{"role": "user", "content": final_prompt}],
        system="Sen Oracle CEO'sun. Son denetimi geçemeyen yanıtı kendin tamamlayıp mükemmelleştir. Türkçe, net, somut.",
        model=settings.llm_model_heavy,
        max_tokens=6000,
    )
