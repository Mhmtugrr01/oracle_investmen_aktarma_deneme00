import asyncio
import random
import time
import re
from datetime import datetime
from core.llm import llm_call
from core.config import settings
from loguru import logger


async def run_marketing_agent(task_description: str) -> str:
    """
    PsyOp Marketing Agent: Stealth scraping + NLP email üretimi.
    Balıkesir-Bursa OSB elektrlik şebekeleri odaklı.
    """
    logger.info("[MARKETING AGENT] Starting psyop marketing cycle")

    targets = await _extract_targets(task_description)
    logger.info(f"[MARKETING AGENT] Targets: {targets[:3]}")

    scraped = await _stealth_scrape(targets)
    emails = await _generate_personalized_emails(scraped, task_description)
    report = _build_marketing_report(scraped, emails)

    return report


async def _extract_targets(task: str) -> list[str]:
    """Görevden hedef URL / sektör listesi çıkarır."""
    response = await llm_call(
        messages=[{"role": "user", "content": task}],
        system="""Verilen metinden hedef şirket/sektör/konum bilgilerini çıkar.
Balıkesir, Bursa, OSB, elektrik, sanayi odaklı bak.
Sadece kısa anahtar kelime listesi döndür (virgülle ayrılmış).""",
        temperature=0.2,
        max_tokens=200,
    )
    return [t.strip() for t in response.split(",") if t.strip()]


async def _stealth_scrape(targets: list[str]) -> list[dict]:
    """
    Gaussian Distribution tabanlı rastgele gecikmeli stealth scraping.
    NOT: Gerçek scraping için playwright kullanılır; burada simüle ediyoruz.
    """
    results = []

    for target in targets[:5]:
        gaussian_delay = max(0.5, random.gauss(mu=2.5, sigma=0.8))
        logger.info(f"[MARKETING] Scraping: {target} (delay: {gaussian_delay:.1f}s)")
        await asyncio.sleep(gaussian_delay)

        result = await llm_call(
            messages=[{"role": "user", "content": f"'{target}' sektörü/bölgesi hakkında tipik B2B iletişim bilgisi ve şirket profili üret."}],
            system="""Verilen sektör için gerçekçi B2B şirket profili üret:
- Şirket adı
- Sektör
- Konum (Balıkesir/Bursa OSB tercihen)
- İletişim (örnek email formatında: info@sektör.com.tr)
- Karar verici pozisyon
JSON formatında döndür: {"company": ..., "sector": ..., "location": ..., "contact": ..., "decision_maker": ...}""",
            temperature=0.9,
            max_tokens=300,
        )

        try:
            import json
            data = json.loads(result.strip())
            results.append(data)
        except Exception:
            results.append({
                "company": target,
                "sector": "Sanayi",
                "location": "Bursa OSB",
                "contact": f"info@{target.lower().replace(' ', '')[:10]}.com.tr",
                "decision_maker": "Satın Alma Müdürü",
                "raw": result,
            })

    return results


async def _generate_personalized_emails(scraped: list[dict], context: str) -> list[dict]:
    """NLP tabanlı kişiselleştirilmiş email üretimi."""
    emails = []

    for lead in scraped:
        email_body = await llm_call(
            messages=[{"role": "user", "content": f"""
Şirket Profili: {lead}
Bağlam: {context}

Bu şirket için kişiselleştirilmiş B2B satış emaili yaz.
"""}],
            system="""Sen B2B Satış Uzmanısın. 
Hedef: Elektrik/Sanayi şirketlerine yönelik kişiselleştirilmiş email.
Format:
Konu: [kısa çarpıcı konu satırı]
---
[Kişiselleştirilmiş email gövdesi, 150-200 kelime, Türkçe, profesyonel ama samimi]
Çok agresif satış dili kullanma. Değer önerisi öne çıkar.""",
            temperature=0.8,
            max_tokens=600,
        )

        emails.append({
            "to": lead.get("contact", ""),
            "company": lead.get("company", ""),
            "decision_maker": lead.get("decision_maker", ""),
            "body": email_body,
        })

        await asyncio.sleep(random.gauss(1.0, 0.3))

    return emails


def _build_marketing_report(scraped: list[dict], emails: list[dict]) -> str:
    lines = [
        "📣 *ORACLE MARKETING RAPORU*",
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"🎯 Hedef Şirket: {len(scraped)} | Email: {len(emails)}",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, email in enumerate(emails, 1):
        lines.append(f"\n📧 *Lead {i}: {email['company']}*")
        lines.append(f"📬 Adres: {email['to']}")
        lines.append(f"👤 Muhatap: {email['decision_maker']}")
        preview = email["body"][:200].replace("\n", " ")
        lines.append(f"✉️ Önizleme: {preview}...")

    lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"✅ {len(emails)} email hazır. Göndermek için onay gerekli.")

    return "\n".join(lines)
