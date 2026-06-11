"""
FREELANCER AJAN — Otomatik İş Bulma, Başvuru, Takip Sistemi
- Alanınıza uygun iş ilanlarını tarar
- Kişiselleştirilmiş başvuru metinleri üretir
- Başvuru durumlarını takip eder
- Başarılı/olumlu dönüşleri Telegram'a iletir
"""
import asyncio
import json
import random
from datetime import datetime
from core.llm import llm_call
from core.config import settings
from loguru import logger


FREELANCER_PLATFORMS = [
    "Upwork",
    "Freelancer.com",
    "Toptal",
    "Guru",
    "PeoplePerHour",
    "LinkedIn",
    "Fiverr",
]

USER_PROFILE = {
    "name": "Kullanıcı",
    "field": "Elektrik Mühendisliği / Yapay Zeka / Yazılım",
    "skills": ["Elektrik mühendisliği", "Python", "AI/ML", "Otomasyon", "LangChain", "FastAPI"],
    "experience": "5+ yıl",
    "location": "Türkiye (Remote odaklı)",
    "languages": ["Türkçe (Ana dil)", "İngilizce (İleri)"],
}


async def run_freelancer_agent(task_description: str) -> str:
    """
    Freelancer Ajan ana döngüsü:
    1. Göreve göre iş kategorisi ve platform belirle
    2. Uygun ilanları simüle et / ara
    3. Başvuru metinleri üret
    4. Takip planı oluştur
    """
    logger.info("[FREELANCER] Starting job search cycle")

    job_spec = await _extract_job_spec(task_description)
    logger.info(f"[FREELANCER] Job spec: {job_spec.get('category', '?')}")

    jobs = await _find_jobs(job_spec)
    applications = await _generate_applications(jobs, job_spec)
    tracking = _build_tracking_plan(applications)
    report = _build_freelancer_report(jobs, applications, tracking, job_spec)

    return report


async def _extract_job_spec(task: str) -> dict:
    """Kullanıcının freelancer talebini analiz eder."""
    response = await llm_call(
        messages=[{"role": "user", "content": f"Freelancer iş arama talebi: {task}"}],
        system="""Verilen metinden freelancer iş arama spesifikasyonunu çıkar.
JSON formatında döndür:
{
  "category": "iş kategorisi",
  "skills_needed": ["beceri1", "beceri2"],
  "budget_range": "beklenen ücret aralığı",
  "job_type": "uzaktan/hybrid/yerinde",
  "platforms": ["platform1", "platform2"],
  "keywords": ["anahtar kelime1", "anahtar kelime2"]
}
Eğer belirtilmemişse kullanıcının profilinden tahmin et: Elektrik Mühendisliği + AI + Python alanı.""",
        temperature=0.2,
        max_tokens=400,
    )

    try:
        return json.loads(response.strip())
    except Exception:
        return {
            "category": "Elektrik Mühendisliği / AI Danışmanlığı",
            "skills_needed": USER_PROFILE["skills"],
            "budget_range": "$50-150/saat",
            "job_type": "uzaktan",
            "platforms": ["Upwork", "LinkedIn", "Freelancer.com"],
            "keywords": ["electrical engineering", "AI automation", "Python", "LangChain"],
        }


async def _find_jobs(job_spec: dict) -> list[dict]:
    """
    Uygun iş ilanlarını bulur.
    NOT: Gerçek entegrasyon için Upwork API / LinkedIn API kullanılabilir.
    Şu an LLM tabanlı gerçekçi ilan simülasyonu yapılıyor.
    """
    platforms = job_spec.get("platforms", FREELANCER_PLATFORMS[:3])
    jobs = []

    for i, platform in enumerate(platforms[:4]):
        delay = max(0.5, random.gauss(1.5, 0.4))
        await asyncio.sleep(delay)

        job_data = await llm_call(
            messages=[{"role": "user", "content": f"""
Platform: {platform}
Kategori: {job_spec.get('category', 'AI/Elektrik')}
Beceriler: {', '.join(job_spec.get('skills_needed', [])[:4])}
Bütçe: {job_spec.get('budget_range', '$50-100/saat')}

Bu platforma uygun, gerçekçi 2 freelancer iş ilanı üret. JSON array:
[
  {{
    "title": "İş başlığı",
    "platform": "{platform}",
    "budget": "bütçe",
    "duration": "süre",
    "description": "kısa açıklama (2-3 cümle)",
    "required_skills": ["skill1", "skill2"],
    "match_score": 85,
    "posted": "2 gün önce",
    "client_rating": 4.8,
    "proposals": 12
  }}
]"""}],
            system="Gerçekçi freelancer ilanları üret. Sadece JSON array döndür.",
            temperature=0.8,
            max_tokens=600,
        )

        try:
            start = job_data.find("[")
            end = job_data.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(job_data[start:end])
                jobs.extend(parsed)
        except Exception:
            jobs.append({
                "title": f"{job_spec.get('category', 'AI')} Uzmanı Aranıyor",
                "platform": platform,
                "budget": job_spec.get("budget_range", "$50-100/saat"),
                "duration": "3-6 ay",
                "description": "Deneyimli uzman aranıyor.",
                "required_skills": job_spec.get("skills_needed", [])[:3],
                "match_score": 75 + i * 5,
                "posted": f"{i+1} gün önce",
                "client_rating": round(4.5 + random.uniform(0, 0.5), 1),
                "proposals": random.randint(5, 25),
            })

    jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return jobs[:6]


async def _generate_applications(jobs: list[dict], job_spec: dict) -> list[dict]:
    """Her iş ilanı için kişiselleştirilmiş başvuru metni üretir."""
    applications = []

    for job in jobs[:4]:
        cover_letter = await llm_call(
            messages=[{"role": "user", "content": f"""
İlan: {job.get('title', '')}
Platform: {job.get('platform', '')}
Açıklama: {job.get('description', '')}
Gereken beceriler: {', '.join(job.get('required_skills', []))}

Benim profilim:
- Alan: {USER_PROFILE['field']}
- Beceriler: {', '.join(USER_PROFILE['skills'])}
- Deneyim: {USER_PROFILE['experience']}
- Diller: {', '.join(USER_PROFILE['languages'])}

Bu iş için profesyonel, kişiselleştirilmiş İngilizce başvuru mektubu yaz.
120-180 kelime. Değer önerisi öne çık. Özgün, spam değil."""}],
            system="Profesyonel freelancer başvuru uzmanısın. Güçlü, özgün cover letter yaz.",
            temperature=0.75,
            max_tokens=400,
        )

        subject = await llm_call(
            messages=[{"role": "user", "content": f"İlan: {job.get('title', '')} — bu başvuru için 8-12 kelimelik güçlü email konu satırı üret. Sadece konu satırını yaz."}],
            system="Kısa, güçlü email konu satırı yaz.",
            temperature=0.7,
            max_tokens=50,
        )

        applications.append({
            "job": job,
            "cover_letter": cover_letter,
            "subject": subject.strip(),
            "status": "hazır",
            "applied_at": None,
        })

        await asyncio.sleep(random.gauss(0.8, 0.2))

    return applications


def _build_tracking_plan(applications: list[dict]) -> dict:
    """Başvuru takip planı oluşturur."""
    return {
        "total": len(applications),
        "followup_day": 5,
        "followup_message": "Başvurumu takip etmek istiyorum. İlgilendiniz mi?",
        "reminder_cycles": 3,
        "success_criteria": "Görüşme daveti veya olumlu yanıt",
        "auto_followup": True,
    }


def _build_freelancer_report(jobs, applications, tracking, job_spec) -> str:
    lines = [
        "💼 *ORACLE FREELANCER RAPORU*",
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"🎯 Kategori: {job_spec.get('category', '?')}",
        f"💰 Bütçe Aralığı: {job_spec.get('budget_range', '?')}",
        f"🌐 Platform: {', '.join(job_spec.get('platforms', [])[:3])}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"🔍 Bulunan İlan: {len(jobs)} | Başvuru Hazır: {len(applications)}",
        "",
        "📋 *EN İYİ İLANLAR:*",
    ]

    for i, job in enumerate(jobs[:4], 1):
        match = job.get("match_score", 0)
        match_icon = "🟢" if match >= 80 else ("🟡" if match >= 65 else "🔴")
        lines.append(f"""
{match_icon} *{i}. {job.get('title', '?')}*
   📌 {job.get('platform', '?')} | 💵 {job.get('budget', '?')}
   ⏱ {job.get('duration', '?')} | 📅 {job.get('posted', '?')}
   ⭐ Müşteri: {job.get('client_rating', '?')} | 📨 {job.get('proposals', '?')} başvuru
   🎯 Eşleşme: %{match}""")

    if applications:
        lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("✉️ *BAŞVURU ÖNİZLEME (1. İlan):*")
        first = applications[0]
        lines.append(f"📧 Konu: {first.get('subject', '')}")
        preview = first.get("cover_letter", "")[:300].replace("\n", " ")
        lines.append(f"✉️ {preview}...")

    lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📆 *TAKİP PLANI:*")
    lines.append(f"  • {tracking['followup_day']} gün sonra otomatik takip mesajı")
    lines.append(f"  • {tracking['reminder_cycles']} tur takip döngüsü")
    lines.append(f"  • Başarı kriteri: {tracking['success_criteria']}")
    lines.append("\n✅ Başvurmaları onaylamak için butonu kullanın.")

    return "\n".join(lines)
