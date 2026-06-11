from openai import AsyncOpenAI
from core.config import settings
from loguru import logger


def get_openai_client() -> AsyncOpenAI:
    client = AsyncOpenAI(
        api_key=settings.openai_api_key or "dummy-key",
        base_url=settings.openai_base_url,
    )
    return client


async def llm_call(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    client = get_openai_client()
    final_model = model or settings.llm_model

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    try:
        response = await client.chat.completions.create(
            model=final_model,
            messages=full_messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except TypeError:
        response = await client.chat.completions.create(
            model=final_model,
            messages=full_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


async def extrapolate_prompt(user_input: str) -> str:
    system = """Sen Oracle CEO Ajan sisteminin Extrapolation (Tümevarım) motorusun.
Kullanıcıdan 1-10 kelimelik kısa bir fikir veya şikayet alıyorsun.
Bu kısa girdiyi KAPSAMLI, DETAYLI ve UYGULANABILIR bir iş direktifine dönüştür.
Şunları içermelidir:
1. Hedef ve Amaç
2. Yapılacak Adımlar (en az 5 adım)
3. Beklenen Çıktı
4. Başarı Kriterleri
5. Hangi ajana yönlendirilmeli (SWE/QUANT/MARKETING/EDGE)

Türkçe yanıt ver. Profesyonel ve net ol."""

    expanded = await llm_call(
        messages=[{"role": "user", "content": f"Kısa girdi: {user_input}"}],
        system=system,
        model=settings.llm_model_heavy,
        max_tokens=8192,
    )
    return expanded


async def detect_intent(expanded_prompt: str) -> str:
    system = """Aşağıdaki iş direktifini analiz et ve hangi ajana yönlendirileceğini belirt.
Sadece şu değerlerden birini döndür: SWE | QUANT | MARKETING | EDGE | CEO

SWE: Yazılım geliştirme, kod yazma, sistem kurma
QUANT: Borsa analizi, kripto, finansal veri
MARKETING: Scraping, email, satış, lead generation
EDGE: Sistem temizleme, lokal cihaz kontrolü
CEO: Genel yönetim, strateji, rapor"""

    intent = await llm_call(
        messages=[{"role": "user", "content": expanded_prompt}],
        system=system,
        temperature=0.1,
        max_tokens=20,
    )
    intent = intent.strip().upper()
    valid = {"SWE", "QUANT", "MARKETING", "EDGE", "CEO"}
    return intent if intent in valid else "CEO"
