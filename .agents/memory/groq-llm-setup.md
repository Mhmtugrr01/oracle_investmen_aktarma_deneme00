---
name: Groq LLM Fallback
description: Groq ücretsiz LLM entegrasyonu ve OpenAI fallback zinciri
---
# Groq LLM Fallback Zinciri

## Kural
LLM çağrısı sırası: Cache → Groq (GROQ_API_KEY varsa) → OpenAI (exponential backoff)

## Neden
OpenAI 429/quota hatası yaşandı. Groq tamamen ücretsizdir ve llama-3.3-70b-versatile çalışır.

## Nasıl uygulanır
- GROQ_API_KEY Replit Secrets'a eklenmezse OpenAI'a düşer
- Groq key yoksa config.llm_model = gpt-4o-mini (OpenAI)
- Groq key varsa config.llm_model = llama-3.3-70b-versatile (Groq)
- OpenAI 429 → 10s/20s/40s exponential backoff, 3 deneme
- "insufficient_quota" → kural tabanlı fallback_response döner

## Kullanıcıya bilgi
console.groq.com → ücretsiz kayıt → API key al → Replit Secrets → GROQ_API_KEY
