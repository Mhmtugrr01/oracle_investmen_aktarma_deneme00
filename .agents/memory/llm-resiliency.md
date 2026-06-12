---
name: LLM Resiliency Architecture
description: Tüm LLM'ler başarısız olsa bile sistemin çalışmaya devam etmesi için mimarı kararlar
---

## Kural

QUANT ve EDGE ajanları LLM olmadan tam çalışır. LLM zinciri: Gemini → Groq → OpenAI → kural tabanlı yanıt.

**Why:** OpenAI kotası tükendi (2026-06). Gemini free tier günlük kota doldu. Groq key yoktu. Sistem çökmemeli.

**How to apply:**
- QUANT: `run_quant_agent()` sadece yfinance + CoinGecko + Fear&Greed kullanır, LLM çağrısı yok
- EDGE: `_rule_identify_action()` kural tabanlı aksiyon tespiti, LLM çağrısı yok
- Graph: QUANT node kritik katmanı (audit) atlar
- CEO fallback: LLM başarısız olursa statik yardım menüsü gösterir
- SWE/Marketing/Freelancer: LLM gerektirir; başarısız olursa kullanıcıya bilgi verir

## Sağlayıcı Önceliği

1. Cache (5dk in-memory)
2. Gemini `gemini-2.0-flash` — `google-genai` paketi (NOT `google-generativeai` — deprecated)
3. Groq `llama-3.3-70b-versatile` — GROQ_API_KEY env var
4. OpenAI `gpt-4o-mini` — 3 deneme, exponential backoff
5. "" (boş string) — çağrı yapan yer fallback üretmeli
