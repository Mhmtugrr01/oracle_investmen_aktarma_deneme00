---
name: Rule-Based Intent Routing
description: LLM olmadan ajan seçimi; kural tabanlı skorlama sistemi
---

## Kural

`rule_based_intent(text)` fonksiyonu LLM olmadan 11/11 test geçer.
Yönlendirme: puan ≥ 2 veya tek net kazanan → kesin yönlendirme; belirsizse LLM'e bırak.

**Why:** LLM'ler kota aşımı döneminde bot tamamen yanıtsız kalıyordu.

**How to apply:**
- "email" ve "mail" MARKETING keywords listesinde — "email gönder" → MARKETING ✅
- Tek sembol (BTC, ETH) → extrapolation_node'da direkt QUANT
- Kelime listesi: QUANT_KW, SWE_KW, MARKETING_KW, EDGE_KW, FREELANCER_KW — core/llm.py
- Yeni keyword eklemek için: ilgili set'e string ekle, test: `rule_based_intent("yeni kelime")`
