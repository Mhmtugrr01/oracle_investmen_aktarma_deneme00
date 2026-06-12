---
name: Conversation Context + Followup Detection
description: Kullanıcı başına bağlam saklama ve takip sorusu tespiti
---

## Kural

Kullanıcı "ne yapmalıyım?" yazdığında tam analizi yeniden çalıştırma — saklanan QUANT verisinden direkt karar ver.

**Why:** Telegram'da kullanıcı "yani sonuç olarak btc'de ne yapmalıyım!?!?" yazınca aynı rapor tekrar geldi. LLM zaman + kota israfı.

**How to apply:**
- `_user_context[user_id_str]` — son agent ve sonuç saklanır
- `_is_followup(text, user_id)` — son agent QUANT ise ve trigger kelime varsa True
- `_build_direct_decision(user_id, text)` — saklanan veriden entry/SL/target döner
- Trigger kelimeler: "ne yapmalıyım", "aksiyon ne", "long mu short mu", "entry nerede", "stop nerede", "sonuç olarak" vb.
- `_store_quant_context()` — QUANT raporu regex ile parse edilip bağlama kaydedilir

**Sınırlama:** Bağlam sadece oturum süresince (in-memory) saklanır — bot yeniden başlatınca sıfırlanır.
