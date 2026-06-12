---
name: Gemini API Integration Quirks
description: google-genai paketi kurulum ve kullanım sorunları
---

## Kural

`google-genai` paketini kullan, `google-generativeai` DEPRECATED ve çalışmıyor.

**Why:** google-generativeai kaldırıldı, artık google.genai kullanılması gerekiyor.

**How to apply:**
```python
from google import genai
from google.genai import types
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    config=types.GenerateContentConfig(max_output_tokens=8192, temperature=0.7),
)
```

- Free tier günlük kota dolabilir → limit:0 hatası 429 olarak gelir → zincirdeki sonraki LLM'e geç
- gemini-1.5-flash → 404 (yeni SDK'da bu model adı değişti)
- GEMINI_API_KEY shared env var olarak kayıtlı (Replit Secrets'ta değil)
