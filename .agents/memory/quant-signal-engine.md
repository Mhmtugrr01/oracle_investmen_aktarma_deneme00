---
name: QUANT Signal Engine V2
description: Sinyal motoru yeniden yazıldı — RSI<25+F&G<20 artık GÜÇLÜ ALIM veriyor
---

## Kural

RSI + Fear&Greed birlikte skorlanmalı. Eski motor RSI=24+F&G=12 için "NÖTR %50" veriyordu — YANLIŞ.

**Why:** Telegram'da kullanıcı RSI=24, F&G=12 iken "NÖTR ↔️ %50" aldı. Bu tarihsel olarak en güçlü kripto alım setuplarından biridir.

**Yeni skor sistemi:**
- RSI ≤ 25: +38 puan (nadir sinyal)
- RSI ≤ 20: +45 puan (ekstrim aşırı satım)
- F&G ≤ 15: +30 puan (Buffett: "Herkes korkarken al")
- F&G ≤ 25: +22 puan
- BB alt bandı (%10): +20 puan
- Stochastic ≤ 15: +15 puan
- 4H RSI uyumu: +12 puan
- MACD negatif penaltısı: sadece -5 (eski: -10) — ekstrim RSI'da MACD gecikebilir

**Eşikler:**
- score ≥ 65: GÜÇLÜ ALIM 🚀 (güven ≤ 95%)
- score ≥ 40: ALIM 📈
- score ≤ -65: GÜÇLÜ SATIM 🔴
- NÖTR sadece -18 < score < 18 aralığında

**Test:** RSI=24, F&G=12, VIX=19, Stoch=15 → Skor=+87 → GÜÇLÜ ALIM %95 güven ✅
