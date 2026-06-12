---
name: Ücretsiz Dış API'ler
description: API key gerektirmeyen ücretsiz veri kaynakları
---
# Ücretsiz Harici API'ler

## CoinGecko Global (BTC Dominance, Market Cap)
- URL: https://api.coingecko.com/api/v3/global
- API key gerektirmez
- Yanıt: btc dominance, eth dominance, total market cap, 24h volume

## Alternative.me Fear & Greed Index
- URL: https://api.alternative.me/fng/?limit=1
- API key gerektirmez
- Yanıt: 0-100 arası değer, sınıflandırma (Extreme Fear/Greed vb.)

## yfinance (Python paketi)
- API key gerektirmez
- VIX, DXY, BTC, ETH, S&P500, altın, hisse verileri

**Why:** Özellikle zamanlayıcı (sabah brifing) LLM çağrısı yapmadan bu kaynaklardan veri çeker.
