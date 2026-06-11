# Oracle Master-Swarm Ecosystem V4.0

Telegram üzerinden yönetilen, 5 uzman yapay zeka ajanından oluşan otonom bilişsel işletim sistemi. Kısa bir mesaj yazın — sistem genişletir, doğru ajana yönlendirir ve sonucu Telegram'a gönderir.

## Run & Operate

- `Oracle Swarm Bot` workflow'u — Ana bot süreci (Python, Telegram polling)
- `pnpm --filter @workspace/api-server run dev` — API server (port 5000)
- `pnpm run typecheck` — TypeScript kontrolü
- Bot başlatma: `cd oracle-swarm && python3 main.py`

## Stack

- **Dil:** Python 3.11
- **API:** FastAPI + Uvicorn
- **Ajan Orkestrasyon:** LangGraph (StateGraph)
- **Kullanıcı Arayüzü:** Telegram Bot (python-telegram-bot v22)
- **LLM Beyin:** OpenAI API (gpt-4o-mini / gpt-4o)
- **Veritabanı / Bellek:** Supabase (PostgreSQL + pgvector)
- **Async Kuyruk:** Celery + Redis (isteğe bağlı)
- **Node.js altyapısı:** pnpm workspaces, Express 5, Drizzle ORM

## Where things live

- `oracle-swarm/main.py` — Giriş noktası, sistem başlatma
- `oracle-swarm/core/graph.py` — LangGraph state machine (CEO Router)
- `oracle-swarm/core/llm.py` — OpenAI LLM çağrıları + Extrapolation motoru
- `oracle-swarm/core/memory.py` — Supabase kalıcı bellek
- `oracle-swarm/core/config.py` — Ortam değişkenleri (pydantic-settings)
- `oracle-swarm/agents/` — 5 ajan modülü
- `oracle-swarm/bot_handler/` — Telegram bot handler + klavyeler
- `oracle-swarm/db/schema.sql` — Supabase tablo şeması (manuel çalıştır)
- `oracle-swarm/tasks/` — Celery async görev kuyruğu

## Architecture decisions

- `telegram/` klasörü `bot_handler/` olarak adlandırıldı — python-telegram-bot kütüphanesiyle isim çakışmasını önlemek için.
- LangGraph StateGraph kullanıldı — her ajan bir node, CEO Router conditional edge ile yönlendiriyor.
- Extrapolation Engine: Kısa (10 kelime) kullanıcı girdisi gpt-4o ile 10 sayfalık iş direktifine dönüştürülüyor.
- HFT Quant ajanı ASLA otomatik alım/satım yapmaz — sadece analiz, Telegram inline onay butonu üretiyor.
- Edge Daemon cloud-side çalışıyor; approved_commands whitelist'i dışında hiçbir işlem yapmaz.

## Product

- **CEO Router:** Kısa mesajı genişletip doğru ajana yönlendirir
- **SWE Agent:** Kod üretir, sandbox'ta test eder, hataları otomatik düzeltir (Zero-Defect Loop)
- **QUANT Agent:** BTC/ETH/hisse teknik analizi (RSI, EMA), Telegram inline onay butonu
- **Marketing Agent:** Stealth scraping + NLP email üretimi (Balıkesir/Bursa OSB odaklı)
- **Edge Agent:** Sistem durumu ve disk/bellek raporları (cloud-side, güvenli)

## User preferences

- Sistem tamamen cloud'da çalışmalı, bilgisayarda kurulum gerektirmemeli
- Kullanıcı arayüzü yalnızca Telegram olmalı
- HFT ajanı asla otomatik işlem açmamalı
- Tüm aksiyonlar Telegram inline butonuyla onaya sunulmalı

## Gotchas

- Supabase tabloları ilk kurulumda `oracle-swarm/db/schema.sql` ile manuel oluşturulmalı
- `telegram/` klasörü python-telegram-bot ile çakışır — klasör adı `bot_handler/` olmalı
- OpenAI API key `OPENAI_API_KEY` secret'ı olarak tanımlı olmalı
- LangGraph 1.x API'sinde `StateGraph` ve `END` import yolları değişti: `from langgraph.graph import StateGraph, END`

## Secrets gerekli

- `TELEGRAM_BOT_TOKEN` — BotFather token (sadece sayı:harf formatı)
- `SUPABASE_URL` — https://xxx.supabase.co
- `SUPABASE_SERVICE_KEY` — Supabase service role key
- `OPENAI_API_KEY` — OpenAI API key

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
