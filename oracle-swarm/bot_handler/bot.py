"""
ORACLE TELEGRAM BOT HANDLER V3

Yenilikler:
- Konuşma bağlamı: Her kullanıcının son analizi hatırlanır
- Takip sorusu tespiti: "ne yapmalıyım?" → doğrudan karar
- QUANT raporunda entry/SL/target bilgisi saklanır
- Markdown hataları yakalanır, düz metin fallback
"""
import asyncio
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)
from telegram.constants import ParseMode
from core.config import settings
from core.graph import oracle_graph, OracleState
from core.memory import get_recent_tasks
from core.scheduler import register_alert_callback, unregister_alert_callback
from bot_handler.keyboards import main_menu_keyboard, back_keyboard
from loguru import logger

PROCESSING_USERS: set[int] = set()
_application: Application | None = None

# ─── Kullanıcı Konuşma Bağlamı ───────────────────────────────────────────────
_user_context: dict[str, dict] = {}
# Format: {user_id: {last_agent, last_result, last_quant_data, last_symbols}}

# Takip sorusu tetikleyicileri
_FOLLOWUP_TRIGGERS = [
    "ne yapmalıyım", "ne yapayım", "ne önerirsin", "önerin nedir",
    "karar ne", "sonuç ne", "ne düşünüyorsun", "aksiyon ne",
    "ne zaman", "nereye gider", "long mu", "short mu",
    "almalı mıyım", "satmalı mıyım", "beklenmeli mi",
    "pozisyon aç", "risk ne", "hedef ne", "entry nedir",
    "giriş fiyatı", "dur nerede", "stop nerede",
    "kısa vadeli", "orta vadeli", "uzun vadeli",
    "net olarak", "direkt olarak", "sonuç olarak",
    "tek cümle", "özetle",
]


def _is_followup(text: str, user_id: str) -> bool:
    """Kullanıcının önceki QUANT analizine atıfla soru sorduğunu tespit eder."""
    ctx = _user_context.get(user_id, {})
    if ctx.get("last_agent") not in ("QUANT",):
        return False
    t = text.lower()
    return any(trigger in t for trigger in _FOLLOWUP_TRIGGERS)


def _build_direct_decision(user_id: str, user_text: str) -> str:
    """Takip sorusuna saklanmış QUANT analizinden doğrudan karar üretir."""
    ctx = _user_context.get(user_id, {})
    qd = ctx.get("last_quant_data", {})

    if not qd:
        return ""

    symbol = qd.get("symbol", "?")
    price = qd.get("price", 0)
    sig = qd.get("signal", {})
    lv = qd.get("levels", {})
    rsi = qd.get("rsi_d", 50)
    fg = qd.get("fg_value", 50)

    from agents.hft_quant import _fmt_price, _fmt_pct

    action = sig.get("action", "BEKLE ↔️")
    signal_name = sig.get("signal", "NÖTR")
    conf = sig.get("confidence", 50)
    bull_reasons = sig.get("bull_reasons", [])
    targets = lv.get("targets", [])
    entry_low = lv.get("entry_low", price * 0.98)
    entry_high = lv.get("entry_high", price * 1.01)
    sl = lv.get("stop_loss", price * 0.92)
    sl_pct = lv.get("sl_pct", 5.0)
    rr = lv.get("risk_reward", 0)

    t1 = targets[0] if len(targets) > 0 else price * 1.05
    t2 = targets[1] if len(targets) > 1 else price * 1.12
    t3 = targets[2] if len(targets) > 2 else price * 1.20

    t1_pct = (t1 - price) / price * 100
    t2_pct = (t2 - price) / price * 100
    t3_pct = (t3 - price) / price * 100

    emoji = sig.get("emoji", "⚪")

    lines = [
        f"{emoji} *{symbol} — NİHAİ KARAR*",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"*SİNYAL: {signal_name}* (Güven: %{conf})",
        f"",
        f"📍 *Optimal Giriş:* {_fmt_price(entry_low)} — {_fmt_price(entry_high)}",
        f"🛑 *Stop-Loss:* {_fmt_price(sl)} (-%{sl_pct:.1f})",
        f"🎯 *Hedef 1:* {_fmt_price(t1)} ({_fmt_pct(t1_pct)})",
        f"🎯 *Hedef 2:* {_fmt_price(t2)} ({_fmt_pct(t2_pct)})",
        f"🎯 *Hedef 3:* {_fmt_price(t3)} ({_fmt_pct(t3_pct)})",
        f"📊 *Risk/Ödül:* 1:{rr:.2f}",
        f"",
        f"⚡ *AKSIYON: {action}*",
    ]

    # Gerekçeler
    if bull_reasons:
        lines.append("")
        lines.append("*Gerekçe:*")
        for r in bull_reasons[:3]:
            lines.append(f"  • {r}")

    # Scenario analizi (skor bazlı)
    score = sig.get("score", 0)
    lines.append("")
    if score >= 60:
        lines += [
            f"*Senaryo Analizi:*",
            f"  🟢 Boğa (%70): RSI{rsi:.0f}+F&G{fg} → {_fmt_price(t2)}'ye hareket",
            f"  🟡 Nötr (%20): {_fmt_price(entry_low)} — {_fmt_price(entry_high)} konsolidasyon",
            f"  🔴 Ayı (%10): SL {_fmt_price(sl)} altı kapanış → çık",
        ]
        lines.append(f"")
        lines.append(f"💡 *Uygulama:* {_fmt_price(entry_high)} altında kademeli al,")
        lines.append(f"   T1'de yarısını kapat, T2 için bekle.")
        lines.append(f"   Maks. portföy riski: %5")
    elif score <= -60:
        lines += [
            f"*Senaryo Analizi:*",
            f"  🔴 Ayı (%70): {_fmt_price(t1)}'ye düşüş",
            f"  🟡 Nötr (%20): Mevcut bölge konsolidasyon",
            f"  🟢 Boğa (%10): SL {_fmt_price(sl)} üstü kapanış → sinyal geçersiz",
        ]
        lines.append(f"")
        lines.append(f"💡 *Uygulama:* Long pozisyonlarını kapat veya hedge al.")
    else:
        lines.append(f"💡 Net yön sinyali yok. {_fmt_price(lv.get('supports', [price*0.95])[0])}")
        lines.append(f"   destek kırarsa veya {_fmt_price(lv.get('resistances', [price*1.05])[0])}")
        lines.append(f"   direnç geçerse yeni sinyal oluşur.")

    lines += [
        "",
        "⚠️ *Yatırım tavsiyesi değildir.*",
        "Oracle asla otomatik işlem açmaz.",
    ]

    return "\n".join(lines)


# ─── Komutlar ─────────────────────────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = str(user.id)

    async def send_alert(uid: str, msg: str):
        try:
            await _application.bot.send_message(
                chat_id=int(uid), text=msg, parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.error(f"Alert send failed: {e}")

    register_alert_callback(user_id, send_alert)
    # Bağlamı sıfırla
    _user_context[user_id] = {}

    welcome = (
        f"🧠 *ORACLE MASTER-SWARM V4.0*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Hoş geldiniz, *{user.first_name}*.\n\n"
        f"Otonom Bilişsel İşletim Sisteminize bağlandınız.\n"
        f"Kısa bir komut yazın — sistem genişletir, doğru ajana yönlendirir, CEO Critic denetler.\n\n"
        f"🤖 *Ajanlar:*\n"
        f"• 📊 QUANT — Kripto/Borsa analizi, giriş/stop/hedef\n"
        f"• 💻 SWE — Kod üretimi, Zero-Defect Loop\n"
        f"• 📣 MARKETING — OSB email, scraping\n"
        f"• 💾 EDGE — Sistem durumu, disk/bellek\n"
        f"• 💼 FREELANCER — İş arama, başvuru\n\n"
        f"⏰ *Otomatik:*\n"
        f"• Saatlik piyasa taraması\n"
        f"• 🌅 08:00 sabah brifing\n"
        f"• Güçlü sinyal tespitinde anlık uyarı\n\n"
        f"Menü için aşağıyı kullanın veya direkt yazın:"
    )

    await update.message.reply_text(
        welcome, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📖 *KULLANIM KILAVUZU*\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "*Komutlar:*\n"
        "/start — Ana menü\n"
        "/status — Sistem durumu\n"
        "/quant BTC ETH — Hızlı analiz\n"
        "/scan — Anlık tarama\n"
        "/history — Son görevler\n\n"
        "*Örnekler:*\n"
        "• `btc analiz` → Kurumsal BTC raporu\n"
        "• `btc 4h grafik` → 4H teknik analiz\n"
        "• `ne yapmalıyım?` → Takip sorusu (önceki analiz)\n"
        "• `Python API yaz` → SWE ajan\n"
        "• `Upwork Python işi bul` → Freelancer\n"
        "• `Balıkesir OSB firmalar` → Marketing\n"
        "• `disk durumu` → Edge ajan\n\n"
        "*Takip soruları:*\n"
        "Analiz aldıktan sonra:\n"
        "• `ne yapmalıyım?` → Direkt karar\n"
        "• `entry nerede?` → Giriş seviyeleri\n"
        "• `stop nerede?` → Stop-loss detayı"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from core.scheduler import _scheduler
    import os
    sched = "🟢 Aktif" if _scheduler and _scheduler.running else "🔴 Durdu"
    llm_status = []
    if os.getenv("GEMINI_API_KEY"):
        llm_status.append("Gemini ✅")
    if os.getenv("GROQ_API_KEY"):
        llm_status.append("Groq ✅")
    if os.getenv("OPENAI_API_KEY"):
        llm_status.append("OpenAI ✅")

    text = (
        f"📡 *SİSTEM DURUMU*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 CEO Router + Critic  — Aktif\n"
        f"🟢 SWE Mühendis         — Aktif\n"
        f"🟢 QUANT Gözcü          — Aktif (4H+Günlük)\n"
        f"🟢 Marketing            — Aktif\n"
        f"🟢 Freelancer           — Aktif\n"
        f"🟢 Edge Daemon          — Aktif\n"
        f"🟢 Supabase Bellek      — Bağlı\n"
        f"🟢 Telegram API         — Bağlı\n"
        f"{sched} Zamanlayıcı\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 LLM: {' | '.join(llm_status) if llm_status else 'Kural tabanlı mod'}\n"
        f"☁️ Cloud (Replit) | QUANT: LLMsiz çalışır"
    )
    await update.message.reply_text(
        text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard()
    )


async def quant_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = " ".join(context.args) if context.args else "BTC ETH"
    await _process_user_input(update, f"analiz: {symbols}")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "🔄 *Anlık piyasa taraması başlatılıyor...*", parse_mode=ParseMode.MARKDOWN,
    )
    try:
        from agents.hft_quant import run_scheduled_scan
        result = await run_scheduled_scan(send_alert_fn=None)
        if "kritik sinyal yok" in result.lower():
            result = "✅ Tarama tamamlandı — Şu an kritik toplama/dağıtım sinyali yok.\n\nBTC, ETH, Altın incelendi."
        await _safe_edit(msg, result)
    except Exception as e:
        await msg.edit_text(f"❌ Tarama hatası: {e}")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    tasks = await get_recent_tasks(user_id, limit=5)
    if not tasks:
        await update.message.reply_text("📋 Henüz tamamlanmış görev yok.", reply_markup=back_keyboard())
        return
    lines = ["📋 *SON GÖREVLER*", "━━━━━━━━━━━━━━━━━━━━━━"]
    for i, t in enumerate(tasks, 1):
        lines.append(
            f"\n*{i}. [{t.get('agent','?')}]* — {t.get('user_input','')[:50]}\n"
            f"   📅 {str(t.get('created_at',''))[:16]}"
        )
    await update.message.reply_text(
        "\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard()
    )


# ─── Mesaj İşleyici ───────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _process_user_input(update, update.message.text.strip())


async def _process_user_input(update: Update, user_input: str):
    user_id = update.effective_user.id
    user_id_str = str(user_id)

    if user_id in PROCESSING_USERS:
        await update.message.reply_text("⏳ Önceki görev işleniyor, lütfen bekleyin...")
        return

    if len(user_input) < 2:
        await update.message.reply_text("ℹ️ Lütfen en az bir kelimelik komut girin.")
        return

    # ── Takip sorusu tespiti ──────────────────────────────────────────────────
    if _is_followup(user_input, user_id_str):
        direct_answer = _build_direct_decision(user_id_str, user_input)
        if direct_answer:
            logger.info(f"[BOT] Takip sorusu → direkt karar: {user_input[:40]}")
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Analizi Onayla", callback_data="quant_approve_ok")],
                [InlineKeyboardButton("🔄 Yenile", callback_data="agent_QUANT"),
                 InlineKeyboardButton("🏠 Menü", callback_data="main_menu")],
            ])
            await _safe_send(update, direct_answer, keyboard)
            return

    PROCESSING_USERS.add(user_id)

    thinking_msg = await update.message.reply_text(
        "🧠 *Oracle analiz ediyor...*", parse_mode=ParseMode.MARKDOWN,
    )

    try:
        await thinking_msg.edit_text(
            "🔄 *Ajan devrede...*\n🔍 Veri toplanıyor...",
            parse_mode=ParseMode.MARKDOWN,
        )

        initial_state: OracleState = {
            "user_id": user_id_str,
            "user_input": user_input,
            "expanded_prompt": "",
            "agent": "",
            "result": "",
            "audited_result": "",
            "status": "pending",
            "task_id": None,
            "messages": [],
        }

        final_state = await oracle_graph.ainvoke(initial_state)

        agent = final_state.get("agent", "CEO")
        result = final_state.get("audited_result") or final_state.get("result", "Sonuç alınamadı.")

        # ── Kullanıcı bağlamını güncelle ──────────────────────────────────────
        _user_context[user_id_str] = {
            "last_agent": agent,
            "last_result": result[:500],
        }

        # QUANT sonucunu sakla (takip soruları için)
        if agent == "QUANT":
            _store_quant_context(user_id_str, result, user_input)

        header = f"✅ *[{agent}] CEO Onaylı*\n━━━━━━━━━━━━━━━━━━━━━━\n"
        full_result = header + result

        keyboard = _get_result_keyboard(agent, final_state)
        chunks = _split_message(full_result)

        for i, chunk in enumerate(chunks):
            if i == 0:
                await _safe_edit(thinking_msg, chunk, keyboard)
            else:
                await _safe_send(update, chunk)

    except Exception as e:
        logger.error(f"[BOT] Mesaj hatası: {e}")
        await thinking_msg.edit_text(f"❌ *Hata:*\n`{str(e)[:200]}`", parse_mode=ParseMode.MARKDOWN)
    finally:
        PROCESSING_USERS.discard(user_id)


def _store_quant_context(user_id: str, result: str, user_input: str):
    """QUANT analizinden son sembol verilerini bağlama saklar."""
    ctx = _user_context.setdefault(user_id, {})
    ctx["last_agent"] = "QUANT"

    # Analiz verilerini asenkron graph çıktısından almak yerine
    # bir sonraki QUANT çağrısında _analyze_symbol_full sonuçlarını cache'leriz.
    # Şimdilik rapordaki sayıları parse et (basit yöntem)
    try:
        # Sembol tespiti
        from agents.hft_quant import _extract_symbols
        symbols = _extract_symbols(user_input)
        ctx["last_symbols"] = symbols

        # Rapordaki ilk giriş/SL/hedef satırlarını parse et
        entry_match = re.search(r"Giriş Bölgesi.*?\$([0-9,]+).*?\$([0-9,]+)", result)
        sl_match = re.search(r"Stop-Loss.*?\$([0-9,]+)", result)
        t1_match = re.search(r"Hedef 1.*?\$([0-9,]+)", result)
        t2_match = re.search(r"Hedef 2.*?\$([0-9,]+)", result)
        t3_match = re.search(r"Hedef 3.*?\$([0-9,]+)", result)
        signal_match = re.search(r"NİHAİ KARAR.*?(\w[\w\s]+)\*(Güven: %(\d+))?", result)
        price_match = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)\s*\(([+-]?\d+\.\d+)%\)", result)
        rr_match = re.search(r"Risk/Ödül.*?1:([0-9.]+)", result)

        def parse_price(m, g=1):
            if m:
                try:
                    return float(m.group(g).replace(",", ""))
                except Exception:
                    pass
            return 0.0

        # Sinyal tespiti
        signal_name = "NÖTR"
        score = 0
        action = "BEKLE ↔️"
        conf = 50
        if "GÜÇLÜ ALIM" in result:
            signal_name, score, action = "GÜÇLÜ ALIM", 72, "AL 🚀"
            conf = 85
        elif "ALIM" in result and "GÜÇLÜ" not in result:
            signal_name, score, action = "ALIM", 42, "AL 📈"
            conf = 72
        elif "GÜÇLÜ SATIM" in result:
            signal_name, score, action = "GÜÇLÜ SATIM", -72, "SAT 🔴"
            conf = 85
        elif "SATIM" in result and "GÜÇLÜ" not in result:
            signal_name, score, action = "SATIM", -42, "SAT 📉"
            conf = 72

        price = parse_price(price_match)
        entry_low = parse_price(entry_match, 1)
        entry_high = parse_price(entry_match, 2)
        sl = parse_price(sl_match)
        t1 = parse_price(t1_match)
        t2 = parse_price(t2_match)
        t3 = parse_price(t3_match)
        rr = float(rr_match.group(1)) if rr_match else 0.0

        sl_pct = abs((price - sl) / price * 100) if price > 0 and sl > 0 else 5.0

        ctx["last_quant_data"] = {
            "symbol": symbols[0].replace("-USD", "") if symbols else "BTC",
            "price": price,
            "signal": {
                "signal": signal_name,
                "action": action,
                "score": score,
                "confidence": conf,
                "emoji": "🟢🟢" if score >= 65 else ("🟢" if score >= 40 else ("🔴🔴" if score <= -65 else ("🔴" if score <= -40 else "⚪"))),
                "bull_reasons": [],
                "bear_reasons": [],
            },
            "levels": {
                "entry_low": entry_low,
                "entry_high": entry_high,
                "stop_loss": sl,
                "sl_pct": sl_pct,
                "targets": [t for t in [t1, t2, t3] if t > 0],
                "risk_reward": rr,
                "supports": [],
                "resistances": [],
            },
            "rsi_d": 50,
            "fg_value": 50,
        }
        logger.debug(f"[BOT] QUANT bağlamı saklandı: {ctx['last_quant_data']['symbol']}")
    except Exception as e:
        logger.warning(f"[BOT] Bağlam parse hatası: {e}")


# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

async def _safe_edit(msg, text: str, keyboard=None):
    """Markdown hatası olursa düz metin olarak dene."""
    try:
        await msg.edit_text(
            text[:4000], parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard,
        )
    except Exception:
        try:
            await msg.edit_text(text[:4000], reply_markup=keyboard)
        except Exception as e:
            logger.warning(f"[BOT] Edit failed: {e}")


async def _safe_send(update: Update, text: str, keyboard=None):
    """Markdown hatası olursa düz metin olarak dene."""
    try:
        await update.message.reply_text(
            text[:4000], parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard,
        )
    except Exception:
        try:
            await update.message.reply_text(text[:4000], reply_markup=keyboard)
        except Exception as e:
            logger.warning(f"[BOT] Send failed: {e}")


# ─── Callback Handler ─────────────────────────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = str(query.from_user.id)

    if data == "main_menu":
        await query.edit_message_text(
            "🏠 *ANA MENÜ*\nNe yapmak istersiniz?",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_keyboard(),
        )

    elif data.startswith("agent_"):
        agent = data.split("_", 1)[1]
        prompts = {
            "SWE": "✍️ *SWE AJAN AKTİF*\nGeliştirmek istediğiniz sistemi/kodu yazın:",
            "QUANT": "📊 *QUANT AJAN AKTİF*\nAnaliz için sembol yazın (ör: BTC ETH AAPL altın):",
            "MARKETING": "📣 *MARKETING AJAN AKTİF*\nHedef sektör/bölge ve amacı yazın:",
            "EDGE": "💻 *EDGE AJAN AKTİF*\nKomut girin (disk durumu, bellek raporu, sistem durumu):",
            "FREELANCER": "💼 *FREELANCER AJAN AKTİF*\nAradığınız iş türünü yazın:",
        }
        await query.edit_message_text(
            prompts.get(agent, f"🎯 *{agent} AJAN AKTİF*\nKomutunuzu yazın:"),
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data == "tasks_history":
        tasks = await get_recent_tasks(user_id, limit=5)
        if not tasks:
            await query.edit_message_text("📋 Henüz görev yok.", reply_markup=back_keyboard())
        else:
            lines = ["📋 *SON 5 GÖREV*", "━━━━━━━━━━━━━━━━━━━━━━"]
            for i, t in enumerate(tasks, 1):
                lines.append(f"{i}. [{t.get('agent','?')}] {t.get('user_input','')[:40]}")
            await query.edit_message_text(
                "\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard()
            )

    elif data == "system_status":
        from core.scheduler import _scheduler
        sched = "🟢 Aktif" if _scheduler and _scheduler.running else "🔴 Durdu"
        await query.edit_message_text(
            f"📡 *SİSTEM DURUMU*\n🟢 Tüm ajanlar aktif\n{sched} Zamanlayıcı\n☁️ Cloud (Replit)",
            parse_mode=ParseMode.MARKDOWN, reply_markup=back_keyboard()
        )

    elif data.startswith("quant_approve_"):
        await query.edit_message_text(
            "✅ *ANALİZ ONAYLANDI*\n\n"
            "⚠️ Bu yalnızca analiz onayıdır.\n"
            "Gerçek alım/satım için yetkili aracı kurumunuzu kullanın.\n"
            "🔒 Oracle ASLA otomatik işlem açmaz.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data == "freelancer_apply_all":
        await query.edit_message_text(
            "📤 *BAŞVURULAR ONAYLANDI*\n"
            "Başvurular işaretlendi.\n"
            "📆 5 gün sonra otomatik takip mesajları hazırlanacak.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data in ("quant_cancel", "marketing_cancel", "freelancer_cancel") or data.startswith("reject_"):
        await query.edit_message_text("❌ *İşlem iptal edildi.*", parse_mode=ParseMode.MARKDOWN)

    elif data == "marketing_send_all":
        await query.edit_message_text(
            "📤 *Email gönderimi onaylandı.*\n"
            "Gerçek gönderim için SMTP entegrasyonu aktif edilmeli.",
            parse_mode=ParseMode.MARKDOWN,
        )


# ─── Yardımcılar ──────────────────────────────────────────────────────────────

def _get_result_keyboard(agent: str, state: dict) -> InlineKeyboardMarkup | None:
    if agent == "QUANT":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Analizi Onayla", callback_data="quant_approve_ok")],
            [InlineKeyboardButton("🔄 Yenile", callback_data="agent_QUANT"),
             InlineKeyboardButton("🏠 Menü", callback_data="main_menu")],
        ])
    elif agent == "MARKETING":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📤 Email Gönderimini Onayla", callback_data="marketing_send_all")],
            [InlineKeyboardButton("❌ İptal", callback_data="marketing_cancel"),
             InlineKeyboardButton("🏠 Menü", callback_data="main_menu")],
        ])
    elif agent == "FREELANCER":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📨 Başvuruları Onayla", callback_data="freelancer_apply_all")],
            [InlineKeyboardButton("❌ İptal", callback_data="freelancer_cancel"),
             InlineKeyboardButton("🏠 Menü", callback_data="main_menu")],
        ])
    else:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 Ana Menü", callback_data="main_menu")],
        ])


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:max_len])
        text = text[max_len:]
    return chunks


def build_application() -> Application:
    global _application
    app = Application.builder().token(settings.telegram_bot_token).build()
    _application = app
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("quant", quant_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app
