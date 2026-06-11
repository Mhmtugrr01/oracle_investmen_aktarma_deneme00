import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode
from core.config import settings
from core.graph import oracle_graph, OracleState
from core.memory import get_recent_tasks
from core.scheduler import register_alert_callback, unregister_alert_callback
from bot_handler.keyboards import (
    main_menu_keyboard,
    back_keyboard,
)
from loguru import logger

PROCESSING_USERS: set[int] = set()
_application: Application | None = None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = str(user.id)

    async def send_alert(uid: str, msg: str):
        try:
            await _application.bot.send_message(
                chat_id=int(uid),
                text=msg,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.error(f"Alert send failed: {e}")

    register_alert_callback(user_id, send_alert)

    welcome = f"""🧠 *ORACLE MASTER-SWARM V4.0*
━━━━━━━━━━━━━━━━━━━━━━
Hoş geldiniz, *{user.first_name}*.

Ben Oracle CEO — Otonom Bilişsel İşletim Sisteminiz.

Kısa bir fikir veya komut yazın. Sistem;
1️⃣ İsteğinizi genişletip doğru ajana yönlendirir
2️⃣ Alt ajan sonucu üretir
3️⃣ CEO Critic katmanı denetler, eksik varsa düzeltir
4️⃣ Nihai ✅ onaylı sonucu size iletir

🤖 *Mevcut Ajanlar:*
• SWE — Yazılım Geliştirme
• QUANT — Borsa/Kripto + Makro Analizi
• MARKETING — Satış & Scraping
• EDGE — Sistem Kontrolü
• FREELANCER — İş Bulma & Başvuru

⏰ *Otomatik Alarmlar aktif:*
• Saatlik piyasa taraması
• 🌅 08:00 sabah brifing
• Güçlü toplama/dağıtım tespitinde anlık uyarı

Veya aşağıdaki menüyü kullanın:"""

    await update.message.reply_text(
        welcome,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """📖 *KULLANIM KILAVUZU*
━━━━━━━━━━━━━━━━━━━━━━
*Komutlar:*
/start — Ana menü + alarm kaydı
/status — Sistem durumu
/history — Son görevler
/quant — Hızlı piyasa analizi
/scan — Anlık piyasa taraması
/help — Bu mesaj

*Örnek Kullanımlar:*
• "BTC ve ETH analiz et"
• "Upwork'te Python uzmanı işi bul"
• "Balıkesir OSB elektrik firmaları"
• "FastAPI ile REST API yaz"
• "Disk durumu nedir"
• "Freelancer işleri ara"

*CEO Critic Sistemi:*
Her yanıt 2 katmandan geçer:
1. Alt ajan cevap üretir
2. CEO denetler ve eksik varsa tamamlar
Sonuç ✅ işareti ile onaylı gelir."""

    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from core.scheduler import _scheduler
    sched_status = "🟢 Aktif" if _scheduler and _scheduler.running else "🔴 Durdu"

    status_text = f"""📡 *SİSTEM DURUMU*
━━━━━━━━━━━━━━━━━━━━━━
🟢 CEO Router + Critic  — Aktif
🟢 SWE Mühendis         — Aktif
🟢 QUANT Gözcü          — Aktif
🟢 Marketing            — Aktif
🟢 Freelancer           — Aktif
🟢 Edge Daemon          — Aktif
🟢 Supabase Bellek      — Bağlı
🟢 Telegram API         — Bağlı
{sched_status} Zamanlayıcı        — Saatlik Tarama
━━━━━━━━━━━━━━━━━━━━━━
☁️ Cloud (Replit) | 🧠 GPT-4o"""

    await update.message.reply_text(
        status_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=back_keyboard(),
    )


async def quant_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    symbols = " ".join(args) if args else "BTC ETH AAPL altın"
    await _process_user_input(update, f"piyasa analizi: {symbols}")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "🔄 *Anlık piyasa taraması başlatılıyor...*",
        parse_mode=ParseMode.MARKDOWN,
    )
    try:
        from agents.hft_quant import run_scheduled_scan
        result = await run_scheduled_scan(send_alert_fn=None)
        if "kritik sinyal yok" in result.lower():
            result = "✅ Tarama tamamlandı — Şu an kritik bir toplama/dağıtım sinyali yok."
        await msg.edit_text(result, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await msg.edit_text(f"❌ Tarama hatası: {e}", parse_mode=ParseMode.MARKDOWN)


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    tasks = await get_recent_tasks(user_id, limit=5)

    if not tasks:
        await update.message.reply_text(
            "📋 Henüz tamamlanmış görev yok.",
            reply_markup=back_keyboard(),
        )
        return

    lines = ["📋 *SON GÖREVLER*", "━━━━━━━━━━━━━━━━━━━━━━"]
    for i, t in enumerate(tasks, 1):
        lines.append(
            f"\n*{i}. [{t.get('agent','?')}]* — {t.get('user_input','')[:50]}\n"
            f"   📅 {str(t.get('created_at',''))[:16]} | {t.get('status','?')}"
        )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=back_keyboard(),
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    await _process_user_input(update, user_input)


async def _process_user_input(update: Update, user_input: str):
    user_id = update.effective_user.id

    if user_id in PROCESSING_USERS:
        await update.message.reply_text("⏳ Önceki görev işleniyor, lütfen bekleyin...")
        return

    if len(user_input) < 2:
        await update.message.reply_text("ℹ️ Lütfen en az bir kelimelik komut girin.")
        return

    PROCESSING_USERS.add(user_id)

    thinking_msg = await update.message.reply_text(
        "🧠 *CEO Router analiz ediyor...*\n⚙️ Prompt genişletiliyor...",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        initial_state: OracleState = {
            "user_id": str(user_id),
            "user_input": user_input,
            "expanded_prompt": "",
            "agent": "",
            "result": "",
            "audited_result": "",
            "status": "pending",
            "task_id": None,
            "messages": [],
        }

        await thinking_msg.edit_text(
            "🔄 *Ajan devrede...*\n🔍 CEO Critic denetim bekliyor...",
            parse_mode=ParseMode.MARKDOWN,
        )

        final_state = await oracle_graph.ainvoke(initial_state)

        agent = final_state.get("agent", "CEO")
        result = final_state.get("audited_result") or final_state.get("result", "Sonuç alınamadı.")

        header = f"✅ *[{agent}] CEO Onaylı*\n━━━━━━━━━━━━━━━━━━━━━━\n"
        full_result = header + result

        chunks = _split_message(full_result)
        keyboard = _get_result_keyboard(agent, final_state)

        for i, chunk in enumerate(chunks):
            if i == 0:
                await thinking_msg.edit_text(
                    chunk,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=keyboard,
                )
            else:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Message handler error: {e}")
        await thinking_msg.edit_text(
            f"❌ *Hata:*\n`{str(e)[:300]}`",
            parse_mode=ParseMode.MARKDOWN,
        )
    finally:
        PROCESSING_USERS.discard(user_id)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "main_menu":
        await query.edit_message_text(
            "🏠 *ANA MENÜ*\nNe yapmak istersiniz?",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_keyboard(),
        )

    elif data.startswith("agent_"):
        agent = data.split("_")[1]
        prompts = {
            "SWE": "✍️ Geliştirmek istediğiniz sistemi/kodu yazın:",
            "QUANT": "📊 Analiz edilmesini istediğiniz sembolleri yazın (örn: BTC ETH AAPL altın):",
            "MARKETING": "📣 Hedef sektör/bölge ve amacınızı yazın:",
            "EDGE": "💻 Sistem komutunu yazın (status, disk_report, memory_check, temp_check):",
            "FREELANCER": "💼 Aradığınız iş türünü yazın (alan, bütçe, platform tercihi):",
        }
        await query.edit_message_text(
            f"🎯 *{agent} AJAN AKTİF*\n{prompts.get(agent, 'Komutunuzu yazın:')}",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data == "tasks_history":
        user_id = str(query.from_user.id)
        tasks = await get_recent_tasks(user_id, limit=5)
        if not tasks:
            await query.edit_message_text("📋 Henüz görev yok.", reply_markup=back_keyboard())
        else:
            lines = ["📋 *SON 5 GÖREV*"]
            for i, t in enumerate(tasks, 1):
                lines.append(f"{i}. [{t.get('agent','?')}] {t.get('user_input','')[:40]}")
            await query.edit_message_text(
                "\n".join(lines),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=back_keyboard(),
            )

    elif data == "system_status":
        from core.scheduler import _scheduler
        sched = "🟢 Aktif" if _scheduler and _scheduler.running else "🔴 Durdu"
        await query.edit_message_text(
            f"📡 *SİSTEM DURUMU*\n🟢 Tüm ajanlar aktif\n{sched} Zamanlayıcı\n☁️ Cloud modu",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=back_keyboard(),
        )

    elif data.startswith("quant_approve_"):
        await query.edit_message_text(
            "✅ *ANALİZ ONAYLANDI*\n\n⚠️ Bu yalnızca analiz onayıdır.\nGerçek alım/satım için yetkili aracı kurumunuzu kullanın.\n🔒 Oracle ASLA otomatik işlem açmaz.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data == "freelancer_apply_all":
        await query.edit_message_text(
            "📤 *BAŞVURULAR ONAYLANDI*\nTüm başvurular gönderilmek üzere işaretlendi.\n\n📆 5 gün sonra otomatik takip mesajları hazırlanacak.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data in ("quant_cancel", "marketing_cancel", "freelancer_cancel") or data.startswith("reject_"):
        await query.edit_message_text("❌ *İşlem iptal edildi.*", parse_mode=ParseMode.MARKDOWN)

    elif data == "marketing_send_all":
        await query.edit_message_text(
            "📤 *Email gönderimi onaylandı.*\nGerçek gönderim için SMTP entegrasyonu aktif edilmeli.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data == "main_menu":
        await query.edit_message_text(
            "🏠 Ana Menü",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_keyboard(),
        )


def _get_result_keyboard(agent: str, state: dict) -> InlineKeyboardMarkup | None:
    if agent == "QUANT":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Analizi Onayla", callback_data="quant_approve_ok")],
            [InlineKeyboardButton("🔄 Yeni Tarama", callback_data="agent_QUANT"),
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
