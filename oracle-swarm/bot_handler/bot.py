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
from core.memory import get_recent_tasks, init_tables
from bot_handler.keyboards import (
    main_menu_keyboard,
    back_keyboard,
    quant_action_keyboard,
    marketing_send_keyboard,
)
from loguru import logger


PROCESSING_USERS: set[int] = set()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome = f"""🧠 *ORACLE MASTER-SWARM V4.0*
━━━━━━━━━━━━━━━━━━━━━━
Hoş geldiniz, *{user.first_name}*.

Ben Oracle CEO — Otonom Bilişsel İşletim Sisteminiz.

Kısa bir fikir veya komut yazın (10 kelime yeterli).
Sistemi genişletip doğru ajana yönlendireceğim.

🤖 *Mevcut Ajanlar:*
• SWE — Yazılım Geliştirme
• QUANT — Borsa/Kripto Analizi
• MARKETING — Satış & Scraping
• EDGE — Sistem Kontrolü
• CEO — Strateji & Rapor

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
/start — Ana menü
/status — Sistem durumu
/history — Son görevler
/help — Bu mesaj

*Kullanım:*
Sadece ne istediğinizi yazın:
• "BTC analiz et"
• "Bursa OSB elektrik firmaları bul"
• "Telegram botu yaz"
• "Disk durumu nedir"

Sistem otomatik genişletip doğru ajana yönlendirir."""

    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_text = """📡 *SİSTEM DURUMU*
━━━━━━━━━━━━━━━━━━━━━━
🟢 CEO Router — Aktif
🟢 SWE Mühendis — Aktif
🟢 QUANT Gözcü — Aktif
🟢 Marketing — Aktif
🟢 Edge Daemon — Aktif
🟢 Supabase Bellek — Bağlı
🟢 Telegram API — Bağlı
━━━━━━━━━━━━━━━━━━━━━━
☁️ Ortam: Cloud (Replit)
🔒 Güvenlik: Aktif"""

    await update.message.reply_text(
        status_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=back_keyboard(),
    )


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
    user_id = update.effective_user.id
    user_input = update.message.text.strip()

    if user_id in PROCESSING_USERS:
        await update.message.reply_text("⏳ Önceki görev işleniyor, lütfen bekleyin...")
        return

    if len(user_input) < 2:
        await update.message.reply_text("ℹ️ Lütfen en az bir kelimelik komut girin.")
        return

    PROCESSING_USERS.add(user_id)

    thinking_msg = await update.message.reply_text(
        "🧠 *CEO Router analiz ediyor...*\n⚙️ Prompt genişletiliyor ve ajan belirleniyor...",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        initial_state: OracleState = {
            "user_id": str(user_id),
            "user_input": user_input,
            "expanded_prompt": "",
            "agent": "",
            "result": "",
            "status": "pending",
            "task_id": None,
            "messages": [],
        }

        await thinking_msg.edit_text(
            "🔄 *LangGraph işliyor...*\n⚡ Ajan devreye girdi.",
            parse_mode=ParseMode.MARKDOWN,
        )

        final_state = await oracle_graph.ainvoke(initial_state)

        agent = final_state.get("agent", "CEO")
        result = final_state.get("result", "Sonuç alınamadı.")
        task_id = final_state.get("task_id", "")

        header = f"✅ *[{agent} AJAN] TAMAMLANDI*\n━━━━━━━━━━━━━━━━━━━━━━\n"
        full_result = header + result

        chunks = _split_message(full_result)
        for i, chunk in enumerate(chunks):
            if i == 0:
                keyboard = _get_result_keyboard(agent, task_id, final_state)
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
            f"❌ *Hata oluştu:*\n`{str(e)[:300]}`\n\nLütfen tekrar deneyin.",
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
            "SWE": "SWE modunu seçtiniz. Geliştirmek istediğiniz sistemi/kodu yazın:",
            "QUANT": "QUANT modunu seçtiniz. Analiz edilmesini istediğiniz sembolleri yazın (örn: BTC, AAPL):",
            "MARKETING": "MARKETING modunu seçtiniz. Hedef sektör/bölge ve amacınızı yazın:",
            "EDGE": "EDGE modunu seçtiniz. Sistem komutunu yazın (disk_report, status, temp_check):",
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
                lines.append(f"{i}. [{t.get('agent','?')}] {t.get('user_input','')[:40]} — {t.get('status','?')}")
            await query.edit_message_text(
                "\n".join(lines),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=back_keyboard(),
            )

    elif data == "system_status":
        await query.edit_message_text(
            "📡 *SİSTEM DURUMU*\n🟢 Tüm sistemler aktif\n☁️ Cloud modu",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=back_keyboard(),
        )

    elif data.startswith("quant_approve_"):
        parts = data.split("_")
        symbol = parts[2] if len(parts) > 2 else "?"
        signal = parts[3] if len(parts) > 3 else "?"
        await query.edit_message_text(
            f"✅ *ONAY KAYDEDİLDİ*\n{symbol} — {signal}\n\n⚠️ Bu sadece analiz onayıdır. Gerçek işlem açılmamıştır.\nGerçek alım/satım için yetkili aracı kurumunuzu kullanın.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data == "quant_cancel" or data.startswith("reject_"):
        await query.edit_message_text("❌ *İşlem iptal edildi.*", parse_mode=ParseMode.MARKDOWN)

    elif data == "marketing_send_all":
        await query.edit_message_text(
            "📤 *Email gönderimi onaylandı.*\nGerçek gönderim için SMTP entegrasyonu eklenmelidir.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data == "marketing_cancel":
        await query.edit_message_text("❌ *Marketing görevi iptal edildi.*", parse_mode=ParseMode.MARKDOWN)


def _get_result_keyboard(agent: str, task_id: str, state: dict) -> InlineKeyboardMarkup | None:
    if agent == "QUANT":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Analizi Onayla", callback_data=f"quant_approve_MARKET_LONG_75")],
            [InlineKeyboardButton("🏠 Ana Menü", callback_data="main_menu")],
        ])
    elif agent == "MARKETING":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📤 Email Gönderimini Onayla", callback_data="marketing_send_all")],
            [InlineKeyboardButton("🏠 Ana Menü", callback_data="main_menu")],
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
    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app
