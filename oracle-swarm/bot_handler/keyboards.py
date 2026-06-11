from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🤖 SWE Geliştir", callback_data="agent_SWE"),
            InlineKeyboardButton("📊 Quant Analiz", callback_data="agent_QUANT"),
        ],
        [
            InlineKeyboardButton("📣 Marketing", callback_data="agent_MARKETING"),
            InlineKeyboardButton("💻 Edge Sistem", callback_data="agent_EDGE"),
        ],
        [
            InlineKeyboardButton("📋 Son Görevler", callback_data="tasks_history"),
            InlineKeyboardButton("📊 Sistem Durumu", callback_data="system_status"),
        ],
    ])


def approval_keyboard(task_id: str, action: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ ONAYLA & ÇALIŞTIRIL", callback_data=f"approve_{action}_{task_id}"),
            InlineKeyboardButton("❌ REDDET", callback_data=f"reject_{task_id}"),
        ],
    ])


def quant_action_keyboard(symbol: str, signal: str, confidence: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                f"✅ {signal} %{confidence} — ONAY VER",
                callback_data=f"quant_approve_{symbol}_{signal}_{confidence}"
            ),
        ],
        [
            InlineKeyboardButton("🔄 Yenile", callback_data=f"quant_refresh_{symbol}"),
            InlineKeyboardButton("❌ İptal", callback_data="quant_cancel"),
        ],
    ])


def marketing_send_keyboard(lead_count: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"📤 {lead_count} Email Gönder", callback_data="marketing_send_all"),
            InlineKeyboardButton("❌ İptal", callback_data="marketing_cancel"),
        ],
    ])


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Ana Menü", callback_data="main_menu")],
    ])
