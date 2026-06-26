"""PROJECT OLYMPUS — Telegram matrix bridge (FAZ 5)."""

from __future__ import annotations

import asyncio
import os
from typing import Final

from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from core.config import load_oracle_config
from core.graph import compile_oracle_graph
from core.types import OracleState, SignalDirection

_PROGRESS_STEPS: Final[list[str]] = [
    "Agent 1: Makro tarama başlatıldı...",
    "Agent 2: Quant motoru piyasa yapısını tarıyor...",
    "Agent 3: Wyckoff balina avı / CVD paraziti inceleniyor...",
    "Agent 4: Fundamental süzgeç çalışıyor...",
    "Agent 5: Sentiment radar canlı...",
    "Agent 6: Red-Team kusur avında...",
    "The Oracle: Nihai direktif üretiliyor...",
]


def _normalize_symbol(raw: str) -> str:
    token = raw.strip().upper()
    if not token:
        return "BTC/USDT"
    if "/" in token:
        return token
    return f"{token}/USDT"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_100_from_norm(score: float) -> float:
    return round((_clamp(score, -1.0, 1.0) + 1.0) * 50.0, 1)


def _format_percent_from_norm(score: float) -> str:
    return f"%{_to_100_from_norm(score):.1f}"


def _format_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 0.005:
        return f"${int(round(rounded)):,}"
    return f"${rounded:,.2f}"


def _macro_label(score: float) -> str:
    if score >= 0.35:
        return "Güçlü"
    if score >= 0.1:
        return "Kararlı"
    if score <= -0.35:
        return "Baskılı"
    if score <= -0.1:
        return "Kırılgan"
    return "Dengeli"


def _whale_label(score: float) -> str:
    if score >= 0.35:
        return "Birikim"
    if score >= 0.1:
        return "Olumlu"
    if score <= -0.35:
        return "Dağıtım"
    if score <= -0.1:
        return "Zayıf"
    return "Nötr"


def _technical_label(score: float) -> str:
    if score >= 0.45:
        return "Güçlü Yükseliş"
    if score >= 0.15:
        return "Toparlanıyor"
    if score <= -0.45:
        return "Ayı (Düşüş)"
    if score <= -0.15:
        return "Zayıf"
    return "Kararsız"


def _oracle_label(direction: SignalDirection) -> str:
    if direction == SignalDirection.LONG:
        return "GÜÇLÜ AL 🟢"
    if direction == SignalDirection.SHORT:
        return "KISA SHORT 🔴"
    if direction == SignalDirection.NO_TRADE:
        return "POZİSYON YOK ⚫"
    return "BEKLE ⚪"


def _build_area_context(state: OracleState) -> str:
    return (
        f"Makro Koşullar {_macro_label(state.macro_score)} 🌐 ({_format_percent_from_norm(state.macro_score)}) | "
        f"Balina Aktivitesi: {_whale_label(state.whale_score)} 🐋 ({_format_percent_from_norm(state.whale_score)}) | "
        f"Teknik Eğilim: {_technical_label(state.quant_score)} 📉 ({_format_percent_from_norm(state.quant_score)})"
    )


def _build_manager_summary(state: OracleState, score_100: float) -> str:
    if state.signal_direction == SignalDirection.LONG:
        return (
            f"Piyasa lehine toparlanma sinyali var (Skor: %{score_100:.1f}). "
            f"Kompozit güven oranı {state.confidence:.0%} seviyesinde ve işlem ancak disiplinli risk yönetimiyle anlamlı."
        )
    if state.signal_direction == SignalDirection.SHORT:
        return (
            f"Aşağı yön baskısı belirgin (Skor: %{score_100:.1f}). "
            f"Kompozit görünüm kırılgan olduğu için savunmacı pozisyon yönetimi şart."
        )
    return (
        f"Piyasa yönü kararsız (Skor: %{score_100:.1f}). "
        "Kompozit güven oranı sınırlı olduğu için işleme girmek yüksek risk barındırıyor."
    )


def _extract_red_team_note(state: OracleState) -> str:
    if state.red_team_objections:
        return state.red_team_objections[0].replace("LLM notu: ", "").strip()
    if state.red_team_verdict:
        return state.red_team_verdict.strip()
    return "Kurumsal risk denetiminde ek bir itiraz üretilmedi."


def _trend_suffix(state: OracleState) -> str:
    if abs(state.quant_score) >= 0.35 or abs(state.macro_score) >= 0.35:
        return "Günlük + 4H Trend"
    return "Günlük + 4H Görünüm"


def _format_abort_message(state: OracleState, reason: str) -> str:
    asset = state.symbol or "?"
    composite = float(state.composite_score or 0.0)
    composite_pct = int(abs(composite) * 100)
    confidence = float(state.confidence or 0.0)
    confidence_pct = int(confidence * 100)
    base_rr = state.base_rr
    trade_type = state.trade_type or "BILINMIYOR"

    tf_biases = _safe_biases(state)
    weekly = tf_biases.get("1w", "?")
    daily = tf_biases.get("1d", "?")
    h4 = tf_biases.get("4h", "?")
    h1 = tf_biases.get("1h", "?")
    alignment = float(state.timeframe_alignment_score or 0.0)
    alignment_pct = int(alignment * 100)

    scores = [state.macro_score, state.quant_score, state.fundamental_score, state.sentiment_score]
    if state.whale_score is not None:
        scores.append(state.whale_score)
    consensus_variance = max(scores) - min(scores) if scores else 0.0

    macro_score = state.macro_score
    macro_pct = int(abs(macro_score) * 100) if macro_score else 0

    warnings = state.cross_asset_warnings or []
    warning_text = ""
    if warnings:
        warning_lines = "\n".join(f"  • {w}" for w in warnings[:3])
        warning_text = f"\n\n⚠️ UYARILAR:\n{warning_lines}"

    hist_bias = state.pattern_outcome_bias or ""
    hist_score = getattr(state, "historical_similarity_score", None)
    hist_text = ""
    if hist_bias and hist_score is not None:
        hist_text = f"\n🔄 Tarihsel Benzerlik: {int(hist_score)}/100 → {hist_bias}"

    if "KOMPOZİT" in reason or composite < 0.60:
        neden = (
            f"Kompozit skor {composite_pct}% (minimum 60% gerekli). "
            f"Ajanlar arası fikir ayrılığı yüksek (variance: {consensus_variance:.2f}). "
            f"Timeframe'ler hizalanmamış ({alignment_pct}%)."
        )
    elif "GÜVEN" in reason or confidence < 0.60:
        neden = (
            f"Sistem güven skoru {confidence_pct}% (minimum 60% gerekli). "
            f"Sinyalin güvenilirliği yetersiz."
        )
    elif "R:R" in reason:
        rr_val = f"{base_rr:.2f}" if base_rr is not None else "hesaplanamadı"
        neden = (
            f"Risk/Ödül oranı {rr_val} (minimum 3.0 gerekli). "
            f"Bu kurulum asimetrik getiri sunmuyor."
        )
    else:
        neden = reason

    alignment_tf_count = int(alignment * 4)
    consistency_pct = int((1 - min(consensus_variance, 1.0)) * 100)
    rr_display = f"{base_rr:.2f}" if base_rr is not None else "N/A"

    msg = f"""🔔 OLYMPUS ORACLE — ANALİZ TAMAMLANDI

📌 VARLIK: {asset}
⚪ KARAR: İŞLEM YAPILMADI

❌ NEDEN:
{neden}

📊 MEVCUT DURUM:
  Kompozit Skor : {composite_pct}%  (eşik: 60%)
  Sistem Güveni : {confidence_pct}%  (eşik: 60%)
  Ajan Uyumu    : {alignment_pct}%  ({alignment_tf_count}/4 TF hizalı)
  Ajan Tutarlılık: {consistency_pct}%  (yüksek = iyi)
"""

    if base_rr is not None:
        msg += f"\n  R:R Oranı     : {rr_display}  (eşik: 3.0)"
    if trade_type != "BILINMIYOR":
        msg += f"\n  Trade Tipi    : {trade_type}"
    if macro_score:
        msg += f"\n  Makro Skor    : {macro_pct}%"

    msg += (
        f"\n\n📈 TIMEFRAME:\n"
        f"  Haftalık → {weekly}\n"
        f"  Günlük   → {daily}\n"
        f"  4 Saatlik → {h4}\n"
        f"  1 Saatlik → {h1}"
        f"{hist_text}"
        f"{warning_text}\n\n"
        "💡 CEO NOTU:\n"
        f'"Piyasa şu an çatışmalı sinyaller üretiyor. '
        f"{('Tarihsel döngü benzerliği iyimser işaret veriyor ancak teknik yapı henüz hazır değil. ' if 'BULLISH' in hist_bias else '')}"
        "Sistem minimum confluence eşiğine ulaşılana kadar bekliyor. "
        "Alım bölgesi oluştuğunda otomatik bildirim gelecek.\"\n\n"
        "⏳ DURUM: İzlemeye devam ediliyor.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return msg


def _format_abort_response(state: OracleState) -> str:
    reason = state.fatal_error or "Belirtilmemiş teknik hata"
    return _format_abort_message(state, reason)


def _signal_emoji(signal: str) -> str:
    signal_emojis = {
        "STRONG_BUY": "🟢🟢",
        "ACCUMULATE": "🟢",
        "HOLD": "🟡",
        "WATCH": "🔵",
        "REDUCE": "🟠",
        "STRONG_SELL": "🔴🔴",
        "SHORT": "🔴",
        "AVOID": "⚪",
        "WATCHLIST_PREMIUM": "💜",
    }
    return signal_emojis.get(signal, "⚪")


def _safe_biases(state: OracleState) -> dict:
    biases = state.timeframe_biases or {}
    return {
        "1w": biases.get("1w", "N/A"),
        "1d": biases.get("1d", "N/A"),
        "4h": biases.get("4h", "N/A"),
        "1h": biases.get("1h", "N/A"),
    }


def _confluence_count(state: OracleState) -> int:
    points = 0
    if (state.timeframe_alignment_score or 0.0) >= 0.75:
        points += 1
    if (state.base_rr or 0.0) >= 3.0:
        points += 1
    if (state.cross_asset_score or 0.0) >= 60.0:
        points += 1
    if state.divergence_daily == "POSITIVE_DIVERGENCE":
        points += 1
    if state.pattern_outcome_bias == "HISTORICALLY_BULLISH":
        points += 1
    return points


def _cross_lines_from_messages(state: OracleState) -> tuple[str, str, float]:
    macro_msg = next((m for m in state.messages if m.startswith("[MACRO_SENTINEL]")), "")
    btc_d = "N/A"
    usdt_d = "N/A"
    dxy = "N/A"
    for part in macro_msg.split():
        if part.startswith("btc_d="):
            btc_d = part.split("=", 1)[1]
        elif part.startswith("usdt_d="):
            usdt_d = part.split("=", 1)[1]
        elif part.startswith("dxy_trend="):
            dxy = part.split("=", 1)[1]
    vix = "N/A"
    if "VIX=" in macro_msg:
        try:
            vix = macro_msg.split("VIX=")[1].split()[0]
        except Exception:
            vix = "N/A"
    return f"BTC.D: {btc_d} | USDT.D: {usdt_d}", f"DXY: {dxy} | VIX: {vix}", float(state.cross_asset_score or 0.0)


def format_oracle_response(state: OracleState) -> str:
    if state.fatal_error:
        return _format_abort_response(state)

    score_pct = state.composite_score * 100.0
    area_context = _build_area_context(state)
    manager_summary = _build_manager_summary(state, score_pct)
    red_team_note = _extract_red_team_note(state)

    signal = state.signal_label or "WATCH"
    signal_emoji = _signal_emoji(signal)
    confluence_count = _confluence_count(state)
    tf_align = float(state.timeframe_alignment_score or 0.0)
    aligned_count = int(round(tf_align * 4)) if tf_align <= 1.0 else 0
    biases = _safe_biases(state)
    cross_line_1, cross_line_2, cross_score = _cross_lines_from_messages(state)
    warnings = state.cross_asset_warnings or []
    cross_warning_lines = "\n".join([f"   - {w}" for w in warnings]) if warnings else "   - Uyarı yok"

    exchange = "BINANCE"
    timestamp = state.updated_at.strftime("%Y-%m-%d %H:%M UTC")
    validity_period = "24 saat"

    entry = _format_price(state.entry_price)
    stop = _format_price(state.stop_loss)
    rr_value = state.base_rr if state.base_rr is not None else state.risk_reward_ratio
    rr = f"{rr_value:.2f}" if rr_value is not None else "N/A"

    level_lines = []
    if state.entry_zone_low is not None and state.entry_zone_high is not None:
        level_lines.append(f"   ┌ Giriş Bölgesi: ${state.entry_zone_low:.4f} — ${state.entry_zone_high:.4f}")
    if state.stop_loss is not None:
        inv = f"{state.invalidation_level:.4f}" if state.invalidation_level is not None else "N/A"
        level_lines.append(f"   ├ Stop-Loss:     ${state.stop_loss:.4f}  (İptal Seviyesi: ${inv})")
    if state.t1 is not None and state.t1_rr is not None:
        level_lines.append(f"   ├ Hedef 1 (T1): ${state.t1:.4f}  [R:R 1:{state.t1_rr:.1f}]")
    if state.t2 is not None and state.t2_rr is not None:
        level_lines.append(f"   ├ Hedef 2 (T2): ${state.t2:.4f}  [R:R 1:{state.t2_rr:.1f}]")
    if state.t3 is not None and state.t3_rr is not None:
        level_lines.append(f"   └ Hedef 3 (T3): ${state.t3:.4f}  [R:R 1:{state.t3_rr:.1f}] ← Uzun Vade")
    level_block = "\n".join(level_lines) if level_lines else "   - Seviye hesaplanamadı"

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "𝗢𝗟𝗬𝗠𝗣𝗨𝗦 𝗢𝗥𝗔𝗖𝗟𝗘 — SİNYAL KARTI\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 VARLIK: {state.symbol} | {exchange}\n"
        f"🕐 ANALİZ: {timestamp}\n\n"
        f"⚡ KARAR: {signal_emoji} {signal}\n"
        f"📊 KOMPOZİT SKOR: {score_pct:.1f}%\n"
        f"🎯 CONFLUENCE: {confluence_count}/5\n"
        f"📐 TIMEFRAME HIZALAMASI: {tf_align*100:.1f}% ({aligned_count}/4)\n\n"
        "📈 TIMEFRAME TABLO:\n"
        f"   Haftalık → {biases['1w']}\n"
        f"   Günlük   → {biases['1d']}\n"
        f"   4 Saatlik → {biases['4h']}\n"
        f"   1 Saatlik → {biases['1h']}\n"
        f"   RSI Uyumsuzluk (Günlük): {state.divergence_daily or 'NONE'}\n\n"
        "🌐 CROSS-ASSET:\n"
        f"   {cross_line_1}\n"
        f"   {cross_line_2}\n"
        f"   Cross-Asset Skoru: {cross_score:.1f}/100\n"
        f"{cross_warning_lines}\n\n"
        "🔄 TARİHSEL DÖNGÜ:\n"
        f"   Geçmiş Benzerlik: {state.historical_pattern or 'N/A'}\n"
        f"   Tarihsel Eğilim: {state.pattern_outcome_bias or 'N/A'}\n\n"
        "🎯 İŞLEM SEVİYELERİ:\n"
        f"{level_block}\n\n"
        "💡 CEO ANALİZİ:\n"
        f"\"{manager_summary}\"\n\n"
        "⚔️ KATİL SAVCI:\n"
        f"\"{red_team_note}\"\n\n"
        "⚠️ UYARILAR:\n"
        f"{area_context}\n\n"
        f"📅 GEÇERLİLİK: {validity_period}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


class TelegramHandler:
    def __init__(self, token: str, allowed_user_id: int | None) -> None:
        self._token = token
        self._allowed_user_id = allowed_user_id
        self._app: Application | None = None
        self._graph = None

    def _is_authorized(self, update: Update) -> bool:
        user = update.effective_user
        if self._allowed_user_id is None:
            return True
        if user is None:
            return False
        return int(user.id) == int(self._allowed_user_id)

    async def _deny_if_unauthorized(self, update: Update) -> bool:
        if self._is_authorized(update):
            return False
        if update.effective_message:
            await update.effective_message.reply_text("Yetkisiz erişim.")
        return True

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._deny_if_unauthorized(update):
            return
        if not update.effective_message:
            return
        await update.effective_message.reply_text(
            "👑 PROJECT OLYMPUS - The Oracle\n"
            "Canli matrix baglantisi aktif.\n"
            "Komut: /oracle BTC"
        )

    async def command_oracle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._deny_if_unauthorized(update):
            return
        if not update.effective_message:
            return
        symbol = _normalize_symbol(context.args[0] if context.args else "BTC")

        progress_message = await update.effective_message.reply_text(
            f"/oracle {symbol} ateşlendi. Sistem hazırlanıyor..."
        )

        user_id = str(update.effective_user.id) if update.effective_user else ""
        chat_id = int(update.effective_chat.id) if update.effective_chat else 0
        query = update.effective_message.text or f"/oracle {symbol}"

        task = asyncio.create_task(
            self._run_pipeline(symbol=symbol, user_id=user_id, chat_id=chat_id, query=query)
        )

        step_idx = 0
        while not task.done():
            await progress_message.edit_text(_PROGRESS_STEPS[step_idx % len(_PROGRESS_STEPS)])
            step_idx += 1
            await asyncio.sleep(1.25)

        final_state = await task
        await progress_message.edit_text(
            format_oracle_response(final_state),
            disable_web_page_preview=True,
        )

    async def command_analiz(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._deny_if_unauthorized(update):
            return
        if not update.effective_message:
            return

        args = context.args
        if not args:
            await update.effective_message.reply_text(
                "Kullanim: /analiz SEMBOL\n"
                "Ornekler: /analiz BTC  /analiz NVDA  /analiz THYAO.IS\n"
                "Kripto icin: /analiz BTC/USDT veya kisaca /analiz BTC"
            )
            return

        raw = args[0].upper().strip()
        crypto_map = {
            "BTC": "BTC/USDT",
            "ETH": "ETH/USDT",
            "INJ": "INJ/USDT",
            "RNDR": "RNDR/USDT",
            "FET": "FET/USDT",
        }
        symbol = crypto_map.get(raw, raw)

        await update.effective_message.reply_text(f"[ORACLE] {symbol} analizi baslatiliyor... (~2 dakika)")

        try:
            user_id = str(update.effective_user.id) if update.effective_user else ""
            chat_id = int(update.effective_chat.id) if update.effective_chat else 0
            query = update.effective_message.text or f"/analiz {symbol}"
            result_state = await self._run_pipeline(symbol=symbol, user_id=user_id, chat_id=chat_id, query=query)
            if result_state and result_state.signal_label not in (None, "AVOID", "WATCH"):
                msg = format_oracle_response(result_state)
            else:
                label = getattr(result_state, "signal_label", "AVOID") if result_state else "HATA"
                composite = getattr(result_state, "composite_score", 0.0) if result_state else 0.0
                base_rr = getattr(result_state, "base_rr", 0.0) if result_state else 0.0
                msg = (
                    f"[{symbol}] Sinyal: {label}\n"
                    f"Kompozit: {composite:.2f} | R:R: {base_rr:.2f}\n"
                    "Islem kalitesi yetersiz — pozisyon yok."
                )
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
        except Exception as exc:
            await update.effective_message.reply_text(f"Analiz hatasi: {exc}")

    async def command_tarama(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._deny_if_unauthorized(update):
            return
        if not update.effective_message:
            return

        await update.effective_message.reply_text("[SCANNER] 21 varlik taraniyor... (~10 dakika)")
        try:
            from core.scanner import OracleScanner

            async def _pipeline_runner(asset: str):
                return await self._run_pipeline(
                    symbol=asset,
                    user_id=str(update.effective_user.id) if update.effective_user else "",
                    chat_id=update.effective_chat.id,
                    query=f"/tarama {asset}",
                )

            async def _telegram_sender(text: str):
                await context.bot.send_message(chat_id=update.effective_chat.id, text=text)

            conf = await load_oracle_config()
            scanner = OracleScanner(_pipeline_runner, _telegram_sender, conf.model_dump())
            await scanner._run_scan_once()
        except Exception as exc:
            await update.effective_message.reply_text(f"Tarama hatasi: {exc}")

    async def _run_pipeline(
        self,
        *,
        symbol: str,
        user_id: str,
        chat_id: int,
        query: str,
    ) -> OracleState:
        if self._graph is None:
            await load_oracle_config()
            self._graph = compile_oracle_graph()

        initial_state = OracleState(
            query=query,
            symbol=symbol,
            user_id=user_id,
            chat_id=chat_id,
        )
        raw_result = await self._graph.ainvoke(initial_state)
        if isinstance(raw_result, OracleState):
            return raw_result
        return OracleState.model_validate(raw_result)

    def build(self) -> Application:
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("oracle", self.command_oracle))
        app.add_handler(CommandHandler("analiz", self.command_analiz))
        app.add_handler(CommandHandler("tarama", self.command_tarama))
        self._app = app
        return app

    async def start(self) -> None:
        if self._app is None:
            self.build()
        assert self._app is not None
        logger.info("Telegram bot polling başlatılıyor...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self._app is None:
            return
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram bot durduruldu.")


def create_handler() -> TelegramHandler:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN ortam değişkeni tanımlı değil.")
    raw_allowed = os.getenv("ALLOWED_USER_ID", "").strip()
    allowed_user_id = int(raw_allowed) if raw_allowed else None
    return TelegramHandler(token=token, allowed_user_id=allowed_user_id)
