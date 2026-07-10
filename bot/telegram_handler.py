from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Final

from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from core.config import get_oracle_config_cached, load_oracle_config
from core.graph import compile_oracle_graph
from core.types import OracleState, PipelineStatus, SignalDirection

_PROGRESS_STEPS: Final[list[str]] = [
    "Agent 1: Makro tarama başlatıldı...",
    "Agent 2: Quant motoru piyasa yapısını tarıyor...",
    "Agent 3: Wyckoff balina avı / CVD paraziti inceleniyor...",
    "Agent 4: Fundamental süzgeç çalışıyor...",
    "Agent 5: Sentiment radar canlı...",
    "Agent 6: Red-Team kusur avında...",
    "The Oracle: Nihai direktif üretiliyor...",
]

_RESERVED_ORACLE_TOKENS: Final[set[str]] = {
    "TARAMA",
    "SCAN",
    "ALL",
    "TUM",
    "TÜM",
    "LISTE",
    "LIST",
    "HELP",
    "YARDIM",
}


def _normalize_symbol(raw: str) -> str:
    # ── GÜVENLİK VE ARINDIRMA SÜZGECİ (Sanitizer v2.0) ──
    token = raw.strip().replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("$", "").replace(" ", "").upper()
    if not token:
        return "BTC/USDT"
    if token in _RESERVED_ORACLE_TOKENS:
        return "__SCAN__"
        
    # Eğer kullanıcı zaten "FETUSDT" yazdıysa sondaki "USDT"yi kırp (FETUSDT/USDT hatasını engelle!)
    if token.endswith("USDT") and "/" not in token:
        token = token[:-4] # "USDT" kısmını atar (FETUSDT -> FET kalır)
        
    if "/" in token:
        return token
    return f"{token}/USDT"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_100_from_norm(score: float) -> float:
    return round((_clamp(score, -1.0, 1.0) + 1.0) * 50.0, 1)


def _format_percent_from_norm(score: float) -> str:
    value = 0.0 if score is None else float(score)
    return f"%{_to_100_from_norm(value):.1f}"


def _format_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 0.005:
        return f"${int(round(rounded)):,}"
    return f"${rounded:,.2f}"


def _macro_label(score: float | None) -> str:
    if score is None:
        return "Veri Yok"
    if score >= 0.35:
        return "Güçlü"
    if score >= 0.1:
        return "Kararlı"
    if score <= -0.35:
        return "Baskılı"
    if score <= -0.1:
        return "Kırılgan"
    return "Dengeli"


def _whale_label(score: float | None) -> str:
    if score is None:
        return "Veri Yok"
    if score >= 0.35:
        return "Birikim"
    if score >= 0.1:
        return "Olumlu"
    if score <= -0.35:
        return "Dağıtım"
    if score <= -0.1:
        return "Zayıf"
    return "Nötr"


def _technical_label(score: float | None) -> str:
    if score is None:
        return "Veri Yok"
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
    macro_score = state.macro_score
    whale_score = state.whale_score
    quant_score = state.quant_score
    return (
        f"Makro Koşullar {_macro_label(macro_score)} 🌐 ({_format_percent_from_norm(macro_score)}) | "
        f"Balina Aktivitesi: {_whale_label(whale_score)} 🐋 ({_format_percent_from_norm(whale_score)}) | "
        f"Teknik Eğilim: {_technical_label(quant_score)} 📉 ({_format_percent_from_norm(quant_score)})"
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

    min_composite = 0.60
    min_confidence = 0.60
    min_rr = 3.0
    try:
        conf = get_oracle_config_cached()
        min_composite = float(conf.ceo.min_composite_score)
        min_confidence = float(conf.ceo.confidence_threshold)
        min_rr = float(conf.risk.min_risk_reward_ratio)
    except Exception:
        # Config cache hazır değilse güvenli varsayılanları koru.
        pass

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

    if "KOMPOZİT" in reason or composite < min_composite:
        neden = (
            f"Kompozit skor {composite_pct}% (minimum {int(min_composite * 100)}% gerekli). "
            f"Ajanlar arası fikir ayrılığı yüksek (variance: {consensus_variance:.2f}). "
            f"Timeframe'ler hizalanmamış ({alignment_pct}%)."
        )
    elif "GÜVEN" in reason or confidence < min_confidence:
        neden = (
            f"Sistem güven skoru {confidence_pct}% (minimum {int(min_confidence * 100)}% gerekli). "
            f"Sinyalin güvenilirliği yetersiz."
        )
    elif "R:R" in reason:
        rr_val = f"{base_rr:.2f}" if base_rr is not None else "hesaplanamadı"
        neden = (
            f"Risk/Ödül oranı {rr_val} (minimum {min_rr:.1f} gerekli). "
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
    Kompozit Skor : {composite_pct}%  (eşik: {int(min_composite * 100)}%)
    Sistem Güveni : {confidence_pct}%  (eşik: {int(min_confidence * 100)}%)
  Ajan Uyumu    : {alignment_pct}%  ({alignment_tf_count}/4 TF hizalı)
  Ajan Tutarlılık: {consistency_pct}%  (yüksek = iyi)
    Ajan Skorları  : Makro {state.macro_score:+.2f} | Quant {state.quant_score:+.2f} | Fundamental {state.fundamental_score:+.2f} | Sentiment {state.sentiment_score:+.2f}
"""

    if base_rr is not None:
        msg += f"\n  R:R Oranı     : {rr_display}  (eşik: {min_rr:.1f})"
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

    # Whale strateji notu (Remora/Çakal)
    whale_strat_msg = next((m for m in (state.messages or []) if m.startswith("[WHALE_STRATEGY]")), None)
    whale_strategy_line = ""
    if whale_strat_msg:
        whale_strategy_line = f"\n\n🐋 BALİNA STRATEJİSİ:\n\"{whale_strat_msg.replace('[WHALE_STRATEGY] ', '')}\"\n"

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
    fib_lines = []
    if state.fib_382 is not None and state.fib_500 is not None and state.fib_618 is not None:
        fib_lines = [
            "\n🎯 ALIM BÖLGELERİ (FİBONACCI RETRACEMent):",
            f"   0.382 → ${state.fib_382:.4f} (İlk destek)",
            f"   0.500 → ${state.fib_500:.4f} (Güçlü destek)",
            f"   0.618 → ${state.fib_618:.4f} ⭐ (Altın oran)",
        ]
        if state.fib_ext_1272 is not None:
            fib_lines += [
                "\n📤 TP HEDEFLERİ (FİBONACCI UZANTI):",
                f"   1.272 → ${state.fib_ext_1272:.4f}  [T1 hedef]",
                f"   1.618 → ${state.fib_ext_1618:.4f}  [T2 hedef] ⭐",
                f"   2.618 → ${state.fib_ext_2618:.4f}  [T3 uzak hedef]",
            ]
    level_block = "\n".join(level_lines) if level_lines else "   - Seviye hesaplanamadı"
    fib_block = "\n".join(fib_lines)

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
        f"{fib_block}\n\n"
        "💡 CEO ANALİZİ:\n"
        f"\"{manager_summary}\"\n\n"
        "⚔️ KATİL SAVCI:\n"
        f"\"{red_team_note}\"\n"
        f"{whale_strategy_line}"
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
            "Komutlar: /oracle BTC | /tarama | /stats [SEMBOL]"
        )

    async def command_oracle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._deny_if_unauthorized(update):
            return
        if not update.effective_message:
            return
        symbol = _normalize_symbol(context.args[0] if context.args else "BTC")
        if symbol == "__SCAN__":
            await update.effective_message.reply_text(
                "⚠️ /oracle yalnızca tek sembol analizi içindir.\n\n"
                "Kullanım: /oracle BTC | /oracle ETH | /oracle NVDA\n"
                "21 varlık tam tarama için: /tarama"
            )
            return

        progress_message = await update.effective_message.reply_text(
            f"/oracle {symbol} ateşlendi. Sistem hazırlanıyor..."
        )

        user_id = str(update.effective_user.id) if update.effective_user else ""
        chat_id = int(update.effective_chat.id) if update.effective_chat else 0
        query = update.effective_message.text or f"/oracle {symbol}"

        task = asyncio.create_task(
            self._run_pipeline(symbol=symbol, user_id=user_id, chat_id=chat_id, query=query)
        )

        # ── 🛡️ ZAMAN AYARLI ZORUNLU EŞİK (REAL TIMEOUT LOOP) ──
        import time
        start_time = time.time()
        step_idx = 0
        
        # Animasyon devam etsin diye while döngüsünü zorla 90 Saniyeye Kilitledim! (Şu an Koma Riski 0'dır)
        while not task.done():
            if time.time() - start_time > 90.0:
                task.cancel()  # İçeride ölü yatıyorsa İptali Kes ve çık!
                break
                
            await progress_message.edit_text(_PROGRESS_STEPS[step_idx % len(_PROGRESS_STEPS)])
            step_idx += 1
            await asyncio.sleep(1.25)

        # ── 🛡️ TITANIUM EXCEPTION SHIELD (HATA SÖKÜCÜ MİMARİ) ──
        try:
            if task.cancelled():
                raise asyncio.TimeoutError()
            final_state = await task
            if not isinstance(final_state, OracleState):
                final_state = OracleState.model_validate(final_state)
        except asyncio.CancelledError:
            fail_str = "⏱️ OLYMPUS ACİL ZİRVE UYARISI:\n\nAnaliz görevi zaman aşımı nedeniyle iptal edildi."
            await progress_message.edit_text(f"❌ BAĞLANTI İPTAL PROTOKOLÜ\n\n{fail_str}", disable_web_page_preview=True)
            logger.error("[TELEGRAM VETO] Görev iptal edildi (CancelledError).")
            return
        except asyncio.TimeoutError:
            fail_str = "⏱️ OLYMPUS ACİL ZİRVE UYARISI:\n\nHedef Ajanlardan (Data/LLM) dönüş kilitlenmiş, API Asılı Kalmıştır!\nAsimetrik Kurallara aykırı olduğu için Sistem Risk Almamak Üzere Operasyonu İnfaz Etmiştir."
            await progress_message.edit_text(f"❌ BAĞLANTI İPTAL PROTOKOLÜ\n\n{fail_str}", disable_web_page_preview=True)
            logger.error("[TELEGRAM VETO] 90 Saniye Cıdar aşıldığı için Ölümcül (hang) Zorla Çıkış Vuruldu.")
            return
        except Exception as critical_crash:
            crash_err = f"💥 FATAL SİSTEM DONANIMI: Pazar İzinlerinden Verisizlik ve Sıkışma Patlak Verdi:\n({critical_crash})\nMühürler kapalıdır."
            await progress_message.edit_text(f"❌ KORUMA KESİCİ\n\n{crash_err}", disable_web_page_preview=True)
            return

        # ── 🛡️ PORTFOLIO AUTO-TRACKER HOOK (R03 Phase 5) ──
        # Normal, kazasız onay süreci:
        status_str = str(final_state.status.value).upper()
        if "ABORT" not in status_str and "FAIL" not in status_str and not final_state.fatal_error:
            try:
                from core.tracker import save_signal
                if all(v is not None for v in (final_state.entry_price, final_state.stop_loss, final_state.t1, final_state.t2, final_state.t3)):
                    save_signal(
                        asset=final_state.symbol,
                        direction=str(final_state.signal_direction),
                        entry=float(final_state.entry_price),
                        sl=float(final_state.stop_loss),
                        t1=float(final_state.t1),
                        t2=float(final_state.t2),
                        t3=float(final_state.t3),
                    )
            except Exception as e:
                logger.error(f"[TELEGRAM] Sinyal veritabanına kaydedilemedi: {e}")

        try:
            await progress_message.edit_text(
                format_oracle_response(final_state),
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.error(f"[TELEGRAM] Nihai mesaj gönderimi başarısız: {exc}")
            await progress_message.edit_text(
                f"⚠️ {symbol} analizi tamamlandı ancak çıktı mesajı üretilemedi.\n"
                f"Hata: {str(exc)[:300]}",
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
            await scanner._run_scan_once(notify_start=False)
        except Exception as exc:
            await update.effective_message.reply_text(f"Tarama hatasi: {exc}")

    async def command_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._deny_if_unauthorized(update):
            return
        if not update.effective_message:
            return

        asset_filter = _normalize_symbol(context.args[0]) if context.args else None
        db_path = Path("data") / "portfolio.db"

        if not db_path.exists():
            await update.effective_message.reply_text(
                "📊 /stats\nHenüz takip verisi yok. Önce en az bir sinyal üretilmeli."
            )
            return

        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            params: tuple = ()
            where = ""
            if asset_filter:
                where = "WHERE asset = ?"
                params = (asset_filter,)

            cur.execute(
                f"SELECT asset, direction, entry_price, stop_loss, t2, status, pnl, timestamp "
                f"FROM trades {where} ORDER BY id DESC LIMIT 30",
                params,
            )
            rows = cur.fetchall()
            conn.close()

            if not rows:
                target = asset_filter or "genel"
                await update.effective_message.reply_text(f"📊 /stats {target}\nKayıt bulunamadı.")
                return

            closed = [r for r in rows if str(r[5]).upper() in {"WIN", "LOSS"}]
            wins = sum(1 for r in closed if str(r[5]).upper() == "WIN")
            losses = sum(1 for r in closed if str(r[5]).upper() == "LOSS")
            closed_count = len(closed)
            win_rate = (wins / closed_count * 100.0) if closed_count else 0.0

            rr_values: list[float] = []
            for r in rows:
                entry, stop, t2 = float(r[2]), float(r[3]), float(r[4])
                risk = abs(entry - stop)
                if risk <= 0:
                    continue
                rr = abs(t2 - entry) / risk
                rr_values.append(rr)

            avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0.0
            best_pnl = max(float(r[6]) for r in rows)
            worst_pnl = min(float(r[6]) for r in rows)
            last = rows[0]
            last_status = "✅" if str(last[5]).upper() == "WIN" else ("❌" if str(last[5]).upper() == "LOSS" else "⏳")

            scope = asset_filter or "GENEL"
            msg = (
                f"📊 OLYMPUS STATS — {scope}\n"
                f"Son 30 kayıt: {len(rows)} | Kapanan: {closed_count}\n"
                f"İsabet: {wins} doğru / {losses} yanlış (%{win_rate:.1f})\n"
                f"Ort. R:R: {avg_rr:.2f}\n"
                f"En iyi PnL: %{best_pnl:+.2f} | En kötü PnL: %{worst_pnl:+.2f}\n"
                f"Son sinyal: {last[7]} | {last[0]} | {last[1]} | {last_status} {last[5]}"
            )
            await update.effective_message.reply_text(msg)
        except Exception as exc:
            await update.effective_message.reply_text(f"/stats hatası: {exc}")

    async def command_backtesting(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Geçmiş sinyal win-rate, ortalama R:R ve varlık bazlı başarı özeti."""
        if await self._deny_if_unauthorized(update):
            return
        if not update.effective_message:
            return
        db_path = Path("data") / "portfolio.db"
        if not db_path.exists():
            await update.effective_message.reply_text("📈 /backtesting\nHenüz kayıtlı sinyal yok.")
            return
        try:
            import sqlite3 as _sq
            conn = _sq.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT asset, direction, entry_price, stop_loss, t2, status, pnl FROM trades ORDER BY id DESC")
            rows = cur.fetchall()
            conn.close()
            if not rows:
                await update.effective_message.reply_text("📈 /backtesting\nKayıt bulunamadı.")
                return
            total = len(rows)
            closed = [r for r in rows if str(r[5]).upper() in {"WIN","LOSS"}]
            wins = sum(1 for r in closed if str(r[5]).upper() == "WIN")
            losses = len(closed) - wins
            win_rate = wins / len(closed) * 100 if closed else 0.0
            rr_vals = []
            for r in rows:
                entry, sl, t2 = float(r[2]), float(r[3]), float(r[4])
                risk = abs(entry - sl)
                if risk > 0:
                    rr_vals.append(abs(t2 - entry) / risk)
            avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 0.0
            # Per-asset breakdown
            from collections import defaultdict
            asset_wins: dict = defaultdict(int)
            asset_total: dict = defaultdict(int)
            for r in closed:
                asset_total[r[0]] += 1
                if str(r[5]).upper() == "WIN":
                    asset_wins[r[0]] += 1
            best_asset = max(asset_wins, key=lambda a: asset_wins[a] / max(asset_total[a],1), default="—")
            pnl_vals = [float(r[6]) for r in rows if r[6] is not None]
            cum_pnl = sum(pnl_vals)
            msg = (
                "📈 OLYMPUS BACKTESTING RAPORU\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Toplam Sinyal : {total}\n"
                f"Kapanan       : {len(closed)}  (Kazanç: {wins} | Kayıp: {losses})\n"
                f"Win-Rate      : %{win_rate:.1f}\n"
                f"Ort. R:R      : {avg_rr:.2f}\n"
                f"Kümülatif PnL : %{cum_pnl:+.2f}\n"
                f"En İyi Varlık : {best_asset}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 Detaylı tablo için /stats kullanın."
            )
            await update.effective_message.reply_text(msg)
        except Exception as exc:
            await update.effective_message.reply_text(f"/backtesting hatası: {exc}")

    async def command_detay(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/detay BTC — Bir varlık için tüm ajan skorları ve Fibonacci + Remora/Çakal detayı."""
        if await self._deny_if_unauthorized(update):
            return
        if not update.effective_message:
            return
        symbol = _normalize_symbol(context.args[0] if context.args else "BTC")
        if symbol == "__SCAN__":
            await update.effective_message.reply_text("Kullanım: /detay BTC")
            return
        progress = await update.effective_message.reply_text(f"⚙️ {symbol} detaylı analiz başlatıldı...")
        user_id = str(update.effective_user.id) if update.effective_user else ""
        chat_id = int(update.effective_chat.id) if update.effective_chat else 0
        try:
            state = await self._run_pipeline(symbol=symbol, user_id=user_id, chat_id=chat_id, query=f"/detay {symbol}")
            lines = [
                f"🔬 DETAYLI ANALİZ — {symbol}",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"Kompozit Skor : {state.composite_score*100:.1f}%",
                f"Güven         : {state.confidence*100:.1f}%",
                "",
                "📊 AJAN SKORLARI:",
                f"  Makro        : {state.macro_score:+.3f}  ({state.macro_score*100:+.1f}%)",
                f"  Quant/Teknik : {state.quant_score:+.3f}  ({state.quant_score*100:+.1f}%)",
                f"  Balina       : {state.whale_score:+.3f}  ({(state.whale_score or 0)*100:+.1f}%)" if state.whale_score is not None else "  Balina       : Veri yok (SSL fallback)",
                f"  Fundamental  : {state.fundamental_score:+.3f}  ({state.fundamental_score*100:+.1f}%)",
                f"  Sentiment    : {state.sentiment_score:+.3f}  ({state.sentiment_score*100:+.1f}%)",
                "",
                "📈 TIMEFRAME:",
            ]
            biases = state.timeframe_biases or {}
            for tf, label in [("1w","Haftalık"),("1d","Günlük"),("4h","4 Saat"),("1h","1 Saat")]:
                lines.append(f"  {label:10s}: {biases.get(tf,'?')}")
            lines += [
                "",
                "🎯 FİBONACCI SEVİYELERİ:",
                f"  Retracement 0.382 : {_format_price(state.fib_382)}",
                f"  Retracement 0.500 : {_format_price(state.fib_500)}",
                f"  Retracement 0.618 : {_format_price(state.fib_618)}  ⭐ Altın oran",
                f"  Uzantı 1.272 (T1) : {_format_price(state.fib_ext_1272)}",
                f"  Uzantı 1.618 (T2) : {_format_price(state.fib_ext_1618)}  ⭐ Altın uzantı",
                f"  Uzantı 2.618 (T3) : {_format_price(state.fib_ext_2618)}",
            ]
            # Whale strategy
            whale_strat = next((m for m in (state.messages or []) if m.startswith("[WHALE_STRATEGY]")), None)
            if whale_strat:
                lines += ["", "🐋 BALİNA STRATEJİSİ:", f"  {whale_strat.replace('[WHALE_STRATEGY] ','')}"]
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            await progress.edit_text("\n".join(lines), disable_web_page_preview=True)
        except Exception as exc:
            logger.error(f"[TELEGRAM] /detay hatası: {exc}")
            await progress.edit_text(f"⚠️ /detay {symbol} hatası: {str(exc)[:300]}")

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
        try:
            raw_result = await asyncio.wait_for(
                self._graph.ainvoke(initial_state),
                timeout=300.0,
            )
        except asyncio.TimeoutError:
            return OracleState(
                query=query,
                symbol=symbol,
                user_id=user_id,
                chat_id=chat_id,
                status=PipelineStatus.ABORTED,
                fatal_error=f"{symbol} analizi 5 dakika limitini aştı — API yanıtsız.",
                signal_direction=SignalDirection.NO_TRADE,
            )
        if isinstance(raw_result, OracleState):
            return raw_result
        return OracleState.model_validate(raw_result)

    def build(self) -> Application:
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("oracle", self.command_oracle))
        app.add_handler(CommandHandler("analiz", self.command_analiz))
        app.add_handler(CommandHandler("tarama", self.command_tarama))
        app.add_handler(CommandHandler("stats", self.command_stats))
        app.add_handler(CommandHandler("backtesting", self.command_backtesting))
        app.add_handler(CommandHandler("detay", self.command_detay))
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
