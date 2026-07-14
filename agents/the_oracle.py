"""DÜĞÜM 0 — The Oracle (CEO Yönetici)."""

from __future__ import annotations

import asyncio

from core.config import load_oracle_config
from core.console import CYAN, GREEN, YELLOW, agent_print, warn_print
from core.types import AgentNode, OracleState, PipelineStatus, SignalDirection


async def run_the_oracle(state: OracleState) -> OracleState:
    # CEO Raporlama Notunun Tanımsızlık (NameError) Kalkanı
    note = "Piyasa süzgeçleri kararlı. Sinyal kalitesi onaylandı."
    agent_print(
        "THE_ORACLE",
        f"CEO denetimi → {state.symbol} | Tüm ajan raporları birleştiriliyor…",
        CYAN,
    )
    await asyncio.sleep(0.2)

    conf = await load_oracle_config()
    ceo_conf = conf.ceo
    risk_conf = conf.risk
    conf_map = conf.model_dump()

    # Integrate new kinetic score (if present) as a soft augmentation to quant_score
    kinetic = float(getattr(state, "kinetic_score", 0.0) or 0.0)
    quant_base = float(getattr(state, "quant_score", 0.0) or 0.0)
    # conservative blend: 80% original quant_score, 20% kinetic influence
    quant_combined = (quant_base * 0.80) + (kinetic * 0.20)

    scores = [
        state.macro_score,
        quant_combined,
        state.whale_score if state.whale_score is not None else 0.0,
        state.fundamental_score,
        state.sentiment_score,
    ]
    consensus_variance = max(scores) - min(scores)
    composite = state.composite_score
    base_rr = state.base_rr

    # Tüm ajan skorları bilindikten sonra confidence yeniden hesapla
    _alignment = float(state.timeframe_alignment_score or 0.5)
    _composite_abs = abs(float(composite))
    _hist = float(getattr(state, "historical_similarity_score", 0.0) or 0.0)
    _div_d = getattr(state, "divergence_daily", "NONE") or "NONE"
    _div_w = getattr(state, "divergence_weekly", "NONE") or "NONE"
    _base_conf = (_alignment * 0.50) + (_composite_abs * 0.30)
    _var_pen = min(consensus_variance * 0.08, 0.20)
    _div_bon = (0.06 if _div_d in ["POSITIVE_DIVERGENCE", "NEGATIVE_DIVERGENCE"] else 0.0)
    _div_bon += (0.08 if _div_w in ["POSITIVE_DIVERGENCE", "NEGATIVE_DIVERGENCE"] else 0.0)
    _hist_bon = (_hist / 100.0) * 0.10
    _actual_conf = round(max(0.0, min(1.0, _base_conf - _var_pen + _div_bon + _hist_bon)), 3)

    # ── Multi-TF hiyerarşi bonusu: haftalık + günlük aynı yön → güçlü teyit ─
    _tf_biases = state.timeframe_biases or {}
    _w_bias = str(_tf_biases.get("1w", "NEUTRAL")).upper()
    _d_bias = str(_tf_biases.get("1d", "NEUTRAL")).upper()
    _BULL_SET = {"BULLISH", "OVERSOLD", "ACCUMULATING"}
    _BEAR_SET = {"BEARISH", "OVERBOUGHT", "DISTRIBUTING"}
    if (_w_bias in _BULL_SET and _d_bias in _BULL_SET) or (_w_bias in _BEAR_SET and _d_bias in _BEAR_SET):
        _actual_conf = round(min(1.0, _actual_conf + 0.08), 3)   # HTF consensus bonus
    elif (_w_bias in _BULL_SET and _d_bias in _BEAR_SET) or (_w_bias in _BEAR_SET and _d_bias in _BULL_SET):
        _actual_conf = round(max(0.0, _actual_conf - 0.05), 3)   # HTF conflict penalty

    # ── Extreme Fear / Greed Contrarian Bonusu (composite ile simetrik) ──────
    _fg = getattr(state, "fear_greed_value", None)
    if _fg is not None:
        if int(_fg) <= 25:
            _actual_conf = round(min(1.0, _actual_conf + 0.06), 3)   # Extreme Fear = güçlü contrarian
        elif int(_fg) >= 75:
            _actual_conf = round(max(0.0, _actual_conf - 0.04), 3)   # Extreme Greed = dikkat
    # Tarihsel benzerlik bonusu
    if _hist >= 75.0 and str(getattr(state, "pattern_outcome_bias", "") or "").upper() == "HISTORICALLY_BULLISH":
        _actual_conf = round(min(1.0, _actual_conf + 0.03), 3)
    elif _hist >= 75.0 and str(getattr(state, "pattern_outcome_bias", "") or "").upper() == "HISTORICALLY_BEARISH":
        _actual_conf = round(min(1.0, _actual_conf + 0.02), 3)  # uncertain ama tarihsel pattern var

    state = state.model_copy(update={"confidence": _actual_conf})

    if (state.confidence or 0.0) == 0.0:
        _comp = abs(state.composite_score)
        _align = state.timeframe_alignment_score if state.timeframe_alignment_score is not None else 0.5
        state = state.model_copy(
            update={
                "confidence": min((_comp * 0.6) + (_align * 0.4), 1.0),
            }
        )

    if base_rr is None:
        reason = "ATR tabanli base_rr bulunamadi."
        warn_print(f"CEO RED → {reason}")
        return state.model_copy(
            update={
                "current_node": AgentNode.THE_ORACLE,
                "status": PipelineStatus.ABORTED,
                "fatal_error": reason,
                "ceo_approved": False,
                "ceo_revision_reason": reason,
                "messages": [f"[THE_ORACLE] FATAL {reason}"],
            }
        )

    agent_print(
        "THE_ORACLE",
        f"Consensus variance={consensus_variance:.2f} | Kompozit={composite:+.2f} | base_rr={base_rr}",
        CYAN,
    )

    inconsistent = consensus_variance > ceo_conf.max_score_spread
    low_rr = base_rr < risk_conf.min_risk_reward_ratio
    low_composite = composite < (ceo_conf.min_composite_score - 1e-9)
    effective_confidence_threshold = (
        0.50 if float(state.base_rr or 0.0) > 5.0
        else ceo_conf.confidence_threshold
    )
    low_confidence = _actual_conf < (effective_confidence_threshold - 1e-9)

    # ── Gri Bölge: 0.52-0.60 arası = "Piyasa Kararsız" — net sinyal değil ──────
    # Yüksek R:R (>7.0) veya makro düşük risk ortamı bu bölgeyi geçebilir.
    in_grey_zone = (
        ceo_conf.min_composite_score <= composite <= 0.60
        and (base_rr or 0.0) < 7.0
        and not low_composite
    )
    if in_grey_zone:
        grey_reason = (
            f"• Gri Bölge (Kararsız Piyasa): Kompozit skor (%{composite*100:.0f}) netlik eşiğinin (%60) altında. "
            f"R:R {base_rr:.2f} ile mevcut sinyal yeterince asimetrik değil. "
            "Net kırılım bekleniyor — işlem yapılmadı."
        )
        reason_parts = [grey_reason]
        reason = "\n".join(reason_parts)
        new_retry = state.retry_count + 1
        warn_print(f"CEO GRİ BÖLGE → {reason} | Rötuş #{new_retry}/1")
        return state.model_copy(
            update={
                "current_node": AgentNode.THE_ORACLE,
                "status": PipelineStatus.ABORTED,
                "fatal_error": reason,
                "ceo_approved": False,
                "ceo_revision_reason": reason,
                "messages": [f"[THE_ORACLE] GREY_ZONE composite={composite:.3f} rr={base_rr}"],
            }
        )

    # ── Ekonomik Takvim Engeli: Kritik veri günleri eşiği yükseltilir ──────────
    high_impact_event_today = any(
        "[EKONOMİK TAKVİM]" in m and "YÜKSEK" in m
        for m in state.messages
    )
    if high_impact_event_today and composite < 0.70:
        econ_reason = (
            f"• Ekonomik Takvim Engeli: Bugün yüksek etkili makro veri açıklaması var. "
            f"Kritik veri günlerinde kompozit eşiği %70'e yükseltildi. "
            f"Mevcut kompozit (%{composite*100:.0f}) bu eşiği geçemiyor."
        )
        return state.model_copy(
            update={
                "current_node": AgentNode.THE_ORACLE,
                "status": PipelineStatus.ABORTED,
                "fatal_error": econ_reason,
                "ceo_approved": False,
                "messages": [f"[THE_ORACLE] ECON_CALENDAR_VETO composite={composite:.3f}"],
            }
        )

    if inconsistent or low_rr or low_composite or low_confidence:
        reason_parts = []
        if inconsistent:
            reason_parts.append(f"• Ajan Tutarsızlığı: Ajan kararları arasında yüksek sapma (variance: {consensus_variance:.2f}) saptandı.")
        if low_rr:
            reason_parts.append(f"• Yetersiz Asimetri: Mevcut R:R oranı ({base_rr:.1f}), belirlenen minimum asimetri eşiğinin ({risk_conf.min_risk_reward_ratio}) altında.")
        if low_composite:
            reason_parts.append(f"• Düşük Kompozit Güven: Yapay Zeka kompozit skoru (%{composite*100:.0f}), asgari mühürleme limiti olan (%{ceo_conf.min_composite_score*100:.0f})'un altında.")
        if low_confidence:
            reason_parts.append(f"• Güven Sınırı Engeli: Sistem güven oranı (%{state.confidence*100:.0f}), dinamik risk eşiğinin ({effective_confidence_threshold:.2f}) altında kaldı.")
            
        # ── 🛡️ ŞEFFAF İPTAL RAPORLAMA PROTOKOLÜ: LİKİDİTE ENGELİ (R03) ──
        # Eğer USDT.D veya Yen carry-trade baskısı yüzünden kompozit skor ezildiyse şeffafça bildir!
        for msg in state.messages:
            if "USDT.D YÜKSELİYOR" in msg:
                reason_parts.append("• Tether Dominans Engeli: USDT.D yükseliyor, likidite nakde kaçtığı için katsayı tırpanlandı.")
            if "Japon Yeni" in msg or "Yen" in msg:
                reason_parts.append("• Küresel Likidite Engeli: JPY/USD carry trade tasfiyesi nedeniyle makro kilit devrede.")
                
        reason = "\n".join(reason_parts)

        new_retry = state.retry_count + 1
        warn_print(
            f"CEO RED → {reason} | Rötuş #{new_retry}/3"
        )
        agent_print(
            "THE_ORACLE",
            "Koşullu edge: analiz döngüsü başa sarılıyor…",
            YELLOW,
        )

        return state.model_copy(
            update={
                "current_node": AgentNode.THE_ORACLE,
                "status": PipelineStatus.RUNNING,
                "retry_count": new_retry,
                "ceo_approved": False,
                "ceo_revision_reason": reason,
                "base_rr": base_rr,
                "risk_reward_ratio": base_rr,
                "confidence": abs(composite),
                "messages": [f"[THE_ORACLE] RED retry={new_retry} reason={reason}"],
            }
        )

    tf_biases = state.timeframe_biases or {}
    # ── Bias oylarından yön belirle (trade_type/signal_label GÖRMEZDEN GEL) ──
    _b = [
        (tf_biases.get("1w", "NEUTRAL") or "NEUTRAL").upper(),
        (tf_biases.get("1d", "NEUTRAL") or "NEUTRAL").upper(),
        (tf_biases.get("4h", "NEUTRAL") or "NEUTRAL").upper(),
        (tf_biases.get("1h", "NEUTRAL") or "NEUTRAL").upper(),
    ]
    _BULL = {"BULLISH", "STRONGLY_BULLISH", "OVERSOLD", "ACCUMULATING"}
    _BEAR = {"BEARISH", "STRONGLY_BEARISH", "OVERBOUGHT", "DISTRIBUTING"}
    _nb = sum(1 for x in _b if x in _BULL)
    _ns = sum(1 for x in _b if x in _BEAR)

    # ── Yön Kararı: Composite Skoru Ŝırpılayıcı (Direction Gate) ───────────────
    # Composite > 0.50 = net bullish ortam → SHORT üretilemez.
    # SHORT için hem bias çoğunluğu hem composite < 0.50 şartı aranacak.
    composite_is_bullish = float(composite) > 0.50
    if _nb > _ns or (composite_is_bullish and _nb == _ns):
        direction = SignalDirection.LONG
        _sig_label = "LONG_FIRSAT"
    elif _ns > _nb and not composite_is_bullish:
        direction = SignalDirection.SHORT
        _sig_label = "SHORT_FIRSAT"
    else:
        # Tüm diğer durumlar: composite yönü baz al
        direction = SignalDirection.LONG if composite_is_bullish else SignalDirection.SHORT
        _sig_label = "LONG_FIRSAT" if composite_is_bullish else "SHORT_FIRSAT"

    # Veri Şatılını çöz ve CEO'nun seçtiği yöne ait matematiksel seviyeleri yükle!
    # Veri Şatılını çöz ve CEO'nun seçtiği yöne ait matematiksel seviyeleri yükle!
    entry_val, stop_val, t1_val, t2_val, t3_val, base_rr_val = None, None, None, None, None, None
    dynamic_target = None
    usdt_d_modifier = 1.0
    
    for msg in state.messages:
        if msg.startswith("[LEVELS_DATA]"):
            try:
                import json
                # Quant sends pure JSON: {"LONG": {...}, "SHORT": {...}, "FIB": {...}}
                raw_json = msg[len("[LEVELS_DATA] "):]
                data = json.loads(raw_json)
                direction_key = "LONG" if direction == SignalDirection.LONG else "SHORT"
                chosen_data = data[direction_key]

                entry_val = chosen_data.get("entry_zone_low")
                stop_val = chosen_data.get("stop_loss")
                t1_val = chosen_data.get("t1")
                t2_val = chosen_data.get("t2")
                t3_val = chosen_data.get("t3")
                base_rr_val = chosen_data.get("base_rr")
            except Exception as e:
                logger.error(f"[THE_ORACLE] LEVELS_DATA çözme hatası: {e}")
        elif msg.startswith("[DYNAMIC_TARGET]"):
            try:
                dynamic_target = float(msg.replace("[DYNAMIC_TARGET] ", ""))
            except Exception:
                pass
        elif msg.startswith("[USDT_D_MODIFIER]"):
            try:
                usdt_d_modifier = float(msg.replace("[USDT_D_MODIFIER] ", ""))
            except Exception:
                pass

    # ── 🛡️ GEOMETRİK EXIT & TEPE KOKLAMA KONTROLÜ (MSTR Target Devrimi) ──
    # Eğer fiyat yukarıdan gelen düşen dirence çarpmak üzereyse kâr almayı zorunlu kıl!
    note = "Piyasa süzgeçleri kararlı. Pozisyon disiplinli risk yönetimiyle anlamlı."
    if direction == SignalDirection.LONG and dynamic_target is not None and state.entry_price is not None:
        distance_pct = abs(state.entry_price - dynamic_target) / dynamic_target * 100.0
        # Fiyat dirence %1.5 yakınsa veya Günlük RSI Negatif Uyumsuzluğu varsa kâr aldır!
        if distance_pct <= 1.5 or _div_d == "NEGATIVE_DIVERGENCE":
            _sig_label = "REDUCE_EXPOSURE"
            note = f"⚠️ DÜŞEN DİRENÇ SINIRINDASINIZ ({dynamic_target:.1f})! RSI negatif uyumsuzluğu var. Kâr alım (Exit) planı devrede."

    htf_warnings: list[str] = []

    merged_warnings = list(state.cross_asset_warnings or [])
    merged_warnings.extend(htf_warnings)

    rr_for_alpha = float(getattr(state, "base_rr", base_rr) or base_rr or 0.0)
    alpha = (
        f"{_sig_label} {state.symbol} | R:R={rr_for_alpha:.2f}"
        f" | Güven={_actual_conf:.0%}"
        f" | Pattern={getattr(state, 'historical_pattern', 'N/A')}"
    )

    agent_print("THE_ORACLE", "CEO ONAY → Red Team'e sevk ediliyor.", GREEN)
    agent_print("THE_ORACLE", f"Alpha taslağı: {alpha}", GREEN)

    return state.model_copy(
        update={
            "current_node": AgentNode.THE_ORACLE,
            "status": PipelineStatus.RUNNING,
            "composite_score": composite,
            "consensus_variance": consensus_variance,
            "confidence": _actual_conf,
            "ceo_approved": True,
            "signal_direction": direction,
            "signal_label": _sig_label,
            "oracle_note": note,
            "entry_price": entry_val if entry_val is not None else state.entry_price,
            "stop_loss": stop_val if stop_val is not None else state.stop_loss,
            "t1": t1_val if t1_val is not None else state.t1,
            "t2": t2_val if t2_val is not None else state.t2,
            "t3": t3_val if t3_val is not None else state.t3,
            "base_rr": base_rr_val if base_rr_val is not None else state.base_rr,
            "risk_reward_ratio": base_rr_val if base_rr_val is not None else state.base_rr,
        }
    )
