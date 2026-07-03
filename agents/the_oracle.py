"""DÜĞÜM 0 — The Oracle (CEO Yönetici)."""

from __future__ import annotations

import asyncio

from core.config import load_oracle_config
from core.console import CYAN, GREEN, YELLOW, agent_print, warn_print
from core.types import AgentNode, OracleState, PipelineStatus, SignalDirection


async def run_the_oracle(state: OracleState) -> OracleState:
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

    scores = [
        state.macro_score,
        state.quant_score,
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

    if inconsistent or low_rr or low_composite or low_confidence:
        reason_parts = []
        if inconsistent:
            reason_parts.append(f"ajan skorları tutarsız (consensus_variance={consensus_variance:.2f})")
        if low_rr:
            reason_parts.append(f"R:R yetersiz ({base_rr} < {risk_conf.min_risk_reward_ratio})")
        if low_composite:
            reason_parts.append(f"kompozit skor düşük ({composite:.2f} < {ceo_conf.min_composite_score})")
        if low_confidence:
            reason_parts.append(
                f"güven eşiği düşük ({state.confidence:.2f} < {effective_confidence_threshold})"
            )
        reason = " + ".join(reason_parts)

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
    _BULL = {"BULLISH", "STRONGLY_BULLISH", "OVERBOUGHT"}
    _BEAR = {"BEARISH", "STRONGLY_BEARISH", "OVERSOLD"}
    _nb = sum(1 for x in _b if x in _BULL)
    _ns = sum(1 for x in _b if x in _BEAR)
    # ── 🛡️ DUAL-CONCURRENCE LOOP: YÖN VE SEVİYE MUTABAKATI (R03 Phase 2) ──
    # CEO yönü, bias oylarına ve kompozit skora göre özgürce mühürler!
    if _nb > _ns:
        direction = SignalDirection.LONG
        _sig_label = "LONG_FIRSAT"
    elif _ns > _nb:
        direction = SignalDirection.SHORT
        _sig_label = "SHORT_FIRSAT"
    else:
        direction = SignalDirection.LONG if float(composite) >= 0 else SignalDirection.SHORT
        _sig_label = "LONG_FIRSAT" if float(composite) >= 0 else "SHORT_FIRSAT"

    # Veri Şatılını çöz ve CEO'nun seçtiği yöne ait matematiksel seviyeleri yükle!
    entry_val, stop_val, t1_val, t2_val, t3_val, base_rr_val = None, None, None, None, None, None
    for msg in state.messages:
        if msg.startswith("[LEVELS_DATA]"):
            try:
                import json
                parts = msg.replace("[LEVELS_DATA] ", "").split("|")
                long_part = parts[0].replace("LONG:", "")
                short_part = parts[1].replace("SHORT:", "")
                
                chosen_data = json.loads(long_part) if direction == SignalDirection.LONG else json.loads(short_part)
                
                entry_val = chosen_data["entry_zone_low"]
                stop_val = chosen_data["stop_loss"]
                t1_val = chosen_data["t1"]
                t2_val = chosen_data["t2"]
                t3_val = chosen_data["t3"]
                base_rr_val = chosen_data["base_rr"]
            except Exception as e:
                logger.error(f"[THE_ORACLE] Şatıl çözme hatası: {e}")
            break

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
            "consensus_variance": variance,
            "confidence": confidence,
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
