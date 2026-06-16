"""DÜĞÜM 2 — Kantitatif Teknik Motor (Quant Technical)."""

from __future__ import annotations

from core.console import BLUE, GREEN, agent_print, error_print
from core.indicators import (
    build_quant_score,
    calculate_atr_stop_loss,
    calculate_fibonacci_levels,
    calculate_rsi,
    detect_rsi_divergence,
    ma_ema_cross,
    nearest_fib_level,
    normalized_from_score,
)
from core.types import AgentNode, OracleState, PipelineStatus
from tools.market_data import fetch_crypto_ohlcv


async def run_quant_engine(state: OracleState) -> OracleState:
    agent_print(
        "QUANT_ENGINE",
        f"Devrede -> {state.symbol} | Gercek OHLCV + indikator analizi...",
        GREEN,
    )

    try:
        df = await fetch_crypto_ohlcv(state.symbol, timeframe="4h", limit=200)

        rsi = calculate_rsi(df)
        cross = ma_ema_cross(df, fast_period=9, slow_period=21, ema=True)
        divergence = detect_rsi_divergence(df, rsi_series=rsi["series"])

        lookback = min(60, len(df))
        swing_high = float(df["high"].tail(lookback).max())
        swing_low = float(df["low"].tail(lookback).min())
        trend = (
            "up"
            if float(df["close"].iloc[-1]) >= float(df["close"].iloc[-lookback])
            else "down"
        )
        fib = calculate_fibonacci_levels(swing_high, swing_low, trend=trend)

        last_price = float(df["close"].iloc[-1])
        fib_key, fib_price, fib_dist = nearest_fib_level(last_price, fib)

        side = "long" if cross["signal"] in ("golden_cross", "bullish") else "short"
        if divergence["bullish"]:
            side = "long"
        elif divergence["bearish"]:
            side = "short"

        atr = calculate_atr_stop_loss(df, period=14, multiplier=2.0, side=side)

        score_0_100, notes = build_quant_score(
            rsi=rsi,
            cross=cross,
            divergence=divergence,
            fib_proximity_pct=fib_dist,
            atr_rr=atr["risk_reward"],
        )
        quant_score = normalized_from_score(score_0_100)

        agent_print(
            "QUANT_ENGINE",
            f"OHLCV={len(df)} bar (4h) | Son fiyat={last_price:.6f}",
            GREEN,
        )
        agent_print(
            "QUANT_ENGINE",
            f"RSI={rsi['rsi']} ({rsi['zone']}) | {cross['type']}={cross['signal']}",
            BLUE,
        )
        agent_print(
            "QUANT_ENGINE",
            f"Fib={fib_key} @ {fib_price:.6f} (uzaklik %{fib_dist}) | Div={divergence['divergence']}",
            BLUE,
        )
        agent_print(
            "QUANT_ENGINE",
            f"ATR stop={atr['stop_loss']:.6f} TP={atr['take_profit']:.6f} R:R={atr['risk_reward']}",
            GREEN,
        )
        agent_print(
            "QUANT_ENGINE",
            f"Quant Skor={score_0_100:.1f}/100 -> {quant_score:+.3f}",
            GREEN,
        )
        for note in notes:
            agent_print("QUANT_ENGINE", note, BLUE)

        return state.model_copy(
            update={
                "current_node": AgentNode.QUANT_ENGINE,
                "status": PipelineStatus.RUNNING,
                "quant_score": quant_score,
                "entry_price": atr["entry"],
                "stop_loss": atr["stop_loss"],
                "take_profit": atr["take_profit"],
                "risk_reward_ratio": atr["risk_reward"],
                "messages": [
                    f"[QUANT_ENGINE] RSI={rsi['rsi']} {cross['signal']} "
                    f"score={score_0_100:.1f} norm={quant_score:+.3f} "
                    f"fib={fib_key}"
                ],
            }
        )

    except Exception as exc:
        msg = f"Quant Hata Olustu: {exc}"
        error_print(msg)
        return state.model_copy(
            update={
                "current_node": AgentNode.QUANT_ENGINE,
                "status": PipelineStatus.RUNNING,
                "fatal_error": msg,
                "messages": [f"[QUANT_ENGINE] ERROR {msg}"],
            }
        )
