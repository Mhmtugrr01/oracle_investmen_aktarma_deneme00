"""DÜĞÜM 1 — Makro Likidite Ajanı (Macro Sentinel)."""

from __future__ import annotations

from core.console import BLUE, CYAN, agent_print, error_print
from core.types import AgentNode, OracleState, PipelineStatus
from core.indicators import normalized_from_score
from tools.market_data import fetch_macro_bundle, pct_change_over


def _compute_macro_score_0_100(
    dxy_chg: float,
    vix_chg: float,
    spy_chg: float,
    vix_level: float,
) -> tuple[float, list[str]]:
    """
    Makro skor (0-100).
    DXY/VIX yukselis = risk-off (dusuk skor), dusus = risk-on (yuksek skor).
    """
    score = 50.0
    notes: list[str] = []

    # DXY: guclu dolar kripto/risk varlik baskisi
    if dxy_chg > 1.0:
        score -= min(20.0, dxy_chg * 4)
        notes.append(f"DXY +{dxy_chg:.2f}% (risk-off)")
    elif dxy_chg < -1.0:
        score += min(20.0, abs(dxy_chg) * 4)
        notes.append(f"DXY {dxy_chg:.2f}% (dolar zayif)")

    # VIX: korku endeksi
    if vix_chg > 10.0:
        score -= min(18.0, vix_chg * 0.8)
        notes.append(f"VIX +{vix_chg:.2f}% (korku artisi)")
    elif vix_chg < -10.0:
        score += min(18.0, abs(vix_chg) * 0.8)
        notes.append(f"VIX {vix_chg:.2f}% (korku azalisi)")

    if vix_level > 30:
        score -= 10
        notes.append(f"VIX seviye {vix_level:.1f} (yuksek)")
    elif vix_level < 18:
        score += 6
        notes.append(f"VIX seviye {vix_level:.1f} (dusuk)")

    # SPY: genel risk iştahı proxy
    if spy_chg > 2.0:
        score += min(12.0, spy_chg * 2)
        notes.append(f"SPY +{spy_chg:.2f}% (risk-on)")
    elif spy_chg < -2.0:
        score -= min(12.0, abs(spy_chg) * 2)
        notes.append(f"SPY {spy_chg:.2f}% (risk-off)")

    return round(max(0.0, min(100.0, score)), 2), notes


async def run_macro_sentinel(state: OracleState) -> OracleState:
    cycle = state.retry_count + 1
    agent_print(
        "MACRO_SENTINEL",
        f"Devrede -> {state.symbol} | Rötüs dongusu #{cycle}",
        CYAN,
    )

    try:
        bundle = await fetch_macro_bundle()

        dxy_df = bundle["DXY"]
        vix_df = bundle["VIX"]
        spy_df = bundle["SPY"]

        dxy_chg = pct_change_over(dxy_df, bars=5)
        vix_chg = pct_change_over(vix_df, bars=5)
        spy_chg = pct_change_over(spy_df, bars=5)
        vix_level = float(vix_df["close"].iloc[-1])
        dxy_price = float(dxy_df["close"].iloc[-1])

        score_0_100, notes = _compute_macro_score_0_100(
            dxy_chg, vix_chg, spy_chg, vix_level
        )
        macro_score = normalized_from_score(score_0_100)

        agent_print(
            "MACRO_SENTINEL",
            f"DXY={dxy_price:.2f} ({dxy_chg:+.2f}%) | VIX={vix_level:.2f} ({vix_chg:+.2f}%)",
            BLUE,
        )
        agent_print(
            "MACRO_SENTINEL",
            f"SPY 5g degisim={spy_chg:+.2f}% | Makro Skor={score_0_100:.1f}/100 -> {macro_score:+.3f}",
            BLUE,
        )
        for note in notes:
            agent_print("MACRO_SENTINEL", note, CYAN)

        return state.model_copy(
            update={
                "current_node": AgentNode.MACRO_SENTINEL,
                "status": PipelineStatus.RUNNING,
                "macro_score": macro_score,
                "messages": [
                    f"[MACRO_SENTINEL] DXY={dxy_price:.2f} VIX={vix_level:.2f} "
                    f"score={score_0_100:.1f} norm={macro_score:+.3f}"
                ],
            }
        )

    except Exception as exc:
        msg = f"Makro Hata Olustu: {exc}"
        error_print(msg)
        return state.model_copy(
            update={
                "current_node": AgentNode.MACRO_SENTINEL,
                "status": PipelineStatus.RUNNING,
                "fatal_error": msg,
                "messages": [f"[MACRO_SENTINEL] ERROR {msg}"],
            }
        )
