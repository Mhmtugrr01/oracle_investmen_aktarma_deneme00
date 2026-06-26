"""Tek seferlik ham skor diagnostigi.

- oracle_config.yaml -> asset_universe tum varliklari tarar
- compile_oracle_graph ile pipeline'i calistirir
- CEO revize dongusune girmeden (retry yok) bir gecis yapar
- Ham skorlari data/diagnostic_run.csv dosyasina yazar
- Skorlar icin min/max/ortalama/std ozetini ekrana basar
"""

from __future__ import annotations

import asyncio
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from core.config import load_oracle_config
from core.graph import compile_oracle_graph
from core.types import OracleState


CSV_COLUMNS = [
    "symbol",
    "macro_score",
    "quant_score",
    "whale_score",
    "fundamental_score",
    "sentiment_score",
    "composite_score",
    "consensus_variance",
    "confidence",
    "base_rr",
    "timeframe_alignment_score",
    "fatal_error",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_asset_universe(asset_universe: dict[str, list[str]]) -> list[str]:
    symbols: list[str] = []
    for _, assets in asset_universe.items():
        symbols.extend(assets)
    return symbols


def _calc_consensus_variance(state: OracleState) -> float:
    scores = [
        _safe_float(state.macro_score) or 0.0,
        _safe_float(state.quant_score) or 0.0,
        _safe_float(state.whale_score) or 0.0,
        _safe_float(state.fundamental_score) or 0.0,
        _safe_float(state.sentiment_score) or 0.0,
    ]
    return float(max(scores) - min(scores))


def _format_metric(name: str, values: list[float | None]) -> str:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return f"{name}: no_data"
    mn = min(clean)
    mx = max(clean)
    avg = mean(clean)
    sd = stdev(clean) if len(clean) > 1 else 0.0
    return f"{name}: min={mn:.4f} max={mx:.4f} mean={avg:.4f} std={sd:.4f}"


async def run_diagnostic() -> Path:
    conf = await load_oracle_config(force_reload=True)
    graph = compile_oracle_graph()

    symbols = _flatten_asset_universe(conf.asset_universe)
    max_retries = int(conf.ceo.max_retries)

    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        initial = OracleState(
            symbol=symbol,
            query="diagnostic_run",
            retry_count=max(0, max_retries - 1),
        )

        raw = await graph.ainvoke(initial)
        state = raw if isinstance(raw, OracleState) else OracleState.model_validate(raw)

        row = {
            "symbol": symbol,
            "macro_score": _safe_float(state.macro_score),
            "quant_score": _safe_float(state.quant_score),
            "whale_score": _safe_float(state.whale_score),
            "fundamental_score": _safe_float(state.fundamental_score),
            "sentiment_score": _safe_float(state.sentiment_score),
            "composite_score": _safe_float(state.composite_score),
            "consensus_variance": _safe_float(getattr(state, "consensus_variance", None))
            or _calc_consensus_variance(state),
            "confidence": _safe_float(state.confidence),
            "base_rr": _safe_float(state.base_rr),
            "timeframe_alignment_score": _safe_float(state.timeframe_alignment_score),
            "fatal_error": state.fatal_error or "",
        }
        rows.append(row)

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnostic_run.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDiagnostic tamamlandi. Dosya: {out_path}")
    print(f"Toplam varlik: {len(rows)}")

    metric_names = [
        "macro_score",
        "quant_score",
        "whale_score",
        "fundamental_score",
        "sentiment_score",
        "composite_score",
        "consensus_variance",
        "confidence",
        "base_rr",
        "timeframe_alignment_score",
    ]

    print("\nSkor ozetleri:")
    for name in metric_names:
        print(_format_metric(name, [_safe_float(r.get(name)) for r in rows]))

    whale_values = [_safe_float(r.get("whale_score")) for r in rows]
    whale_zero = sum(1 for v in whale_values if v is not None and v == 0.0)
    whale_neg_0360 = sum(1 for v in whale_values if v is not None and v == -0.360)

    print("\nWhale donma kontrolleri:")
    print(f"whale_score == 0.0 sayisi: {whale_zero}")
    print(f"whale_score == -0.360 sayisi: {whale_neg_0360}")

    return out_path


def main() -> None:
    asyncio.run(run_diagnostic())


if __name__ == "__main__":
    main()
