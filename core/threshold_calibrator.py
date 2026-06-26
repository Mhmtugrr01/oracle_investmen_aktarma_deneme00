"""Test set (2025+) uzerinde esik kalibrasyonu.

- composite_4factor, confidence, consensus_variance metriklerini decile bazli inceler.
- Pozitif beklenen degerin basladigi dilime gore esik onerir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BACKTEST_PATH = Path("data/backtest_history.csv")
WEIGHTS_PATH = Path("data/optimal_weights.json")
OUT_PATH = Path("data/threshold_recommendations.json")


def _load() -> tuple[pd.DataFrame, dict]:
    if not BACKTEST_PATH.exists():
        raise FileNotFoundError(f"Eksik dosya: {BACKTEST_PATH}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Eksik dosya: {WEIGHTS_PATH}")

    df = pd.read_csv(BACKTEST_PATH)
    weights_payload = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    return df, weights_payload


def _calc_weighted(df: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    return (
        df["macro_score"] * float(w.get("macro", 0.0))
        + df["quant_score"] * float(w.get("quant", 0.0))
        + df["fundamental_score"] * float(w.get("fundamental", 0.0))
        + df["sentiment_score"] * float(w.get("sentiment", 0.0))
    )


def _decile_table(df: pd.DataFrame, metric: str, higher_better: bool) -> tuple[pd.DataFrame, float | None]:
    d = df[[metric, "forward_return_30d"]].dropna().copy()
    if d.empty:
        return pd.DataFrame(), None

    d["decile"] = pd.qcut(d[metric], q=10, duplicates="drop")
    table = (
        d.groupby("decile", observed=True)
        .agg(
            count=("forward_return_30d", "size"),
            mean_forward_return_30d=("forward_return_30d", "mean"),
            metric_min=(metric, "min"),
            metric_max=(metric, "max"),
        )
        .reset_index(drop=True)
    )

    recommendation = None
    if higher_better:
        positive = table[table["mean_forward_return_30d"] > 0]
        if not positive.empty:
            recommendation = float(positive["metric_min"].iloc[0])
    else:
        positive = table[table["mean_forward_return_30d"] > 0]
        if not positive.empty:
            recommendation = float(positive["metric_max"].max())

    return table, recommendation


def run_calibration() -> Path:
    df, payload = _load()

    if "date" not in df.columns:
        raise ValueError("backtest_history.csv dosyasinda date kolonu yok")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    test = df[df["date"].dt.year >= 2025].copy()

    w = payload.get("weights", {})
    test["composite_4factor_opt"] = _calc_weighted(test, w)

    # confidence / consensus_variance kolonlari yoksa toleransli fallback
    if "confidence" not in test.columns:
        test["confidence"] = pd.NA
    if "consensus_variance" not in test.columns:
        test["consensus_variance"] = pd.NA

    comp_table, comp_thr = _decile_table(test, "composite_4factor_opt", higher_better=True)
    conf_table, conf_thr = _decile_table(test, "confidence", higher_better=True)
    var_table, var_thr = _decile_table(test, "consensus_variance", higher_better=False)

    print("\nComposite decile analizi (test 2025+):")
    print(comp_table.to_string(index=False) if not comp_table.empty else "veri yok")

    print("\nConfidence decile analizi (test 2025+):")
    print(conf_table.to_string(index=False) if not conf_table.empty else "veri yok")

    print("\nConsensus variance decile analizi (test 2025+):")
    print(var_table.to_string(index=False) if not var_table.empty else "veri yok")

    print("\nEsik onerileri:")
    print(f"min_composite_score onerisi: {comp_thr}")
    print(f"confidence_threshold onerisi: {conf_thr}")
    print(f"max_consensus_variance onerisi: {var_thr}")

    out = {
        "min_composite_score_suggested": comp_thr,
        "confidence_threshold_suggested": conf_thr,
        "max_consensus_variance_suggested": var_thr,
        "weights_used": w,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kaydedildi: {OUT_PATH}")

    return OUT_PATH


def main() -> None:
    run_calibration()


if __name__ == "__main__":
    main()
