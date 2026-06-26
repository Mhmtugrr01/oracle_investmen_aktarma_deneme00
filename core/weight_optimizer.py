"""4 faktor agirlik optimizasyonu (grid search, Spearman IC)."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd


BACKTEST_PATH = Path("data/backtest_history.csv")
OUT_PATH = Path("data/optimal_weights.json")


def _spearman_ic(df: pd.DataFrame, weights: dict[str, float]) -> float:
    d = df.copy()
    d = d.dropna(
        subset=[
            "macro_score",
            "quant_score",
            "fundamental_score",
            "sentiment_score",
            "forward_return_30d",
        ]
    )
    if d.empty:
        return float("nan")

    score = (
        d["macro_score"] * weights["macro"]
        + d["quant_score"] * weights["quant"]
        + d["fundamental_score"] * weights["fundamental"]
        + d["sentiment_score"] * weights["sentiment"]
    )
    return float(score.corr(d["forward_return_30d"], method="spearman"))


def run_optimization() -> Path:
    if not BACKTEST_PATH.exists():
        raise FileNotFoundError(f"Eksik dosya: {BACKTEST_PATH}")

    df = pd.read_csv(BACKTEST_PATH)
    if "date" not in df.columns:
        raise ValueError("backtest_history.csv dosyasinda date kolonu yok")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    train = df[(df["date"].dt.year >= 2022) & (df["date"].dt.year <= 2024)].copy()
    test = df[df["date"].dt.year >= 2025].copy()

    step_values = [i / 20 for i in range(21)]  # 0.00..1.00, 0.05 adim

    best_weights: dict[str, float] | None = None
    best_train_ic = float("-inf")

    for w_macro, w_quant, w_fund, w_sent in product(step_values, repeat=4):
        if abs((w_macro + w_quant + w_fund + w_sent) - 1.0) > 1e-9:
            continue

        weights = {
            "macro": w_macro,
            "quant": w_quant,
            "fundamental": w_fund,
            "sentiment": w_sent,
        }
        train_ic = _spearman_ic(train, weights)
        if pd.isna(train_ic):
            continue

        if train_ic > best_train_ic:
            best_train_ic = train_ic
            best_weights = weights

    if best_weights is None:
        raise RuntimeError("Uygun agirlik kombinasyonu bulunamadi")

    test_ic = _spearman_ic(test, best_weights)

    result = {
        "weights": best_weights,
        "train_ic_spearman": float(best_train_ic),
        "test_ic_spearman": float(test_ic) if not pd.isna(test_ic) else None,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("En iyi agirliklar:", best_weights)
    print(f"Train IC (Spearman): {best_train_ic:.6f}")
    print(
        "Test IC (Spearman): "
        + (f"{test_ic:.6f}" if not pd.isna(test_ic) else "N/A")
    )
    print(f"Kaydedildi: {OUT_PATH}")

    return OUT_PATH


def main() -> None:
    run_optimization()


if __name__ == "__main__":
    main()
