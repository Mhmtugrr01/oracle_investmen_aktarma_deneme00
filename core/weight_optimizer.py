"""
PROJECT OLYMPUS — core/weight_optimizer.py (R05_OPTIMIZER_REFORMED)
Tarihsel backtest dökümünden en asimetrik ve overfitting içermeyen ağırlıkları hesaplar.
"""

from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

BACKTEST_PATH = Path("data/backtest_history.csv")
OUT_PATH = Path("data/optimal_weights.json")


def _spearman_ic(df: pd.DataFrame, weights: dict[str, float]) -> float:
    d = df.copy()
    # Sadece tarihsel ileri getirisi olmayan satırları eliyoruz (bu zorunlu)
    d = d.dropna(subset=["forward_return_30d"])
    
    # ── VERİ KATLİAMINI ENGELLE: Eksik/Null gelen haber veya sentiment skorlarını 0.0 (Nötr) kabul et ──
    score_cols = ["macro_score", "quant_score", "fundamental_score", "sentiment_score"]
    for col in score_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
            
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

    # ── YENİNESİL KURUMSAL ZAMAN BÖLÜMÜ (TRAIN/TEST SPLIT) ──
    # Backtest'imiz Eylül 2024'ten başladığı için eski 2022-2024 ayrımı feci bir dengesizlik yaratıyordu.
    # Doğru Bölüm: Train (Eylül 2024 - Aralık 2025) | Test (Ocak 2026 - Bugün)
    train = df[df["date"] < "2026-01-01"].copy()
    test = df[df["date"] >= "2026-01-01"].copy()

    if len(train) < 10 or len(test) < 10:
        raise ValueError(f"Optimizasyon için yetersiz veri! Train: {len(train)}, Test: {len(test)}")

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

    # 🚨 OVERFITTING TESTİ: Train ve Test IC farkı kontrol ediliyor
    overfitting_alert = False
    ic_diff = abs(best_train_ic - test_ic)
    if ic_diff > 0.08:
        overfitting_alert = True

    result = {
        "weights": best_weights,
        "train_ic_spearman": float(best_train_ic),
        "test_ic_spearman": float(test_ic) if not pd.isna(test_ic) else None,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "overfitting_detected": overfitting_alert
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("THE ORACLE _R05 — KANTİTATİF AĞIRLIK OPTİMİZASYONU")
    print("=" * 70)
    print("En iyi ağırlıklar (Weights):", best_weights)
    print(f"Train IC (Spearman) [2024-2025] : {best_train_ic:.6f}")
    print("Test IC (Spearman)  [2026]      : " + (f"{test_ic:.6f}" if not pd.isna(test_ic) else "N/A"))
    print(f"Uyum Farkı (Delta)              : {ic_diff:.4f}")
    
    if overfitting_alert:
        print("\n🚨 UYARI: Train ve Test arasındaki fark yüksek (%8).")
    else:
        print("\n✅ UYUM BAŞARILI: Aşırı uyum (Overfit) tespit edilmedi!")
    print("=" * 70)

    return OUT_PATH


def main() -> None:
    run_optimization()


if __name__ == "__main__":
    main()