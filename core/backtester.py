"""Haftalik ham skor backtest araci.

- 2022-01-01'den bugune haftalik adimlarla calisir.
- Her varlik ve her tarih icin 5 ajani (CEO gate olmadan) kosar.
- Whale skorunu referans icin whale_score_raw kolonu ile saklar.
- Whale donmasi (diagnostic_run) %50 ustuyse composite hesapta whale agirligini 0 kabul eder.
"""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
import os

from agents.fundamental_filter import run_fundamental_filter
from agents.macro_sentinel import run_macro_sentinel
from agents.quant_engine import run_quant_engine
from agents.sentiment_reader import run_sentiment_reader
from agents.whale_hunter import run_whale_hunter
from core.config import load_oracle_config
from core.types import OracleState


OUT_PATH = Path("data/backtest_history.csv")
DIAG_PATH = Path("data/diagnostic_run.csv")


def _flatten_assets(asset_universe: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for assets in asset_universe.values():
        out.extend(assets)
    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _symbol_to_yf(symbol: str) -> str | None:
    if "/" in symbol:
        crypto_map = {
            "BTC/USDT": "BTC-USD",
            "ETH/USDT": "ETH-USD",
            "INJ/USDT": "INJ-USD",
            "FET/USDT": "FET-USD",
        }
        return crypto_map.get(symbol)
    return symbol


def _load_price_series(symbol: str) -> pd.Series | None:
    ticker = _symbol_to_yf(symbol)
    if not ticker:
        return None
    try:
        raw = yf.download(
            ticker,
            start="2021-12-01",
            end=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if raw is None or raw.empty:
            return None
        close = raw["Close"].dropna().copy()
        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 0:
                return None
            close = close.iloc[:, 0]
        if close.empty:
            return None
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close
    except Exception:
        return None


def _forward_return(close: pd.Series, dt: pd.Timestamp, days: int) -> float | None:
    if close.empty:
        return None
    date = pd.Timestamp(dt).tz_localize(None)

    past = close.loc[close.index <= date]
    if past.empty:
        return None
    p0 = float(past.iloc[-1])
    if p0 == 0:
        return None

    target = date + pd.Timedelta(days=days)
    fut = close.loc[close.index >= target]
    if fut.empty:
        return None
    p1 = float(fut.iloc[0])

    return (p1 - p0) / p0


def _consensus_variance(macro: float, quant: float, whale: float | None, fundamental: float, sentiment: float) -> float:
    vals = [macro, quant, whale if whale is not None else 0.0, fundamental, sentiment]
    return float(max(vals) - min(vals))


def _load_whale_freeze_ratio() -> float:
    if not DIAG_PATH.exists():
        return 0.0

    rows = list(csv.DictReader(DIAG_PATH.open("r", encoding="utf-8", newline="")))
    if not rows:
        return 0.0

    frozen = 0
    total = 0
    for r in rows:
        v = _to_float(r.get("whale_score"))
        if v is None:
            continue
        total += 1
        if v == 0.0 or v == -0.360:
            frozen += 1

    if total == 0:
        return 0.0
    return frozen / total


async def _run_five_agents(symbol: str, backtest_date: pd.Timestamp) -> OracleState:
    state = OracleState(symbol=symbol, query=f"backtest:{backtest_date.date().isoformat()}")
    for fn in (
        run_macro_sentinel,
        run_quant_engine,
        run_whale_hunter,
        run_fundamental_filter,
        run_sentiment_reader,
    ):
        state = await fn(state)
    return state


async def run_backtest() -> Path:
    conf = await load_oracle_config(force_reload=True)
    symbols = _flatten_assets(conf.asset_universe)

    whale_freeze_ratio = _load_whale_freeze_ratio()
    whale_excluded = whale_freeze_ratio > 0.50

    w_macro = float(conf.analysis.weights.get("macro", 0.0))
    w_quant = float(conf.analysis.weights.get("quant", 0.0))
    w_fund = float(conf.analysis.weights.get("fundamental", 0.0))
    w_sent = float(conf.analysis.weights.get("sentiment", 0.0))

    # SSL nedeniyle geçici dışlandı: whale agirligi backtest skorda kullanilmiyor.
    four_sum = w_macro + w_quant + w_fund + w_sent
    if four_sum <= 0:
        w_macro = w_quant = w_fund = w_sent = 0.25
    else:
        w_macro /= four_sum
        w_quant /= four_sum
        w_fund /= four_sum
        w_sent /= four_sum

    # yfinance saatlik (1h/4h) veri kısıtı nedeniyle (en fazla 730 gün geriye gider) backtest başlangıcını Eylül 2024'e çekiyoruz.
    weekly_dates = pd.date_range(start="2024-09-01", end=pd.Timestamp.now("UTC").tz_localize(None), freq="W-FRI")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "date",
        "symbol",
        "macro_score",
        "quant_score",
        "fundamental_score",
        "sentiment_score",
        "whale_score_raw",
        "composite_4factor",
        "forward_return_7d",
        "forward_return_30d",
        "confidence",
        "consensus_variance",
        "fatal_error",
    ]

    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        close = _load_price_series(symbol)
        if close is None:
            continue

        for dt in weekly_dates:
            # Mevcut ajanlar tarih-parametresi almadigi icin skorlari sembol bazinda tek kez hesapliyoruz.
            import os
            # AJANLARA TAKVİM ZERK ETME (MAKAS KOMUTUNU HAZIRLA):
            os.environ["BACKTEST_AS_OF"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Gerçek Asimetri: Carkların İÇİNDE her haftanın datası tek tek çalışır, o haftaya kördür.
                state = await _run_five_agents(symbol, dt)
                

                macro = _to_float(state.macro_score) or 0.0
                quant = _to_float(state.quant_score) or 0.0
                fund = _to_float(state.fundamental_score) or 0.0
                sent = _to_float(state.sentiment_score) or 0.0
                whale = _to_float(state.whale_score)

                composite_4factor = (
                    (macro * w_macro)
                    + (quant * w_quant)
                    + (fund * w_fund)
                    + (sent * w_sent)
                )

                row = {
                    "date": dt.date().isoformat(),
                    "symbol": symbol,
                    "macro_score": macro,
                    "quant_score": quant,
                    "fundamental_score": fund,
                    "sentiment_score": sent,
                    "whale_score_raw": whale,
                    "composite_4factor": round(float(composite_4factor), 6),
                    "forward_return_7d": _forward_return(close, dt, 7),
                    "forward_return_30d": _forward_return(close, dt, 30),
                    "confidence": _to_float(state.confidence),
                    "consensus_variance": _consensus_variance(macro, quant, whale, fund, sent),
                    "fatal_error": state.fatal_error or "",
                }
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "date": dt.date().isoformat(),
                        "symbol": symbol,
                        "macro_score": None,
                        "quant_score": None,
                        "fundamental_score": None,
                        "sentiment_score": None,
                        "whale_score_raw": None,
                        "composite_4factor": None,
                        "forward_return_7d": _forward_return(close, dt, 7),
                        "forward_return_30d": _forward_return(close, dt, 30),
                        "confidence": None,
                        "consensus_variance": None,
                        "fatal_error": str(exc),
                    }
                )

    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"backtest_history yazildi: {OUT_PATH}")
    print(f"satir sayisi: {len(rows)}")
    print(f"whale donma orani (diagnostic): {whale_freeze_ratio:.2%}")
    if whale_excluded:
        print("not: whale agirligi backtest skorunda 0 kabul edildi (SSL nedeniyle gecici dislandi)")

    return OUT_PATH


def main() -> None:
    asyncio.run(run_backtest())


if __name__ == "__main__":
    main()
