"""Merkezi Oracle konfigurasyonu (YAML -> Pydantic)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisConfig(BaseModel):
    weights: dict[str, float]

    @model_validator(mode="after")
    def _validate_weights(self) -> "AnalysisConfig":
        required = {"macro", "quant", "whale", "fundamental", "sentiment"}
        missing = required - set(self.weights.keys())
        if missing:
            raise ValueError(f"Eksik analiz agirliklari: {sorted(missing)}")

        total = sum(float(v) for v in self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Agirlik toplami 1.0 olmali, bulundu: {total}")
        return self


class KellyConfig(BaseModel):
    max_fraction: float = Field(gt=0.0, le=1.0)
    min_fraction: float = Field(gt=0.0, le=1.0)


class AtrRiskConfig(BaseModel):
    stop_loss_multiplier: float = Field(gt=0.0)
    take_profit_multiplier: float = Field(gt=0.0)


class RiskConfig(BaseModel):
    kelly: KellyConfig
    min_risk_reward_ratio: float = Field(gt=0.0)
    atr: AtrRiskConfig


class CeoConfig(BaseModel):
    max_score_spread: float = Field(gt=0.0)
    long_threshold: float
    short_threshold: float
    min_composite_score: float = Field(ge=0.0, le=1.0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    confidence_boost: float = Field(ge=0.0, le=1.0)
    max_retries: int = Field(ge=1, le=10)


class RedTeamConfig(BaseModel):
    black_swan_spread: float = Field(gt=0.0)
    min_llm_confidence: float = Field(ge=0.0, le=1.0)
    fail_on_fatal: bool = True


class QuantScoreConfig(BaseModel):
    rsi_oversold_bonus: float
    rsi_overbought_penalty: float
    rsi_bullish_zone_bonus: float
    rsi_bearish_zone_penalty: float
    cross_golden_bonus: float
    cross_bullish_bonus: float
    cross_bearish_penalty: float
    cross_death_penalty: float
    divergence_bullish_bonus: float
    divergence_bearish_penalty: float
    fib_proximity_bonus: float
    atr_rr_bonus: float
    atr_rr_penalty: float
    atr_rr_high_threshold: float
    atr_rr_low_threshold: float


class QuantConfig(BaseModel):
    timeframe: str
    ohlcv_limit: int = Field(ge=100, le=5000)
    rsi_period: int = Field(ge=5, le=100)
    ma_fast_period: int = Field(ge=2, le=200)
    ma_slow_period: int = Field(ge=3, le=400)
    fib_lookback_bars: int = Field(ge=20, le=500)
    atr_period: int = Field(ge=5, le=100)
    fib_proximity_threshold_pct: float = Field(gt=0.0)
    score: QuantScoreConfig


class WhaleConfig(BaseModel):
    timeframe: str
    ohlcv_limit: int = Field(ge=100, le=5000)
    sweep_lookback_bars: int = Field(ge=10, le=500)
    cvd_lookback_bars: int = Field(ge=10, le=500)
    wick_ratio_threshold: float = Field(gt=0.0, lt=1.0)
    body_ratio_threshold: float = Field(gt=0.0, lt=1.0)
    rr_target_multiplier: float = Field(gt=0.5)


class LlmConfig(BaseModel):
    model: str
    temperature: float = Field(ge=0.0, le=1.0)
    timeout_seconds: int = Field(ge=5, le=180)
    max_retries: int = Field(ge=1, le=10)
    base_url_env: str
    api_key_env: str


class ScanScheduleConfig(BaseModel):
    full_scan_interval_hours: int = Field(default=4, ge=1, le=168)
    watchlist_check_interval_min: int = Field(default=15, ge=1, le=1440)
    daily_briefing_hour: int = Field(default=8, ge=0, le=23)


class HTFFilterConfig(BaseModel):
    enabled: bool = True
    block_long_if_weekly_bearish: bool = True
    downgrade_long_if_daily_bearish: bool = True
    require_full_alignment_for_strong_buy: bool = True


class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: AnalysisConfig
    risk: RiskConfig
    ceo: CeoConfig
    red_team: RedTeamConfig
    quant: QuantConfig
    whale: WhaleConfig
    llm: LlmConfig
    asset_universe: dict[str, list[str]] = Field(default_factory=dict)
    scan_schedule: ScanScheduleConfig = Field(default_factory=ScanScheduleConfig)
    higher_timeframe_filter: HTFFilterConfig = Field(default_factory=HTFFilterConfig)


class OracleConfigLoader:
    """Thread-safe ve async YAML config loader."""

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._lock = asyncio.Lock()
        self._cached: OracleConfig | None = None

    @property
    def path(self) -> Path:
        return self._path

    async def load(self, force_reload: bool = False) -> OracleConfig:
        if self._cached is not None and not force_reload:
            return self._cached

        async with self._lock:
            if self._cached is not None and not force_reload:
                return self._cached
            payload = await asyncio.to_thread(self._read_yaml)
            self._cached = OracleConfig.model_validate(payload)
            return self._cached

    async def as_dict(self, force_reload: bool = False) -> dict[str, Any]:
        conf = await self.load(force_reload=force_reload)
        return conf.model_dump()

    def cached(self) -> OracleConfig:
        if self._cached is None:
            raise RuntimeError(
                "Oracle config cache bos. Once await load_oracle_config() cagirin."
            )
        return self._cached

    def cached_dict(self) -> dict[str, Any]:
        return self.cached().model_dump()

    def _read_yaml(self) -> dict[str, Any]:
        if not self._path.exists():
            raise FileNotFoundError(f"Config bulunamadi: {self._path}")
        with self._path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("oracle_config.yaml root objesi map/dict olmali.")
        return data


_CONFIG_LOADER = OracleConfigLoader(Path(__file__).resolve().parents[1] / "oracle_config.yaml")


async def load_oracle_config(force_reload: bool = False) -> OracleConfig:
    return await _CONFIG_LOADER.load(force_reload=force_reload)


async def load_oracle_config_dict(force_reload: bool = False) -> dict[str, Any]:
    return await _CONFIG_LOADER.as_dict(force_reload=force_reload)


def get_oracle_config_cached() -> OracleConfig:
    return _CONFIG_LOADER.cached()


def get_oracle_config_cached_dict() -> dict[str, Any]:
    return _CONFIG_LOADER.cached_dict()
