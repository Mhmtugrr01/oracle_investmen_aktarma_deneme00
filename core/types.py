"""
PROJECT OLYMPUS — The Oracle
Pydantic tabanlı LangGraph Agent-State şeması.
Dict yerine validate edilmiş modeller; hallucination kaynaklı çöküşleri önler.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from core.config import get_oracle_config_cached


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentNode(str, Enum):
    """LangGraph düğüm sırası — DÜĞÜM 0 → 6."""

    THE_ORACLE = "the_oracle"
    MACRO_SENTINEL = "macro_sentinel"
    QUANT_ENGINE = "quant_engine"
    WHALE_HUNTER = "whale_hunter"
    FUNDAMENTAL_FILTER = "fundamental_filter"
    SENTIMENT_READER = "sentiment_reader"
    RED_TEAM = "red_team"
    END = "end"


class PipelineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    NO_TRADE = "no_trade"


VALID_SIGNALS = [
    "STRONG_BUY",
    "ACCUMULATE",
    "HOLD",
    "REDUCE",
    "STRONG_SELL",
    "SHORT",
    "WATCH",
    "AVOID",
]


def _merge_messages(existing: list[str], new: list[str]) -> list[str]:
    return existing + new


class OracleState(BaseModel):
    """The Oracle'ın merkezi graph-state modeli."""

    # ── Kimlik & oturum ──────────────────────────────────────────────
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    chat_id: int = 0
    symbol: str = "BTC/USDT"
    query: str = ""

    # ── Pipeline kontrol ─────────────────────────────────────────────
    current_node: AgentNode = AgentNode.THE_ORACLE
    status: PipelineStatus = PipelineStatus.IDLE
    fatal_error: Optional[str] = None
    retry_count: int = Field(default=0, ge=0, le=3)
    ceo_approved: bool = False
    ceo_revision_reason: Optional[str] = None

    # ── Ajan skorları (−1.0 … +1.0) ──────────────────────────────────
    macro_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    quant_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    whale_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    fundamental_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)

    # ── Red Team (DÜĞÜM 6) ───────────────────────────────────────────
    red_team_verdict: Optional[str] = None
    red_team_passed: bool = False
    red_team_objections: list[str] = Field(default_factory=list)

    # ── Alpha çıktısı ────────────────────────────────────────────────
    signal_direction: SignalDirection = SignalDirection.NEUTRAL
    signal_label: Optional[str] = None
    alpha_signal: Optional[str] = None
    base_rr: Optional[float] = Field(default=None, ge=0.0)
    risk_reward_ratio: Optional[float] = Field(default=None, ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entry_zone_low: Optional[float] = Field(default=None)
    entry_zone_high: Optional[float] = Field(default=None)
    entry_price: Optional[float] = Field(default=None, ge=0.0)
    stop_loss: Optional[float] = Field(default=None, ge=0.0)
    take_profit: Optional[float] = Field(default=None, ge=0.0)
    t1: Optional[float] = Field(default=None)
    t1_rr: Optional[float] = Field(default=None)
    t2: Optional[float] = Field(default=None)
    t2_rr: Optional[float] = Field(default=None)
    t3: Optional[float] = Field(default=None)
    t3_rr: Optional[float] = Field(default=None)
    fib_382: Optional[float] = Field(default=None)
    fib_500: Optional[float] = Field(default=None)
    fib_618: Optional[float] = Field(default=None)
    invalidation_level: Optional[float] = Field(default=None)
    trade_type: Optional[str] = Field(default=None)
    timeframe_alignment_score: Optional[float] = Field(default=None)
    timeframe_biases: Optional[dict] = Field(default=None)
    divergence_daily: Optional[str] = Field(default=None)
    divergence_weekly: Optional[str] = Field(default=None)
    cross_asset_score: Optional[float] = Field(default=None)
    cross_asset_warnings: Optional[list] = Field(default_factory=list)
    historical_pattern: Optional[str] = Field(default=None)
    pattern_outcome_bias: Optional[str] = Field(default=None)
    historical_similarity_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    fundamental_data_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    news_article_count: Optional[int] = Field(default=None, ge=0)
    news_sentiment: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    fear_greed_value: Optional[int] = Field(default=None)
    consensus_variance: Optional[float] = Field(default=None, ge=0.0)
    ma_fallback_used: Optional[bool] = Field(default=None)

    # ── Denetim izi ──────────────────────────────────────────────────
    messages: Annotated[list[str], _merge_messages] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"validate_assignment": True}

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()

    def mark_running(self, node: AgentNode) -> OracleState:
        return self.model_copy(
            update={
                "current_node": node,
                "status": PipelineStatus.RUNNING,
                "updated_at": _utcnow(),
            }
        )

    def mark_completed(self) -> OracleState:
        return self.model_copy(
            update={
                "status": PipelineStatus.COMPLETED,
                "current_node": AgentNode.END,
                "updated_at": _utcnow(),
            }
        )

    def mark_failed(self, error: str) -> OracleState:
        return self.model_copy(
            update={
                "status": PipelineStatus.FAILED,
                "fatal_error": error,
                "updated_at": _utcnow(),
            }
        )

    def append_log(self, message: str) -> OracleState:
        return self.model_copy(
            update={
                "messages": [message],
                "updated_at": _utcnow(),
            }
        )

    @property
    def composite_score(self) -> float:
        def _to_unit(value: Optional[float]) -> float:
            if value is None:
                return 0.5
            clipped = max(-1.0, min(1.0, float(value)))
            return (clipped + 1.0) / 2.0

        try:
            config_weights = get_oracle_config_cached().analysis.weights
        except Exception:
            config_weights = {
                "macro": 0.15,
                "quant": 0.40,
                "whale": 0.10,
                "fundamental": 0.25,
                "sentiment": 0.10,
            }

        macro_weight = float(config_weights.get("macro", 0.15))
        quant_weight = float(config_weights.get("quant", 0.40))
        whale_weight = float(config_weights.get("whale", 0.10))
        fundamental_weight = float(config_weights.get("fundamental", 0.25))
        sentiment_weight = float(config_weights.get("sentiment", 0.10))

        macro_component = _to_unit(self.macro_score)
        if "/" in self.symbol and self.cross_asset_score is not None:
            cross_unit = max(0.0, min(100.0, float(self.cross_asset_score))) / 100.0
            macro_component = macro_component * 0.5 + cross_unit * 0.5

        score = (
            macro_component * macro_weight
            + _to_unit(self.whale_score) * whale_weight
            + _to_unit(self.quant_score) * quant_weight
            + _to_unit(self.fundamental_score) * fundamental_weight
            + max(0.0, min(1.0, float(self.timeframe_alignment_score or 0.0))) * 0.10
            + _to_unit(self.sentiment_score) * sentiment_weight
        )
        if self.divergence_daily == "POSITIVE_DIVERGENCE":
            score += 0.05
        elif self.divergence_daily == "NEGATIVE_DIVERGENCE":
            score -= 0.05

        hist = float(self.historical_similarity_score or 0.0)
        pattern_bias = str(self.pattern_outcome_bias or "")
        if hist >= 75.0 and pattern_bias == "HISTORICALLY_BULLISH":
            score += 0.04
        elif hist >= 75.0 and pattern_bias == "HISTORICALLY_BEARISH":
            score -= 0.04

        # ── Extreme Fear / Greed Contrarian Bonusu ──────────────────────
        # F&G < 25 → tarihsel birikim zonu → sistematik contrarian fırsat
        # Şu an -0.34 sentiment olarak composite'i düşürüyor; bu bunu dengeler
        fg = self.fear_greed_value
        if fg is not None:
            if int(fg) <= 25:
                score += 0.06   # Extreme Fear = tarihsel alım bölgesi
            elif int(fg) >= 75:
                score -= 0.04   # Extreme Greed = dikkat

        return round(max(0.0, min(1.0, score)), 4)

    @property
    def is_halted(self) -> bool:
        return self.fatal_error is not None or self.status == PipelineStatus.ABORTED


class OracleStateUpdate(BaseModel):
    """LangGraph düğümlerinin kısmi state güncellemesi."""

    current_node: Optional[AgentNode] = None
    status: Optional[PipelineStatus] = None
    fatal_error: Optional[str] = None
    retry_count: Optional[int] = Field(default=None, ge=0, le=3)
    ceo_approved: Optional[bool] = None
    ceo_revision_reason: Optional[str] = None
    macro_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    quant_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    whale_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    fundamental_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    sentiment_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    red_team_verdict: Optional[str] = None
    red_team_passed: Optional[bool] = None
    red_team_objections: Optional[list[str]] = None
    signal_direction: Optional[SignalDirection] = None
    signal_label: Optional[str] = None
    alpha_signal: Optional[str] = None
    base_rr: Optional[float] = Field(default=None, ge=0.0)
    risk_reward_ratio: Optional[float] = Field(default=None, ge=0.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    entry_price: Optional[float] = Field(default=None, ge=0.0)
    stop_loss: Optional[float] = Field(default=None, ge=0.0)
    take_profit: Optional[float] = Field(default=None, ge=0.0)
    t1: Optional[float] = None
    t1_rr: Optional[float] = None
    t2: Optional[float] = None
    t2_rr: Optional[float] = None
    t3: Optional[float] = None
    t3_rr: Optional[float] = None
    fib_382: Optional[float] = None
    fib_500: Optional[float] = None
    fib_618: Optional[float] = None
    invalidation_level: Optional[float] = None
    trade_type: Optional[str] = None
    timeframe_alignment_score: Optional[float] = None
    timeframe_biases: Optional[dict] = None
    divergence_daily: Optional[str] = None
    divergence_weekly: Optional[str] = None
    cross_asset_score: Optional[float] = None
    cross_asset_warnings: Optional[list] = None
    historical_pattern: Optional[str] = None
    pattern_outcome_bias: Optional[str] = Field(default=None)
    historical_similarity_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    fundamental_data_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    news_article_count: Optional[int] = Field(default=None, ge=0)
    news_sentiment: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    fear_greed_value: Optional[int] = None
    consensus_variance: Optional[float] = Field(default=None, ge=0.0)
    ma_fallback_used: Optional[bool] = None
    oracle_summary: Optional[str] = None
    messages: Optional[list[str]] = None

