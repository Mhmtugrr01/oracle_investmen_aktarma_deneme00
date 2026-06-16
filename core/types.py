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
    whale_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    fundamental_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)

    # ── Red Team (DÜĞÜM 6) ───────────────────────────────────────────
    red_team_verdict: Optional[str] = None
    red_team_passed: bool = False
    red_team_objections: list[str] = Field(default_factory=list)

    # ── Alpha çıktısı ────────────────────────────────────────────────
    signal_direction: SignalDirection = SignalDirection.NEUTRAL
    alpha_signal: Optional[str] = None
    risk_reward_ratio: Optional[float] = Field(default=None, ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entry_price: Optional[float] = Field(default=None, ge=0.0)
    stop_loss: Optional[float] = Field(default=None, ge=0.0)
    take_profit: Optional[float] = Field(default=None, ge=0.0)

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
        weights = {
            "macro": 0.20,
            "quant": 0.25,
            "whale": 0.15,
            "fundamental": 0.20,
            "sentiment": 0.20,
        }
        return (
            self.macro_score * weights["macro"]
            + self.quant_score * weights["quant"]
            + self.whale_score * weights["whale"]
            + self.fundamental_score * weights["fundamental"]
            + self.sentiment_score * weights["sentiment"]
        )

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
    alpha_signal: Optional[str] = None
    risk_reward_ratio: Optional[float] = Field(default=None, ge=0.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    entry_price: Optional[float] = Field(default=None, ge=0.0)
    stop_loss: Optional[float] = Field(default=None, ge=0.0)
    take_profit: Optional[float] = Field(default=None, ge=0.0)
    messages: Optional[list[str]] = None
