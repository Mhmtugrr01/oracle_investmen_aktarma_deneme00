"""PROJECT OLYMPUS — ajan modülü."""

from agents.the_oracle import run_the_oracle
from agents.macro_sentinel import run_macro_sentinel
from agents.quant_engine import run_quant_engine
from agents.whale_hunter import run_whale_hunter
from agents.fundamental_filter import run_fundamental_filter
from agents.sentiment_reader import run_sentiment_reader
from agents.red_team import run_red_team

__all__ = [
    "run_the_oracle",
    "run_macro_sentinel",
    "run_quant_engine",
    "run_whale_hunter",
    "run_fundamental_filter",
    "run_sentiment_reader",
    "run_red_team",
]
