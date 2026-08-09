"""
PROJECT OLYMPUS — Scan Store
============================
Gece taramasının kesintiye dayanıklı (resume-capable) altyapısı:
  - Rejim anlık görüntülerinin günlük tarihçesi (USDT.D zamanla birikir),
  - Tarama koşuları (run) durum takibi,
  - Tarama sonuçlarının kalıcı kaydı (Render restart'ta yeniden teslim).

signal_tracker.py ile aynı SQLite desenini kullanır; tüm IO asyncio.to_thread
ile yapılır (event loop bloke olmaz).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:  # yalnızca tip için — runtime döngüyü önler
    from core.regime_engine import RegimeSnapshot

DB_PATH = Path("data/scan_store.db")


class ScanStore:
    """Rejim + tarama sonuçları için kalıcı SQLite deposu."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Senkron altyapı (to_thread ile çağrılır) ──────────────────────────────
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    primary_trend TEXT,
                    intraday_timing TEXT,
                    risk_appetite REAL,
                    usdt_d REAL,
                    btc_d REAL,
                    total_market_cap REAL,
                    dxy REAL,
                    vix REAL,
                    snapshot_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_runs (
                    run_id TEXT PRIMARY KEY,
                    trigger TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT DEFAULT 'running',
                    candidate_count INTEGER DEFAULT 0,
                    found_count INTEGER DEFAULT 0,
                    digest_sent INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    scanned_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT,
                    composite REAL,
                    base_rr REAL,
                    t1 REAL,
                    t2 REAL,
                    t3 REAL,
                    stop_loss REAL,
                    trade_type TEXT,
                    oracle_summary TEXT,
                    tf_bias_json TEXT,
                    regime_json TEXT,
                    correlation_json TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_regime_captured ON regime_history(captured_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scan_results_run ON scan_results(run_id)")

    def _record_regime_sync(self, snap: RegimeSnapshot) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO regime_history
                (captured_at, primary_trend, intraday_timing, risk_appetite,
                 usdt_d, btc_d, total_market_cap, dxy, vix, snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap.captured_at)),
                    snap.primary_trend,
                    snap.intraday_timing,
                    snap.risk_appetite,
                    snap.usdt_d,
                    snap.btc_d,
                    snap.total_market_cap,
                    snap.dxy,
                    snap.vix,
                    json.dumps(
                        {
                            "dominance": snap.dominance.__dict__,
                            "source_flags": snap.source_flags,
                            "warnings": snap.warnings,
                            "econ_events_today": snap.econ_events_today,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                ),
            )

    def _get_regime_history_sync(self, limit: int = 30) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT captured_at, primary_trend, intraday_timing, risk_appetite,"
                " usdt_d, btc_d, total_market_cap, dxy, vix FROM regime_history"
                " ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _start_run_sync(self, run_id: str, trigger: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scan_runs (run_id, trigger, started_at, status)"
                " VALUES (?, ?, ?, 'running')",
                (run_id, trigger, time.strftime("%Y-%m-%d %H:%M:%S")),
            )

    def _finish_run_sync(self, run_id: str, status: str, candidate_count: int, found_count: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE scan_runs SET finished_at = ?, status = ?, candidate_count = ?, found_count = ?"
                " WHERE run_id = ?",
                (time.strftime("%Y-%m-%d %H:%M:%S"), status, candidate_count, found_count, run_id),
            )

    def _get_last_run_sync(self) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM scan_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def _record_result_sync(
        self,
        run_id: str,
        symbol: str,
        signal: str,
        composite: float,
        base_rr: float,
        t1: float,
        t2: float,
        t3: float,
        stop_loss: float,
        trade_type: str,
        oracle_summary: str,
        tf_bias_json: str,
        regime_json: str,
        correlation_json: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO scan_results
                (run_id, scanned_at, symbol, signal, composite, base_rr, t1, t2, t3,
                 stop_loss, trade_type, oracle_summary, tf_bias_json, regime_json, correlation_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    symbol,
                    signal,
                    composite,
                    base_rr,
                    t1,
                    t2,
                    t3,
                    stop_loss,
                    trade_type,
                    oracle_summary,
                    tf_bias_json,
                    regime_json,
                    correlation_json,
                ),
            )

    def _get_results_sync(self, run_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT symbol, signal, composite, base_rr, t1, t2, t3, stop_loss,"
                " trade_type, oracle_summary, tf_bias_json, regime_json, correlation_json"
                " FROM scan_results WHERE run_id = ? ORDER BY composite DESC",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Async sarmalayıcılar ──────────────────────────────────────────────────
    async def record_regime(self, snap: RegimeSnapshot) -> None:
        try:
            await asyncio.to_thread(self._record_regime_sync, snap)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SCAN_STORE] rejim kaydı başarısız: {exc}")

    async def get_regime_history(self, limit: int = 30) -> list[dict[str, Any]]:
        try:
            return await asyncio.to_thread(self._get_regime_history_sync, limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SCAN_STORE] rejim tarihçesi okunamadı: {exc}")
            return []

    async def start_run(self, run_id: str, trigger: str) -> None:
        await asyncio.to_thread(self._start_run_sync, run_id, trigger)

    async def finish_run(self, run_id: str, status: str, candidate_count: int, found_count: int) -> None:
        await asyncio.to_thread(self._finish_run_sync, run_id, status, candidate_count, found_count)

    async def get_last_run(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_last_run_sync)

    async def record_result(
        self,
        run_id: str,
        symbol: str,
        signal: str,
        composite: float,
        base_rr: float,
        t1: float,
        t2: float,
        t3: float,
        stop_loss: float,
        trade_type: str,
        oracle_summary: str = "",
        tf_bias_json: str = "{}",
        regime_json: str = "{}",
        correlation_json: str = "{}",
    ) -> None:
        try:
            await asyncio.to_thread(
                self._record_result_sync,
                run_id, symbol, signal, composite, base_rr, t1, t2, t3,
                stop_loss, trade_type, oracle_summary, tf_bias_json, regime_json, correlation_json,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SCAN_STORE] sonuç kaydı başarısız: {exc}")

    async def get_results(self, run_id: str) -> list[dict[str, Any]]:
        try:
            return await asyncio.to_thread(self._get_results_sync, run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SCAN_STORE] sonuçlar okunamadı: {exc}")
            return []


# ── Modül seviyesi singleton + yardımcılar ────────────────────────────────────
_STORE: ScanStore | None = None


def get_scan_store() -> ScanStore:
    global _STORE
    if _STORE is None:
        _STORE = ScanStore()
    return _STORE


async def record_regime_snapshot(snap: RegimeSnapshot) -> None:
    """regime_engine tarafından lazy-import edilen kayıt noktası."""
    await get_scan_store().record_regime(snap)
