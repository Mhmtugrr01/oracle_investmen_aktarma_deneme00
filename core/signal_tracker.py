"""
PROJECT OLYMPUS — The Signal Tracker
Sinyal sonuç takibi: Her üretilen sinyalin win/loss durumunu takip eder.
SQLite tabanlı, sistem performansının gerçek zamanlı kanıtını sunar.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yfinance as yf
from loguru import logger


def _last_close_price(df) -> float | None:
    """DataFrame'den SON KAPANIŞ fiyatını güvenli skaler olarak döndürür.

    FAZ 0 — Series Bug'ı: yfinance bazı sürümlerde çok sütunlu/çok katmanlı
    DataFrame döndürür; `df["Close"]` bir Series değil DataFrame olabilir ve
    `.iloc[-1]` Series döndürür → `float(Series)` patlar. Bu helper her iki
    durumu da (Series + DataFrame) sıkıştırıp skaler döndürür.
    """
    if df is None or df.empty:
        return None
    close = df.get("Close") if "Close" in df.columns else df.get("close")
    if close is None:
        close = df.get("close")
    if close is None:
        return None
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if close is None or len(close) == 0:
        return None
    try:
        val = close.iloc[-1]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        val = float(val)
        if pd.isna(val):
            return None
        return val
    except Exception:  # noqa: BLE001
        return None


class SignalStatus(str, Enum):
    """Sinyal durumu."""
    PENDING = "pending"      # Henüz giriş yapılmadı
    OPEN = "open"            # Pozisyon açık
    TP1_HIT = "tp1_hit"      # T1 hedefi tuttu
    TP2_HIT = "tp2_hit"      # T2 hedefi tuttu
    TP3_HIT = "tp3_hit"      # T3 hedefi tuttu
    SL_HIT = "sl_hit"        # Stop loss tetiklendi
    EXPIRED = "expired"      # Zaman aşımı (7 gün)
    CANCELLED = "cancelled"  # İptal edildi


class SignalOutcome(str, Enum):
    """Sinyal sonucu."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"


@dataclass
class TrackedSignal:
    """Takip edilen sinyal kaydı."""
    signal_id: str
    symbol: str
    direction: str  # LONG | SHORT
    entry_price: float
    stop_loss: float
    t1: float
    t2: Optional[float]
    t3: Optional[float]
    signal_time: datetime
    confidence: float
    composite_score: float
    choch_detected: bool
    rsi_trendline_break: bool
    cluster_leader_rank: Optional[int]
    status: SignalStatus
    outcome: SignalOutcome
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    pnl_percent: Optional[float]
    max_favorable_excursion: Optional[float]  # En yüksek kâr
    max_adverse_excursion: Optional[float]    # En yüksek zarar


class SignalTracker:
    """
    Sinyal takip motoru.
    
    Her sinyali SQLite veritabanına kaydeder, periyodik olarak
    açık pozisyonların durumunu kontrol eder ve win/loss istatistikleri üretir.
    """

    DB_PATH = Path("data/signal_history.db")
    MAX_SIGNAL_AGE_DAYS = 7  # 7 gün sonra sinyal expire olur

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or self.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Veritabanı tablosunu oluştur."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    t1 REAL NOT NULL,
                    t2 REAL,
                    t3 REAL,
                    signal_time TEXT NOT NULL,
                    confidence REAL,
                    composite_score REAL,
                    choch_detected INTEGER,
                    rsi_trendline_break INTEGER,
                    cluster_leader_rank INTEGER,
                    status TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl_percent REAL,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(signal_time)
            """)
            conn.commit()

    def record_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        t1: float,
        t2: float | None = None,
        t3: float | None = None,
        confidence: float = 0.0,
        composite_score: float = 0.0,
        choch_detected: bool = False,
        rsi_trendline_break: bool = False,
        cluster_leader_rank: int | None = None,
    ) -> str:
        """
        Yeni sinyal kaydet.
        
        Returns:
            signal_id: UUID formatında benzersiz ID
        """
        import uuid
        signal_id = str(uuid.uuid4())[:8]
        
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO signals (
                    signal_id, symbol, direction, entry_price, stop_loss,
                    t1, t2, t3, signal_time, confidence, composite_score,
                    choch_detected, rsi_trendline_break, cluster_leader_rank,
                    status, outcome, max_favorable_excursion, max_adverse_excursion
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id, symbol, direction, entry_price, stop_loss,
                t1, t2, t3, now, confidence, composite_score,
                1 if choch_detected else 0,
                1 if rsi_trendline_break else 0,
                cluster_leader_rank,
                SignalStatus.PENDING.value,
                SignalOutcome.PENDING.value,
                0.0, 0.0
            ))
            conn.commit()
        
        logger.info(f"[TRACKER] Sinyal kaydedildi: {signal_id} | {symbol} {direction} @ {entry_price}")
        return signal_id

    async def update_open_signals(self) -> int:
        """
        Açık sinyallerin güncel fiyatlarını kontrol et ve durumlarını güncelle.
        
        Returns:
            Güncellenen sinyal sayısı
        """
        updated = 0
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM signals 
                WHERE status IN (?, ?)
                AND signal_time > ?
            """, (
                SignalStatus.PENDING.value,
                SignalStatus.OPEN.value,
                (datetime.now(timezone.utc) - timedelta(days=self.MAX_SIGNAL_AGE_DAYS)).isoformat()
            ))
            
            rows = cursor.fetchall()
        
        for row in rows:
            try:
                signal = self._row_to_signal(row)
                new_status, outcome, exit_price, exit_time, pnl = await self._check_signal_status(signal)
                
                if new_status != signal.status:
                    self._update_signal_status(
                        signal.signal_id, new_status, outcome, exit_price, exit_time, pnl
                    )
                    updated += 1
                    
            except Exception as e:
                logger.error(f"[TRACKER] Sinyal güncelleme hatası {row['signal_id']}: {e}")
        
        return updated

    async def _check_signal_status(
        self, signal: TrackedSignal
    ) -> tuple[SignalStatus, SignalOutcome, float | None, str | None, float | None]:
        """
        Tek bir sinyalin durumunu kontrol et.
        
        Returns:
            (new_status, outcome, exit_price, exit_time, pnl_percent)
        """
        try:
            ticker = self._to_yf_ticker(signal.symbol)
            if not ticker:
                return signal.status, signal.outcome, None, None, None
            
            # Güncel fiyatı çek
            data = await asyncio.to_thread(
                yf.download,
                ticker,
                period="1d",
                interval="1m",
                progress=False
            )
            
            if data is None or data.empty:
                return signal.status, signal.outcome, None, None, None

            current_price = _last_close_price(data)
            if current_price is None:
                return signal.status, signal.outcome, None, None, None
            current_time = datetime.now(timezone.utc).isoformat()
            
            # LONG pozisyon kontrolü
            if signal.direction == "LONG":
                # TP kontrolü
                if current_price >= signal.t3 and signal.t3:
                    return SignalStatus.TP3_HIT, SignalOutcome.WIN, current_price, current_time, \
                           ((current_price - signal.entry_price) / signal.entry_price) * 100
                elif current_price >= signal.t2 and signal.t2:
                    return SignalStatus.TP2_HIT, SignalOutcome.WIN, current_price, current_time, \
                           ((current_price - signal.entry_price) / signal.entry_price) * 100
                elif current_price >= signal.t1:
                    return SignalStatus.TP1_HIT, SignalOutcome.WIN, current_price, current_time, \
                           ((current_price - signal.entry_price) / signal.entry_price) * 100
                # SL kontrolü
                elif current_price <= signal.stop_loss:
                    return SignalStatus.SL_HIT, SignalOutcome.LOSS, current_price, current_time, \
                           ((current_price - signal.entry_price) / signal.entry_price) * 100
            
            # SHORT pozisyon kontrolü
            elif signal.direction == "SHORT":
                # TP kontrolü (short için ters)
                if current_price <= signal.t3 and signal.t3:
                    return SignalStatus.TP3_HIT, SignalOutcome.WIN, current_price, current_time, \
                           ((signal.entry_price - current_price) / signal.entry_price) * 100
                elif current_price <= signal.t2 and signal.t2:
                    return SignalStatus.TP2_HIT, SignalOutcome.WIN, current_price, current_time, \
                           ((signal.entry_price - current_price) / signal.entry_price) * 100
                elif current_price <= signal.t1:
                    return SignalStatus.TP1_HIT, SignalOutcome.WIN, current_price, current_time, \
                           ((signal.entry_price - current_price) / signal.entry_price) * 100
                # SL kontrolü
                elif current_price >= signal.stop_loss:
                    return SignalStatus.SL_HIT, SignalOutcome.LOSS, current_price, current_time, \
                           ((signal.entry_price - current_price) / signal.entry_price) * 100
            
            # Zaman aşımı kontrolü
            signal_age = datetime.now(timezone.utc) - datetime.fromisoformat(signal.signal_time.replace('Z', '+00:00'))
            if signal_age.days >= self.MAX_SIGNAL_AGE_DAYS:
                pnl = 0.0
                if signal.direction == "LONG":
                    pnl = ((current_price - signal.entry_price) / signal.entry_price) * 100
                else:
                    pnl = ((signal.entry_price - current_price) / signal.entry_price) * 100
                
                outcome = SignalOutcome.WIN if pnl > 0 else SignalOutcome.LOSS if pnl < 0 else SignalOutcome.BREAKEVEN
                return SignalStatus.EXPIRED, outcome, current_price, current_time, pnl
            
            # Pozisyon hala açık
            return SignalStatus.OPEN, SignalOutcome.PENDING, None, None, None
            
        except Exception as e:
            logger.error(f"[TRACKER] Fiyat kontrol hatası {signal.symbol}: {e}")
            return signal.status, signal.outcome, None, None, None

    def _update_signal_status(
        self,
        signal_id: str,
        status: SignalStatus,
        outcome: SignalOutcome,
        exit_price: float | None,
        exit_time: str | None,
        pnl: float | None
    ) -> None:
        """Sinyal durumunu veritabanında güncelle."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE signals 
                SET status = ?, outcome = ?, exit_price = ?, exit_time = ?, 
                    pnl_percent = ?, updated_at = ?
                WHERE signal_id = ?
            """, (
                status.value, outcome.value, exit_price, exit_time, pnl,
                datetime.now(timezone.utc).isoformat(), signal_id
            ))
            conn.commit()

    def get_statistics(self, days: int = 30) -> dict[str, Any]:
        """
        Son N günün sinyal istatistiklerini getir.
        
        Returns:
            dict: {
                'total_signals': int,
                'closed_signals': int,
                'open_signals': int,
                'win_rate': float,
                'avg_pnl': float,
                'avg_rr': float,
                'by_direction': {...},
                'by_choch': {...},
                'by_rsi_break': {...}
            }
        """
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Toplam sinyal sayısı
            total = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE signal_time > ?",
                (since,)
            ).fetchone()[0]
            
            # Kapanan sinyaller
            closed = conn.execute(
                """SELECT COUNT(*) FROM signals 
                   WHERE signal_time > ? AND status IN (?, ?, ?, ?)""",
                (since, SignalStatus.TP1_HIT.value, SignalStatus.TP2_HIT.value,
                 SignalStatus.TP3_HIT.value, SignalStatus.SL_HIT.value)
            ).fetchone()[0]
            
            # Açık sinyaller
            open_count = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE signal_time > ? AND status IN (?, ?)",
                (since, SignalStatus.PENDING.value, SignalStatus.OPEN.value)
            ).fetchone()[0]
            
            # Win rate
            wins = conn.execute(
                """SELECT COUNT(*) FROM signals 
                   WHERE signal_time > ? AND outcome = ?""",
                (since, SignalOutcome.WIN.value)
            ).fetchone()[0]
            
            win_rate = (wins / closed * 100) if closed > 0 else 0.0
            
            # Ortalama PnL
            avg_pnl_result = conn.execute(
                """SELECT AVG(pnl_percent) FROM signals 
                   WHERE signal_time > ? AND outcome != ?""",
                (since, SignalOutcome.PENDING.value)
            ).fetchone()[0]
            avg_pnl = float(avg_pnl_result) if avg_pnl_result else 0.0
            
            # Direction bazlı istatistikler
            by_direction = {}
            for direction in ["LONG", "SHORT"]:
                dir_total = conn.execute(
                    """SELECT COUNT(*) FROM signals 
                       WHERE signal_time > ? AND direction = ? AND outcome != ?""",
                    (since, direction, SignalOutcome.PENDING.value)
                ).fetchone()[0]
                dir_wins = conn.execute(
                    """SELECT COUNT(*) FROM signals 
                       WHERE signal_time > ? AND direction = ? AND outcome = ?""",
                    (since, direction, SignalOutcome.WIN.value)
                ).fetchone()[0]
                by_direction[direction] = {
                    'total': dir_total,
                    'wins': dir_wins,
                    'win_rate': (dir_wins / dir_total * 100) if dir_total > 0 else 0.0
                }
            
            # CHoCH bazlı istatistikler
            choch_yes = conn.execute(
                """SELECT COUNT(*) FROM signals 
                   WHERE signal_time > ? AND choch_detected = 1 AND outcome != ?""",
                (since, SignalOutcome.PENDING.value)
            ).fetchone()[0]
            choch_wins = conn.execute(
                """SELECT COUNT(*) FROM signals 
                   WHERE signal_time > ? AND choch_detected = 1 AND outcome = ?""",
                (since, SignalOutcome.WIN.value)
            ).fetchone()[0]
            
            by_choch = {
                'with_choch': {
                    'total': choch_yes,
                    'wins': choch_wins,
                    'win_rate': (choch_wins / choch_yes * 100) if choch_yes > 0 else 0.0
                }
            }
            
            # RSI Trendline Break bazlı istatistikler
            rsi_yes = conn.execute(
                """SELECT COUNT(*) FROM signals 
                   WHERE signal_time > ? AND rsi_trendline_break = 1 AND outcome != ?""",
                (since, SignalOutcome.PENDING.value)
            ).fetchone()[0]
            rsi_wins = conn.execute(
                """SELECT COUNT(*) FROM signals 
                   WHERE signal_time > ? AND rsi_trendline_break = 1 AND outcome = ?""",
                (since, SignalOutcome.WIN.value)
            ).fetchone()[0]
            
            by_rsi_break = {
                'with_rsi_break': {
                    'total': rsi_yes,
                    'wins': rsi_wins,
                    'win_rate': (rsi_wins / rsi_yes * 100) if rsi_yes > 0 else 0.0
                }
            }
        
        return {
            'period_days': days,
            'total_signals': total,
            'closed_signals': closed,
            'open_signals': open_count,
            'win_rate': round(win_rate, 2),
            'avg_pnl': round(avg_pnl, 2),
            'by_direction': by_direction,
            'by_choch': by_choch,
            'by_rsi_break': by_rsi_break,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

    def _row_to_signal(self, row: sqlite3.Row) -> TrackedSignal:
        """SQLite satırını TrackedSignal nesnesine çevir."""
        return TrackedSignal(
            signal_id=row['signal_id'],
            symbol=row['symbol'],
            direction=row['direction'],
            entry_price=row['entry_price'],
            stop_loss=row['stop_loss'],
            t1=row['t1'],
            t2=row['t2'],
            t3=row['t3'],
            signal_time=datetime.fromisoformat(row['signal_time'].replace('Z', '+00:00')),
            confidence=row['confidence'] or 0.0,
            composite_score=row['composite_score'] or 0.0,
            choch_detected=bool(row['choch_detected']),
            rsi_trendline_break=bool(row['rsi_trendline_break']),
            cluster_leader_rank=row['cluster_leader_rank'],
            status=SignalStatus(row['status']),
            outcome=SignalOutcome(row['outcome']),
            exit_price=row['exit_price'],
            exit_time=datetime.fromisoformat(row['exit_time'].replace('Z', '+00:00')) if row['exit_time'] else None,
            pnl_percent=row['pnl_percent'],
            max_favorable_excursion=row['max_favorable_excursion'],
            max_adverse_excursion=row['max_adverse_excursion']
        )

    def _to_yf_ticker(self, symbol: str) -> str | None:
        """Sembolü yfinance formatına çevir."""
        symbol = symbol.upper()
        crypto_map = {
            "BTC/USDT": "BTC-USD",
            "ETH/USDT": "ETH-USD",
            "INJ/USDT": "INJ-USD",
            "RNDR/USDT": "RNDR-USD",
            "FET/USDT": "FET-USD",
        }
        if symbol in crypto_map:
            return crypto_map[symbol]
        if symbol.endswith(".IS"):
            return symbol
        if symbol.isalpha():
            return symbol
        return None


# Singleton
_tracker: SignalTracker | None = None


def get_signal_tracker() -> SignalTracker:
    """Global SignalTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = SignalTracker()
    return _tracker
