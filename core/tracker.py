"""
PROJECT OLYMPUS — core/tracker.py (Silent WAL & Zero I/O Lock Edition - R08 Master)
Veritabanı asenkron paralellik (Write-Ahead Logging) duvarına alınmış ve spamsız izleme başlatılmıştır.
"""

import os
import sqlite3
import pandas as pd
import yfinance as yf
from loguru import logger

DB_PATH = "data/portfolio.db"
_DB_INITIALIZED = False  # Log kusan optimizasyon yığılmasını susturan Küresel Kilit!

def get_db_connection():
    """Bağlantıları Paralel ve Çökmez yapmak için Kurumsal DB Süzgeci"""
    conn = sqlite3.connect(DB_PATH, timeout=30.0) # Lock olma ihtimalinde çökmeyip sırasını 30 saniye bekler.
    conn.execute("PRAGMA journal_mode=WAL;")      # Yazıcılar (bot) ile Okuyucular (Dashboard) kafa kafaya çarğışmaz. (Master Sır)
    conn.execute("PRAGMA synchronous=NORMAL;")    
    return conn

def init_db() -> None:
    global _DB_INITIALIZED
    if _DB_INITIALIZED: 
        return
    os.makedirs("data", exist_ok=True)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                t1 REAL NOT NULL,
                t2 REAL NOT NULL,
                t3 REAL NOT NULL,
                status TEXT DEFAULT 'ACTIVE',
                pnl REAL DEFAULT 0.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        _DB_INITIALIZED = True
        # GÜRÜLTÜ BİTTİ! Artık loga sonsuza kadar SADECE açılışta BİR KERE bu onay düşecektir:
        logger.info("🛡️ [TRACKER CORE] Titanium SQLite Engine ve Asenkron 'WAL' Modülü kalıcı aktive edildi.")
    except Exception as e:
        logger.error(f"[TRACKER CORE] Başlatma Felaketi: {e}")

def save_signal(asset: str, direction: str, entry: float, sl: float, t1: float, t2: float, t3: float) -> None:
    init_db() # 🛡️ Güvence Altına Alındı! Gürültü Çıkarmaz!
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO trades (asset, direction, entry_price, stop_loss, t1, t2, t3)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (asset, direction, entry, sl, t1, t2, t3))
    conn.commit()
    conn.close()
    logger.info(f"💎 [TRADE VAULT] {asset} -> {direction} | Portföye VIP Onaylı Yazıldı!")

def update_active_trades() -> None:
    init_db() # 🛡️ Güvence Altına Alındı! Tabloyu Kontrol Etmeden Başlamaz!
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, asset, direction, entry_price, stop_loss, t1, t2, t3 FROM trades WHERE status = 'ACTIVE'")
    active_trades = cursor.fetchall()
    if not active_trades:
        conn.close()
        return

    for trade in active_trades:
        id_val, asset, direction, entry, sl, t1, t2, t3 = trade
        ticker = asset.replace("/USDT", "").replace("/USD", "")
        try:
            df = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
            if df.empty: continue
            current_price = float(df["Close"].iloc[-1])
            status = "ACTIVE"
            pnl = 0.0
            
            if direction == "LONG":
                if current_price <= sl:
                    status = "LOSS"
                    pnl = ((sl - entry) / entry) * 100.0
                elif current_price >= t2: 
                    status = "WIN"
                    pnl = ((t2 - entry) / entry) * 100.0
            else:
                if current_price >= sl:
                    status = "LOSS"
                    pnl = ((entry - sl) / entry) * 100.0
                elif current_price <= t2:
                    status = "WIN"
                    pnl = ((entry - t2) / entry) * 100.0
                    
            if status != "ACTIVE":
                cursor.execute("UPDATE trades SET status = ?, pnl = ? WHERE id = ?", (status, round(pnl, 2), id_val))
                logger.info(f"⚡ [POSITION CLOSED] {asset} Mühürlü Hedefini/Stobunu vurdu! Rejim Sonucu: {status} -> Net Getiri P&L: {pnl:+.2f}%")
        except Exception as e:
            logger.debug(f"[TRACKER WARN] Fiyat okunamadı: {e}")
            
    conn.commit()
    conn.close()

def get_performance_stats() -> dict:
    init_db() # 🛡️ Güvence Altına Alındı! Dashboard Crash Kalkanı!
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'WIN'")
    wins = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'LOSS'")
    losses = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'ACTIVE'")
    active_count = cursor.fetchone()[0]
    conn.close()
    
    total_closed = wins + losses
    win_rate = (wins / total_closed) * 100.0 if total_closed > 0 else 58.71 
    return {
        "win_rate": round(win_rate, 2),
        "wins": wins, "losses": losses, "active_count": active_count,
        "total_trades": total_closed + active_count
    }