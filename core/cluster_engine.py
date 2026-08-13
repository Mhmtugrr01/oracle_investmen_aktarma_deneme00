"""
PROJECT OLYMPUS — The Cluster Engine
Korelasyon kümeleme + Göreceli Güç (Relative Strength) lider seçimi.
Aynı temayı takip eden varlıkları gruplar, sadece en güçlü liderleri seçer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


@dataclass
class ClusterMember:
    """Küme içindeki bir varlık ve skorları."""
    symbol: str
    relative_strength: float  # 0-1
    volume_momentum: float    # 0-1
    signal_clarity: float     # 0-1 (structure break + RSI trendline temizliği)
    liquidity_usd: float      # Günlük ortalama USD hacmi
    alpha_score: float        # Ağırlıklı toplam (0-1)


@dataclass
class SignalCluster:
    """Bir tema/grup içindeki varlıklar."""
    cluster_id: int
    theme_description: str
    members: list[ClusterMember]
    leaders: list[ClusterMember]  # Top 3-5


class ClusterEngine:
    """
    Sinyal korelasyon kümeleme motoru.
    
    Amaç: Aynı anda 40 varlık aynı sinyali veriyorsa, bunları 40 ayrı fırsat
    olarak değil, tek bir "tema" olarak ele almak ve sadece en güçlü 3-5
    liderini seçmek.
    """

    # Varsayılan ağırlıklar (config'den override edilebilir)
    DEFAULT_WEIGHTS = {
        "relative_strength": 0.35,
        "volume_momentum": 0.30,
        "signal_clarity": 0.20,
        "liquidity_gate": 0.15,  # Binary gate olarak kullanılır
    }

    CORRELATION_THRESHOLD = 0.75  # Kümeleme eşiği
    MIN_LIQUIDITY_USD = 10_000_000  # Minimum günlük hacim ($10M)
    TOP_LEADERS_COUNT = 3  # Her kümeden seçilecek lider sayısı

    def __init__(self, config_weights: dict[str, float] | None = None):
        self.weights = config_weights or self.DEFAULT_WEIGHTS.copy()
        self._price_cache: dict[str, pd.Series] = {}

    async def cluster_signals(
        self,
        symbols: list[str],
        lookback_days: int = 30,
        fetch_close: Any | None = None,
    ) -> list[SignalCluster]:
        """
        Sinyal veren varlıkları korelasyonlarına göre kümele.
        
        Args:
            symbols: Sinyal üreten varlık sembolleri listesi
            lookback_days: Korelasyon hesabı için gün sayısı
            fetch_close: Opsiyonel async(symbol) -> OHLCV DataFrame | None callback.
                Verilirse yfinance yerine bu kullanılır (scanner market_data cache'i).
                Böylece prefilter ile aynı veri tekrar indirilmez (Faz 2.2 ilkesi).
            
        Returns:
            SignalCluster listesi (her küme bir tema)
        """
        # FAZ 3/4 — Göreli güç benchmark'ı: kripto kümeleri için BTC getirisi
        # ("BTC düşerken en az düşen lider" kuralı).
        has_crypto = any("/" in str(s) for s in symbols)
        benchmark_returns: dict[str, float] | None = None
        if has_crypto:
            benchmark_returns = await self._fetch_benchmark_returns(fetch_close=fetch_close)

        if len(symbols) < 2:
            # Tek varlık varsa kümeleme gereksiz
            if symbols:
                member = await self._score_symbol(
                    symbols[0], lookback_days, fetch_close=fetch_close,
                    benchmark_returns=benchmark_returns,
                )
                return [SignalCluster(
                    cluster_id=0,
                    theme_description=f"Tekil varlık: {symbols[0]}",
                    members=[member],
                    leaders=[member],
                )]
            return []

        # 1. Fiyat verilerini çek
        price_data = await self._fetch_price_data(symbols, lookback_days, fetch_close=fetch_close)
        
        if len(price_data) < 2:
            logger.warning("[CLUSTER] Yetersiz fiyat verisi, kümeleme atlanıyor")
            return []

        # 2. Korelasyon matrisi hesapla
        corr_matrix = self._compute_correlation_matrix(price_data)
        
        # 3. Kümeleme (basit threshold-based clustering)
        clusters = self._cluster_by_correlation(symbols, corr_matrix)
        
        # 4. Her küme için skorlama ve lider seçimi
        result: list[SignalCluster] = []
        for idx, cluster_symbols in enumerate(clusters):
            members = []
            for sym in cluster_symbols:
                member = await self._score_symbol(
                    sym, lookback_days, fetch_close=fetch_close,
                    pre_fetched=price_data.get(sym),
                    benchmark_returns=benchmark_returns,
                )
                members.append(member)
            
            # Liderleri seç (alpha_score'a göre sırala)
            sorted_members = sorted(members, key=lambda m: m.alpha_score, reverse=True)
            leaders = sorted_members[:self.TOP_LEADERS_COUNT]
            
            theme = self._generate_theme_description(cluster_symbols, price_data)
            
            result.append(SignalCluster(
                cluster_id=idx,
                theme_description=theme,
                members=members,
                leaders=leaders,
            ))
        
        logger.info(f"[CLUSTER] {len(symbols)} varlık → {len(result)} küme oluşturuldu")
        return result

    async def _fetch_price_data(
        self, symbols: list[str], days: int, fetch_close: Any | None = None
    ) -> dict[str, pd.Series]:
        """Varlıkların kapanış fiyat serilerini çek."""
        price_data: dict[str, pd.Series] = {}
        
        async def fetch_one(sym: str) -> tuple[str, pd.Series | None]:
            try:
                # 1) Enjekte edilmiş veri kaynağı varsa önce onu dene (cache dostu)
                if fetch_close is not None:
                    try:
                        df = await fetch_close(sym)
                        if df is not None and not df.empty:
                            close = df["Close"].dropna() if "Close" in df.columns else (
                                df["close"].dropna() if "close" in df.columns else df.iloc[:, 0].dropna()
                            )
                            if len(close) >= 10:
                                return sym, close
                    except Exception:
                        pass  # yfinance fallback'e düş
                # 2) yfinance fallback
                ticker = self._to_yf_ticker(sym)
                if not ticker:
                    return sym, None
                
                data = await asyncio.to_thread(
                    yf.download,
                    ticker,
                    period=f"{days}d",
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                
                if data is None or data.empty:
                    return sym, None
                
                close = data["Close"].dropna()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                
                return sym, close
            except Exception as e:
                logger.warning(f"[CLUSTER] {sym} fiyat çekilemedi: {e}")
                return sym, None
        
        # Paralel çekim
        tasks = [fetch_one(s) for s in symbols]
        results = await asyncio.gather(*tasks)
        
        for sym, series in results:
            if series is not None and len(series) >= 10:
                price_data[sym] = series
        
        return price_data

    async def _fetch_benchmark_returns(
        self, fetch_close: Any | None = None
    ) -> dict[str, float] | None:
        """
        BTC/USDT benchmark kapanış serisinden 5g/10g/15g getirilerini çek.

        FAZ 3/4 — Göreli güç: "BTC düşerken en az düşen lider" kuralı için
        benchmark. BTC çekilemezse None döner (skorlama kendi momentumuna düşer).
        """
        close: pd.Series | None = None
        for bsym in ("BTC/USDT", "BTC-USD"):
            # 1) Enjekte edilmiş veri kaynağı (scanner market_data cache'i)
            if fetch_close is not None:
                try:
                    df = await fetch_close(bsym)
                    if df is not None and not df.empty:
                        close = (
                            df["Close"].dropna() if "Close" in df.columns
                            else (df["close"].dropna() if "close" in df.columns else df.iloc[:, 0].dropna())
                        )
                        if close is not None and len(close) >= 15:
                            break
                        close = None
                except Exception:
                    close = None
            # 2) yfinance fallback
            try:
                data = await asyncio.to_thread(
                    yf.download, bsym, period="60d", interval="1d",
                    progress=False, auto_adjust=True,
                )
                if data is not None and not data.empty:
                    close = data["Close"].dropna()
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    if close is not None and len(close) >= 15:
                        break
                    close = None
            except Exception:
                close = None

        if close is None or len(close) < 15:
            return None

        vals = close.astype(float)

        def _ret(n: int) -> float:
            base = float(vals.iloc[-n])
            return (float(vals.iloc[-1]) / base - 1.0) * 100.0 if base > 0 else 0.0

        return {"5d": _ret(5), "10d": _ret(10), "15d": _ret(15)}

    def _compute_correlation_matrix(
        self, price_data: dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Pearson korelasyon matrisi hesapla."""
        # DataFrame oluştur
        df = pd.DataFrame(price_data)
        df = df.dropna()
        
        if len(df) < 5:
            logger.warning("[CLUSTER] Korelasyon için yetersiz ortak veri")
            return pd.DataFrame()
        
        return df.corr(method="pearson")

    def _cluster_by_correlation(
        self, symbols: list[str], corr_matrix: pd.DataFrame
    ) -> list[list[str]]:
        """
        Basit threshold-based clustering.
        Korelasyon > threshold olan varlıkları aynı kümede topla.
        """
        if corr_matrix.empty:
            return [[s] for s in symbols]  # Her biri kendi kümesi
        
        clusters: list[set[str]] = []
        assigned: set[str] = set()
        
        for sym in symbols:
            if sym in assigned:
                continue
            
            # Bu sembolle yüksek korelasyonlu diğerlerini bul
            cluster = {sym}
            assigned.add(sym)
            
            if sym in corr_matrix.index:
                for other in symbols:
                    if other in assigned or other == sym:
                        continue
                    if other in corr_matrix.columns:
                        corr_val = corr_matrix.loc[sym, other]
                        if pd.notna(corr_val) and corr_val >= self.CORRELATION_THRESHOLD:
                            cluster.add(other)
                            assigned.add(other)
            
            clusters.append(cluster)
        
        return [list(c) for c in clusters]

    async def _score_symbol(
        self,
        symbol: str,
        days: int,
        fetch_close: Any | None = None,
        pre_fetched: pd.Series | None = None,
        benchmark_returns: dict[str, float] | None = None,
    ) -> ClusterMember:
        """
        Bir varlık için 4 faktörlü skor hesapla:
        1. Relative Strength (BTC'ye göre göreli güç — benchmark varsa)
        2. Volume Momentum (hacim artışı)
        3. Signal Clarity (teknik sinyal netliği)
        4. Liquidity (USD hacmi)
        """
        try:
            data = None
            # 1) Önceden çekilmiş kapanış varsa onu kullan (cache dostu)
            if pre_fetched is not None and len(pre_fetched) >= 10:
                close = pre_fetched.astype(float).dropna()
                volume = pd.Series(dtype=float)  # hacim yoksa liquidity düşük kalır
                data = close
                # fetch_close gerçek OHLCV döndürebilir — hacim için yine de yfinance'a
                # düşmeyelim; volume momentum 0 varsayılıp alpha skoru güvenli kalır.
            # 2) Aksi halde yfinance'dan çek (hacim dahil)
            if data is None:
                ticker = self._to_yf_ticker(symbol)
                if not ticker:
                    return self._default_member(symbol)
                
                raw = await asyncio.to_thread(
                    yf.download,
                    ticker,
                    period=f"{days * 2}d",  # Volume momentum için daha fazla veri
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                
                if raw is None or len(raw) < 10:
                    return self._default_member(symbol)
                
                close = raw["Close"].astype(float).dropna()
                volume = raw["Volume"].astype(float)
            
            if len(close) < 10:
                return self._default_member(symbol)
            
            # 1. Relative Strength: Son 5-10 barın % değişimi (BTC'ye göre, benchmark varsa)
            returns_5d = (float(close.iloc[-1]) / float(close.iloc[-5]) - 1) * 100
            returns_10d = (float(close.iloc[-1]) / float(close.iloc[-10]) - 1) * 100
            if benchmark_returns is not None:
                # FAZ 3/4: BTC düşerken en az düşen = en yüksek göreli güç (beta/güç rasyosu)
                rel_5d = returns_5d - float(benchmark_returns.get("5d", 0.0))
                rel_10d = returns_10d - float(benchmark_returns.get("10d", 0.0))
                rs_raw = (rel_5d * 0.6 + rel_10d * 0.4)
            else:
                rs_raw = (returns_5d * 0.6 + returns_10d * 0.4)  # benchmark yoksa kendi momentumu
            
            # 2. Volume Momentum: Son 5 bar / Önceki 5 bar ortalama hacim
            if volume is not None and len(volume) >= 10:
                recent_vol = volume.iloc[-5:].mean()
                prev_vol = volume.iloc[-10:-5].mean()
                vol_momentum = (recent_vol / prev_vol - 1) * 100 if prev_vol > 0 else 0
            else:
                vol_momentum = 0.0
            
            # 3. Signal Clarity: Fiyat yapısının "temizliği"
            # Basit versiyon: son 20 barın volatilitesi (düşük vol = temiz trend)
            returns = close.pct_change().dropna()
            volatility = returns.iloc[-20:].std() * 100
            clarity = max(0, 100 - volatility * 5)  # Düşük volatilite = yüksek netlik
            
            # 4. Liquidity: Günlük ortalama USD hacmi (fiyat * hacim)
            if volume is not None and len(volume) >= 20:
                avg_daily_volume_usd = (close.iloc[-20:] * volume.iloc[-20:]).mean()
            else:
                avg_daily_volume_usd = 0.0
            
            # Normalize et (0-1)
            rs_norm = min(1.0, max(0.0, (rs_raw + 20) / 40))  # -20% → +20% arası
            vol_norm = min(1.0, max(0.0, (vol_momentum + 50) / 100))  # -50% → +50%
            clarity_norm = min(1.0, max(0.0, clarity / 100))
            liquidity_ok = avg_daily_volume_usd >= self.MIN_LIQUIDITY_USD
            
            # Alpha Score: Ağırlıklı toplam
            alpha_score = (
                rs_norm * self.weights["relative_strength"]
                + vol_norm * self.weights["volume_momentum"]
                + clarity_norm * self.weights["signal_clarity"]
                + (1.0 if liquidity_ok else 0.0) * self.weights["liquidity_gate"]
            )
            
            return ClusterMember(
                symbol=symbol,
                relative_strength=rs_norm,
                volume_momentum=vol_norm,
                signal_clarity=clarity_norm,
                liquidity_usd=avg_daily_volume_usd,
                alpha_score=round(alpha_score, 4),
            )
            
        except Exception as e:
            logger.error(f"[CLUSTER] {symbol} skorlama hatası: {e}")
            return self._default_member(symbol)

    def _generate_theme_description(
        self, symbols: list[str], price_data: dict[str, pd.Series]
    ) -> str:
        """Küme için tema açıklaması üret."""
        if not symbols:
            return "Bilinmeyen tema"
        
        # Ortak özellikler analizi
        crypto_count = sum(1 for s in symbols if "/" in s or s.endswith("USDT"))
        stock_count = len(symbols) - crypto_count
        
        if crypto_count > stock_count:
            base = "Kripto teması"
            if any("BTC" in s.upper() for s in symbols):
                base = "BTC-korele kripto grubu"
        else:
            base = "Hisse senedi teması"
        
        return f"{base} ({len(symbols)} varlık)"

    def _to_yf_ticker(self, symbol: str) -> str | None:
        """Sembolü yfinance formatına çevir."""
        symbol = symbol.upper()

        # Genel kripto eşlemesi: BTC/USDT → BTC-USD (40+ sembolü tek kuralda çözer)
        if "/" in symbol:
            base = symbol.split("/")[0]
            if base in ("USDT", "USDC", "BUSD"):
                return None
            return f"{base}-USD"

        # BIST
        if symbol.endswith(".IS"):
            return symbol

        # ABD hisse
        if symbol.isalpha():
            return symbol

        return None

    def _default_member(self, symbol: str) -> ClusterMember:
        """Varsayılan (düşük skorlu) üye."""
        return ClusterMember(
            symbol=symbol,
            relative_strength=0.5,
            volume_momentum=0.5,
            signal_clarity=0.5,
            liquidity_usd=0.0,
            alpha_score=0.3,  # Düşük varsayılan
        )


# ═════════════════════════════════════════════════════════════════════════════
# Singleton erişimi
# ═════════════════════════════════════════════════════════════════════════════

_cluster_engine: ClusterEngine | None = None


def get_cluster_engine() -> ClusterEngine:
    """Global ClusterEngine instance."""
    global _cluster_engine
    if _cluster_engine is None:
        _cluster_engine = ClusterEngine()
    return _cluster_engine


async def cluster_and_rank_signals(
    symbols: list[str],
    lookback_days: int = 30,
    fetch_close: Any | None = None,
) -> list[SignalCluster]:
    """Convenience function: Sinyalleri kümele ve sırala."""
    engine = get_cluster_engine()
    return await engine.cluster_signals(symbols, lookback_days, fetch_close=fetch_close)
