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
    ) -> list[SignalCluster]:
        """
        Sinyal veren varlıkları korelasyonlarına göre kümele.
        
        Args:
            symbols: Sinyal üreten varlık sembolleri listesi
            lookback_days: Korelasyon hesabı için gün sayısı
            
        Returns:
            SignalCluster listesi (her küme bir tema)
        """
        if len(symbols) < 2:
            # Tek varlık varsa kümeleme gereksiz
            if symbols:
                member = await self._score_symbol(symbols[0], lookback_days)
                return [SignalCluster(
                    cluster_id=0,
                    theme_description=f"Tekil varlık: {symbols[0]}",
                    members=[member],
                    leaders=[member],
                )]
            return []

        # 1. Fiyat verilerini çek
        price_data = await self._fetch_price_data(symbols, lookback_days)
        
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
                member = await self._score_symbol(sym, lookback_days)
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
        self, symbols: list[str], days: int
    ) -> dict[str, pd.Series]:
        """Varlıkların kapanış fiyat serilerini çek."""
        price_data: dict[str, pd.Series] = {}
        
        async def fetch_one(sym: str) -> tuple[str, pd.Series | None]:
            try:
                # Kripto için yfinance formatına çevir
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

    async def _score_symbol(self, symbol: str, days: int) -> ClusterMember:
        """
        Bir varlık için 4 faktörlü skor hesapla:
        1. Relative Strength (küme ortalamasına göre güç)
        2. Volume Momentum (hacim artışı)
        3. Signal Clarity (teknik sinyal netliği)
        4. Liquidity (USD hacmi)
        """
        try:
            ticker = self._to_yf_ticker(symbol)
            if not ticker:
                return self._default_member(symbol)
            
            data = await asyncio.to_thread(
                yf.download,
                ticker,
                period=f"{days * 2}d",  # Volume momentum için daha fazla veri
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            
            if data is None or len(data) < 10:
                return self._default_member(symbol)
            
            close = data["Close"]
            volume = data["Volume"]
            
            # 1. Relative Strength: Son 5-10 barın % değişimi
            returns_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100
            returns_10d = (close.iloc[-1] / close.iloc[-10] - 1) * 100
            rs_raw = (returns_5d * 0.6 + returns_10d * 0.4)  # 0-100 arası normalize edilecek
            
            # 2. Volume Momentum: Son 5 bar / Önceki 5 bar ortalama hacim
            recent_vol = volume.iloc[-5:].mean()
            prev_vol = volume.iloc[-10:-5].mean()
            vol_momentum = (recent_vol / prev_vol - 1) * 100 if prev_vol > 0 else 0
            
            # 3. Signal Clarity: Fiyat yapısının "temizliği"
            # Basit versiyon: son 20 barın volatilitesi (düşük vol = temiz trend)
            returns = close.pct_change().dropna()
            volatility = returns.iloc[-20:].std() * 100
            clarity = max(0, 100 - volatility * 5)  # Düşük volatilite = yüksek netlik
            
            # 4. Liquidity: Günlük ortalama USD hacmi (fiyat * hacim)
            avg_daily_volume_usd = (close.iloc[-20:] * volume.iloc[-20:]).mean()
            
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
        
        # Kripto mapping
        crypto_map = {
            "BTC/USDT": "BTC-USD",
            "ETH/USDT": "ETH-USD",
            "INJ/USDT": "INJ-USD",
            "RNDR/USDT": "RNDR-USD",
            "FET/USDT": "FET-USD",
        }
        
        if symbol in crypto_map:
            return crypto_map[symbol]
        
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
) -> list[SignalCluster]:
    """Convenience function: Sinyalleri kümele ve sırala."""
    engine = get_cluster_engine()
    return await engine.cluster_signals(symbols, lookback_days)
