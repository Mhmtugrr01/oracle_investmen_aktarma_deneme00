"""Merkezi varlık sınıflandırıcı — tüm ajanların is_crypto / asset tipi tespiti buradan yapılır."""
from __future__ import annotations

from enum import Enum


class AssetType(str, Enum):
    CRYPTO = "crypto"
    US_STOCK = "us_stock"
    COMMODITY_GOLD = "commodity_gold"
    BIST_STOCK = "bist_stock"
    INDEX = "index"
    UNKNOWN = "unknown"


# Sabit eşleme kümeleri
_GOLD_SYMBOLS: frozenset[str] = frozenset({"GC=F", "XAUUSD", "GLD", "IAU", "GOLD"})
_SILVER_SYMBOLS: frozenset[str] = frozenset({"SI=F", "XAGUSD", "SLV"})
_INDEX_SYMBOLS: frozenset[str] = frozenset({"SPY", "QQQ", "DIA", "IWM", "^GSPC", "^NDX", "^DJI", "VIX", "^VIX"})
_BIST_SUFFIX = ".IS"
_CRYPTO_QUOTE_CURRENCIES: frozenset[str] = frozenset({"USDT", "USD", "BTC", "ETH", "BUSD", "USDC"})

# Bilinen ABD hisse sembolleri (genişletilebilir)
_KNOWN_US_STOCKS: frozenset[str] = frozenset({
    "TSLA", "NVDA", "MSTR", "COIN", "INTC", "AMD", "AAPL", "MSFT", "GOOGL",
    "META", "AMZN", "NFLX", "AMAT", "SOXX", "ARKK",
})


def classify_asset(symbol: str) -> AssetType:
    """
    Sembolü analiz ederek varlık tipini döndürür.
    Tüm ajanlar is_crypto / asset tipi kontrolü için bu fonksiyonu kullanır.
    """
    s = symbol.strip().upper()

    # Kripto: BTC/USDT formatı (CCXT) veya BTC-USD formatı (yfinance)
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2 and parts[1] in _CRYPTO_QUOTE_CURRENCIES:
            return AssetType.CRYPTO

    if s.endswith("-USD") or s.endswith("-USDT"):
        base = s.rsplit("-", 1)[0]
        # Altın için GC=F-USD gibi garip durum olmamalı ama güvenlik için kontrol
        if base not in _GOLD_SYMBOLS:
            return AssetType.CRYPTO

    # Altın ve gümüş emtia
    if s in _GOLD_SYMBOLS or s in _SILVER_SYMBOLS:
        return AssetType.COMMODITY_GOLD

    # BIST hisseleri (.IS uzantısı)
    if s.endswith(_BIST_SUFFIX):
        return AssetType.BIST_STOCK

    # Endeks sembolleri
    if s in _INDEX_SYMBOLS:
        return AssetType.INDEX

    # Bilinen ABD hisseleri
    if s in _KNOWN_US_STOCKS:
        return AssetType.US_STOCK

    # Bilinmeyen ama büyük harf alfanümerik → varsayılan ABD hissesi
    if s.isalpha() and s.isupper() and len(s) <= 5:
        return AssetType.US_STOCK

    return AssetType.UNKNOWN


def is_crypto(symbol: str) -> bool:
    return classify_asset(symbol) == AssetType.CRYPTO


def is_gold(symbol: str) -> bool:
    return classify_asset(symbol) == AssetType.COMMODITY_GOLD


def is_bist(symbol: str) -> bool:
    return classify_asset(symbol) == AssetType.BIST_STOCK


def is_us_stock(symbol: str) -> bool:
    return classify_asset(symbol) in (AssetType.US_STOCK, AssetType.INDEX)


def is_risk_asset(symbol: str) -> bool:
    """Kripto + hisse = risk varlığı (makro risk-off cezası uygulanır)."""
    return classify_asset(symbol) in (AssetType.CRYPTO, AssetType.US_STOCK, AssetType.INDEX)
