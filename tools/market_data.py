"""
PROJECT OLYMPUS — Asenkron piyasa veri kancaları.
CCXT (kripto) + yfinance (makro/hisse) — bloke etmeyen yapı.
Bağlantı Havuzu (Global Exchange Cache) ile optimize edilmiş sürüm.
Kapsam, hata yönetimi ve yedek borsa algoritmaları %100 korunmuştur.
"""

from __future__ import annotations

import asyncio
import os
import ssl
from typing import Any

import aiohttp
import ccxt.async_support as ccxt_async
import certifi
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

RETRYABLE = (
    ccxt_async.NetworkError,
    ccxt_async.RequestTimeout,
    ccxt_async.ExchangeNotAvailable,
    ccxt_async.RateLimitExceeded,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    aiohttp.ClientConnectorCertificateError,
)

MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "SPY": "SPY",
    "NVDA": "NVDA",
}

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_DATA_URL = "https://data-api.binance.vision/api/v3/klines"
CRYPTO_FALLBACK_EXCHANGES = ("kraken", "okx", "kucoin")

# ── GLOBAL BAĞLANTI HAVUZU (Çirkin Uyarıları ve Bağlantı Kaybını Önler) ──
_EXCHANGES_CACHE: dict[str, Any] = {}


def build_ssl_context(verify: bool | None = None) -> ssl.SSLContext:
    if verify is None:
        verify = os.getenv("SSL_VERIFY", "true").lower() not in ("0", "false", "no")
    if verify:
        try:
            import truststore
            truststore.inject_into_ssl()
            return ssl.create_default_context()
        except ImportError:
            return ssl.create_default_context(cafile=certifi.where())
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _ssl_context(verify: bool | None = None) -> ssl.SSLContext:
    return build_ssl_context(verify)


def _aiohttp_connector(verify: bool | None = None) -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(ssl=_ssl_context(verify))


def _symbol_to_binance(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace("-", "")


def _get_exchange_instance(exchange_id: str) -> Any:
    """Borsayı her seferinde açıp kapatmak yerine global havuzdan çekerek korur."""
    global _EXCHANGES_CACHE
    if exchange_id not in _EXCHANGES_CACHE:
        exchange_cls = getattr(ccxt_async, exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Desteklenmeyen borsa: {exchange_id}")
        _EXCHANGES_CACHE[exchange_id] = exchange_cls(
            {
                "enableRateLimit": True,
                "timeout": 30_000,
                "aiohttp_connector": _aiohttp_connector(),
            }
        )
    return _EXCHANGES_CACHE[exchange_id]


def _normalize_yfinance_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(f"{ticker} icin veri bos dondu.")

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    rename_map = {
        "adj close": "close",
        "adjclose": "close",
    }
    df = df.rename(columns=rename_map)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ticker} OHLCV eksik kolonlar: {missing}")

    df = df.reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(f"{ticker} yetersiz bar sayisi: {len(df)}")

    return df


def _ohlcv_to_dataframe(raw: list[list[Any]]) -> pd.DataFrame:
    if not raw:
        raise ValueError("OHLCV verisi bos.")
    df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df.reset_index(drop=True)


def _is_crypto_symbol(symbol: str) -> bool:
    token = symbol.upper()
    return "/" in token or token.endswith("USDT") or token.endswith("USD")


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.set_index("timestamp")
    resampled = frame.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    return resampled.reset_index()


async def _fetch_yfinance_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    import yfinance as yf

    interval_map = {
        "1h": ("730d", "1h"),
        "4h": ("730d", "1h"),
        "1d": ("5y", "1d"),
        "1w": ("10y", "1wk"),
    }
    period, interval = interval_map.get(timeframe, ("5y", "1d"))
    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    df = _normalize_yfinance_df(raw, symbol)

    if timeframe == "4h":
        df = _resample_ohlcv(df, "4H")
    elif timeframe in ("1w", "1wk"):
        df = _resample_ohlcv(df, "W-FRI")

    return df.tail(max(limit, 20)).reset_index(drop=True)


async def _fetch_binance_klines_public(
    symbol: str,
    timeframe: str,
    limit: int,
    *,
    verify_ssl: bool | None = None,
    base_url: str = BINANCE_KLINES_URL,
) -> pd.DataFrame:
    """Binance public klines — CCXT fallback, markets yuklemez."""
    params = {
        "symbol": _symbol_to_binance(symbol),
        "interval": timeframe,
        "limit": limit,
    }
    timeout = aiohttp.ClientTimeout(total=35)
    async with aiohttp.ClientSession(
        connector=_aiohttp_connector(verify_ssl),
        timeout=timeout,
    ) as session:
        async with session.get(base_url, params=params) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ConnectionError(
                    f"Binance klines HTTP {resp.status} ({base_url}): {body[:200]}"
                )
            raw = await resp.json()

    rows = [
        [
            int(c[0]),
            float(c[1]),
            float(c[2]),
            float(c[3]),
            float(c[4]),
            float(c[5]),
        ]
        for c in raw
    ]
    return _ohlcv_to_dataframe(rows)


async def _fetch_binance_klines_with_ssl_fallback(
    symbol: str,
    timeframe: str,
    limit: int,
    base_url: str = BINANCE_KLINES_URL,
) -> pd.DataFrame:
    try:
        return await _fetch_binance_klines_public(
            symbol, timeframe, limit, verify_ssl=True, base_url=base_url
        )
    except aiohttp.ClientConnectorCertificateError:
        logger.warning(
            "SSL sertifika dogrulamasi basarisiz — dogrulamasiz fallback aktif. "
            "Uretimde SSL_VERIFY veya kurumsal CA yapilandirin."
        )
        return await _fetch_binance_klines_public(
            symbol, timeframe, limit, verify_ssl=False, base_url=base_url
        )


async def _fetch_crypto_multi_source(
    symbol: str,
    timeframe: str,
    limit: int,
    primary_exchange: str,
) -> pd.DataFrame:
    errors: list[str] = []

    try:
        return await _fetch_crypto_ohlcv_ccxt(symbol, timeframe, limit, primary_exchange)
    except Exception as exc:
        errors.append(f"{primary_exchange}:{exc}")

    for label, url in (
        ("binance_data", BINANCE_DATA_URL),
        ("binance_api", BINANCE_KLINES_URL),
    ):
        try:
            return await _fetch_binance_klines_with_ssl_fallback(
                symbol, timeframe, limit, base_url=url
            )
        except Exception as exc:
            errors.append(f"{label}:{exc}")

    for exchange_id in CRYPTO_FALLBACK_EXCHANGES:
        try:
            return await _fetch_crypto_ohlcv_ccxt(symbol, timeframe, limit, exchange_id)
        except Exception as exc:
            errors.append(f"{exchange_id}:{exc}")

    raise RuntimeError("Kripto OHLCV alinamadi — " + " | ".join(errors))


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=12),
    reraise=True,
)
async def _fetch_crypto_ohlcv_ccxt(
    symbol: str,
    timeframe: str,
    limit: int,
    exchange_id: str,
) -> pd.DataFrame:
    # ── GLOBAL BAĞLANTI HAVUZUNDAN ÇEKİLİR (REUSE) ──
    exchange = _get_exchange_instance(exchange_id)
    raw = await asyncio.wait_for(
        exchange.fetch_ohlcv(symbol.upper(), timeframe, limit=limit),
        timeout=35.0,
    )
    return _ohlcv_to_dataframe(raw)


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=12),
    reraise=True,
)
async def fetch_crypto_ohlcv(
    symbol: str,
    timeframe: str = "4h",
    limit: int = 200,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """
    Crypto semboller için CCXT/public klines kullanır.
    Crypto olmayan semboller için yfinance tabanlı OHLCV döndürür.
    """
    logger.debug(f"fetch_crypto_ohlcv: {symbol} {timeframe} limit={limit}")
    if _is_crypto_symbol(symbol):
        df = await _fetch_crypto_multi_source(symbol, timeframe, limit, exchange_id)
    else:
        df = await _fetch_yfinance_ohlcv(symbol, timeframe, limit)

    if len(df) < 20:
        raise ValueError(f"{symbol} yetersiz bar: {len(df)}")
    return df


def _download_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    return _normalize_yfinance_df(raw, ticker)


@retry(
    retry=retry_if_exception_type((RETRYABLE, ValueError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=12),
    reraise=True,
)
async def fetch_stock_macro_data(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Yahoo Finance verisini asyncio.to_thread ile bloke etmeden ceker.
    DXY (DX-Y.NYB), VIX (^VIX), SPY, NVDA vb. destekler.
    """
    yahoo_ticker = MACRO_TICKERS.get(ticker.upper(), ticker)
    logger.debug(f"yfinance download: {yahoo_ticker} period={period}")
    df = await asyncio.to_thread(_download_yfinance, yahoo_ticker, period, interval)
    return df


async def fetch_macro_bundle() -> dict[str, pd.DataFrame]:
    """DXY + VIX + SPY verisini paralel ceker."""
    tasks = {
        "DXY": fetch_stock_macro_data("DXY"),
        "VIX": fetch_stock_macro_data("VIX"),
        "SPY": fetch_stock_macro_data("SPY"),
    }
    keys = list(tasks.keys())
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    bundle: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for key, result in zip(keys, results):
        if isinstance(result, Exception):
            errors.append(f"{key}: {result}")
        else:
            bundle[key] = result

    if errors:
        raise RuntimeError("Makro veri hatasi — " + " | ".join(errors))
    return bundle


def pct_change_over(df: pd.DataFrame, bars: int = 5) -> float:
    """Son N barlik yuzde degisim."""
    if len(df) <= bars:
        bars = max(1, len(df) - 1)
    start = float(df["close"].iloc[-bars - 1])
    end = float(df["close"].iloc[-1])
    if start == 0:
        return 0.0
    return ((end - start) / start) * 100.0
