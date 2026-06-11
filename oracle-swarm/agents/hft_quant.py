import asyncio
from datetime import datetime, timedelta
from core.llm import llm_call
from core.config import settings
from loguru import logger


async def run_quant_agent(task_description: str) -> str:
    """
    HFT Quant Gözcü: Piyasa analizi yapar, ASLA otomatik işlem açmaz.
    Sadece analiz sonucunu Telegram inline butonu ile sunar.
    """
    logger.info("[QUANT AGENT] Starting market analysis")

    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return "⚠️ yfinance kurulu değil. `pip install yfinance` çalıştırın."

    symbols = await _extract_symbols(task_description)
    logger.info(f"[QUANT AGENT] Analyzing symbols: {symbols}")

    results = []
    for symbol in symbols[:5]:
        try:
            analysis = await _analyze_symbol(symbol)
            results.append(analysis)
        except Exception as e:
            logger.error(f"[QUANT AGENT] {symbol} error: {e}")
            results.append({"symbol": symbol, "error": str(e)})

    report = await _generate_quant_report(results, task_description)
    return report


async def _extract_symbols(task: str) -> list[str]:
    """Görev metninden sembol/ticker çıkarır."""
    response = await llm_call(
        messages=[{"role": "user", "content": f"Bu metinden borsa sembollerini çıkar (sadece virgülle ayrılmış liste): {task}"}],
        system="Sadece geçerli borsa sembollerini döndür (örn: BTC/USDT, AAPL, ETH/USDT). Bulamazsan BTCUSDT,AAPL,ETH-USD döndür.",
        temperature=0.1,
        max_tokens=100,
    )
    symbols = [s.strip() for s in response.split(",") if s.strip()]
    return symbols[:5] if symbols else ["BTC-USD", "AAPL", "ETH-USD"]


async def _analyze_symbol(symbol: str) -> dict:
    """Tek bir sembol için teknik analiz yapar."""
    import yfinance as yf
    import pandas as pd

    clean_symbol = symbol.replace("/", "-").replace("USDT", "-USD").replace("USD", "-USD")
    if clean_symbol.endswith("-USD-USD"):
        clean_symbol = clean_symbol[:-4]

    loop = asyncio.get_event_loop()
    ticker = await loop.run_in_executor(None, lambda: yf.Ticker(clean_symbol))
    hist = await loop.run_in_executor(None, lambda: ticker.history(period="30d"))

    if hist.empty:
        return {"symbol": symbol, "error": "Veri bulunamadı"}

    close = hist["Close"]
    volume = hist["Volume"]

    rsi = _calculate_rsi(close)
    ema_20 = float(close.ewm(span=20).mean().iloc[-1])
    ema_50 = float(close.ewm(span=50).mean().iloc[-1]) if len(close) >= 50 else ema_20
    current_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2]) if len(close) > 1 else current_price
    price_change_pct = ((current_price - prev_price) / prev_price) * 100

    volume_avg = float(volume.mean())
    volume_current = float(volume.iloc[-1])
    volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1.0

    signal, confidence = _generate_signal(rsi, ema_20, ema_50, current_price, price_change_pct)

    return {
        "symbol": symbol,
        "price": current_price,
        "change_pct": round(price_change_pct, 2),
        "rsi": round(rsi, 1),
        "ema_20": round(ema_20, 2),
        "ema_50": round(ema_50, 2),
        "volume_ratio": round(volume_ratio, 2),
        "signal": signal,
        "confidence": confidence,
    }


def _calculate_rsi(prices, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.001)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _generate_signal(rsi: float, ema20: float, ema50: float, price: float, change: float) -> tuple[str, int]:
    bullish = 0
    if rsi < 40:
        bullish += 30
    elif rsi > 70:
        bullish -= 30

    if ema20 > ema50:
        bullish += 25
    elif ema20 < ema50:
        bullish -= 25

    if price > ema20:
        bullish += 20
    else:
        bullish -= 20

    if change > 1:
        bullish += 15
    elif change < -1:
        bullish -= 15

    confidence = min(max(50 + bullish, 10), 95)

    if bullish > 20:
        return "LONG 📈", confidence
    elif bullish < -20:
        return "SHORT 📉", 100 - confidence
    else:
        return "NÖTR ↔️", 50


async def _generate_quant_report(results: list[dict], task: str) -> str:
    lines = [
        "📊 *ORACLE QUANT RAPORU*",
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for r in results:
        if "error" in r:
            lines.append(f"\n❌ *{r['symbol']}*: {r['error']}")
            continue

        signal_icon = "🟢" if "LONG" in r["signal"] else ("🔴" if "SHORT" in r["signal"] else "🟡")
        lines.append(f"""
{signal_icon} *{r['symbol']}*
💰 Fiyat: ${r['price']:,.2f} ({r['change_pct']:+.1f}%)
📊 RSI: {r['rsi']} | EMA20: ${r['ema_20']:,.2f}
📈 Sinyal: {r['signal']} — Güven: %{r['confidence']}
📦 Hacim: {r['volume_ratio']:.1f}x ortalama""")

    lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("⚠️ *Bu analiz yatırım tavsiyesi değildir.*")
    lines.append("✅ Onay için aşağıdaki butonu kullanın.")

    return "\n".join(lines)
