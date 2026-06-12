"""
HFT QUANT GÖZCÜ AJANI — Gelişmiş Piyasa Analizi
- USDT/BTC Dominance
- VIX, DXY, Total Market Cap
- RSI, EMA, MACD, Bollinger Bantları
- Otomatik Toplama/Dağıtım Bölgesi Tespiti
- ASLA otomatik işlem yapmaz — sadece analiz + Telegram inline onay
"""
import asyncio
from datetime import datetime
from core.llm import llm_call
from core.config import settings
from loguru import logger


async def run_quant_agent(task_description: str) -> str:
    logger.info("[QUANT AGENT] Starting advanced market analysis")

    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return "⚠️ yfinance kurulu değil."

    symbols = await _extract_symbols(task_description)
    logger.info(f"[QUANT AGENT] Symbols: {symbols}")

    macro, cg_data, fg_data = await asyncio.gather(
        _fetch_macro_data(),
        _fetch_coingecko_data(),
        _fetch_fear_greed(),
        return_exceptions=True,
    )
    if isinstance(macro, Exception):
        macro = {}
    if isinstance(cg_data, Exception):
        cg_data = {}
    if isinstance(fg_data, Exception):
        fg_data = {}

    analyses = []
    for sym in symbols[:5]:
        try:
            a = await _analyze_symbol(sym)
            analyses.append(a)
        except Exception as e:
            logger.error(f"[QUANT] {sym} error: {e}")
            analyses.append({"symbol": sym, "error": str(e)})

    report = await _build_full_report(macro, analyses, task_description, cg_data, fg_data)
    return report


async def _extract_symbols(task: str) -> list[str]:
    """Metinden analiz edilecek sembolleri çıkarır — LLM kullanmaz, kural tabanlı."""
    task_upper = task.upper()
    known_map = {
        "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
        "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
        "BNB": "BNB-USD", "SOL": "SOL-USD", "ADA": "ADA-USD",
        "XRP": "XRP-USD", "DOGE": "DOGE-USD",
        "AAPL": "AAPL", "MSFT": "MSFT", "GOOGL": "GOOGL",
        "TSLA": "TSLA", "NVDA": "NVDA", "AMZN": "AMZN",
        "GOLD": "GC=F", "ALTIN": "GC=F",
        "SILVER": "SI=F", "OIL": "CL=F",
        "SP500": "^GSPC", "NASDAQ": "^IXIC",
    }

    found = []
    for keyword, ticker in known_map.items():
        if keyword in task_upper and ticker not in found:
            found.append(ticker)

    if not found:
        found = ["BTC-USD", "ETH-USD"]

    return found[:5]


async def _fetch_coingecko_data() -> dict:
    """CoinGecko ücretsiz API — gerçek BTC dominance ve market cap."""
    import aiohttp
    result = {}
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.coingecko.com/api/v3/global"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    gd = data.get("data", {})
                    btc_dom = gd.get("market_cap_percentage", {}).get("btc", 0)
                    eth_dom = gd.get("market_cap_percentage", {}).get("eth", 0)
                    total_mc = gd.get("total_market_cap", {}).get("usd", 0)
                    total_vol = gd.get("total_volume", {}).get("usd", 0)
                    result = {
                        "btc_dominance": round(btc_dom, 2),
                        "eth_dominance": round(eth_dom, 2),
                        "total_market_cap_b": round(total_mc / 1e9, 1),
                        "total_volume_b": round(total_vol / 1e9, 1),
                    }
                    logger.debug(f"[QUANT COINGECKO] BTC Dom: %{btc_dom:.1f}, Total: ${total_mc/1e9:.0f}B")
    except Exception as e:
        logger.warning(f"[QUANT COINGECKO] Failed: {e}")
    return result


async def _fetch_fear_greed() -> dict:
    """Alternative.me Fear & Greed Index — ücretsiz, API key yok."""
    import aiohttp
    result = {}
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.alternative.me/fng/?limit=1"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    entry = data.get("data", [{}])[0]
                    value = int(entry.get("value", 0))
                    classification = entry.get("value_classification", "Nötr")
                    result = {"value": value, "classification": classification}
                    logger.debug(f"[QUANT F&G] {value} — {classification}")
    except Exception as e:
        logger.warning(f"[QUANT F&G] Failed: {e}")
    return result


async def _fetch_macro_data() -> dict:
    """Makro göstergeler: VIX, DXY, BTC/USDT Dominance, Total Market Cap."""
    import yfinance as yf

    macro = {}
    macro_tickers = {
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
        "SP500": "^GSPC",
        "GOLD": "GC=F",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "TOTAL_CRYPTO": "BTC-USD",
    }

    loop = asyncio.get_event_loop()

    for name, ticker_sym in macro_tickers.items():
        try:
            t = await loop.run_in_executor(None, lambda s=ticker_sym: yf.Ticker(s))
            hist = await loop.run_in_executor(None, lambda tt=t: tt.history(period="5d"))
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
                chg = ((price - prev) / prev) * 100
                macro[name] = {"price": price, "change_pct": round(chg, 2)}
        except Exception as e:
            logger.warning(f"[QUANT MACRO] {name} failed: {e}")
            macro[name] = {"price": 0, "change_pct": 0}

    try:
        btc_price = macro.get("BTC", {}).get("price", 0)
        eth_price = macro.get("ETH", {}).get("price", 0)
        total_approx = btc_price * 19_700_000 + eth_price * 120_000_000
        btc_dom = (btc_price * 19_700_000 / total_approx * 100) if total_approx > 0 else 0
        macro["BTC_DOM_APPROX"] = {"price": round(btc_dom, 1), "change_pct": 0}
    except Exception:
        macro["BTC_DOM_APPROX"] = {"price": 0, "change_pct": 0}

    return macro


async def _analyze_symbol(symbol: str) -> dict:
    """Tek sembol için tam teknik analiz: RSI, EMA, MACD, Bollinger, hacim."""
    import yfinance as yf
    import numpy as np

    clean = symbol.replace("/", "-").replace("USDT", "-USD")
    if clean.endswith("-USD-USD"):
        clean = clean[:-4]

    loop = asyncio.get_event_loop()
    ticker = await loop.run_in_executor(None, lambda: yf.Ticker(clean))
    hist = await loop.run_in_executor(None, lambda: ticker.history(period="90d"))

    if hist.empty:
        return {"symbol": symbol, "error": "Veri yok"}

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    rsi = _calc_rsi(close)
    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1]) if len(close) >= 50 else ema20
    ema200 = float(close.ewm(span=200).mean().iloc[-1]) if len(close) >= 200 else ema50

    macd_line, signal_line, histogram = _calc_macd(close)

    bb_upper, bb_mid, bb_lower = _calc_bollinger(close)

    current = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else current
    change_pct = ((current - prev) / prev) * 100

    vol_avg = float(volume.mean())
    vol_curr = float(volume.iloc[-1])
    vol_ratio = vol_curr / vol_avg if vol_avg > 0 else 1.0

    zone, zone_strength = _detect_zone(
        rsi=rsi, ema20=ema20, ema50=ema50, ema200=ema200,
        price=current, bb_upper=bb_upper, bb_lower=bb_lower,
        macd_hist=histogram, vol_ratio=vol_ratio, change_pct=change_pct,
    )

    signal, confidence = _generate_signal(
        rsi=rsi, ema20=ema20, ema50=ema50, price=current,
        change_pct=change_pct, macd_hist=histogram, vol_ratio=vol_ratio,
        bb_upper=bb_upper, bb_lower=bb_lower,
    )

    bb_pct = ((current - bb_lower) / (bb_upper - bb_lower) * 100) if (bb_upper - bb_lower) > 0 else 50

    return {
        "symbol": symbol,
        "price": current,
        "change_pct": round(change_pct, 2),
        "rsi": round(rsi, 1),
        "ema20": round(ema20, 4),
        "ema50": round(ema50, 4),
        "ema200": round(ema200, 4),
        "macd": round(macd_line, 4),
        "macd_signal": round(signal_line, 4),
        "macd_hist": round(histogram, 4),
        "bb_upper": round(bb_upper, 4),
        "bb_lower": round(bb_lower, 4),
        "bb_pct": round(bb_pct, 1),
        "vol_ratio": round(vol_ratio, 2),
        "zone": zone,
        "zone_strength": zone_strength,
        "signal": signal,
        "confidence": confidence,
    }


def _calc_rsi(prices, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.001)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if not (val != val) else 50.0


def _calc_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal).mean()
    hist = macd - sig
    return float(macd.iloc[-1]), float(sig.iloc[-1]), float(hist.iloc[-1])


def _calc_bollinger(prices, period=20, std_dev=2):
    mid = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = float((mid + std_dev * std).iloc[-1])
    lower = float((mid - std_dev * std).iloc[-1])
    mid_val = float(mid.iloc[-1])
    return upper, mid_val, lower


def _detect_zone(
    rsi, ema20, ema50, ema200, price,
    bb_upper, bb_lower, macd_hist, vol_ratio, change_pct
) -> tuple[str, str]:
    """
    Otomatik Toplama/Dağıtım Bölgesi Tespiti.
    Birden fazla göstergenin kesişimine göre karar verir.
    """
    acc_score = 0
    dist_score = 0

    if rsi < 35:
        acc_score += 3
    elif rsi < 45:
        acc_score += 1
    if rsi > 65:
        dist_score += 3
    elif rsi > 55:
        dist_score += 1

    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_pos = (price - bb_lower) / bb_range
        if bb_pos < 0.2:
            acc_score += 3
        elif bb_pos < 0.35:
            acc_score += 1
        if bb_pos > 0.80:
            dist_score += 3
        elif bb_pos > 0.65:
            dist_score += 1

    if macd_hist > 0 and macd_hist > abs(macd_hist) * 0.1:
        acc_score += 1
    elif macd_hist < 0:
        dist_score += 1

    if vol_ratio > 1.5 and change_pct > 0:
        acc_score += 2
    elif vol_ratio > 1.5 and change_pct < 0:
        dist_score += 2
    elif vol_ratio < 0.7:
        acc_score += 1

    if price > ema200:
        acc_score += 1
    else:
        dist_score += 1

    if acc_score >= 5:
        strength = "GÜÇLÜ" if acc_score >= 7 else "ORTA"
        return "🟢 TOPLAMA BÖLGESİ", strength
    elif dist_score >= 5:
        strength = "GÜÇLÜ" if dist_score >= 7 else "ORTA"
        return "🔴 DAĞITIM BÖLGESİ", strength
    else:
        return "🟡 NÖTR BÖLGE", "ZAYIF"


def _generate_signal(rsi, ema20, ema50, price, change_pct, macd_hist, vol_ratio, bb_upper, bb_lower) -> tuple[str, int]:
    bullish = 0
    if rsi < 40:
        bullish += 25
    elif rsi > 70:
        bullish -= 25

    if ema20 > ema50:
        bullish += 20
    else:
        bullish -= 20

    if price > ema20:
        bullish += 15
    else:
        bullish -= 15

    if macd_hist > 0:
        bullish += 15
    else:
        bullish -= 10

    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_pos = (price - bb_lower) / bb_range
        if bb_pos < 0.3:
            bullish += 10
        elif bb_pos > 0.7:
            bullish -= 10

    if vol_ratio > 1.5 and change_pct > 0:
        bullish += 15

    if change_pct > 1:
        bullish += 10
    elif change_pct < -1:
        bullish -= 10

    confidence = min(max(50 + bullish, 10), 95)
    if bullish > 25:
        return "LONG 📈", confidence
    elif bullish < -25:
        return "SHORT 📉", 100 - confidence
    else:
        return "NÖTR ↔️", 50


async def _build_full_report(
    macro: dict, analyses: list, task: str,
    cg_data: dict | None = None, fg_data: dict | None = None
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cg_data = cg_data or {}
    fg_data = fg_data or {}

    # Fear & Greed ikonu
    fg_val = fg_data.get("value", 0)
    fg_class = fg_data.get("classification", "")
    if fg_val < 25:
        fg_icon = "😱"
    elif fg_val < 40:
        fg_icon = "😨"
    elif fg_val < 60:
        fg_icon = "😐"
    elif fg_val < 75:
        fg_icon = "😊"
    else:
        fg_icon = "🤑"

    # Gerçek BTC dominance (CoinGecko varsa onu kullan)
    btc_dom_real = cg_data.get("btc_dominance", macro.get("BTC_DOM_APPROX", {}).get("price", 0))
    eth_dom = cg_data.get("eth_dominance", 0)
    total_mc = cg_data.get("total_market_cap_b", 0)
    total_vol = cg_data.get("total_volume_b", 0)

    lines = [
        "📊 *ORACLE QUANT RAPORU — TAM ANALİZ*",
        f"🕐 {now}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "🌍 *MAKRO GÖSTERGELER*",
    ]

    macro_items = [
        ("VIX", "😨 VIX", ""),
        ("DXY", "💵 DXY", ""),
        ("SP500", "📈 S&P500", ""),
        ("GOLD", "🏅 Altın", "$"),
        ("BTC", "₿ BTC", "$"),
        ("ETH", "Ξ ETH", "$"),
    ]
    for key, label, unit in macro_items:
        d = macro.get(key, {})
        price = d.get("price", 0)
        chg = d.get("change_pct", 0)
        icon = "↑" if chg > 0 else "↓"
        lines.append(f"  {label}: {unit}{price:,.2f} {icon}{abs(chg):.1f}%")

    lines.append("")
    lines.append("🔵 *KRİPTO PİYASASI*")
    lines.append(f"  BTC Dominance: %{btc_dom_real:.1f} | ETH: %{eth_dom:.1f}")
    if total_mc > 0:
        lines.append(f"  Toplam Market Cap: ${total_mc:.0f}B | Hacim: ${total_vol:.0f}B")
    if fg_val > 0:
        lines.append(f"  {fg_icon} Korku & Açgözlülük: {fg_val}/100 ({fg_class})")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("📉 *TEKNİK ANALİZ*")

    for r in analyses:
        if "error" in r:
            lines.append(f"\n❌ *{r['symbol']}*: {r['error']}")
            continue

        zone = r.get("zone", "🟡 NÖTR")
        strength = r.get("zone_strength", "")
        signal = r.get("signal", "NÖTR")
        conf = r.get("confidence", 50)
        sig_icon = "🟢" if "LONG" in signal else ("🔴" if "SHORT" in signal else "🟡")

        lines.append(f"""
{sig_icon} *{r['symbol']}* — {r['price']:,.4g} ({r['change_pct']:+.1f}%)
  📍 Bölge: {zone} ({strength})
  📊 RSI: {r['rsi']} | BB%: {r['bb_pct']}%
  📈 EMA20: {r['ema20']:,.4g} | EMA50: {r['ema50']:,.4g}
  ⚡ MACD Hist: {r['macd_hist']:+.4f}
  📦 Hacim: {r['vol_ratio']:.1f}x ort.
  🎯 Sinyal: {signal} — Güven: %{conf}""")

    vix = macro.get("VIX", {}).get("price", 0)
    if vix > 30:
        risk = "⚠️ VIX >30: Yüksek korku, volatilite artmış. Pozisyon küçültün."
    elif vix > 20:
        risk = "🟡 VIX 20-30: Orta volatilite. Dikkatli olun."
    else:
        risk = "🟢 VIX <20: Piyasa sakin. Risk iştahı normal."

    lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("🧠 *RİSK DEĞERLENDİRMESİ*")
    lines.append(risk)
    if fg_val > 0:
        if fg_val < 25:
            lines.append("😱 Aşırı Korku: Tarihsel olarak iyi alım fırsatı olabilir.")
        elif fg_val > 75:
            lines.append("🤑 Aşırı Açgözlülük: Dikkat — düzeltme riski yüksek.")
    lines.append("\n⚠️ *Bu analiz yatırım tavsiyesi değildir.*")
    lines.append("✅ Aksiyona geçmek için aşağıdaki butonu kullanın.")

    return "\n".join(lines)


async def run_scheduled_scan(send_alert_fn=None) -> str:
    """Zamanlayıcı tarafından çağrılır. Kritik sinyal varsa alert gönderir."""
    logger.info("[QUANT SCHEDULED] Running auto market scan")

    watchlist = ["BTC-USD", "ETH-USD", "AAPL", "GC=F"]
    macro = await _fetch_macro_data()
    alerts = []

    for sym in watchlist:
        try:
            a = await _analyze_symbol(sym)
            if "TOPLAMA" in a.get("zone", "") and "GÜÇLÜ" in a.get("zone_strength", ""):
                alerts.append(f"🟢 *{a['symbol']}* GÜÇLÜ TOPLAMA BÖLGESİNDE! RSI:{a['rsi']} Güven:%{a['confidence']}")
            elif "DAĞITIM" in a.get("zone", "") and "GÜÇLÜ" in a.get("zone_strength", ""):
                alerts.append(f"🔴 *{a['symbol']}* GÜÇLÜ DAĞITIM BÖLGESİNDE! RSI:{a['rsi']} Güven:%{a['confidence']}")
        except Exception as e:
            logger.error(f"[QUANT SCHEDULED] {sym}: {e}")

    if alerts and send_alert_fn:
        alert_msg = "🚨 *ORACLE QUANT ALARMI*\n━━━━━━━━━━━━━━━━\n" + "\n".join(alerts)
        await send_alert_fn(alert_msg)
        return alert_msg

    return "Scan OK — kritik sinyal yok"
