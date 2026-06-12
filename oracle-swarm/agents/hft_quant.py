"""
ORACLE HFT QUANT AJAN V3 — Kurumsal Seviye Piyasa Analizi

Özellikler:
- Çok zaman çerçeveli analiz: Günlük + 4H (1H veriden türetilir)
- RSI < 25 + F&G < 20 = GÜÇLÜ ALIM (tarihsel kanıta dayalı doğru skorlama)
- Destek/Direnç: Fibonacci, EMA seviyeleri, psikolojik bölgeler
- Kesin Giriş Bölgesi + Stop-Loss + Kâr Hedefleri (T1/T2/T3)
- Risk/Ödül oranı + Pozisyon büyüklüğü önerisi
- ASLA otomatik işlem yapmaz — sadece analiz + Telegram inline onay
"""
import asyncio
from datetime import datetime
from loguru import logger


# ─── Ana Giriş ───────────────────────────────────────────────────────────────

async def run_quant_agent(task_description: str) -> str:
    logger.info("[QUANT AGENT] Kurumsal piyasa analizi başlatılıyor")

    try:
        import yfinance as yf
    except ImportError:
        return "⚠️ yfinance kurulu değil: `pip install yfinance`"

    symbols = _extract_symbols(task_description)
    logger.info(f"[QUANT AGENT] Semboller: {symbols}")

    # Paralel veri çekimi
    macro, cg_data, fg_data = await asyncio.gather(
        _fetch_macro_data(),
        _fetch_coingecko_data(),
        _fetch_fear_greed(),
        return_exceptions=True,
    )
    macro = macro if not isinstance(macro, Exception) else {}
    cg_data = cg_data if not isinstance(cg_data, Exception) else {}
    fg_data = fg_data if not isinstance(fg_data, Exception) else {}

    fg_value = fg_data.get("value", 50)
    vix = macro.get("VIX", {}).get("price", 15.0)

    analyses = []
    for sym in symbols[:4]:
        try:
            a = await _analyze_symbol_full(sym, fg_value=fg_value, vix=vix)
            analyses.append(a)
        except Exception as e:
            logger.error(f"[QUANT] {sym} hatası: {e}")
            analyses.append({"symbol": sym, "error": str(e)})

    report = _build_professional_report(macro, analyses, cg_data, fg_data, task_description)
    return report


# ─── Sembol Çıkarma ───────────────────────────────────────────────────────────

def _extract_symbols(task: str) -> list[str]:
    task_upper = task.upper()
    known = {
        "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
        "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
        "BNB": "BNB-USD", "SOL": "SOL-USD", "ADA": "ADA-USD",
        "XRP": "XRP-USD", "DOGE": "DOGE-USD", "AVAX": "AVAX-USD",
        "LINK": "LINK-USD", "DOT": "DOT-USD", "MATIC": "MATIC-USD",
        "AAPL": "AAPL", "MSFT": "MSFT", "GOOGL": "GOOGL",
        "TSLA": "TSLA", "NVDA": "NVDA", "AMZN": "AMZN",
        "GOLD": "GC=F", "ALTIN": "GC=F", "GÜMÜŞ": "SI=F",
        "OIL": "CL=F", "SP500": "^GSPC", "NASDAQ": "^IXIC",
    }
    found = []
    for kw, ticker in known.items():
        if kw in task_upper and ticker not in found:
            found.append(ticker)
    return found[:4] if found else ["BTC-USD", "ETH-USD"]


# ─── Veri Kaynakları ─────────────────────────────────────────────────────────

async def _fetch_coingecko_data() -> dict:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=aiohttp.ClientTimeout(total=8)
            ) as r:
                if r.status == 200:
                    d = (await r.json()).get("data", {})
                    return {
                        "btc_dominance": round(d.get("market_cap_percentage", {}).get("btc", 0), 2),
                        "eth_dominance": round(d.get("market_cap_percentage", {}).get("eth", 0), 2),
                        "total_market_cap_b": round(d.get("total_market_cap", {}).get("usd", 0) / 1e9, 1),
                        "total_volume_b": round(d.get("total_volume", {}).get("usd", 0) / 1e9, 1),
                    }
    except Exception as e:
        logger.warning(f"[QUANT CG] {e}")
    return {}


async def _fetch_fear_greed() -> dict:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=aiohttp.ClientTimeout(total=6)
            ) as r:
                if r.status == 200:
                    entry = (await r.json()).get("data", [{}])[0]
                    return {
                        "value": int(entry.get("value", 50)),
                        "classification": entry.get("value_classification", "Nötr"),
                    }
    except Exception as e:
        logger.warning(f"[QUANT F&G] {e}")
    return {}


async def _fetch_macro_data() -> dict:
    import yfinance as yf
    loop = asyncio.get_event_loop()
    macro = {}
    tickers = {
        "VIX": "^VIX", "DXY": "DX-Y.NYB", "SP500": "^GSPC",
        "GOLD": "GC=F", "BTC": "BTC-USD", "ETH": "ETH-USD",
    }
    for name, sym in tickers.items():
        try:
            t = await loop.run_in_executor(None, lambda s=sym: yf.Ticker(s))
            h = await loop.run_in_executor(None, lambda tt=t: tt.history(period="5d"))
            if not h.empty:
                price = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2]) if len(h) > 1 else price
                macro[name] = {
                    "price": price,
                    "change_pct": round((price - prev) / prev * 100, 2),
                }
        except Exception as e:
            logger.warning(f"[QUANT MACRO] {name}: {e}")
    return macro


# ─── Teknik Analiz — Temel Hesaplamalar ──────────────────────────────────────

def _calc_rsi(prices, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if not (val != val) else 50.0


def _calc_macd(prices, fast=12, slow=26, signal=9):
    ef = prices.ewm(span=fast).mean()
    es = prices.ewm(span=slow).mean()
    macd = ef - es
    sig = macd.ewm(span=signal).mean()
    return float(macd.iloc[-1]), float(sig.iloc[-1]), float((macd - sig).iloc[-1])


def _calc_bollinger(prices, period=20, std_dev=2.0):
    mid = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    return (
        float((mid + std_dev * std).iloc[-1]),
        float(mid.iloc[-1]),
        float((mid - std_dev * std).iloc[-1]),
    )


def _calc_atr(high, low, close, period=14) -> float:
    tr_list = []
    for i in range(1, min(period + 1, len(close))):
        hl = float(high.iloc[-i]) - float(low.iloc[-i])
        hc = abs(float(high.iloc[-i]) - float(close.iloc[-i - 1]))
        lc = abs(float(low.iloc[-i]) - float(close.iloc[-i - 1]))
        tr_list.append(max(hl, hc, lc))
    return sum(tr_list) / len(tr_list) if tr_list else float(close.iloc[-1]) * 0.02


def _calc_stochastic(high, low, close, k_period=14, d_period=3) -> tuple[float, float]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    rng = highest_high - lowest_low
    k = 100 * (close - lowest_low) / rng.replace(0, 1e-10)
    d = k.rolling(d_period).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])


# ─── Sinyal Motoru V2 — Tarihsel Kanıta Dayalı Doğru Skorlama ────────────────

def _generate_signal_v2(
    rsi: float, ema20: float, ema50: float, ema200: float,
    price: float, change_pct: float, macd_hist: float,
    vol_ratio: float, bb_upper: float, bb_lower: float,
    fg_value: float = 50, vix: float = 15,
    rsi_4h: float = 50, stoch_k: float = 50,
) -> dict:
    """
    Kurumsal seviye çok faktörlü sinyal motoru.
    RSI<25 + F&G<20 tarihsel olarak Bitcoin'de en güçlü alım setupu.
    """
    score = 0
    bull_reasons = []
    bear_reasons = []

    # ── RSI (En kritik gösterge, ±45 puan) ──────────────────────────
    if rsi <= 20:
        score += 45
        bull_reasons.append(f"RSI {rsi:.1f} — EKSTRİM AŞIRI SATIM (tarihsel dip bölgesi)")
    elif rsi <= 25:
        score += 38
        bull_reasons.append(f"RSI {rsi:.1f} — Güçlü aşırı satım (nadir sinyal)")
    elif rsi <= 30:
        score += 28
        bull_reasons.append(f"RSI {rsi:.1f} — Aşırı satım bölgesi")
    elif rsi <= 40:
        score += 12
        bull_reasons.append(f"RSI {rsi:.1f} — Hafif oversold")
    elif rsi >= 80:
        score -= 45
        bear_reasons.append(f"RSI {rsi:.1f} — AŞIRI ALIM (dönüş riski kritik)")
    elif rsi >= 70:
        score -= 28
        bear_reasons.append(f"RSI {rsi:.1f} — Aşırı alım bölgesi")
    elif rsi >= 60:
        score -= 12

    # ── Korku & Açgözlülük (Kripto için kritik, ±30 puan) ────────────
    if fg_value <= 15:
        score += 30
        bull_reasons.append(f"Korku&Açgözlülük {fg_value}/100 — EKSTRİM KORKU (Buffett: 'Herkes korkarken al')")
    elif fg_value <= 25:
        score += 22
        bull_reasons.append(f"Korku&Açgözlülük {fg_value}/100 — Korku bölgesi (tarihsel alım fırsatı)")
    elif fg_value <= 40:
        score += 8
    elif fg_value >= 85:
        score -= 30
        bear_reasons.append(f"Korku&Açgözlülük {fg_value}/100 — EKSTRİM AÇGÖZLÜLÜK (Buffett: 'Herkes açgözlüyken sat')")
    elif fg_value >= 75:
        score -= 20
        bear_reasons.append(f"Korku&Açgözlülük {fg_value}/100 — Açgözlülük bölgesi")

    # ── Bollinger Band Konumu (±20 puan) ─────────────────────────────
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_pos = (price - bb_lower) / bb_range
        if bb_pos <= 0.10:
            score += 20
            bull_reasons.append(f"BB% {bb_pos*100:.0f} — Alt banda değiyor (reversal bölgesi)")
        elif bb_pos <= 0.25:
            score += 12
            bull_reasons.append(f"BB% {bb_pos*100:.0f} — Alt BB bölgesinde")
        elif bb_pos >= 0.90:
            score -= 20
            bear_reasons.append(f"BB% {bb_pos*100:.0f} — Üst banda değiyor (dönüş riski)")
        elif bb_pos >= 0.75:
            score -= 12

    # ── Stochastic (%K, ±15 puan) ─────────────────────────────────────
    if stoch_k <= 15:
        score += 15
        bull_reasons.append(f"Stochastic %K {stoch_k:.0f} — Aşırı satım")
    elif stoch_k <= 25:
        score += 8
    elif stoch_k >= 85:
        score -= 15
        bear_reasons.append(f"Stochastic %K {stoch_k:.0f} — Aşırı alım")
    elif stoch_k >= 75:
        score -= 8

    # ── 4H RSI Uyumu (±12 puan) ───────────────────────────────────────
    if rsi_4h <= 30 and rsi <= 35:
        score += 12
        bull_reasons.append(f"4H RSI {rsi_4h:.0f} + Günlük RSI {rsi:.0f} — Çift zaman çerçevesi sinyal uyumu")
    elif rsi_4h >= 70 and rsi >= 60:
        score -= 12

    # ── EMA Yapısı (±12 puan) ─────────────────────────────────────────
    if price > ema20 > ema50:
        score += 12
        bull_reasons.append("Fiyat EMA20 > EMA50 üzerinde — sağlıklı yükseliş trendi")
    elif price > ema200:
        score += 5
        bull_reasons.append("Fiyat EMA200 üzerinde — uzun vadeli yükseliş trendi korunuyor")
    elif price < ema20 < ema50:
        score -= 8  # Küçük penaltı — RSI aşırı satım durumunda override edilebilir

    # ── MACD (±10 puan) ──────────────────────────────────────────────
    if macd_hist > 0:
        score += 10
        bull_reasons.append("MACD pozitif — momentum yükseliş lehine")
    else:
        score -= 5  # Düşük penaltı (RSI extreme iken MACD gecikebilir)

    # ── Hacim Analizi (±15 puan) ──────────────────────────────────────
    if vol_ratio > 1.8 and change_pct > 0:
        score += 15
        bull_reasons.append(f"Hacim {vol_ratio:.1f}x ort. + yükseliş — kurumsal alım sinyali")
    elif vol_ratio > 1.5 and change_pct > 0:
        score += 8
    elif vol_ratio > 1.5 and change_pct < 0:
        score -= 10
        bear_reasons.append(f"Hacim {vol_ratio:.1f}x ort. + düşüş — satış baskısı")

    # ── Makro Bağlam (±10 puan) ───────────────────────────────────────
    if vix < 20:
        score += 5  # Sakin piyasa ortamı
    elif vix > 30:
        score -= 10
        bear_reasons.append(f"VIX {vix:.0f} — Yüksek piyasa korkusu, volatilite riski")

    # ── Sinyal Kararı ─────────────────────────────────────────────────
    if score >= 65:
        signal, action, emoji = "GÜÇLÜ ALIM", "AL 🚀", "🟢🟢"
    elif score >= 40:
        signal, action, emoji = "ALIM", "AL 📈", "🟢"
    elif score >= 18:
        signal, action, emoji = "HAFİF ALIM", "TEMKİNLİ AL ↗️", "🟡"
    elif score <= -65:
        signal, action, emoji = "GÜÇLÜ SATIM", "SAT 🔴", "🔴🔴"
    elif score <= -40:
        signal, action, emoji = "SATIM", "SAT 📉", "🔴"
    elif score <= -18:
        signal, action, emoji = "HAFİF SATIM", "TEMKİNLİ SAT ↘️", "🟡"
    else:
        signal, action, emoji = "NÖTR", "BEKLE ↔️", "⚪"

    confidence = min(int(abs(score) * 1.1 + 35), 95)

    return {
        "score": score,
        "signal": signal,
        "action": action,
        "emoji": emoji,
        "confidence": confidence,
        "bull_reasons": bull_reasons[:4],
        "bear_reasons": bear_reasons[:3],
    }


# ─── Destek/Direnç + Giriş/Çıkış Seviyeleri ─────────────────────────────────

def _calc_key_levels(
    close, high, low, price: float, ema20: float, ema50: float,
    ema200: float, bb_upper: float, bb_lower: float, atr: float, signal_score: int
) -> dict:
    """
    Fibonacci, EMA ve psikolojik seviyelere dayalı destek/direnç hesaplama.
    """
    # Son 60 günlük swing high/low
    n = min(len(close), 60)
    recent_high = float(high.iloc[-n:].max())
    recent_low = float(low.iloc[-n:].min())

    # Fibonacci seviyeleri (son high→low)
    fib_range = recent_high - recent_low
    fibs = {
        "0%": recent_low,
        "23.6%": recent_low + fib_range * 0.236,
        "38.2%": recent_low + fib_range * 0.382,
        "50%": recent_low + fib_range * 0.500,
        "61.8%": recent_low + fib_range * 0.618,
        "78.6%": recent_low + fib_range * 0.786,
        "100%": recent_high,
    }

    # Tüm aday seviyeleri topla
    all_levels = list(fibs.values()) + [ema20, ema50, ema200, bb_upper, bb_lower]

    # Psikolojik yuvarlak sayılar (BTC için $5k, diğerleri için %5 yuvarla)
    if price > 10000:
        step = 5000
    elif price > 1000:
        step = 500
    elif price > 100:
        step = 50
    elif price > 10:
        step = 5
    else:
        step = 0.5
    psych = [round(price / step) * step * m for m in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]]
    all_levels += psych

    # Destek: fiyatın altındaki seviyeler
    supports = sorted([l for l in all_levels if l < price * 0.998 and l > price * 0.6], reverse=True)
    # Direnç: fiyatın üstündeki seviyeler
    resistances = sorted([l for l in all_levels if l > price * 1.002 and l < price * 1.5])

    # Tekrarlı seviyeleri temizle (birbirine %1.5'ten yakın olanları birleştir)
    def deduplicate(levels: list, pct_thresh=0.015) -> list:
        if not levels:
            return []
        out = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - out[-1]) / out[-1] > pct_thresh:
                out.append(lvl)
        return out

    supports = deduplicate(supports)[:4]
    resistances = deduplicate(resistances)[:4]

    # Giriş/Stop-Loss/Hedef hesaplama
    if signal_score > 0:  # Yükseliş sinyali
        entry_low = max(supports[0], price - atr * 0.4) if supports else price - atr * 0.4
        entry_high = price + atr * 0.2
        stop_loss = (supports[1] - atr * 0.2) if len(supports) > 1 else (price - atr * 3.5)
        stop_loss = max(stop_loss, price * 0.88)  # Max %12 SL
        targets = resistances[:3] if resistances else [price * 1.08, price * 1.15, price * 1.22]
    else:  # Düşüş sinyali
        entry_low = price - atr * 0.2
        entry_high = (resistances[0] + atr * 0.3) if resistances else price + atr * 0.5
        stop_loss = (resistances[1] + atr * 0.2) if len(resistances) > 1 else price * 1.08
        targets = supports[:3] if supports else [price * 0.92, price * 0.85, price * 0.78]

    # Risk/Ödül
    risk = abs(price - stop_loss)
    reward = abs((targets[1] if len(targets) > 1 else targets[0]) - price)
    rr = round(reward / risk, 2) if risk > 0 else 0

    return {
        "supports": supports,
        "resistances": resistances,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "targets": targets,
        "risk_reward": rr,
        "sl_pct": round((price - stop_loss) / price * 100, 1) if signal_score > 0 else round((stop_loss - price) / price * 100, 1),
        "atr": atr,
        "period_high": recent_high,
        "period_low": recent_low,
    }


# ─── Sembol Tam Analizi ───────────────────────────────────────────────────────

async def _analyze_symbol_full(symbol: str, fg_value: float = 50, vix: float = 15) -> dict:
    """
    Günlük + 4H çoklu zaman çerçeveli tam teknik analiz.
    """
    import yfinance as yf
    import pandas as pd

    clean = symbol.replace("/", "-")
    if clean.endswith("-USD-USD"):
        clean = clean[:-4]

    loop = asyncio.get_event_loop()
    ticker = await loop.run_in_executor(None, lambda: yf.Ticker(clean))

    # ── Günlük veri (90 gün) ──
    hist_d = await loop.run_in_executor(None, lambda: ticker.history(period="90d"))
    if hist_d.empty:
        return {"symbol": symbol, "error": "Veri alınamadı"}

    close_d = hist_d["Close"]
    high_d = hist_d["High"]
    low_d = hist_d["Low"]
    vol_d = hist_d["Volume"]

    # ── Günlük göstergeler ──
    rsi_d = _calc_rsi(close_d)
    ema20 = float(close_d.ewm(span=20).mean().iloc[-1])
    ema50 = float(close_d.ewm(span=50).mean().iloc[-1]) if len(close_d) >= 50 else ema20
    ema200 = float(close_d.ewm(span=200).mean().iloc[-1]) if len(close_d) >= 200 else ema50
    macd_line, macd_sig, macd_hist = _calc_macd(close_d)
    bb_upper, bb_mid, bb_lower = _calc_bollinger(close_d)
    atr = _calc_atr(high_d, low_d, close_d)
    stoch_k, stoch_d = _calc_stochastic(high_d, low_d, close_d)

    current = float(close_d.iloc[-1])
    prev = float(close_d.iloc[-2]) if len(close_d) > 1 else current
    change_pct = (current - prev) / prev * 100
    vol_avg = float(vol_d.mean())
    vol_curr = float(vol_d.iloc[-1])
    vol_ratio = vol_curr / vol_avg if vol_avg > 0 else 1.0

    bb_pct = ((current - bb_lower) / (bb_upper - bb_lower) * 100) if (bb_upper - bb_lower) > 0 else 50

    # ── 4H veri (son 30 gün, 1h'ten türetilir) ──
    rsi_4h = rsi_d  # Fallback
    try:
        hist_1h = await loop.run_in_executor(
            None, lambda: ticker.history(period="30d", interval="1h")
        )
        if not hist_1h.empty and len(hist_1h) > 20:
            hist_4h = hist_1h.resample("4h").agg({
                "Open": "first", "High": "max",
                "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna()
            if len(hist_4h) >= 14:
                rsi_4h = _calc_rsi(hist_4h["Close"])
                logger.debug(f"[QUANT 4H] {symbol} RSI 4H: {rsi_4h:.1f}")
    except Exception as e:
        logger.debug(f"[QUANT 4H] {symbol} 4H veri alınamadı: {e}")

    # ── Sinyal motoru ──
    sig = _generate_signal_v2(
        rsi=rsi_d, ema20=ema20, ema50=ema50, ema200=ema200,
        price=current, change_pct=change_pct, macd_hist=macd_hist,
        vol_ratio=vol_ratio, bb_upper=bb_upper, bb_lower=bb_lower,
        fg_value=fg_value, vix=vix, rsi_4h=rsi_4h, stoch_k=stoch_k,
    )

    # ── Destek/direnç ve giriş seviyeleri ──
    levels = _calc_key_levels(
        close=close_d, high=high_d, low=low_d, price=current,
        ema20=ema20, ema50=ema50, ema200=ema200,
        bb_upper=bb_upper, bb_lower=bb_lower,
        atr=atr, signal_score=sig["score"],
    )

    return {
        "symbol": symbol,
        "price": current,
        "change_pct": round(change_pct, 2),
        "rsi_d": round(rsi_d, 1),
        "rsi_4h": round(rsi_4h, 1),
        "stoch_k": round(stoch_k, 1),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "ema200": round(ema200, 2),
        "macd_hist": round(macd_hist, 2),
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "bb_pct": round(bb_pct, 1),
        "vol_ratio": round(vol_ratio, 2),
        "atr": round(atr, 2),
        "signal": sig,
        "levels": levels,
    }


# ─── Raporlama ────────────────────────────────────────────────────────────────

def _fmt_price(p: float) -> str:
    """Fiyatı okunabilir biçimde formatlar. Bilimsel gösterim yok."""
    if p >= 10000:
        return f"${p:,.0f}"
    elif p >= 100:
        return f"${p:,.1f}"
    elif p >= 1:
        return f"${p:,.3f}"
    else:
        return f"${p:.6f}"


def _fmt_pct(p: float, sign: bool = True) -> str:
    prefix = "+" if sign and p > 0 else ""
    return f"{prefix}{p:.1f}%"


def _build_professional_report(
    macro: dict, analyses: list, cg_data: dict,
    fg_data: dict, task: str
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fg_val = fg_data.get("value", 0)
    fg_class = fg_data.get("classification", "")

    fg_icon = (
        "😱" if fg_val < 20 else
        "😨" if fg_val < 35 else
        "😐" if fg_val < 55 else
        "😊" if fg_val < 70 else "🤑"
    )

    btc_dom = cg_data.get("btc_dominance", 0)
    eth_dom = cg_data.get("eth_dominance", 0)
    total_mc = cg_data.get("total_market_cap_b", 0)
    total_vol = cg_data.get("total_volume_b", 0)

    lines = [
        "📊 *ORACLE QUANT — KURUMSAL ANALİZ*",
        f"🕐 {now}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "🌍 *MAKRO BAĞLAM*",
    ]

    macro_items = [
        ("VIX", "😨 VIX", False),
        ("DXY", "💵 DXY", False),
        ("SP500", "📈 S&P500", True),
        ("GOLD", "🏅 Altın", True),
        ("BTC", "₿ BTC", True),
        ("ETH", "Ξ ETH", True),
    ]
    for key, label, use_dollar in macro_items:
        d = macro.get(key, {})
        p = d.get("price", 0)
        c = d.get("change_pct", 0)
        icon = "↑" if c > 0 else "↓"
        pstr = f"${p:,.2f}" if use_dollar and p > 100 else f"{p:.2f}"
        lines.append(f"  {label}: {pstr} {icon}{abs(c):.1f}%")

    lines += ["", "🔵 *KRİPTO PİYASASI*"]
    lines.append(f"  BTC Dom: %{btc_dom:.1f} | ETH: %{eth_dom:.1f}")
    if total_mc > 0:
        lines.append(f"  Total MC: ${total_mc:.0f}B | 24h Hacim: ${total_vol:.0f}B")
    if fg_val > 0:
        lines.append(f"  {fg_icon} Korku&Açgözlülük: {fg_val}/100 ({fg_class})")

    # ── Makro Yorum ──
    vix = macro.get("VIX", {}).get("price", 0)
    sp_chg = macro.get("SP500", {}).get("change_pct", 0)
    macro_comment = []
    if vix < 20:
        macro_comment.append("VIX<20 → Piyasa sakin, panik yok")
    elif vix > 30:
        macro_comment.append("⚠️ VIX>30 → Yüksek belirsizlik")
    if sp_chg > 1:
        macro_comment.append("S&P500 güçlü → risk iştahı yüksek")
    if fg_val < 20:
        macro_comment.append("Ekstrim korku = tarihsel alım fırsatı")
    if macro_comment:
        lines.append(f"  💡 {' | '.join(macro_comment)}")

    lines += ["", "━━━━━━━━━━━━━━━━━━━━━━", "📉 *TEKNİK ANALİZ*"]

    for r in analyses:
        if "error" in r:
            lines.append(f"\n❌ {r['symbol']}: {r['error']}")
            continue

        sig = r["signal"]
        lv = r["levels"]
        price = r["price"]

        sig_display = f"{sig['emoji']} *{sig['signal']}*"
        conf = sig["confidence"]

        lines.append(f"""
{sig['emoji']} *{r['symbol']}* — {_fmt_price(price)} ({_fmt_pct(r['change_pct'])})
━━━━━━━━━━━━━━━━━━━━━━
📊 *Göstergeler:*
  RSI Günlük: {r['rsi_d']} | RSI 4H: {r['rsi_4h']}
  Stochastic %K: {r['stoch_k']}
  BB%: {r['bb_pct']:.0f} (alt: {_fmt_price(r['bb_lower'])} | üst: {_fmt_price(r['bb_upper'])})
  EMA20: {_fmt_price(r['ema20'])} | EMA50: {_fmt_price(r['ema50'])} | EMA200: {_fmt_price(r['ema200'])}
  MACD Hist: {r['macd_hist']:+.1f} | Hacim: {r['vol_ratio']:.1f}x ort.

📍 *Bölge Analizi:*""")

        if sig["bull_reasons"]:
            for reason in sig["bull_reasons"]:
                lines.append(f"  🟢 {reason}")
        if sig["bear_reasons"]:
            for reason in sig["bear_reasons"]:
                lines.append(f"  🔴 {reason}")

        lines.append(f"""
🎯 *NİHAİ KARAR — {sig_display}*
   Güven: %{conf} | Skor: {sig['score']:+d}/100

📍 *Giriş Bölgesi:* {_fmt_price(lv['entry_low'])} — {_fmt_price(lv['entry_high'])}
🛑 *Stop-Loss:* {_fmt_price(lv['stop_loss'])} (-%{lv['sl_pct']:.1f})""")

        for i, t in enumerate(lv["targets"][:3], 1):
            t_pct = (t - price) / price * 100
            lines.append(f"🎯 *Hedef {i}:* {_fmt_price(t)} ({_fmt_pct(t_pct)})")

        lines.append(f"📊 *Risk/Ödül (T2):* 1:{lv['risk_reward']:.2f}")

        # Destek/Direnç seviyeleri
        if lv["supports"]:
            sup_str = " → ".join([_fmt_price(s) for s in lv["supports"][:3]])
            lines.append(f"🔵 *Destek:* {sup_str}")
        if lv["resistances"]:
            res_str = " → ".join([_fmt_price(r2) for r2 in lv["resistances"][:3]])
            lines.append(f"🔴 *Direnç:* {res_str}")

        # Aksiyon özeti
        action_lines = [
            "",
            f"⚡ *AKSIYON: {sig['action']}*",
        ]
        if sig["score"] >= 40:
            action_lines += [
                f"  → {_fmt_price(lv['entry_low'])} — {_fmt_price(lv['entry_high'])} arasında kademeli giriş",
                f"  → SL: {_fmt_price(lv['stop_loss'])} (sabit tut, asla genişletme)",
                f"  → T1 ({_fmt_price(lv['targets'][0])}) karı kilitle, kalana T2'yi hedefle",
                f"  → Pozisyon büyüklüğü: Maks. %5 portföy riski",
            ]
        elif sig["score"] <= -40:
            action_lines += [
                f"  → Short veya nakde çekil",
                f"  → SL: {_fmt_price(lv['stop_loss'])}",
                f"  → Uzun pozisyon açmaktan kaçın",
            ]
        else:
            action_lines += [
                f"  → Net sinyal yok, kenar bekleniyor",
                f"  → {_fmt_price(lv['supports'][0]) if lv['supports'] else '?'} destek kırarsa dikkat",
            ]
        lines.extend(action_lines)

    # ── Genel Risk Notu ──
    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️ *Bu analiz yatırım tavsiyesi değildir.*",
        "Oracle ASLA otomatik işlem açmaz.",
    ]

    return "\n".join(lines)


# ─── Zamanlanmış Tarama ───────────────────────────────────────────────────────

async def run_scheduled_scan(send_alert_fn=None) -> str:
    logger.info("[QUANT SCHEDULED] Otomatik piyasa taraması")
    watchlist = ["BTC-USD", "ETH-USD", "GC=F"]

    macro = await _fetch_macro_data()
    fg_data = await _fetch_fear_greed()
    fg_value = fg_data.get("value", 50)
    vix = macro.get("VIX", {}).get("price", 15.0)

    alerts = []
    for sym in watchlist:
        try:
            a = await _analyze_symbol_full(sym, fg_value=fg_value, vix=vix)
            if "error" in a:
                continue
            sig = a["signal"]
            lv = a["levels"]
            if sig["score"] >= 60:
                alerts.append(
                    f"🚨 *{a['symbol']}* GÜÇLÜ ALIM!\n"
                    f"RSI: {a['rsi_d']} | F&G: {fg_value} | Güven: %{sig['confidence']}\n"
                    f"Giriş: {_fmt_price(lv['entry_low'])} — SL: {_fmt_price(lv['stop_loss'])}"
                )
            elif sig["score"] <= -60:
                alerts.append(
                    f"⚠️ *{a['symbol']}* GÜÇLÜ SATIM!\n"
                    f"RSI: {a['rsi_d']} | F&G: {fg_value} | Güven: %{sig['confidence']}"
                )
        except Exception as e:
            logger.error(f"[QUANT SCAN] {sym}: {e}")

    if alerts and send_alert_fn:
        msg = "🔔 *ORACLE QUANT ALARMI*\n━━━━━━━━━━━━━━━━\n" + "\n\n".join(alerts)
        await send_alert_fn(msg)
        return msg

    return f"✅ Tarama OK — {len(watchlist)} sembol incelendi, kritik sinyal yok"
