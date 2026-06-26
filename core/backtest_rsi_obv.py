"""
backtest_rsi_obv.py
Gerçek tarihsel test — look-ahead bias yok.
Sinyal: RSI<35 + OBV yükseliyor + Hacim>1.2x → giriş bir sonraki gün
Çıkış: RSI>58 VEYA stop -10%
"""
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

ASSETS = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "SPY": "SPY",
    "GC=F": "Altın"
}


def run(ticker, name, period="3y"):
    """3 yıllık tarihi test — şu anki tarihten önceki veriler."""
    raw = yf.download(ticker, period=period, interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) < 100:
        print(f"{name}: veri yetersiz")
        return

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df.ffill().dropna()

    # Teknik indikatörler
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["obv"] = ta.obv(df["close"], df["volume"])
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"]
    df["obv_up"] = df["obv"] > df["obv"].shift(5)

    # Sinyal: RSI<35 + OBV yükseliyor + Hacim %20 üstü
    df["signal"] = (df["rsi"] < 35) & df["obv_up"] & (df["vol_ratio"] > 1.2)

    # Backtest: işlem döngüsü
    trades = []
    in_trade = False
    entry = 0.0
    idx = 0
    
    for i in range(25, len(df) - 1):
        if not in_trade and df.iloc[i]["signal"]:
            # Bir sonraki gün aç
            in_trade = True
            entry = float(df.iloc[i + 1]["close"])
            idx = i
        elif in_trade:
            cur = float(df.iloc[i]["close"])
            pct = (cur - entry) / entry * 100
            rsi_exit = float(df.iloc[i]["rsi"]) > 58
            stop = pct < -10
            
            if (rsi_exit or stop) and i - idx >= 3:
                trades.append({
                    "pct": round(pct, 2),
                    "days": i - idx,
                    "exit": "rsi" if rsi_exit else "stop"
                })
                in_trade = False

    if not trades:
        print(f"{name}: sinyal üretilmedi")
        return

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t.pct > 0]
    losses = df_t[df_t.pct <= 0]
    wr = len(wins) / len(df_t) * 100
    aw = wins.pct.mean() if len(wins) else 0
    al = abs(losses.pct.mean()) if len(losses) else 1
    
    print(f"\n{name} | İşlem: {len(df_t)} | WR: {wr:.1f}%"
          f" | Kazanç: +{aw:.1f}% | Kayıp: -{al:.1f}%"
          f" | R:R: 1:{aw/al:.1f}")
    print(f"  En iyi: +{df_t.pct.max():.1f}% | "
          f"En kötü: {df_t.pct.min():.1f}%")


if __name__ == "__main__":
    print("=" * 70)
    print("RSI + OBV Backtest — 3 Yıl Tarihsel Veri")
    print("=" * 70)
    for t, n in ASSETS.items():
        run(t, n, "3y")
    print("\n" + "=" * 70)
    print("Test tamamlandı. Parametreler ayarlamaya ihtiyaç duyarsan:")
    print("  • RSI eşiği: <35 veya <30")
    print("  • OBV lookback: 5 bar veya 7 bar")
    print("  • Hacim ratio: 1.2x veya 1.5x")
    print("=" * 70)
