"""
FASE D — FİYAT BAZLI İŞLEM PLANI (DIGEST v3)
=============================================
Kullanıcı isteği: "Al sinyali oluştuysa; bu sinyal HANGİ BÖLGEDEN alınacak,
hangi bölgede kademeli satılacak, hangi bölgede tekrar kontrol edilecek,
hangi seviyenin altında sinyal geçersiz sayılacak?"

Bu modül sinyal + MTF hizalaması + rejim + seviyeleri birleştirip
ZAMAN değil FİYAT bazlı, yatırımcının önüne koyabileceği tek bir plan üretir.

Kurallar:
  - MTF tam hizalı (BULLISH LONG)     → "FULL"  : tam plan (giriş bölgesi → TP kademeleri)
  - Üst TF zıt / rejim soğuk          → "BOUNCE_ONLY": sadece dip bölgeden TEPKİ ALIMI (limit)
  - Hizalama belirsiz                 → "LIMIT_ONLY": sadece giriş bölgesi + stop, TP kademeleri
  - Hiçbir seviye yok                 → "NO_PLAN" : veri yetersiz
"""

from __future__ import annotations

from typing import Any, Optional


def _fmt_price(value: Optional[float], digits: int = 4) -> str:
    """Fiyatı okunur formatta yazar; None/0 → '-'."""
    if value is None or value <= 0:
        return "-"
    return f"{value:,.{digits}f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"%{value:+.1f}"


def _direction_bias(direction: str) -> str:
    return "BULLISH" if direction == "LONG" else "BEARISH"


def build_trade_plan(
    direction: str,                # "LONG" | "SHORT"
    mtf_bias: Optional[str],       # "BULLISH" | "BEARISH" | "NEUTRAL" | None
    entry_timing: Optional[str],   # "NOW" | "WAIT" | "AVOID" | None
    levels: dict[str, Any],        # entry_zone_low/high, stop_loss, t1/t2/t3, fib_* , invalidation
    price: Optional[float] = None,
    base_rr: Optional[float] = None,
    usdt_d_trend: Optional[str] = None,  # "RISING" | "FALLING" | "FLAT" | "UNKNOWN"
) -> dict[str, Any]:
    """
    Dönüş:
      {
        "plan_type": "FULL" | "BOUNCE_ONLY" | "LIMIT_ONLY" | "NO_PLAN",
        "header": str,                 # tek satır özet başlık
        "lines": list[str],            # mesaj satırları (emoji'li)
        "validity_price": float|None,  # fiyat bu bandın içindeyken plan GEÇERLİ
        "invalidation_price": float|None,
      }
    """
    direction = (direction or "LONG").upper()
    want = _direction_bias(direction)
    bias = (mtf_bias or "").upper()
    timing = (entry_timing or "").upper()

    elow = levels.get("entry_zone_low")
    ehigh = levels.get("entry_zone_high")
    stop = levels.get("stop_loss")
    t1 = levels.get("t1")
    t2 = levels.get("t2")
    t3 = levels.get("t3")
    invalidation = levels.get("invalidation_level")
    fib_382 = levels.get("fib_382")
    fib_500 = levels.get("fib_500")
    fib_618 = levels.get("fib_618")

    has_entry = bool(elow and ehigh and ehigh >= elow)
    if not has_entry:
        return {
            "plan_type": "NO_PLAN",
            "header": "Seviye verisi yetersiz — plan üretilemedi.",
            "lines": ["⚠️ Giriş bölgesi hesaplanamadı (yetersiz veri)."],
            "validity_price": None,
            "invalidation_price": invalidation,
            "validity_strength": 0,
            "usdt_d_note": "",
        }

    # ── Plan tipi seçimi ─────────────────────────────────────────────
    # Öncelik: (1) ÜST TF ZIT ise TEPKİ ALIMI — zamanlama WAIT bile olsa
    #          (2) Hizalı + AVOID değilse TAM PLAN
    #          (3) Geri kalan (belirsiz/WAIT/AVOID) → LİMİT PLAN
    if bias != want and bias not in ("", "NEUTRAL"):
        plan_type = "BOUNCE_ONLY"
    elif bias == want and timing != "AVOID":
        plan_type = "FULL"
    else:
        plan_type = "LIMIT_ONLY"

    # ── FAZ C: USDT.D GEÇERLİLİK GÜCÜ (güçlendirir/zayıflatır, tek başına yön vermez) ──
    validity_strength = 0
    usdt_d_note = ""
    if usdt_d_trend:
        u = (usdt_d_trend or "").upper()
        up = any(k in u for k in ("RISING", "UP", "INCREAS"))
        down = any(k in u for k in ("FALLING", "DOWN", "DECREAS"))
        if direction == "LONG":
            if down:
                validity_strength = 1
                usdt_d_note = "🧭 USDT.D düşüyor → risk-on ortamı LONG'u güçlendiriyor."
            elif up:
                validity_strength = -1
                usdt_d_note = "🧭 USDT.D yükseliyor → stablecoin'e kaçış, LONG zayıf (stop sıkı)."
        else:  # SHORT
            if up:
                validity_strength = 1
                usdt_d_note = "🧭 USDT.D yükseliyor → risk-off ortamı SHORT'u güçlendiriyor."
            elif down:
                validity_strength = -1
                usdt_d_note = "🧭 USDT.D düşüyor → risk-on ortamı SHORT zayıf (stop sıkı)."

    lines: list[str] = []
    entry_digits = 6 if ehigh < 10 else 2

    # 1) GİRİŞ BÖLGESİ
    lines.append(f"📥 GİRİŞ BÖLGESİ: {_fmt_price(elow, entry_digits)} – {_fmt_price(ehigh, entry_digits)}")
    if plan_type == "BOUNCE_ONLY":
        lines.append("   → Üst TF sinyalle zıt: bu bölgeye sadece LİMİT emir (tepki/çevik pozisyon).")
    elif plan_type == "LIMIT_ONLY":
        lines.append("   → Hizalama belirsiz: sadece bu bölgeden LİMİT emir ile gir.")
    else:
        lines.append("   → MTF hizalı: bu bölgeye giriş onaylı.")

    # 2) STOP + GEÇERSİZLİK
    if stop:
        lines.append(f"🛑 STOP LOSS: {_fmt_price(stop, entry_digits)}")
    if invalidation:
        lines.append(f"🚫 SİNYAL GEÇERSİZ: {_fmt_price(invalidation, entry_digits)} altı/üstü kapanışta plan iptal.")

    # 3) KADEMELİ KAR AL (T1 %30 → T2 %30 → T3 %40)
    if t1 or t2 or t3:
        lines.append("🎯 KAR AL KADEMELERİ (30% → 30% → 40%):")
        if t1:
            lines.append(f"   T1 ({_fmt_price(t1, entry_digits)}): pozisyonun %30'u — ilk kar, kalan için stop'u girişe çek.")
        if t2:
            lines.append(f"   T2 ({_fmt_price(t2, entry_digits)}): pozisyonun %30'u — ana hedef (R:R baz alınan nokta).")
        if t3:
            lines.append(f"   T3 ({_fmt_price(t3, entry_digits)}): kalan %40 — trend uzatması, stop trailing.")

    # 4) YENİDEN KONTROL BÖLGESİ (fib 0.382/0.5/0.618)
    recheck_zone = [x for x in (fib_382, fib_500, fib_618) if x]
    if recheck_zone:
        lo = min(recheck_zone)
        hi = max(recheck_zone)
        lines.append(
            f"🔁 YENİDEN KONTROL BÖLGESİ: {_fmt_price(lo, entry_digits)} – {_fmt_price(hi, entry_digits)}"
        )
        lines.append("   → Fiyat bu bölgeye geri çekilirse planı yeniden doğrula; fib %50/%61,8 kırılırsa riski azalt.")

    # 5) R:R + ON-SUYU
    if base_rr:
        lines.append(f"⚖️ Beklenen R:R: 1:{float(base_rr):.1f} (T2 bazlı)")

    # 6) FAZ C: USDT.D geçerlilik notu
    if usdt_d_note:
        lines.append(usdt_d_note)

    header = {
        "FULL": "📗 TAM PLAN (MTF Hizalı)",
        "BOUNCE_ONLY": "📕 TEPKİ ALIMI (Üst TF Zıt)",
        "LIMIT_ONLY": "📘 LİMİT PLAN (Hizalama Belirsiz)",
        "NO_PLAN": "⚠️ PLAN YOK",
    }[plan_type]

    return {
        "plan_type": plan_type,
        "header": header,
        "lines": lines,
        "validity_price": ehigh if direction == "LONG" else elow,
        "invalidation_price": invalidation,
        "validity_strength": validity_strength,
        "usdt_d_note": usdt_d_note,
    }
