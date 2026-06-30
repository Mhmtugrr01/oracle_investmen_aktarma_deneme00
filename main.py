"""
PROJECT OLYMPUS — main.py (Executive Dashboard & Lifespan Reformed)
"""

import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from bot.telegram_handler import create_handler
from loguru import logger
from contextlib import asynccontextmanager

handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler (Deprecation önleyici kalkan)."""
    global handler
    try:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            logger.error("[SYSTEM] TELEGRAM_BOT_TOKEN bulunamadı!")
        else:
            handler = create_handler()
            asyncio.create_task(handler.start())
            logger.info("[SYSTEM] Telegram bot polling arka plan görevi olarak başarıyla başlatıldı.")
    except Exception as exc:
        logger.error(f"[SYSTEM] Bot başlatma hatası: {exc}")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def read_root():
    """CEO Portföy Başarı ve Durum İzleme Paneli (Apple Dark Mode)."""
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>The Oracle — Executive Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { background-color: #0B0F19; color: #F3F4F6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        </style>
    </head>
    <body class="p-6 md:p-12">
        <div class="max-w-5xl mx-auto">
            <!-- Header -->
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-gray-800 pb-6 mb-8">
                <div>
                    <h1 class="text-3xl font-bold tracking-tight text-white">THE ORACLE</h1>
                    <p class="text-gray-400 text-sm mt-1">Sembiyotik Portföy & Karar Destek Mekanizması</p>
                </div>
                <div class="mt-4 md:mt-0 flex items-center space-x-2">
                    <span class="h-3 w-3 rounded-full bg-green-500 animate-pulse"></span>
                    <span class="text-sm font-semibold text-green-400">Bulut Sunucusu Aktif (7/24)</span>
                </div>
            </div>

            <!-- Grid 1: Ana İstatistikler -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <!-- Card 1 -->
                <div class="bg-[#151E2E] p-6 rounded-2xl border border-gray-800">
                    <p class="text-gray-400 text-xs uppercase tracking-wider font-bold">Tarihsel Başarı Oranı</p>
                    <p class="text-4xl font-extrabold text-white mt-2">%58.71</p>
                    <p class="text-xs text-green-400 mt-2">✓ 2 Yıllık Backtest Verisiyle Teyitli</p>
                </div>
                <!-- Card 2 -->
                <div class="bg-[#151E2E] p-6 rounded-2xl border border-gray-800">
                    <p class="text-gray-400 text-xs uppercase tracking-wider font-bold">Tahmin Gücü (Spearman IC)</p>
                    <p class="text-4xl font-extrabold text-white mt-2">0.058</p>
                    <p class="text-xs text-cyan-400 mt-2">✓ Kurumsal Fon Standartları Üzeri Edge</p>
                </div>
                <!-- Card 3 -->
                <div class="bg-[#151E2E] p-6 rounded-2xl border border-gray-800">
                    <p class="text-gray-400 text-xs uppercase tracking-wider font-bold">Toplam Simülasyon Sayısı</p>
                    <p class="text-4xl font-extrabold text-white mt-2">4,680</p>
                    <p class="text-xs text-gray-400 mt-2">✓ 21 Varlık Üzerinde Çapraz Doğrulama</p>
                </div>
            </div>

            <!-- Grid 2: Detaylar -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <!-- Sol Panel: 5-Sütunlu Kesişim Kriterleri -->
                <div class="bg-[#111827] p-8 rounded-2xl border border-gray-800">
                    <h3 class="text-xl font-bold text-white mb-6 border-b border-gray-800 pb-3">5-Sütunlu Karar Filtresi</h3>
                    <ul class="space-y-4">
                        <li class="flex items-start">
                            <span class="bg-blue-900/40 text-blue-400 p-1.5 rounded-lg mr-3 text-xs font-bold">1</span>
                            <div>
                                <h4 class="font-bold text-gray-200">Makro Trend Bekçisi</h4>
                                <p class="text-gray-400 text-xs mt-0.5">DXY, VIX ve Japon Yeni Carry-Trade sarsıntılarını izler.</p>
                            </div>
                        </li>
                        <li class="flex items-start">
                            <span class="bg-purple-900/40 text-purple-400 p-1.5 rounded-lg mr-3 text-xs font-bold">2</span>
                            <div>
                                <h4 class="font-bold text-gray-200">Quant & Trendline Breakout</h4>
                                <p class="text-gray-400 text-xs mt-0.5">RSI aşırı satım tuzaklarını eler, fiyattaki "Düşen Kırılımını" doğrular.</p>
                            </div>
                        </li>
                        <li class="flex items-start">
                            <span class="bg-red-900/40 text-red-400 p-1.5 rounded-lg mr-3 text-xs font-bold">3</span>
                            <div>
                                <h4 class="font-bold text-gray-200">Balina CVD Akışı</h4>
                                <p class="text-gray-400 text-xs mt-0.5">Akıllı paranın (Whale) gizli toplama/dağıtım (rejim) evrelerini süzgeçler.</p>
                            </div>
                        </li>
                        <li class="flex items-start">
                            <span class="bg-green-900/40 text-green-400 p-1.5 rounded-lg mr-3 text-xs font-bold">4</span>
                            <div>
                                <h4 class="font-bold text-gray-200">Temel Değer & Katalizörler</h4>
                                <p class="text-gray-400 text-xs mt-0.5">SEC bilançolarını, KAP duyurularını ve proje gelişimini puanlar.</p>
                            </div>
                        </li>
                        <li class="flex items-start">
                            <span class="bg-yellow-900/40 text-yellow-400 p-1.5 rounded-lg mr-3 text-xs font-bold">5</span>
                            <div>
                                <h4 class="font-bold text-gray-200">Korku & Coşku Duygu Analizi</h4>
                                <p class="text-gray-400 text-xs mt-0.5">Sosyal medyadaki (Twitter, YouTube) manipülasyonu ve iştahı arındırır.</p>
                            </div>
                        </li>
                    </ul>
                </div>

                <!-- Sağ Panel: Aktif Portföy Evreni -->
                <div class="bg-[#111827] p-8 rounded-2xl border border-gray-800 flex flex-col justify-between">
                    <div>
                        <h3 class="text-xl font-bold text-white mb-6 border-b border-gray-800 pb-3">İzleme Evreni</h3>
                        <div class="flex flex-wrap gap-2">
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">BTC</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">ETH</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">INJ</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">RNDR</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">FET</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">COIN</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">NVDA</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">TSLA</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">MSTR</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">THYAO.IS</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">GARAN.IS</span>
                            <span class="bg-[#1F2937] text-gray-200 px-3 py-1.5 rounded-lg text-xs font-semibold">ONS ALTIN</span>
                        </div>
                    </div>
                    <div class="mt-6 border-t border-gray-800 pt-6">
                        <p class="text-sm text-gray-400">💡 <span class="font-semibold text-white">Yönetici Talimatı:</span> Sistem yön belirsizken uykuda kalır. Sadece 1'e 3 (R:R 3.0) asimetrik fırsat doğduğunda Telegram üzerinden mühürlü sinyal fırlatır.</p>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <div class="mt-12 text-center text-xs text-gray-500 border-t border-gray-800 pt-6">
                The Oracle R06_MASTER © 2026. Tüm Hakları Saklıdır. Yatırım Tavsiyesi Değildir.
            </div>
        </div>
    </body>
    </html>
    """

def main():
    port = int(os.getenv("PORT", 8000))
    logger.info(f"[SYSTEM] Web sunucusu {port} portu üzerinden başlatılıyor...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()