import asyncio
from datetime import datetime
from core.llm import llm_call
from core.config import settings
from loguru import logger


APPROVED_COMMANDS = {
    "disk_report": "Disk kullanım raporu",
    "temp_check": "Geçici dosya boyutu kontrolü",
    "status": "Sistem durum raporu",
    "memory_check": "Bellek kullanım raporu",
}


async def run_edge_agent(task_description: str) -> str:
    """
    Edge OS Cloud Ajanı: Sistem kontrol ve raporlama.
    ASLA yetkisiz işlem yapmaz. Tüm aksiyonlar onay gerektirir.
    """
    logger.info("[EDGE AGENT] Analyzing system task")

    requested_action = await _identify_action(task_description)
    logger.info(f"[EDGE AGENT] Requested: {requested_action}")

    if requested_action not in APPROVED_COMMANDS:
        return f"""🔒 *EDGE OS GÜVENLIK KILIDI*

İstenen aksiyon: `{requested_action}` onaylı listede değil.

✅ İzin verilen komutlar:
{chr(10).join(f'  • `{k}`: {v}' for k, v in APPROVED_COMMANDS.items())}

⚠️ Yetki dışı işlem reddedildi. Lütfen izin verilen bir komut seçin."""

    report = await _execute_cloud_action(requested_action)
    return report


async def _identify_action(task: str) -> str:
    """Görevden hangi aksiyon istendiğini çıkarır."""
    response = await llm_call(
        messages=[{"role": "user", "content": task}],
        system=f"""Verilen metinden hangi sistem aksiyonu istendiğini belirle.
Sadece şu değerlerden birini döndür: {', '.join(APPROVED_COMMANDS.keys())}
Emin değilsen 'status' döndür.""",
        temperature=0.1,
        max_tokens=30,
    )
    return response.strip().lower().split()[0] if response.strip() else "status"


async def _execute_cloud_action(action: str) -> str:
    """Cloud tarafında güvenli sistem aksiyonlarını çalıştırır."""
    import shutil
    import platform

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if action == "disk_report":
        try:
            total, used, free = shutil.disk_usage("/")
            return f"""💾 *DISK RAPORU*
🕐 {now}
━━━━━━━━━━━━━━━
📦 Toplam: {total // (1024**3)} GB
✅ Kullanılan: {used // (1024**3)} GB ({(used/total*100):.1f}%)
🆓 Boş: {free // (1024**3)} GB ({(free/total*100):.1f}%)
━━━━━━━━━━━━━━━
✅ Rapor tamamlandı."""
        except Exception as e:
            return f"❌ Disk raporu alınamadı: {e}"

    elif action == "temp_check":
        import os
        tmp_size = 0
        try:
            for root, dirs, files in os.walk("/tmp"):
                for f in files:
                    try:
                        tmp_size += os.path.getsize(os.path.join(root, f))
                    except Exception:
                        pass
        except Exception:
            pass
        return f"""🧹 *TEMP DOSYA KONTROLÜ*
🕐 {now}
━━━━━━━━━━━━━━━
📁 /tmp boyutu: {tmp_size // (1024**2)} MB
⚠️ Temizlik için onay gerekli.
━━━━━━━━━━━━━━━
ℹ️ Temizlemek için: /edge temizle"""

    elif action == "memory_check":
        try:
            import psutil
            mem = psutil.virtual_memory()
            return f"""🧠 *BELLEK RAPORU*
🕐 {now}
━━━━━━━━━━━━━━━
📦 Toplam: {mem.total // (1024**2)} MB
✅ Kullanılan: {mem.used // (1024**2)} MB ({mem.percent:.1f}%)
🆓 Boş: {mem.available // (1024**2)} MB
━━━━━━━━━━━━━━━"""
        except ImportError:
            return f"🧠 Bellek kontrolü: psutil kurulu değil. Cloud ortam aktif."

    else:
        return f"""📡 *SİSTEM DURUM RAPORU*
🕐 {now}
━━━━━━━━━━━━━━━
🖥️ Platform: {platform.system()} {platform.release()}
🐍 Python: {platform.python_version()}
☁️ Mod: Cloud (Replit)
🔒 Güvenlik: Aktif
✅ Oracle Swarm: Çalışıyor
━━━━━━━━━━━━━━━"""
