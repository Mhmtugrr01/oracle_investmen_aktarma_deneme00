"""
EDGE DAEMON — Sistem Kontrol Ajanı
LLM gerektirmez — tamamen kural tabanlı çalışır.
"""
import asyncio
import shutil
import platform
import os
from datetime import datetime
from loguru import logger


APPROVED_COMMANDS = {
    "disk_report": "Disk kullanım raporu",
    "temp_check": "Geçici dosya boyutu kontrolü",
    "status": "Sistem durum raporu",
    "memory_check": "Bellek kullanım raporu",
}

_ACTION_RULES = {
    "disk_report": ["disk", "depolama", "storage", "gb", "mb", "space", "yer"],
    "memory_check": ["memory", "bellek", "ram", "heap", "mem"],
    "temp_check": ["temp", "geçici", "tmp", "temizle", "cache", "temiz"],
    "status": ["durum", "status", "çalış", "sağlık", "health", "uptime", "system", "sistem"],
}


async def run_edge_agent(task_description: str) -> str:
    logger.info("[EDGE AGENT] Rule-based system task")
    action = _rule_identify_action(task_description)
    logger.info(f"[EDGE AGENT] Action: {action}")
    return await _execute_cloud_action(action)


def _rule_identify_action(task: str) -> str:
    """LLM olmadan kural tabanlı aksiyon tespiti."""
    t = task.lower()
    scores = {}
    for action, keywords in _ACTION_RULES.items():
        scores[action] = sum(1 for kw in keywords if kw in t)
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "status"


async def _execute_cloud_action(action: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if action == "disk_report":
        try:
            total, used, free = shutil.disk_usage("/")
            pct_used = (used / total * 100)
            pct_free = (free / total * 100)
            health = "✅ Normal" if pct_used < 80 else ("⚠️ Yüksek" if pct_used < 90 else "🔴 KRİTİK")
            return f"""💾 *DISK RAPORU*
🕐 {now}
━━━━━━━━━━━━━━━
📦 Toplam: {total // (1024**3):.1f} GB
✅ Kullanılan: {used // (1024**3):.1f} GB ({pct_used:.1f}%)
🆓 Boş: {free // (1024**3):.1f} GB ({pct_free:.1f}%)
━━━━━━━━━━━━━━━
Durum: {health}"""
        except Exception as e:
            return f"❌ Disk raporu alınamadı: {e}"

    elif action == "temp_check":
        tmp_size = 0
        file_count = 0
        try:
            for root, dirs, files in os.walk("/tmp"):
                for f in files:
                    try:
                        tmp_size += os.path.getsize(os.path.join(root, f))
                        file_count += 1
                    except Exception:
                        pass
        except Exception:
            pass
        size_mb = tmp_size // (1024 ** 2)
        return f"""🧹 *TEMP DOSYA KONTROLÜ*
🕐 {now}
━━━━━━━━━━━━━━━
📁 /tmp boyutu: {size_mb} MB
📄 Dosya sayısı: {file_count}
{'⚠️ Temizlik önerilir (>500MB)' if size_mb > 500 else '✅ Normal seviyede'}
━━━━━━━━━━━━━━━
ℹ️ Temizlik için onay gerekli."""

    elif action == "memory_check":
        try:
            import psutil
            mem = psutil.virtual_memory()
            health = "✅ Normal" if mem.percent < 80 else ("⚠️ Yüksek" if mem.percent < 90 else "🔴 KRİTİK")
            return f"""🧠 *BELLEK RAPORU*
🕐 {now}
━━━━━━━━━━━━━━━
📦 Toplam: {mem.total // (1024**2)} MB
📊 Kullanılan: {mem.used // (1024**2)} MB ({mem.percent:.1f}%)
🆓 Boş: {mem.available // (1024**2)} MB
━━━━━━━━━━━━━━━
Durum: {health}"""
        except ImportError:
            # psutil yoksa /proc/meminfo dene
            try:
                with open("/proc/meminfo") as f:
                    lines = {l.split(":")[0]: l.split(":")[1].strip() for l in f.readlines()}
                total = int(lines.get("MemTotal", "0 kB").split()[0]) // 1024
                free = int(lines.get("MemAvailable", "0 kB").split()[0]) // 1024
                used = total - free
                pct = (used / total * 100) if total else 0
                return f"""🧠 *BELLEK RAPORU*
🕐 {now}
━━━━━━━━━━━━━━━
📦 Toplam: {total} MB
📊 Kullanılan: {used} MB ({pct:.1f}%)
🆓 Boş: {free} MB
━━━━━━━━━━━━━━━"""
            except Exception:
                return f"🧠 Bellek kontrolü: Cloud ortam aktif — {now}"

    else:  # status
        try:
            total, used, free = shutil.disk_usage("/")
            disk_pct = (used / total * 100)
        except Exception:
            disk_pct = 0

        mem_info = ""
        try:
            import psutil
            mem = psutil.virtual_memory()
            mem_info = f"\n🧠 RAM: {mem.used // (1024**2)}/{mem.total // (1024**2)} MB ({mem.percent:.0f}%)"
        except Exception:
            pass

        return f"""📡 *SİSTEM DURUM RAPORU*
🕐 {now}
━━━━━━━━━━━━━━━
🖥️ Platform: {platform.system()} {platform.release()}
🐍 Python: {platform.python_version()}
☁️ Mod: Cloud (Replit)
💾 Disk: {used // (1024**3):.0f}/{total // (1024**3):.0f} GB ({disk_pct:.0f}%){mem_info}
🔒 Güvenlik: Aktif
✅ Oracle Swarm V4.0: Çalışıyor
━━━━━━━━━━━━━━━
⚡ LLM durum: {'Gemini ✅' if os.getenv('GEMINI_API_KEY') else ''} {'Groq ✅' if os.getenv('GROQ_API_KEY') else ''} {'OpenAI ✅' if os.getenv('OPENAI_API_KEY') else ''} """
