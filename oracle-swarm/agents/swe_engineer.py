import asyncio
import subprocess
import tempfile
import os
from core.llm import llm_call
from core.config import settings
from loguru import logger


MAX_RETRY = 5


async def run_swe_agent(task_description: str) -> str:
    """
    SWE Agent: Kod üretir, test eder, hataları otomatik düzeltir.
    Zero-Defect Recursive Loop.
    """
    logger.info("[SWE AGENT] Starting zero-defect loop")

    system = """Sen Oracle SWE Mühendisi Ajanısın. 
Görev: Verilen iş direktifini analiz et ve çalışan Python kodu üret.
Kurallar:
- Sadece Python kodu üret (markdown olmadan, düz kod)
- Tüm import'ları ekle
- Hata yönetimi (try/except) ekle
- Logları ekle
- Kodu doğrudan çalıştırılabilir yap"""

    code = await llm_call(
        messages=[{"role": "user", "content": f"Görev:\n{task_description}"}],
        system=system,
        model=settings.llm_model_heavy,
        max_tokens=8192,
    )

    code = _clean_code(code)

    for attempt in range(1, MAX_RETRY + 1):
        logger.info(f"[SWE AGENT] Test attempt {attempt}/{MAX_RETRY}")
        success, error_log = await _sandbox_test(code)

        if success:
            logger.success(f"[SWE AGENT] ✅ Code passed on attempt {attempt}")
            return f"✅ SWE AJAN BAŞARILI (Deneme {attempt})\n\n```python\n{code}\n```"

        logger.warning(f"[SWE AGENT] ❌ Attempt {attempt} failed: {error_log[:200]}")

        fix_prompt = f"""Aşağıdaki Python kodu şu hatayı veriyor:

HATA:
{error_log}

MEVCUT KOD:
{code}

Hatayı düzelt ve tam çalışan kodu döndür. Sadece düz Python kodu, markdown yok."""

        code = await llm_call(
            messages=[{"role": "user", "content": fix_prompt}],
            system="Sen bir Python hata düzeltme uzmanısın. Sadece düzeltilmiş tam kodu döndür.",
            model=settings.llm_model_heavy,
            max_tokens=8192,
        )
        code = _clean_code(code)

    return f"⚠️ SWE AJAN: {MAX_RETRY} denemede düzeltilemedi.\n\nSon hata: {error_log}\n\nSon kod:\n```python\n{code}\n```"


def _clean_code(code: str) -> str:
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


async def _sandbox_test(code: str) -> tuple[bool, str]:
    """Kodu izole bir subprocess'te test eder."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["python3", "-c", f"import ast; ast.parse(open('{tmp_path}').read()); print('SYNTAX_OK')"],
                capture_output=True,
                text=True,
                timeout=15,
            ),
        )

        os.unlink(tmp_path)

        if result.returncode == 0 and "SYNTAX_OK" in result.stdout:
            return True, ""
        else:
            return False, result.stderr or result.stdout

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Kod 15 saniyede tamamlanamadı"
    except Exception as e:
        return False, str(e)
