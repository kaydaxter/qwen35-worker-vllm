"""Carga FORZADA del parche qwen35_register en CUALQUIER proceso de Python.

vLLM, en el flujo del worker-vllm de RunPod (motor en subproceso, vLLM v1),
NO dispara el entry-point `vllm.general_plugins`, así que register() nunca corría.

Este módulo se importa al arrancar el intérprete (vía un .pth en site-packages).
Instala un hook sobre __import__ que ejecuta register() en cuanto vLLM está
COMPLETAMENTE cargado (no a medias) y se re-ejecuta en cada subproceso.

CLAVE del timing: no basta con que 'vllm.config.model' aparezca en sys.modules
(eso pasa al EMPEZAR su carga, cuando aún es un módulo parcialmente inicializado
-> ImportError circular al hacer `from vllm.config import ModelConfig`). Hay que
esperar a que ModelConfig sea REALMENTE importable. Probamos a importarlo; si
falla, reintentamos en el siguiente import de vLLM.
"""

import sys
import builtins

_applied = [False]
_busy = [False]

_orig_import = builtins.__import__


def _try_apply():
    if _applied[0] or _busy[0]:
        return
    _busy[0] = True
    try:
        # ¿vLLM ya está del todo cargado? Si ModelConfig no importa todavía
        # (circular import porque vllm.config sigue inicializándose), salimos
        # y reintentamos en el próximo import de vLLM.
        try:
            from vllm.config import ModelConfig  # noqa: F401
        except Exception:
            return
        _applied[0] = True
        print("[qwen35-boot] vLLM listo -> aplicando register()",
              file=sys.stderr, flush=True)
        import qwen35_register
        qwen35_register.register()
    except Exception:
        import traceback
        _applied[0] = False  # permitir reintento si algo raro pasó
        print("[qwen35-boot] register() lanzó excepción:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        _busy[0] = False


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _orig_import(name, globals, locals, fromlist, level)
    if not _applied[0] and not _busy[0] and isinstance(name, str) and name.startswith("vllm"):
        _try_apply()
    return mod


builtins.__import__ = _patched_import
print("[qwen35-boot] import hook instalado (pid %d)" % __import__("os").getpid(),
      file=sys.stderr, flush=True)
