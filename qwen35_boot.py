"""Carga FORZADA del parche qwen35_register en CUALQUIER proceso de Python.

vLLM, en el flujo del worker-vllm de RunPod (motor en subproceso, vLLM v1),
NO dispara el entry-point `vllm.general_plugins`, así que register() nunca corría.

Este módulo se importa al arrancar el intérprete (vía un fichero .pth en
site-packages). Instala un hook sobre __import__ que ejecuta register() en
cuanto el módulo `vllm.config.model` está cargado — temprano, antes de que se
construya el ModelConfig/renderer, y se re-ejecuta en cada subproceso (que
también procesa el .pth al arrancar).
"""

import sys
import builtins

_applied = [False]


def _try_apply():
    if _applied[0]:
        return
    # Espera a que el módulo clave de vLLM exista para no importar a medias.
    if "vllm.config.model" not in sys.modules:
        return
    _applied[0] = True
    print("[qwen35-boot] vllm.config.model cargado -> aplicando register()",
          file=sys.stderr, flush=True)
    try:
        import qwen35_register
        qwen35_register.register()
    except Exception:
        import traceback
        print("[qwen35-boot] register() lanzó excepción:", file=sys.stderr, flush=True)
        traceback.print_exc()


_orig_import = builtins.__import__


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _orig_import(name, globals, locals, fromlist, level)
    if not _applied[0] and isinstance(name, str) and name.startswith("vllm"):
        _try_apply()
    return mod


builtins.__import__ = _patched_import
print("[qwen35-boot] import hook instalado (pid %d)" % __import__("os").getpid(),
      file=sys.stderr, flush=True)
