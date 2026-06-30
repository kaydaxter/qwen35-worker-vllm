"""Inyecta la llamada a qwen35_register.register() en /src/engine.py del
worker-vllm, JUSTO antes de la linea:
    engine = AsyncLLMEngine.from_engine_args(self.engine_args)
En ese punto vLLM esta 100% importado -> el parche aplica sin circular import.
Falla el build (assert) si no encuentra el sitio, para no producir una imagen
silenciosamente rota.
"""
import sys

P = "/src/engine.py"
s = open(P, encoding="utf-8").read()

if "qwen35_register" in s:
    print("[build] /src/engine.py ya estaba parcheado; nada que hacer")
    sys.exit(0)

MARK = "AsyncLLMEngine.from_engine_args"
lines = s.splitlines(keepends=True)
out = []
done = False
for ln in lines:
    if (not done) and MARK in ln:
        indent = ln[:len(ln) - len(ln.lstrip())]
        out.append(f"{indent}import qwen35_register; qwen35_register.register()\n")
        done = True
    out.append(ln)

assert done, f"NO se encontro '{MARK}' en {P}; el worker cambio de estructura"
open(P, "w", encoding="utf-8").write("".join(out))
print(f"[build] {P} parcheado: register() inyectado antes de {MARK}")
