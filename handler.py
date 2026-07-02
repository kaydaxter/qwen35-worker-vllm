"""Worker RunPod serverless sobre vLLM 0.24 (imagen oficial).

Anti-ceguera: cualquier fallo imprime traceback y duerme 60s antes de salir,
para que el recolector de logs de RunPod llegue a capturarlo.
Config por env sin rebuild: MODEL_NAME, DTYPE, MAX_MODEL_LEN,
GPU_MEMORY_UTILIZATION, VLLM_EXTRA_ARGS.
"""
import json
import os
import shlex
import subprocess
import sys
import time
import traceback

print("[worker] handler.py arrancando (python OK)", flush=True)


def fatal(msg, exc=True):
    print(f"[worker][FATAL] {msg}", flush=True)
    if exc:
        traceback.print_exc()
    print("[worker] durmiendo 60s para que el log se capture...", flush=True)
    time.sleep(60)
    sys.exit(1)


try:
    import urllib.request
    import urllib.error
    import runpod
    print("[worker] imports OK (runpod %s)" % getattr(runpod, "__version__", "?"), flush=True)
except Exception:
    fatal("fallo importando dependencias")

MODEL = os.environ.get("MODEL_NAME", "google/gemma-4-31B-it")
PORT = "8000"
BASE = f"http://127.0.0.1:{PORT}"

cmd = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--host", "127.0.0.1", "--port", PORT,
    "--dtype", os.environ.get("DTYPE", "bfloat16"),
    "--max-model-len", os.environ.get("MAX_MODEL_LEN", "131072"),
    "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"),
]
extra = os.environ.get("VLLM_EXTRA_ARGS", "").strip()
if extra:
    try:
        cmd += shlex.split(extra)
    except Exception:
        fatal(f"VLLM_EXTRA_ARGS no parsea: {extra!r}")

print("[worker] lanzando:", " ".join(cmd), flush=True)
t0 = time.time()
try:
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
except Exception:
    fatal("Popen de vllm fallo")

while True:
    rc = proc.poll()
    if rc is not None:
        fatal(f"vllm murio en el arranque con exit code {rc} "
              f"(a los {time.time()-t0:.0f}s) — su error debe estar arriba", exc=False)
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    el = int(time.time() - t0)
    if el and el % 60 < 2:
        print(f"[worker] esperando a vLLM... {el}s", flush=True)
    time.sleep(2)
print(f"[worker] vLLM LISTO en {time.time()-t0:.1f}s", flush=True)


def _post(path, body, timeout=600):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def handler(job):
    inp = (job or {}).get("input", {}) or {}
    sp = inp.get("sampling_params", {}) or {}
    common = {
        "model": MODEL,
        "max_tokens": int(sp.get("max_tokens", 256)),
        "temperature": float(sp.get("temperature", 0.7)),
    }
    try:
        if "messages" in inp:
            r = _post("/v1/chat/completions", {**common, "messages": inp["messages"]})
            text = r["choices"][0]["message"]["content"]
        elif "prompt" in inp:
            r = _post("/v1/completions", {**common, "prompt": inp["prompt"]})
            text = r["choices"][0]["text"]
        else:
            return {"error": "input necesita 'messages' o 'prompt'"}
        return {"text": text, "usage": r.get("usage")}
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode()[:1000]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


try:
    runpod.serverless.start({"handler": handler})
except Exception:
    fatal("runpod.serverless.start fallo")
