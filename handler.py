"""Worker RunPod serverless sobre vLLM moderno (imagen oficial).

Arranca `vllm serve` (API OpenAI) como subproceso y reenvia los jobs de RunPod
a localhost. Todo lo de vLLM es configurable por env sin rebuild:
  MODEL_NAME, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, VLLM_EXTRA_ARGS
"""
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
import urllib.error

import runpod

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
    cmd += shlex.split(extra)

print("[worker] lanzando:", " ".join(cmd), flush=True)
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

# esperar a que el server este listo (o morir si vllm muere)
while True:
    if proc.poll() is not None:
        sys.exit(f"[worker] vllm murio en el arranque (exit {proc.returncode})")
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
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


runpod.serverless.start({"handler": handler})
