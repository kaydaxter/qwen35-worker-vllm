"""Worker RunPod serverless sobre vLLM 0.24 (imagen oficial). v3.

- STREAMING: input {"stream": true} -> yield de deltas de texto (endpoint /stream).
- HIGIENE DE LOGS: jamas se imprime contenido de usuario (prompts/respuestas);
  vllm arranca con --disable-log-requests. Solo metadatos operacionales.
- ANTI-CEGUERA: todo fallo imprime traceback y duerme 60s antes de salir.
- Config por env sin rebuild: MODEL_NAME, DTYPE, MAX_MODEL_LEN,
  GPU_MEMORY_UTILIZATION, VLLM_EXTRA_ARGS.
"""
import json
import os
import shlex
import subprocess
import sys
import time
import traceback

print("[worker] handler.py v3 arrancando (python OK)", flush=True)


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
    # nota: en vLLM 0.24 el log de requests viene APAGADO por defecto
    # (--enable-log-requests es opt-in); no hace falta ningun flag.
]
extra = os.environ.get("VLLM_EXTRA_ARGS", "").strip()
if extra:
    try:
        cmd += shlex.split(extra)
    except Exception:
        fatal(f"VLLM_EXTRA_ARGS no parsea: {extra!r}")

print("[worker] lanzando vLLM (modelo: %s)" % MODEL, flush=True)
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


def _req(path, body, timeout=900):
    return urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")


def _build_body(inp):
    sp = inp.get("sampling_params", {}) or {}
    body = {
        "model": MODEL,
        "max_tokens": int(sp.get("max_tokens", 256)),
        "temperature": float(sp.get("temperature", 0.7)),
    }
    if sp.get("top_p") is not None:
        body["top_p"] = float(sp["top_p"])
    if sp.get("seed") is not None:
        body["seed"] = int(sp["seed"])
    if "messages" in inp:
        return "/v1/chat/completions", {**body, "messages": inp["messages"]}, True
    if "prompt" in inp:
        return "/v1/completions", {**body, "prompt": inp["prompt"]}, False
    return None, None, None


def _stream_gen(path, body, chat):
    body = {**body, "stream": True}
    with urllib.request.urlopen(_req(path, body), timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                ch = json.loads(payload)["choices"][0]
                delta = (ch.get("delta", {}).get("content") if chat
                         else ch.get("text")) or ""
            except Exception:
                continue
            if delta:
                yield delta


def handler(job):
    inp = (job or {}).get("input", {}) or {}
    path, body, chat = _build_body(inp)
    if path is None:
        return {"error": "input necesita 'messages' o 'prompt'"}
    try:
        if inp.get("stream"):
            return _stream_gen(path, body, chat)
        with urllib.request.urlopen(_req(path, body), timeout=900) as r:
            out = json.loads(r.read().decode())
        text = (out["choices"][0]["message"]["content"] if chat
                else out["choices"][0]["text"])
        return {"text": text, "usage": out.get("usage")}
    except urllib.error.HTTPError as e:
        # sin contenido de usuario en logs: solo codigo y cuerpo truncado al caller
        return {"error": f"HTTP {e.code}", "body": e.read().decode()[:800]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


try:
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
except Exception:
    fatal("runpod.serverless.start fallo")
