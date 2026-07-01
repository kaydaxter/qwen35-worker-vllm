"""RunPod serverless worker que embebe el harness oficial `pagestorm`.

Recibe {"input": {"prompt": "..."}} y ejecuta el pipeline por etapas del 24B
(book_plan -> first_chapter_plan -> scene_breakdown -> first_chapter_text) con
el guided decoding y los samplers oficiales, devolviendo el primer capitulo.

El modelo (Mistral 24B, ~48 GB fp16) lo baja vLLM en la PRIMERA generacion y
queda cacheado en el network volume (HF_HOME=/runpod-volume/hf).
"""
import os
import time
import traceback

import runpod

from pagestorm.bundle import load_story_bundle
from pagestorm.profiles import get_profile_by_repo_id
from pagestorm.orchestrator import generate_full_book

REPO = os.environ.get(
    "PAGESTORM_REPO",
    "Pageshift-Entertainment/pagestorm-research-preview-24b-first-chapter-only",
)

print(f"[pagestorm-worker] cargando bundle (config+tokenizer) de {REPO} ...", flush=True)
_profile = get_profile_by_repo_id(REPO)
_bundle = load_story_bundle(profile_name=_profile.name, repo_id=REPO)
print("[pagestorm-worker] bundle listo. A la espera de jobs.", flush=True)


def handler(job):
    inp = (job or {}).get("input", {}) or {}
    prompt = inp.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return {"error": "Falta 'prompt' (string) en input."}

    gmu = float(inp.get("gpu_memory_utilization", 0.95))
    mml = inp.get("max_model_len")
    tp = int(inp.get("tensor_parallel_size", 1))

    t0 = time.time()
    print(f"[pagestorm-worker] generando para prompt: {prompt!r}", flush=True)
    try:
        run = generate_full_book(
            _bundle,
            prompt=prompt,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gmu,
            max_model_len=int(mml) if mml else None,
        )
    except Exception as e:
        print("[pagestorm-worker] ERROR en generacion:", e, flush=True)
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-4000:]}

    stages = {so.role: so.text for so in run.stage_outputs}
    elapsed = round(time.time() - t0, 1)
    print(f"[pagestorm-worker] listo en {elapsed}s "
          f"(validation_success={run.validation_success})", flush=True)
    return {
        "prompt": prompt,
        "model": run.model,
        "validation_success": run.validation_success,
        "failed_stage_role": run.failed_stage_role,
        "elapsed_seconds": elapsed,
        "chapter": stages.get("first_chapter_text", ""),
        "stages": stages,
    }


runpod.serverless.start({"handler": handler})
