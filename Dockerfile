# Worker vLLM 0.20.2 + parche TEXT-ONLY para Qwen3_5MoeForCausalLM.
# El parche se INYECTA en /src/engine.py justo antes de from_engine_args
# (vLLM ya 100% importado -> sin circular import). Sin hooks, sin .pth.
FROM runpod/worker-v1-vllm:v2.22.4

COPY qwen35_register.py patch_engine.py /tmp/

RUN PYDIR="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    cp /tmp/qwen35_register.py "$PYDIR/qwen35_register.py" && \
    echo "[build] register copiado a $PYDIR" && \
    python3 /tmp/patch_engine.py && \
    python3 -c "import qwen35_register; print('[build] register importa OK')" && \
    grep -n "qwen35_register" /src/engine.py
