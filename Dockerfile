# vLLM 0.20.2 (worker base) + parche text-only para Qwen3_5MoeForCausalLM.
# El entry-point de vLLM no se dispara (motor en subproceso), asi que copiamos
# AMBOS modulos (register + boot) directamente a site-packages y forzamos la
# carga con un .pth que se ejecuta en CADA proceso de Python.
FROM runpod/worker-v1-vllm:v2.22.4

COPY qwen35_register.py pyproject.toml qwen35_boot.py /opt/qwen35-register/

RUN pip install --no-cache-dir /opt/qwen35-register/ || true; \
    PYDIR="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    cp /opt/qwen35-register/qwen35_register.py "$PYDIR/qwen35_register.py" && \
    cp /opt/qwen35-register/qwen35_boot.py "$PYDIR/qwen35_boot.py" && \
    printf 'import qwen35_boot\n' > "$PYDIR/zzz_qwen35.pth" && \
    echo "[build] instalado en $PYDIR" && \
    ls -la "$PYDIR/qwen35_register.py" "$PYDIR/qwen35_boot.py" "$PYDIR/zzz_qwen35.pth" && \
    echo "[build] sanity:" && \
    python3 -c "import qwen35_register, qwen35_boot; print('AMBOS modulos importan OK')"
