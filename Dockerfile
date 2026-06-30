# Misma imagen que usa tu endpoint (vLLM 0.20.2) + el plugin text-only.
# El entry-point de vLLM no se dispara en este flujo (motor en subproceso),
# así que ademas forzamos la carga via un .pth de arranque (qwen35_boot),
# que se ejecuta en CADA proceso de Python (principal + subprocesos del motor).
FROM runpod/worker-v1-vllm:v2.22.4

COPY qwen35_register.py pyproject.toml qwen35_boot.py /opt/qwen35-register/

RUN pip install --no-cache-dir /opt/qwen35-register/ && \
    PYDIR="$(python -c 'import site; print(site.getsitepackages()[0])')" && \
    cp /opt/qwen35-register/qwen35_boot.py "$PYDIR/qwen35_boot.py" && \
    printf 'import qwen35_boot\n' > "$PYDIR/zzz_qwen35.pth" && \
    echo "[build] hook instalado en $PYDIR:" && \
    ls -la "$PYDIR/qwen35_boot.py" "$PYDIR/zzz_qwen35.pth" && \
    echo "[build] sanity:" && python -c "import qwen35_boot; print('qwen35_boot OK')"
