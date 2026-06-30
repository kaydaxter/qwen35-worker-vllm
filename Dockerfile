# Misma imagen que ya usa tu endpoint (vLLM 0.20.2) + un mini-plugin que
# registra el handler text-only Qwen3_5MoeForCausalLM que YA existe en vLLM
# pero no estaba en el registry. No cambia vLLM, ni el modelo, ni nada más.
FROM runpod/worker-v1-vllm:v2.22.4

COPY qwen35_register.py pyproject.toml /opt/qwen35-register/
RUN pip install --no-cache-dir /opt/qwen35-register/
