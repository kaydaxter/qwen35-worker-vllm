# Worker serverless que corre el harness oficial `pagestorm` (Mistral 24B).
# Base: imagen oficial de vLLM (>=0.23, requerido por pagestorm). Sin parches:
# Mistral 24B lo soporta vLLM de fabrica.
FROM vllm/vllm-openai:v0.24.0

# La imagen de vLLM trae un ENTRYPOINT que lanza su api-server; lo anulamos
# para correr nuestro handler de RunPod.
ENTRYPOINT []

# runpod SDK + el paquete pagestorm (arrastra vllm>=0.23 -ya presente-,
# transformers>=5.12.1, jinja2, huggingface_hub>=1.21).
RUN pip install --no-cache-dir runpod "git+https://github.com/Pageshift-ai/pagestorm.git" \
 && python -c "import pagestorm, runpod; print('[build] pagestorm + runpod importan OK')"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
