# Worker serverless que corre el harness oficial `pagestorm` (Mistral 24B).
# Base: imagen oficial de vLLM (>=0.23, requerido por pagestorm). Sin parches:
# Mistral 24B lo soporta vLLM de fabrica.
FROM vllm/vllm-openai:v0.24.0

# La imagen de vLLM trae un ENTRYPOINT que lanza su api-server; lo anulamos
# para correr nuestro handler de RunPod.
ENTRYPOINT []

# runpod SDK + el paquete pagestorm (arrastra vllm>=0.23 -ya presente-,
# transformers>=5.12.1, jinja2, huggingface_hub>=1.21).
# Instalamos pagestorm desde el TARBALL (la imagen base no trae git, asi que
# "git+https://..." falla). El tarball no necesita git.
RUN pip install --no-cache-dir runpod "https://github.com/Pageshift-ai/pagestorm/archive/refs/heads/master.tar.gz" \
 && python -c "import pagestorm, runpod; print('[build] pagestorm + runpod importan OK')"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
