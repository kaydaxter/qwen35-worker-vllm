# Worker RunPod serverless sobre vLLM 0.24 oficial (fix de sliding-window KV
# para gemma-4: per_layer_sliding_window + KV compartido entre capas).
FROM vllm/vllm-openai:v0.24.0

ENTRYPOINT []

# logs del SDK de RunPod sin contenido de jobs (higiene: prompts de usuario
# fuera de los logs de la consola). Sobreescribible por env del template.
ENV RUNPOD_DEBUG_LEVEL=WARN

RUN pip install --no-cache-dir runpod \
 && python3 -c "import runpod; print('[build] runpod OK')"

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
