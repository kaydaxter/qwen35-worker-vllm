# Failover: servir gemma-4-31B (128k) fuera de RunPod

Plan de contingencia si RunPod aplica su ToS o se queda sin capacidad.
El objetivo: conmutar en < 1 hora sin tocar el backend (misma API OpenAI).

## Clave de diseño

En Koyeb/Verda NO hace falta el handler de RunPod: son servicios HTTP
normales, así que se usa la **imagen oficial de vLLM directamente** —
`vllm/vllm-openai:v0.24.0` — que expone la API OpenAI en el puerto 8000.
El backend solo cambia la URL base (y el header de auth si se configura).

## Koyeb (primario de failover — ToS limpio, scale-to-zero)

Requisitos: cuenta Pro ($29/mes), CLI `koyeb` con token.

```bash
koyeb service create gemma31b \
  --app produccion \
  --docker vllm/vllm-openai:v0.24.0 \
  --docker-args "--model google/gemma-4-31B-it --dtype bfloat16 --max-model-len 131072 --gpu-memory-utilization 0.95 --disable-log-requests" \
  --instance-type gpu-nvidia-h100 \
  --regions was \
  --ports 8000:http \
  --routes /:8000 \
  --scale-to-zero \
  --min-scale 0 --max-scale 2 \
  --env HF_TOKEN=<token-hf>
```

Notas:
- Cold start SIN snapshots: minutos (Koyeb no tiene sleep de GPU).
  Para failover activo, subir `--min-scale 1` (GPU caliente, ~$1.60-2.50/h).
- Verificar tipo de instancia GPU exacto con `koyeb instances types`.

## Verda (secundario — empresa más sólida, facturación por bloques de 10 min)

Serverless Containers: crear servicio con la misma imagen y args vía consola
o API. Mismo patrón: imagen oficial + args + puerto 8000.

## Simulacro (pendiente de cuenta)

1. Crear cuenta + desplegar con `min-scale 0` (coste ~$0 en reposo).
2. Request de humo: `curl $URL/v1/chat/completions -d '{...}'` → texto.
3. Medir: tiempo desde "decisión de conmutar" hasta primer token servido.
4. Documentar el switch en el backend (variable de entorno con la URL base).
5. Repetir el humo 1 vez/mes (los deploys dormidos se pudren).

## Qué NO cubre esto

- Pesos privados: cuando el modelo sea el vuestro, la imagen debe llevarlos
  horneados o el repo HF privado + token en env (validar ese flujo aparte).
- FlashBoot no existe fuera de RunPod: el failover tiene peor cold start.
  Es un paracaídas, no una réplica exacta.
