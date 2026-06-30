"""Plugin general de vLLM.

Las clases Qwen3_5MoeForCausalLM y Qwen3_5ForCausalLM (handlers text-only)
YA existen en vllm/model_executor/models/qwen3_5.py de vLLM 0.20.2, pero el
registry solo expone las variantes multimodales ...ForConditionalGeneration.
Esto las registra para que vLLM pueda servir modelos cuyo config declara
architectures=["Qwen3_5MoeForCausalLM"] (como el repo de Bahushruth).

No modifica vLLM ni el modelo: solo añade dos entradas al registry.
"""


def register():
    from vllm import ModelRegistry

    pairs = {
        "Qwen3_5MoeForCausalLM": "vllm.model_executor.models.qwen3_5:Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM": "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
    }
    for arch, path in pairs.items():
        try:
            ModelRegistry.register_model(arch, path)
        except Exception as e:  # noqa: BLE001
            # Si una versión futura ya lo trae registrado, no es fatal.
            print(f"[qwen35-register] aviso registrando {arch}: {e}")
    print("[qwen35-register] Qwen3_5(Moe)ForCausalLM registrados")
