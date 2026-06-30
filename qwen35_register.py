"""Parche TEXT-ONLY para bahushruth/qwen3.6-35b-a3b-abliterated-v4 en vLLM 0.20.2.

CAUSA (traceback confirmado en produccion):
  vLLM resuelve la arch Qwen3_5MoeForCausalLM a un handler que comparte el
  ProcessingInfo MULTIMODAL de qwen3_vl. El renderer construye un mm_processor:
    renderers/base.py:119  -> mm_registry.create_processor(...)
    qwen3_vl.py:873        -> get_data_parser -> .vision_config.spatial_merge_size
    qwen3_5.py:115         -> ctx.get_hf_config(Qwen3_5MoeConfig)
  y peta porque el repo trae el config APLANADO text-only:
    TypeError: Expected Qwen3_5MoeConfig, found Qwen3_5MoeTextConfig

FIX (minimo): forzar is_multimodal_model=False / supports_multimodal=False para
nuestras arch, de modo que el renderer NO construya el mm_processor y el modelo
cargue como text-only. NO se toca el modelo, NO se cuantiza, NO fp8.

INYECCION: se llama desde /src/engine.py JUSTO antes de
AsyncLLMEngine.from_engine_args, con vLLM ya 100% importado -> sin circular
import (el fallo del intento anterior con hook sobre __import__).
"""
import sys
import traceback

_OURS = frozenset({"Qwen3_5MoeForCausalLM", "Qwen3_5ForCausalLM"})


def _log(m):
    print("[qwen35]", m, file=sys.stderr, flush=True)


def register():
    _log(">>> register() text-only EJECUTANDO <<<")

    # (A) inspect_model_cls -> supports_multimodal=False.
    #     Asi multimodal_config NO se construye en ModelConfig.__post_init__
    #     (que corre dentro de from_engine_args, DESPUES de este register()).
    try:
        import dataclasses as dc
        from vllm.model_executor.models import registry as reg_mod
        REG = reg_mod.ModelRegistry  # instancia singleton
        if not getattr(REG, "_q35_patched", False):
            _orig_inspect = REG.inspect_model_cls

            def _inspect(architectures, model_config):
                info, arch = _orig_inspect(architectures, model_config)
                if arch in _OURS or getattr(info, "architecture", None) in _OURS:
                    try:
                        info = dc.replace(
                            info,
                            supports_multimodal=False,
                            supports_multimodal_raw_input_only=False,
                            supports_multimodal_encoder_tp_data=False,
                        )
                    except Exception:
                        _log("dataclasses.replace sobre _ModelInfo fallo")
                        traceback.print_exc()
                return info, arch

            REG.inspect_model_cls = _inspect
            REG._q35_patched = True
            _log("inspect_model_cls parcheado (supports_multimodal=False)")
        else:
            _log("inspect_model_cls ya estaba parcheado")
    except Exception:
        _log("(A) inspect_model_cls fallo")
        traceback.print_exc()

    # (B) Cinturon y tirantes: ModelConfig.is_multimodal_model -> False para
    #     nuestras arch, por si multimodal_config se construyo igualmente.
    #     renderers/base.py gatea el mm_processor en is_multimodal_model.
    try:
        from vllm.config.model import ModelConfig
        if not getattr(ModelConfig, "_q35_mm_patched", False):
            _prop = ModelConfig.is_multimodal_model
            _orig_fget = getattr(_prop, "fget", None)
            if _orig_fget is not None:
                def _is_mm(self):
                    try:
                        archs = list(self.architectures)
                    except Exception:
                        archs = []
                    if any(a in _OURS for a in archs):
                        return False
                    return _orig_fget(self)

                ModelConfig.is_multimodal_model = property(_is_mm)
                ModelConfig._q35_mm_patched = True
                _log("ModelConfig.is_multimodal_model parcheado -> False")
            else:
                _log("is_multimodal_model sin .fget; no se parchea property")
        else:
            _log("is_multimodal_model ya estaba parcheado")
    except Exception:
        _log("(B) is_multimodal_model fallo")
        traceback.print_exc()

    _log(">>> register() text-only COMPLETADO <<<")
