"""vLLM general plugin: serve bahushruth/qwen3.6-35b-a3b-abliterated-v4 as
TEXT-ONLY (BF16, no fp8) under vLLM 0.20.2, without touching the model repo.

The HF repo ships a FLATTENED text-only config:
    model_type      = "qwen3_5_moe_text"
    architectures   = ["Qwen3_5MoeForCausalLM"]
    (no text_config / vision_config; transformers 5.x -> Qwen3_5MoeTextConfig)
Weights: lm_head.weight + model.language_model.*  (no model.visual.*, no mtp.*).

vLLM 0.20.2 otherwise:
  - resolves the arch to the vision-capable handler / builds a multimodal
    renderer and dies in renderers/base.py:108 ->
    Qwen3_5MoeProcessingInfo.get_hf_config(Qwen3_5MoeConfig)
    -> TypeError: Expected Qwen3_5MoeConfig, found Qwen3_5MoeTextConfig.
  - even on the text-only handler Qwen3_5MoeForCausalLM:
      * lacks is_hybrid -> GDN/SSM state never allocated (broken MoE-hybrid)
      * lacks get_mamba_state_* classmethods -> AttributeError sizing the
        mamba page (those live only on Qwen3_5(/Next)ForConditionalGeneration)
      * load_weights has no mapper -> all model.language_model.* tensors are
        silently "skip loading" (garbage weights).

This plugin monkey-patches runtime only (imported before the engine is built).
Every patch is wrapped in try/except and is idempotent.

Entry point (pyproject / setup.py of the plugin package):
    [project.entry-points."vllm.general_plugins"]
    qwen35_textonly = "qwen35_register:register"
"""

import logging

logger = logging.getLogger("qwen35_textonly_plugin")

# Architectures this plugin owns. Both are aliased to the text-only handler.
_OURS = frozenset({"Qwen3_5MoeForCausalLM", "Qwen3_5ForCausalLM"})


def register():
    # =====================================================================
    # (1) Aliases by STRING (lazy). Keeps the architecture resolvable even
    #     before the heavier patches run, and matches the already-working
    #     setup ("Resolved architecture: Qwen3_5MoeForCausalLM").
    # =====================================================================
    try:
        from vllm import ModelRegistry

        ModelRegistry.register_model(
            "Qwen3_5MoeForCausalLM",
            "vllm.model_executor.models.qwen3_5:Qwen3_5MoeForCausalLM",
        )
        ModelRegistry.register_model(
            "Qwen3_5ForCausalLM",
            "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
        )
        logger.info("[qwen35] string aliases registered")
    except Exception:
        logger.exception("[qwen35] (1) string register_model failed")

    # =====================================================================
    # (2) TEXT-ONLY load fixes on the handler class, applied BEFORE the
    #     final class-object re-registration in (4) — because
    #     _RegisteredModel.from_model_cls() snapshots _ModelInfo
    #     (is_hybrid / supports_multimodal) at registration time.
    # =====================================================================
    TextCausalLM = None
    try:
        from vllm.model_executor.models import qwen3_5 as _q35
        from vllm.model_executor.models.utils import (
            AutoWeightsLoader,
            WeightsMapper,
        )
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
            MambaStateDtypeCalculator,
            MambaStateShapeCalculator,
        )

        TextCausalLM = _q35.Qwen3_5MoeForCausalLM

        # --- (2a) Mark the handler as a HYBRID (GatedDeltaNet/SSM) model. ---
        # config/model.py:1559 is_hybrid reads _model_info.is_hybrid, which is
        # getattr(cls, "is_hybrid", False). The handler does NOT inherit
        # IsHybrid (only Qwen3_5(/Next)ForConditionalGeneration do), so without
        # this it is treated as a plain decoder, attn_type != "hybrid", and the
        # linear_attention layers never get GDN state allocated.
        TextCausalLM.is_hybrid = True

        # --- (2b) Mamba/GDN state hooks (classmethods on the model class). ---
        # platforms/interface.py and gpu_model_runner call these on the
        # resolved class to size/copy the mamba page. They exist only on the
        # ConditionalGeneration handlers, NOT on Qwen3_5MoeForCausalLM
        # (Qwen3_5ForCausalLMBase + QwenNextMixtureOfExperts define neither).
        # Reimplemented verbatim vs. Qwen3_5ForConditionalGeneration
        # (qwen3_5.py:711-746); with the flattened config hf_text_config IS
        # hf_config, so linear_* attrs are present.
        @classmethod
        def _gdn_state_dtype(cls, vllm_config):
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            )

        @classmethod
        def _gdn_state_shape(cls, vllm_config):
            parallel_config = vllm_config.parallel_config
            hf = vllm_config.model_config.hf_text_config
            num_spec = (
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            )
            return MambaStateShapeCalculator.gated_delta_net_state_shape(
                parallel_config.tensor_parallel_size,
                hf.linear_num_key_heads,
                hf.linear_num_value_heads,
                hf.linear_key_head_dim,
                hf.linear_value_head_dim,
                hf.linear_conv_kernel_dim,
                num_spec,
            )

        @classmethod
        def _gdn_state_copy(cls):
            return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

        TextCausalLM.get_mamba_state_dtype_from_config = _gdn_state_dtype
        TextCausalLM.get_mamba_state_shape_from_config = _gdn_state_shape
        TextCausalLM.get_mamba_state_copy_func = _gdn_state_copy

        # --- (2c) Weight-prefix remap in load_weights. ----------------------
        # Repo weights: lm_head.weight + model.language_model.*  (no mtp.*).
        # Handler module tree: self.model.* + self.lm_head.* (NO language_model
        # wrapper). The base load_weights (qwen3_5.py:544) uses no mapper, so
        # every model.language_model.* tensor is silently skipped -> garbage.
        # Strip the wrapper: "model.language_model." -> "model." (mapper.apply
        # runs first in AutoWeightsLoader.load_weights; lm_head.weight intact).
        _TEXT_WEIGHT_MAPPER = WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "model."}
        )

        def _text_load_weights(self, weights):
            loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
            return loader.load_weights(weights, mapper=_TEXT_WEIGHT_MAPPER)

        TextCausalLM.load_weights = _text_load_weights

        logger.info("[qwen35] text-only load fixes applied (is_hybrid, "
                    "mamba hooks, weight mapper)")
    except Exception:
        logger.exception("[qwen35] (2) text-only load patch failed")

    # =====================================================================
    # (3) Multimodal detection -> force TEXT-ONLY at the inspection
    #     chokepoint, so is_multimodal_model == False and the renderer
    #     never builds a vision processor.
    #
    #     config/model.py:623 builds multimodal_config iff
    #       _model_info.supports_multimodal is True;
    #     :1520 is_multimodal_model == (multimodal_config is not None);
    #     renderers/base.py:108 gates the whole mm branch on it.
    #
    #     ModelRegistry is a singleton INSTANCE; inspect_model_cls is a bound
    #     method returning (info, arch). _ModelInfo is a frozen dataclass ->
    #     rebuild via dataclasses.replace().
    # =====================================================================
    try:
        import dataclasses as _dc
        from vllm.model_executor.models import registry as _reg_mod

        _ModelRegistry = _reg_mod.ModelRegistry  # singleton instance

        if not getattr(_ModelRegistry, "_q35_textonly_patched", False):
            _orig_inspect = _ModelRegistry.inspect_model_cls  # already bound

            def _inspect_model_cls_textonly(architectures, model_config):
                info, arch = _orig_inspect(architectures, model_config)
                if arch in _OURS or getattr(info, "architecture", None) in _OURS:
                    try:
                        info = _dc.replace(
                            info,
                            supports_multimodal=False,
                            supports_multimodal_raw_input_only=False,
                            supports_multimodal_encoder_tp_data=False,
                        )
                    except Exception:
                        logger.exception(
                            "[qwen35] dataclasses.replace on _ModelInfo failed"
                        )
                return info, arch

            # Rebind on the instance (plain function: _orig_inspect is bound,
            # so no double self).
            _ModelRegistry.inspect_model_cls = _inspect_model_cls_textonly
            _ModelRegistry._q35_textonly_patched = True
            logger.info("[qwen35] inspect_model_cls patched (force "
                        "supports_multimodal=False)")
    except Exception:
        logger.exception("[qwen35] (3) inspect_model_cls patch failed")

    # =====================================================================
    # (3b) Belt-and-suspenders: force the ModelConfig.is_multimodal_model
    #      property to False for our arches, in case a multimodal_config was
    #      ever built (e.g. a differently-ordered re-resolution to the
    #      ConditionalGeneration variant before our patches ran).
    # =====================================================================
    try:
        from vllm.config.model import ModelConfig as _ModelConfig

        if not getattr(_ModelConfig, "_q35_textonly_mm_patched", False):
            _prop = _ModelConfig.is_multimodal_model
            _orig_is_mm = getattr(_prop, "fget", None)
            if _orig_is_mm is not None:
                def _is_multimodal_model_textonly(self):
                    try:
                        archs = list(self.architectures)
                    except Exception:
                        archs = []
                    if (any(a in _OURS for a in archs)
                            or getattr(self, "_architecture", None) in _OURS):
                        return False
                    return _orig_is_mm(self)

                _ModelConfig.is_multimodal_model = property(
                    _is_multimodal_model_textonly
                )
                _ModelConfig._q35_textonly_mm_patched = True
                logger.info("[qwen35] ModelConfig.is_multimodal_model patched")
            else:
                logger.warning("[qwen35] is_multimodal_model has no .fget; "
                               "skipping property override")
    except Exception:
        logger.exception("[qwen35] (3b) is_multimodal_model property patch failed")

    # =====================================================================
    # (4) Re-register the PATCHED CLASS-OBJECT (not a string).
    #     The lazy/string registration computes _ModelInfo from an on-disk
    #     cache or a SUBPROCESS that re-imports the module WITHOUT our patches
    #     -> is_hybrid=False / wrong supports_multimodal. Registering the
    #     class-object builds a _RegisteredModel whose _ModelInfo is snapshot
    #     IN-PROCESS from this patched class: is_hybrid=True and
    #     supports_multimodal=False (the real class is not SupportsMultiModal)
    #     -> is_multimodal_model=False at the root.
    #     MUST run AFTER (2) so the class attrs are already set.
    # =====================================================================
    try:
        if TextCausalLM is not None:
            from vllm import ModelRegistry

            ModelRegistry.register_model("Qwen3_5MoeForCausalLM", TextCausalLM)
            logger.info("[qwen35] re-registered Qwen3_5MoeForCausalLM as "
                        "patched class-object")
        else:
            logger.error("[qwen35] (4) TextCausalLM unavailable; "
                         "skipping class re-register")
    except Exception:
        logger.exception("[qwen35] (4) class-object re-register failed")
