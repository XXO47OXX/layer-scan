"""vLLM inference backend for layer-scan.

Uses vLLM for model loading with tensor parallelism support,
then accesses the underlying model's layers for custom duplication ordering.

Advantages over TransformersBackend:
- Tensor parallelism across multiple GPUs
- PagedAttention for better memory efficiency
- Supports larger models on consumer hardware

Usage:
    layer-scan scan --model <path> --backend vllm
    layer-scan scan --model <path> --backend vllm --gpu-split 2
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from layer_scan.backends.base import Backend
from layer_scan.config import DuplicationConfig

logger = logging.getLogger(__name__)


class VLLMBackend(Backend):
    """vLLM backend with tensor parallelism and layer duplication support.

    For baseline (no duplication): uses vLLM's optimized inference.
    For duplication configs: accesses underlying model layers directly.
    """

    def __init__(self) -> None:
        self._llm: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._layers: list[Any] = []
        self._total_layers: int = 0
        self._base_model: Any = None

    def load(self, model_path: str, **kwargs) -> None:
        """Load a model via vLLM.

        Kwargs:
            dtype: Torch dtype string (default: "float16").
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_split: Alias for tensor_parallel_size (list of ints → len).
            trust_remote_code: Whether to trust remote code.
            max_model_len: Maximum sequence length.
            enforce_eager: Disable CUDA graphs (default: True for layer access).
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is required for this backend.\n"
                "Install it with: pip install vllm"
            ) from None

        dtype_str = kwargs.get("dtype", "float16")

        # Determine tensor parallel size
        tp_size = kwargs.get("tensor_parallel_size", 1)
        gpu_split = kwargs.get("gpu_split")
        if gpu_split and isinstance(gpu_split, list):
            tp_size = len(gpu_split)

        logger.info(
            "Loading model via vLLM: %s (dtype=%s, tp=%d)",
            model_path, dtype_str, tp_size,
        )

        self._llm = LLM(
            model=model_path,
            dtype=dtype_str,
            tensor_parallel_size=tp_size,
            trust_remote_code=kwargs.get("trust_remote_code", False),
            max_model_len=kwargs.get("max_model_len", None),
            enforce_eager=kwargs.get("enforce_eager", True),
        )

        # Extract tokenizer
        self._tokenizer = self._llm.get_tokenizer()

        # Access underlying model for layer-level operations
        self._model = self._extract_underlying_model()
        if self._model is not None:
            self._layers = self._find_layers()
            self._total_layers = len(self._layers)
            self._base_model = self._find_base_model()
            logger.info("Model loaded: %d layers discovered", self._total_layers)
        else:
            logger.warning(
                "Could not access underlying model. "
                "Layer duplication will not be available."
            )

    def _extract_underlying_model(self) -> Any | None:
        """Extract the underlying HF model from vLLM internals."""
        try:
            # vLLM >= 0.4.x path
            executor = self._llm.llm_engine.model_executor
            if hasattr(executor, "driver_worker"):
                worker = executor.driver_worker
                if hasattr(worker, "model_runner"):
                    return worker.model_runner.model
            # vLLM with Ray distributed
            if hasattr(executor, "driver_worker_model_runner"):
                return executor.driver_worker_model_runner.model
        except (AttributeError, RuntimeError) as e:
            logger.debug("Could not extract model: %s", e)

        return None

    def _find_base_model(self) -> Any | None:
        """Find the base model object (contains embed_tokens, norm, etc.)."""
        model = self._model
        for attr in ["model", "transformer", "gpt_neox"]:
            if hasattr(model, attr):
                return getattr(model, attr)
        return None

    def _find_layers(self) -> list[Any]:
        """Discover decoder layers in the underlying model."""
        model = self._model

        for attr_path in [
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "transformer.blocks",
            "model.decoder.layers",
        ]:
            obj = model
            found = True
            for part in attr_path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            if found and hasattr(obj, "__len__"):
                return list(obj)

        raise RuntimeError(
            "Could not find decoder layers in vLLM model. "
            "Supported architectures: LLaMA, Mistral, Qwen2, GPT-2, GPT-NeoX."
        )

    def get_total_layers(self) -> int:
        if self._total_layers == 0:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._total_layers

    def get_tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer

    @torch.no_grad()
    def forward_with_duplication(
        self,
        text: str,
        duplication_config: DuplicationConfig | None = None,
    ) -> torch.Tensor:
        """Run forward pass with optional layer duplication.

        Baseline: uses vLLM's optimized inference path.
        Duplication: manual layer-by-layer execution on underlying model.
        """
        if duplication_config is None:
            return self._forward_baseline(text)

        if self._model is None:
            raise RuntimeError(
                "Layer duplication requires access to underlying model, "
                "which is not available in this vLLM configuration."
            )

        return self._forward_duplicated(text, duplication_config)

    def _forward_baseline(self, text: str) -> torch.Tensor:
        """Baseline forward pass through the underlying model.

        vLLM's generate() API doesn't expose raw logits, so we use
        manual layer-by-layer execution on the underlying HF model.
        """
        return self._forward_manual(text)

    def _forward_manual(self, text: str) -> torch.Tensor:
        """Manual forward pass through the underlying model."""
        if self._model is None:
            raise RuntimeError("Underlying model not accessible")

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        base = self._base_model
        if base is None:
            raise RuntimeError("Base model not found")

        # Compute embeddings
        if hasattr(base, "embed_tokens"):
            hidden_states = base.embed_tokens(input_ids)
        elif hasattr(base, "wte"):
            hidden_states = base.wte(input_ids)
        else:
            raise RuntimeError("Cannot find embedding layer")

        # Execute all layers in order
        position_ids = torch.arange(
            input_ids.shape[1], device=device
        ).unsqueeze(0)

        position_embeddings = self._compute_position_embeddings(
            base, hidden_states, position_ids
        )

        for layer in self._layers:
            layer_kwargs: dict[str, Any] = {
                "attention_mask": None,
                "position_ids": position_ids,
            }
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            layer_output = layer(hidden_states, **layer_kwargs)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        # Final norm + LM head
        hidden_states = self._apply_final_norm(base, hidden_states)
        logits = self._apply_lm_head(self._model, base, hidden_states)

        return logits[0, -1, :]

    def _forward_duplicated(
        self,
        text: str,
        config: DuplicationConfig,
    ) -> torch.Tensor:
        """Layer-duplicated forward pass."""
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        base = self._base_model
        if base is None:
            raise RuntimeError("Base model not found")

        # Compute embeddings
        if hasattr(base, "embed_tokens"):
            hidden_states = base.embed_tokens(input_ids)
        elif hasattr(base, "wte"):
            hidden_states = base.wte(input_ids)
        else:
            raise RuntimeError("Cannot find embedding layer")

        exec_order = config.execution_order()

        position_ids = torch.arange(
            input_ids.shape[1], device=device
        ).unsqueeze(0)

        position_embeddings = self._compute_position_embeddings(
            base, hidden_states, position_ids
        )

        for layer_idx in exec_order:
            layer = self._layers[layer_idx]
            layer_kwargs: dict[str, Any] = {
                "attention_mask": None,
                "position_ids": position_ids,
            }
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            layer_output = layer(hidden_states, **layer_kwargs)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        hidden_states = self._apply_final_norm(base, hidden_states)
        logits = self._apply_lm_head(self._model, base, hidden_states)

        return logits[0, -1, :]

    @staticmethod
    def _compute_position_embeddings(
        base_model: Any,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Pre-compute rotary position embeddings if available."""
        if hasattr(base_model, "rotary_emb"):
            return base_model.rotary_emb(hidden_states, position_ids)
        return None

    @staticmethod
    def _apply_final_norm(base: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final layer normalization."""
        if hasattr(base, "norm"):
            return base.norm(hidden_states)
        elif hasattr(base, "ln_f"):
            return base.ln_f(hidden_states)
        return hidden_states

    @staticmethod
    def _apply_lm_head(model: Any, base: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits via LM head."""
        if hasattr(model, "lm_head"):
            return model.lm_head(hidden_states)
        elif hasattr(base, "lm_head"):
            return base.lm_head(hidden_states)
        raise RuntimeError("Cannot find LM head")

    def cleanup(self) -> None:
        """Release resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._model = None
        self._tokenizer = None
        self._layers = []
        self._total_layers = 0
        self._base_model = None

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
