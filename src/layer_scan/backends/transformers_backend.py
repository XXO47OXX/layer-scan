"""HuggingFace Transformers backend — the default reference implementation.

This backend hooks into the model's forward pass at the layer level,
implementing layer duplication by re-executing specified layers without
modifying model weights. It works with any CausalLM model.

Note: This is the reference backend for correctness. For large models
(70B+), use the ExLlamaV2 backend for quantized inference on consumer GPUs.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from layer_scan.backends.base import Backend
from layer_scan.config import DuplicationConfig

logger = logging.getLogger(__name__)


class TransformersBackend(Backend):
    """HuggingFace Transformers backend with runtime layer duplication."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._layers: list[Any] = []
        self._total_layers: int = 0

    def load(self, model_path: str, **kwargs) -> None:
        """Load a CausalLM model from path or HuggingFace ID.

        Kwargs:
            dtype: Torch dtype string (default: "float16").
            device_map: Device map for model parallelism (default: "auto").
            trust_remote_code: Whether to trust remote code (default: False).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_str = kwargs.get("dtype", "float16")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.float16)

        logger.info("Loading tokenizer from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )

        logger.info("Loading model from %s (dtype=%s)", model_path, dtype_str)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=kwargs.get("device_map", "auto"),
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        self._model.eval()

        # Discover layer structure
        self._layers = self._find_layers()
        self._total_layers = len(self._layers)
        logger.info("Model loaded: %d layers discovered", self._total_layers)

    def _find_layers(self) -> list[Any]:
        """Discover the decoder layers in the model.

        Supports common architectures: LLaMA, Mistral, Qwen, GPT-NeoX, etc.
        """
        model = self._model

        # Try common layer container names
        for attr_path in [
            "model.layers",              # LLaMA, Mistral, Qwen2
            "transformer.h",             # GPT-2, GPT-Neo
            "gpt_neox.layers",           # GPT-NeoX, Pythia
            "transformer.blocks",        # MPT
            "model.decoder.layers",      # OPT
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
            "Could not find decoder layers. Supported architectures: "
            "LLaMA, Mistral, Qwen2, GPT-2, GPT-NeoX, MPT, OPT. "
            "For other architectures, use a custom backend."
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
        """Run forward pass, optionally duplicating layers [i..j-1].

        For the baseline (duplication_config=None), uses the standard
        model forward pass. For duplicated configs, hooks into the
        layer execution to replay specified layers.

        Returns:
            Logits tensor of shape (vocab_size,) for the last token.
        """
        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)

        if duplication_config is None:
            # Standard forward pass (baseline)
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return outputs.logits[0, -1, :]

        # Layer-duplicated forward pass
        return self._forward_duplicated(
            input_ids=input_ids,
            attention_mask=attention_mask,
            config=duplication_config,
        )

    def _forward_duplicated(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        config: DuplicationConfig,
    ) -> torch.Tensor:
        """Execute the forward pass with layer duplication.

        Execution order: [0...j-1, i...N-1]
        Layers [i...j-1] execute twice — this gives the model
        additional "thinking time" on the reasoning circuits.
        """
        model = self._model

        # Get the embedding output (before decoder layers)
        if hasattr(model, "model"):
            # LLaMA-style: model.model.embed_tokens
            base = model.model
        elif hasattr(model, "transformer"):
            # GPT-style: model.transformer.wte
            base = model.transformer
        else:
            raise RuntimeError("Cannot determine embedding layer location")

        # Compute embeddings
        if hasattr(base, "embed_tokens"):
            hidden_states = base.embed_tokens(input_ids)
        elif hasattr(base, "wte"):
            hidden_states = base.wte(input_ids)
        else:
            raise RuntimeError("Cannot find embedding layer")

        # Build execution order
        exec_order = config.execution_order()

        # Execute layers in the duplicated order
        # Note: For models with RoPE, position IDs remain based on
        # sequence length — we don't change them for duplicated layers.
        # This matches the RYS methodology.
        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)

        # Pre-compute rotary embeddings for architectures that need them
        # (transformers 5.x moved rotary_emb out of individual layers)
        position_embeddings = self._compute_position_embeddings(
            base, hidden_states, position_ids
        )

        for layer_idx in exec_order:
            layer = self._layers[layer_idx]
            # Most decoder layers accept (hidden_states, attention_mask, position_ids)
            # but the exact signature varies by architecture
            layer_kwargs: dict[str, Any] = {
                "attention_mask": self._prepare_causal_mask(
                    attention_mask, hidden_states
                ),
                "position_ids": position_ids,
            }
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            layer_output = layer(hidden_states, **layer_kwargs)
            # Layer output is typically a tuple; first element is hidden_states
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        # Apply final norm
        if hasattr(base, "norm"):
            hidden_states = base.norm(hidden_states)
        elif hasattr(base, "ln_f"):
            hidden_states = base.ln_f(hidden_states)

        # Project to logits
        if hasattr(model, "lm_head"):
            logits = model.lm_head(hidden_states)
        elif hasattr(base, "lm_head"):
            logits = base.lm_head(hidden_states)
        else:
            raise RuntimeError("Cannot find LM head for logit projection")

        return logits[0, -1, :]

    @torch.no_grad()
    def forward_batch(
        self,
        texts: list[str],
        duplication_config: DuplicationConfig | None = None,
    ) -> list[torch.Tensor]:
        """Run true GPU-batched forward pass on multiple texts.

        Pads all texts to equal length and processes in a single kernel.
        For baseline (no duplication), uses the model's native batching.
        For duplicated configs, falls back to sequential (custom layer
        execution doesn't support batch yet).

        Returns:
            List of logit tensors, one per text, each (vocab_size,).
        """
        if not texts:
            return []

        # Duplicated forward requires custom layer loop — fall back to sequential
        if duplication_config is not None:
            return [
                self.forward_with_duplication(text, duplication_config)
                for text in texts
            ]

        # Batch tokenize with padding
        self._tokenizer.pad_token = self._tokenizer.pad_token or self._tokenizer.eos_token
        batch = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = batch["input_ids"].to(self._model.device)
        attention_mask = batch["attention_mask"].to(self._model.device)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract last-token logits for each sequence (accounting for padding)
        # For left-padded models, last real token is at different positions
        results = []
        for i in range(input_ids.shape[0]):
            # Find the last non-padding position
            seq_mask = attention_mask[i]
            last_pos = seq_mask.sum().item() - 1
            results.append(outputs.logits[i, last_pos, :])

        return results

    @staticmethod
    def _compute_position_embeddings(
        base_model: Any,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Pre-compute rotary position embeddings (cos, sin).

        Transformers 5.x moved rotary_emb out of individual decoder layers.
        The base model computes (cos, sin) once and passes them to each layer.
        Returns None for architectures without rotary embeddings (e.g. GPT-2).
        """
        if hasattr(base_model, "rotary_emb"):
            return base_model.rotary_emb(hidden_states, position_ids)
        return None

    @staticmethod
    def _prepare_causal_mask(
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Prepare attention mask for direct layer calls.

        When calling layers directly (bypassing model.forward), we pass None
        so each layer constructs its own causal mask internally. The raw
        tokenizer mask (long int) is incompatible with SDPA which expects
        bool/float. For single-sequence inference there is no padding, so
        None is correct and safe.
        """
        return None

    def cleanup(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._layers = []
        self._total_layers = 0

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
