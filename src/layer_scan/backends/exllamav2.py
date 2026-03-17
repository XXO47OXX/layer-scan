"""ExLlamaV2 backend — optimized for consumer GPUs with quantized models.

This is the recommended backend for scanning large models (70B+) on
consumer hardware (e.g., 2×RTX 4090). ExLlamaV2 supports GPTQ and EXL2
quantization with excellent memory efficiency.

Requires: pip install layer-scan[exllamav2]
"""

from __future__ import annotations

import logging

import torch

from layer_scan.backends.base import Backend
from layer_scan.config import DuplicationConfig

logger = logging.getLogger(__name__)


class ExLlamaV2Backend(Backend):
    """ExLlamaV2 backend with runtime layer duplication.

    Uses ExLlamaV2's modular architecture to execute individual layers
    in a custom order, enabling layer duplication without model copies.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._cache = None
        self._config = None
        self._total_layers: int = 0

    def load(self, model_path: str, **kwargs) -> None:
        """Load an EXL2/GPTQ quantized model.

        Kwargs:
            gpu_split: List of GPU memory limits in MB (e.g., [22000, 22000]).
            max_seq_len: Maximum sequence length (default: 4096).
            rope_scale: RoPE scaling factor (default: 1.0).
        """
        try:
            from exllamav2 import (
                ExLlamaV2,
                ExLlamaV2Cache,
                ExLlamaV2Config,
                ExLlamaV2Tokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "ExLlamaV2 not installed. Install with: "
                "pip install layer-scan[exllamav2]"
            ) from e

        logger.info("Loading ExLlamaV2 model from %s", model_path)

        self._config = ExLlamaV2Config(model_path)
        self._config.max_seq_len = kwargs.get("max_seq_len", 4096)

        if "rope_scale" in kwargs:
            self._config.scale_pos_emb = kwargs["rope_scale"]

        self._model = ExLlamaV2(self._config)

        gpu_split = kwargs.get("gpu_split")
        if gpu_split:
            self._model.load(gpu_split)
        else:
            self._model.load_autosplit()

        self._tokenizer = ExLlamaV2Tokenizer(self._config)
        self._cache = ExLlamaV2Cache(self._model, max_seq_len=self._config.max_seq_len)

        # Count transformer layers (excluding embedding, head, norms)
        self._total_layers = self._count_decoder_layers()
        logger.info("ExLlamaV2 model loaded: %d decoder layers", self._total_layers)

    def _count_decoder_layers(self) -> int:
        """Count the number of decoder layers in the ExLlamaV2 model."""
        count = 0
        for module in self._model.modules:
            module_name = type(module).__name__
            if "Attention" in module_name or "MLP" in module_name:
                # Each transformer block has attention + MLP
                # We count by attention modules to get block count
                if "Attention" in module_name:
                    count += 1
        return count

    def get_total_layers(self) -> int:
        if self._total_layers == 0:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._total_layers

    def get_tokenizer(self):
        """Return a tokenizer adapter compatible with HuggingFace interface."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return _ExLlamaV2TokenizerAdapter(self._tokenizer)

    @torch.no_grad()
    def forward_with_duplication(
        self,
        text: str,
        duplication_config: DuplicationConfig | None = None,
    ) -> torch.Tensor:
        """Run forward pass with optional layer duplication.

        For ExLlamaV2, we use the module-level forward to control
        execution order of individual transformer blocks.
        """
        input_ids = self._tokenizer.encode(text)
        input_ids = input_ids.to(self._model.modules[0].device())

        self._cache.current_seq_len = 0

        if duplication_config is None:
            # Standard forward pass
            logits = self._model.forward(input_ids, self._cache)
            return logits[0, -1, :]

        # Duplicated forward: execute modules in custom order
        return self._forward_duplicated(input_ids, duplication_config)

    def _forward_duplicated(
        self,
        input_ids: torch.Tensor,
        config: DuplicationConfig,
    ) -> torch.Tensor:
        """Execute ExLlamaV2 modules with layer duplication.

        ExLlamaV2 exposes modules as a flat list. We need to map
        decoder layer indices to module indices and build the
        custom execution order.
        """
        # Build module execution order based on layer duplication config
        # ExLlamaV2 modules: [embedding, layer0_attn, layer0_mlp, ..., norm, head]
        exec_order = config.execution_order()

        # Map layer indices to module ranges
        layer_modules = self._get_layer_module_map()

        # Execute embedding
        hidden = self._model.modules[0].forward(input_ids, self._cache)

        # Execute decoder layers in duplicated order
        for layer_idx in exec_order:
            for mod_idx in layer_modules[layer_idx]:
                hidden = self._model.modules[mod_idx].forward(hidden, self._cache)

        # Execute final norm and head
        for mod_idx in self._get_post_layer_modules():
            hidden = self._model.modules[mod_idx].forward(hidden, self._cache)

        return hidden[0, -1, :]

    def _get_layer_module_map(self) -> dict[int, list[int]]:
        """Map decoder layer indices to ExLlamaV2 module indices."""
        layer_map: dict[int, list[int]] = {}
        current_layer = -1

        for mod_idx, module in enumerate(self._model.modules):
            name = type(module).__name__
            if "Attention" in name:
                current_layer += 1
                layer_map[current_layer] = [mod_idx]
            elif "MLP" in name and current_layer >= 0:
                layer_map[current_layer].append(mod_idx)

        return layer_map

    def _get_post_layer_modules(self) -> list[int]:
        """Get module indices for norm and lm_head (after all decoder layers)."""
        post_modules = []
        found_last_mlp = False

        for mod_idx in range(len(self._model.modules) - 1, -1, -1):
            name = type(self._model.modules[mod_idx]).__name__
            if "MLP" in name or "Attention" in name:
                if not found_last_mlp:
                    found_last_mlp = True
                    # Everything after this index is post-layer
                    post_modules = list(range(mod_idx + 1, len(self._model.modules)))
                break

        return post_modules

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._cache is not None:
            del self._cache
            self._cache = None
        self._tokenizer = None
        self._total_layers = 0

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _ExLlamaV2TokenizerAdapter:
    """Adapter to make ExLlamaV2 tokenizer compatible with HuggingFace interface.

    layer-scan's scoring module expects tokenizer.encode(str) -> list[int].
    """

    def __init__(self, exl2_tokenizer) -> None:
        self._tokenizer = exl2_tokenizer

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self._tokenizer.encode(text)
        return ids[0].tolist() if hasattr(ids, "tolist") else list(ids[0])

    def decode(self, token_ids: list[int]) -> str:
        import torch as _torch

        ids = _torch.tensor([token_ids])
        return self._tokenizer.decode(ids)[0]
