"""Abstract backend interface for layer-scan."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from layer_scan.config import DuplicationConfig


class Backend(ABC):
    """Abstract base for inference backends.

    A backend must provide:
    1. Model loading and tokenization
    2. Forward pass with optional layer duplication
    3. Layer count information
    """

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """Load a model from the given path or HuggingFace ID."""

    @abstractmethod
    def get_total_layers(self) -> int:
        """Return the total number of transformer decoder layers."""

    @abstractmethod
    def get_tokenizer(self):
        """Return the model's tokenizer."""

    @abstractmethod
    def forward_with_duplication(
        self,
        text: str,
        duplication_config: DuplicationConfig | None = None,
    ) -> torch.Tensor:
        """Run forward pass with optional layer duplication.

        Args:
            text: Input text to process.
            duplication_config: If provided, duplicate layers [i..j-1].
                If None, run the standard forward pass (baseline).

        Returns:
            Logits tensor of shape (vocab_size,) for the last token position.
        """

    def forward_batch(
        self,
        texts: list[str],
        duplication_config: DuplicationConfig | None = None,
    ) -> list[torch.Tensor]:
        """Run batched forward pass on multiple texts.

        Default implementation falls back to sequential calls.
        Override in backends that support true GPU batching.

        Returns:
            List of logit tensors, one per text, each (vocab_size,).
        """
        return [
            self.forward_with_duplication(text, duplication_config)
            for text in texts
        ]

    def cleanup(self) -> None:
        """Release resources (GPU memory, etc.). Optional."""
