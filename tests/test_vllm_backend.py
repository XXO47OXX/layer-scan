"""Tests for vLLM backend (mocked — no actual vLLM required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from layer_scan.config import DuplicationConfig


class TestVLLMBackendImport:
    def test_import_error_without_vllm(self):
        """VLLMBackend.load() raises ImportError when vllm not installed."""
        from layer_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        with patch.dict("sys.modules", {"vllm": None}):
            with pytest.raises(ImportError, match="vllm"):
                backend.load("test-model")

    def test_class_instantiates(self):
        """VLLMBackend can be instantiated without vllm installed."""
        from layer_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        assert backend._llm is None
        assert backend._total_layers == 0


class TestVLLMBackendInterface:
    def test_get_total_layers_before_load(self):
        from layer_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.get_total_layers()

    def test_get_tokenizer_before_load(self):
        from layer_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.get_tokenizer()

    def test_cleanup_safe_when_not_loaded(self):
        from layer_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        backend.cleanup()  # Should not raise


class TestVLLMBackendWithMock:
    @pytest.fixture
    def mock_vllm_backend(self):
        """Create a VLLMBackend with mocked internals."""
        from layer_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()

        # Mock the model with LLaMA-style architecture
        mock_model = MagicMock()
        mock_base = MagicMock()
        mock_model.model = mock_base

        # Layers
        mock_layers = [MagicMock() for _ in range(8)]
        for layer in mock_layers:
            layer.return_value = (torch.randn(1, 5, 64),)
        mock_base.layers = mock_layers

        # Embeddings
        mock_base.embed_tokens.return_value = torch.randn(1, 5, 64)
        mock_base.norm.return_value = torch.randn(1, 5, 64)
        mock_model.lm_head.return_value = torch.randn(1, 5, 100)

        # No rotary embeddings
        if hasattr(mock_base, "rotary_emb"):
            del mock_base.rotary_emb

        # Mock parameters for device detection
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.randint(0, 100, (1, 5))}

        def mock_encode(text, add_special_tokens=True):
            if len(text) == 1 and text.isdigit():
                return [48 + int(text)]
            return [100]

        mock_tokenizer.encode = mock_encode

        # Set internals directly
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        backend._layers = mock_layers
        backend._total_layers = 8
        backend._base_model = mock_base

        return backend

    def test_get_total_layers(self, mock_vllm_backend):
        assert mock_vllm_backend.get_total_layers() == 8

    def test_get_tokenizer(self, mock_vllm_backend):
        tok = mock_vllm_backend.get_tokenizer()
        assert tok is not None

    def test_forward_baseline(self, mock_vllm_backend):
        logits = mock_vllm_backend._forward_manual("test text")
        assert logits.shape == (100,)

    def test_forward_duplicated(self, mock_vllm_backend):
        config = DuplicationConfig(i=2, j=6, total_layers=8)
        logits = mock_vllm_backend._forward_duplicated("test text", config)
        assert logits.shape == (100,)

    def test_forward_duplicated_execution_order(self, mock_vllm_backend):
        config = DuplicationConfig(i=3, j=5, total_layers=8)
        expected_order = config.execution_order()
        assert expected_order == [0, 1, 2, 3, 4, 3, 4, 5, 6, 7]

    def test_cleanup(self, mock_vllm_backend):
        mock_vllm_backend.cleanup()
        assert mock_vllm_backend._model is None
        assert mock_vllm_backend._total_layers == 0


class TestVLLMFindLayers:
    def test_finds_llama_layers(self):
        from layer_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        mock_model = MagicMock()
        mock_model.model.layers = [MagicMock() for _ in range(4)]
        backend._model = mock_model

        layers = backend._find_layers()
        assert len(layers) == 4

    def test_finds_gpt2_layers(self):
        from layer_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        mock_model = MagicMock(spec=[])

        # GPT-2 style: model.transformer.h
        mock_transformer = MagicMock(spec=[])
        mock_transformer.h = [MagicMock() for _ in range(6)]
        mock_model.transformer = mock_transformer

        backend._model = mock_model

        layers = backend._find_layers()
        assert len(layers) == 6
