from unittest.mock import MagicMock

import pytest
import torch

from layer_scan.config import DuplicationConfig


class TestTransformersBackendFindLayers:
    """Test _find_layers for various architecture attribute paths."""

    def _make_backend_with_model(self, model_mock):
        from layer_scan.backends.transformers_backend import TransformersBackend
        backend = TransformersBackend()
        backend._model = model_mock
        return backend

    def test_llama_style_layers(self):
        model = MagicMock()
        layers = [MagicMock() for _ in range(32)]
        model.model.layers = layers
        del model.transformer

        backend = self._make_backend_with_model(model)
        found = backend._find_layers()
        assert len(found) == 32

    def test_gpt2_style_layers(self):
        model = MagicMock()
        del model.model
        layers = [MagicMock() for _ in range(12)]
        model.transformer.h = layers
        del model.transformer.blocks
        del model.gpt_neox

        backend = self._make_backend_with_model(model)
        found = backend._find_layers()
        assert len(found) == 12

    def test_gpt_neox_style_layers(self):
        model = MagicMock()
        del model.model
        del model.transformer
        layers = [MagicMock() for _ in range(24)]
        model.gpt_neox.layers = layers

        backend = self._make_backend_with_model(model)
        found = backend._find_layers()
        assert len(found) == 24

    def test_mpt_style_layers(self):
        model = MagicMock()
        del model.model
        del model.gpt_neox
        layers = [MagicMock() for _ in range(16)]
        model.transformer.blocks = layers
        del model.transformer.h

        backend = self._make_backend_with_model(model)
        found = backend._find_layers()
        assert len(found) == 16

    def test_opt_style_layers(self):
        model = MagicMock()
        del model.model.layers
        del model.transformer
        del model.gpt_neox
        layers = [MagicMock() for _ in range(20)]
        model.model.decoder.layers = layers

        backend = self._make_backend_with_model(model)
        found = backend._find_layers()
        assert len(found) == 20

    def test_unknown_architecture_raises(self):
        from layer_scan.backends.transformers_backend import TransformersBackend
        model = MagicMock(spec=[])  # No attributes at all
        backend = TransformersBackend()
        backend._model = model
        with pytest.raises(RuntimeError, match="Could not find decoder layers"):
            backend._find_layers()


class TestTransformersBackendForward:
    """Test forward pass logic."""

    def test_baseline_uses_standard_forward(self):
        from layer_scan.backends.transformers_backend import TransformersBackend
        backend = TransformersBackend()
        backend._tokenizer = MagicMock()
        backend._tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 3, 100)
        backend._model = MagicMock()
        backend._model.device = torch.device("cpu")
        backend._model.return_value = mock_output

        result = backend.forward_with_duplication("test text", duplication_config=None)
        assert result.shape == (100,)
        backend._model.assert_called_once()

    def test_duplicated_forward_calls_layers_in_order(self):
        from layer_scan.backends.transformers_backend import TransformersBackend
        backend = TransformersBackend()

        # Setup tokenizer
        backend._tokenizer = MagicMock()
        backend._tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Setup model with LLaMA-style architecture
        model = MagicMock()
        model.device = torch.device("cpu")

        # Create layers that return (hidden_states,) tuples
        executed_layers = []
        layers = []
        for idx in range(4):
            layer = MagicMock()
            def make_side_effect(layer_idx):
                def side_effect(hidden, **kwargs):
                    executed_layers.append(layer_idx)
                    return (hidden,)
                return side_effect
            layer.side_effect = make_side_effect(idx)
            layers.append(layer)

        backend._model = model
        backend._layers = layers
        backend._total_layers = 4

        # Setup embedding
        model.model.embed_tokens.return_value = torch.randn(1, 3, 64)
        # Setup norm and lm_head
        model.model.norm.return_value = torch.randn(1, 3, 64)
        model.lm_head.return_value = torch.randn(1, 3, 100)

        config = DuplicationConfig(i=1, j=3, total_layers=4)
        # execution_order: [0, 1, 2, 1, 2, 3]
        result = backend.forward_with_duplication("test", duplication_config=config)

        assert executed_layers == [0, 1, 2, 1, 2, 3]
        assert result.shape == (100,)

    def test_forward_duplicated_unknown_embedding_raises(self):
        from layer_scan.backends.transformers_backend import TransformersBackend
        backend = TransformersBackend()
        backend._model = MagicMock(spec=[])  # No model or transformer attr
        backend._layers = []

        config = DuplicationConfig(i=0, j=1, total_layers=2)
        with pytest.raises(RuntimeError, match="Cannot determine embedding"):
            backend._forward_duplicated(
                torch.tensor([[1]]), None, config
            )


class TestTransformersBackendLifecycle:
    """Test load/cleanup lifecycle."""

    def test_get_total_layers_not_loaded(self):
        from layer_scan.backends.transformers_backend import TransformersBackend
        backend = TransformersBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.get_total_layers()

    def test_get_tokenizer_not_loaded(self):
        from layer_scan.backends.transformers_backend import TransformersBackend
        backend = TransformersBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.get_tokenizer()

    def test_cleanup_releases_resources(self):
        from layer_scan.backends.transformers_backend import TransformersBackend
        backend = TransformersBackend()
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()
        backend._layers = [MagicMock()]
        backend._total_layers = 10

        backend.cleanup()

        assert backend._model is None
        assert backend._tokenizer is None
        assert backend._layers == []
        assert backend._total_layers == 0


class TestExLlamaV2Backend:
    """ExLlamaV2 backend tests."""

    def test_count_decoder_layers(self):
        from layer_scan.backends.exllamav2 import ExLlamaV2Backend
        backend = ExLlamaV2Backend()

        # Create mock modules with different type names
        modules = []
        for i in range(5):
            attn = MagicMock()
            type(attn).__name__ = "ExLlamaV2Attention"
            mlp = MagicMock()
            type(mlp).__name__ = "ExLlamaV2MLP"
            modules.extend([attn, mlp])

        # Add embedding and head
        embed = MagicMock()
        type(embed).__name__ = "ExLlamaV2Embedding"
        head = MagicMock()
        type(head).__name__ = "ExLlamaV2Linear"
        modules = [embed] + modules + [head]

        backend._model = MagicMock()
        backend._model.modules = modules

        assert backend._count_decoder_layers() == 5

    def test_get_layer_module_map(self):
        from layer_scan.backends.exllamav2 import ExLlamaV2Backend
        backend = ExLlamaV2Backend()

        modules = []
        embed = MagicMock()
        type(embed).__name__ = "ExLlamaV2Embedding"
        modules.append(embed)

        for i in range(3):
            attn = MagicMock()
            type(attn).__name__ = "ExLlamaV2Attention"
            mlp = MagicMock()
            type(mlp).__name__ = "ExLlamaV2MLP"
            modules.extend([attn, mlp])

        norm = MagicMock()
        type(norm).__name__ = "ExLlamaV2RMSNorm"
        head = MagicMock()
        type(head).__name__ = "ExLlamaV2Linear"
        modules.extend([norm, head])

        backend._model = MagicMock()
        backend._model.modules = modules

        layer_map = backend._get_layer_module_map()
        assert len(layer_map) == 3
        # Layer 0: attn at index 1, mlp at index 2
        assert layer_map[0] == [1, 2]
        assert layer_map[1] == [3, 4]
        assert layer_map[2] == [5, 6]

    def test_get_post_layer_modules(self):
        from layer_scan.backends.exllamav2 import ExLlamaV2Backend
        backend = ExLlamaV2Backend()

        modules = []
        embed = MagicMock()
        type(embed).__name__ = "ExLlamaV2Embedding"
        modules.append(embed)

        attn = MagicMock()
        type(attn).__name__ = "ExLlamaV2Attention"
        mlp = MagicMock()
        type(mlp).__name__ = "ExLlamaV2MLP"
        modules.extend([attn, mlp])

        norm = MagicMock()
        type(norm).__name__ = "ExLlamaV2RMSNorm"
        head = MagicMock()
        type(head).__name__ = "ExLlamaV2Linear"
        modules.extend([norm, head])

        backend._model = MagicMock()
        backend._model.modules = modules

        post = backend._get_post_layer_modules()
        assert post == [3, 4]  # norm and head indices

    def test_get_total_layers_not_loaded(self):
        from layer_scan.backends.exllamav2 import ExLlamaV2Backend
        backend = ExLlamaV2Backend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.get_total_layers()

    def test_get_tokenizer_not_loaded(self):
        from layer_scan.backends.exllamav2 import ExLlamaV2Backend
        backend = ExLlamaV2Backend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.get_tokenizer()

    def test_load_without_exllamav2_raises(self):
        from layer_scan.backends.exllamav2 import ExLlamaV2Backend
        backend = ExLlamaV2Backend()
        with pytest.raises(ImportError, match="ExLlamaV2 not installed"):
            backend.load("/fake/model")


class TestExLlamaV2TokenizerAdapter:
    """Test the tokenizer adapter."""

    def test_encode_tensor_to_list(self):
        from layer_scan.backends.exllamav2 import _ExLlamaV2TokenizerAdapter

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])

        adapter = _ExLlamaV2TokenizerAdapter(mock_tokenizer)
        result = adapter.encode("hello")
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_encode_list_input(self):
        from layer_scan.backends.exllamav2 import _ExLlamaV2TokenizerAdapter

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [[10, 20, 30]]

        adapter = _ExLlamaV2TokenizerAdapter(mock_tokenizer)
        result = adapter.encode("hello")
        assert result == [10, 20, 30]
