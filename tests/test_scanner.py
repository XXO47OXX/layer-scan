from unittest.mock import MagicMock

import torch

from layer_scan.config import ScanConfig
from layer_scan.scanner import _generate_configs, _generate_sparse_configs


class TestGenerateConfigs:
    def test_basic_generation(self):
        configs = _generate_configs(
            total_layers=20,
            min_block_size=7,
            step=1,
            skip_early=0,
            skip_late=0,
        )
        assert len(configs) > 0
        for cfg in configs:
            assert cfg.duplicated_count >= 7
            assert cfg.i >= 0
            assert cfg.j <= 20

    def test_step_reduces_configs(self):
        configs_step1 = _generate_configs(20, 7, step=1, skip_early=0, skip_late=0)
        configs_step2 = _generate_configs(20, 7, step=2, skip_early=0, skip_late=0)
        assert len(configs_step2) < len(configs_step1)

    def test_skip_early_late(self):
        configs = _generate_configs(
            total_layers=20,
            min_block_size=7,
            step=1,
            skip_early=3,
            skip_late=3,
        )
        for cfg in configs:
            assert cfg.i >= 3
            assert cfg.j <= 17

    def test_min_block_size(self):
        configs = _generate_configs(20, min_block_size=10, step=1, skip_early=0, skip_late=0)
        for cfg in configs:
            assert cfg.duplicated_count >= 10

    def test_small_model(self):
        configs = _generate_configs(5, min_block_size=7, step=1, skip_early=0, skip_late=0)
        assert len(configs) == 0

    def test_config_count_formula(self):
        configs = _generate_configs(10, min_block_size=7, step=1, skip_early=0, skip_late=0)
        # i=0: j in [7,8,9,10] → 4 configs
        # i=1: j in [8,9,10] → 3 configs
        # i=2: j in [9,10] → 2 configs
        # i=3: j in [10] → 1 config
        assert len(configs) == 10


class TestGenerateSparseConfigs:
    def test_sparse_fewer_than_dense(self):
        sparse = _generate_sparse_configs(40, 7, sparse_step=4, skip_early=0, skip_late=0)
        dense = _generate_configs(40, 7, step=1, skip_early=0, skip_late=0)
        assert len(sparse) < len(dense)

    def test_sparse_step(self):
        configs = _generate_sparse_configs(40, 7, sparse_step=4, skip_early=0, skip_late=0)
        i_values = {cfg.i for cfg in configs}
        for i in i_values:
            assert i % 4 == 0


class TestScannerIntegration:
    """Integration tests with mock backend."""

    def test_scan_report_structure(self):
        from layer_scan.probes.math_probe import MathProbe
        from layer_scan.scanner import run_scan

        # Create mock backend
        backend = MagicMock()
        backend.get_total_layers.return_value = 16
        backend.get_tokenizer.return_value = self._make_mock_tokenizer()
        backend.forward_with_duplication.return_value = torch.randn(100)

        probe = MathProbe()
        config = ScanConfig(
            model_path="test-model",
            probe_name="math",
            min_block_size=7,
            step=2,
            batch_size=3,
            top_k=3,
        )

        report = run_scan(backend, probe, config)

        assert report.total_layers == 16
        assert report.baseline_score is not None
        assert len(report.top_configs) <= 3
        assert report.heatmap_matrix.shape == (16, 17)
        assert report.total_time_seconds > 0

    @staticmethod
    def _make_mock_tokenizer():
        tokenizer = MagicMock()
        # encode("0") → [48], encode("1") → [49], etc.
        def mock_encode(text, add_special_tokens=True):
            if len(text) == 1 and text.isdigit():
                return [48 + int(text)]
            return [100]

        tokenizer.encode = mock_encode
        return tokenizer
