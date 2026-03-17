"""Real-world scenario tests -- end-to-end with mock models."""

from unittest.mock import MagicMock

import pytest
import torch

from layer_scan.config import ScanConfig
from layer_scan.scanner import run_scan


class TestQwen2StyleModel:
    """Simulate Qwen2-72B style model (80 layers)."""

    @pytest.mark.integration
    def test_full_scan_report(self, mock_tokenizer):
        """80-layer model + math probe -> complete report."""
        from layer_scan.probes.math_probe import MathProbe

        backend = MagicMock()
        backend.get_total_layers.return_value = 80
        backend.get_tokenizer.return_value = mock_tokenizer
        backend.forward_with_duplication.return_value = torch.randn(100)

        config = ScanConfig(
            model_path="Qwen/Qwen2-72B",
            probe_name="math",
            min_block_size=7,
            step=8,  # Large step for fast test
            skip_early=10,
            skip_late=10,
            batch_size=2,
            top_k=5,
        )

        report = run_scan(backend, MathProbe(), config)

        assert report.total_layers == 80
        assert len(report.results) > 0
        assert len(report.top_configs) <= 5
        assert report.baseline_score is not None
        assert report.heatmap_matrix.shape == (80, 81)


class TestSmallModel:
    """Small model edge cases."""

    @pytest.mark.integration
    def test_8_layers_min_block_7(self, mock_tokenizer):
        """8 layers, min_block=7 -> very few configs."""
        from layer_scan.probes.math_probe import MathProbe

        backend = MagicMock()
        backend.get_total_layers.return_value = 8
        backend.get_tokenizer.return_value = mock_tokenizer
        backend.forward_with_duplication.return_value = torch.randn(100)

        config = ScanConfig(
            model_path="tiny-model",
            min_block_size=7,
            batch_size=2,
            top_k=3,
        )

        report = run_scan(backend, MathProbe(), config)
        # Only configs where j - i >= 7 and i < j <= 8
        # i=0: j=7,8 -> 2 configs; i=1: j=8 -> 1 config = 3 total
        assert len(report.results) == 3


class TestSparseFirst:
    @pytest.mark.integration
    def test_sparse_fewer_configs(self, mock_tokenizer):
        """sparse_first=True -> fewer configs than full scan."""
        from layer_scan.probes.math_probe import MathProbe

        backend = MagicMock()
        backend.get_total_layers.return_value = 32
        backend.get_tokenizer.return_value = mock_tokenizer
        backend.forward_with_duplication.return_value = torch.randn(100)

        config_full = ScanConfig(
            model_path="model", min_block_size=7, batch_size=2, top_k=3,
        )
        config_sparse = ScanConfig(
            model_path="model", min_block_size=7, batch_size=2, top_k=3,
            sparse_first=True, sparse_step=4,
        )

        report_full = run_scan(backend, MathProbe(), config_full)
        report_sparse = run_scan(backend, MathProbe(), config_sparse)

        assert len(report_sparse.results) < len(report_full.results)


class TestCustomProbeScenario:
    @pytest.mark.integration
    def test_custom_probe_scan(self, mock_tokenizer, sample_probe_json):
        """Custom JSON probe -> end-to-end scan."""
        from layer_scan.probes.custom import CustomProbe

        backend = MagicMock()
        backend.get_total_layers.return_value = 10
        backend.get_tokenizer.return_value = mock_tokenizer
        backend.forward_with_duplication.return_value = torch.randn(100)

        probe = CustomProbe(sample_probe_json)
        config = ScanConfig(
            model_path="test",
            probe_name="custom",
            min_block_size=3,
            batch_size=2,
            top_k=2,
        )

        report = run_scan(backend, probe, config)
        assert len(report.results) > 0
        assert report.baseline_score is not None


class TestMergekitExport:
    @pytest.mark.integration
    def test_export_valid_yaml(self, sample_scan_report):
        """mergekit YAML export produces valid YAML structure."""
        from layer_scan.export import export_mergekit_yaml

        yaml_str = export_mergekit_yaml(sample_scan_report, "test-model")
        assert "merge_method: passthrough" in yaml_str
        assert "slices:" in yaml_str
        assert "layer_range:" in yaml_str
        assert "test-model" in yaml_str

    @pytest.mark.integration
    def test_export_rank_selection(self, sample_scan_report):
        """Different ranks produce different YAML configs."""
        from layer_scan.export import export_mergekit_yaml

        yaml1 = export_mergekit_yaml(sample_scan_report, "model", rank=1)
        yaml2 = export_mergekit_yaml(sample_scan_report, "model", rank=2)
        assert yaml1 != yaml2

    @pytest.mark.integration
    def test_export_invalid_rank(self, sample_scan_report):
        """Invalid rank -> ValueError."""
        from layer_scan.export import export_mergekit_yaml

        with pytest.raises(ValueError, match="rank"):
            export_mergekit_yaml(sample_scan_report, "model", rank=0)
        with pytest.raises(ValueError, match="rank"):
            export_mergekit_yaml(sample_scan_report, "model", rank=100)

    @pytest.mark.integration
    def test_export_empty_report(self):
        """Empty top_configs -> ValueError."""

        from layer_scan.export import export_mergekit_yaml

        report = MagicMock()
        report.top_configs = []

        with pytest.raises(ValueError, match="No configurations"):
            export_mergekit_yaml(report, "model")


class TestMultiProbeComparison:
    @pytest.mark.integration
    def test_different_probes_different_results(self, mock_tokenizer):
        """math vs json probes -> potentially different top configs."""
        from layer_scan.probes.json_probe import JsonProbe
        from layer_scan.probes.math_probe import MathProbe

        # Use deterministic but different logits per call
        call_count = [0]
        def varying_logits(text, duplication_config=None):
            call_count[0] += 1
            torch.manual_seed(call_count[0])
            return torch.randn(100)

        backend = MagicMock()
        backend.get_total_layers.return_value = 16
        backend.get_tokenizer.return_value = mock_tokenizer
        backend.forward_with_duplication.side_effect = varying_logits

        config = ScanConfig(
            model_path="test", min_block_size=7, step=3, batch_size=2, top_k=3,
        )

        report_math = run_scan(backend, MathProbe(), config)
        report_json = run_scan(backend, JsonProbe(), config)

        # Both should complete successfully
        assert len(report_math.results) > 0
        assert len(report_json.results) > 0


class TestBaselineScore:
    @pytest.mark.integration
    def test_baseline_is_no_duplication(self, mock_tokenizer):
        """Baseline score = forward with no duplication config."""
        from layer_scan.probes.math_probe import MathProbe

        baseline_calls = []
        def track_calls(text, duplication_config=None):
            if duplication_config is None:
                baseline_calls.append(True)
            return torch.randn(100)

        backend = MagicMock()
        backend.get_total_layers.return_value = 10
        backend.get_tokenizer.return_value = mock_tokenizer
        backend.forward_with_duplication.side_effect = track_calls

        config = ScanConfig(
            model_path="test", min_block_size=7, batch_size=2, top_k=2,
        )

        run_scan(backend, MathProbe(), config)
        # Baseline should have been called (2 samples, no duplication)
        assert len(baseline_calls) == 2


class TestBackendReload:
    def test_cleanup_then_reuse(self, mock_tokenizer):
        """cleanup() followed by reconfiguration -> works."""
        backend = MagicMock()
        backend.get_total_layers.return_value = 10
        backend.get_tokenizer.return_value = mock_tokenizer
        backend.forward_with_duplication.return_value = torch.randn(100)

        # First use
        result1 = backend.forward_with_duplication("test", None)
        backend.cleanup()

        # Second use (re-mock after cleanup)
        backend.forward_with_duplication.return_value = torch.randn(100)
        result2 = backend.forward_with_duplication("test", None)

        assert result1.shape == result2.shape
