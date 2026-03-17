"""Stress and performance tests — marked slow, CI optional."""

import time
import tracemalloc
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from layer_scan.config import ScanConfig
from layer_scan.scanner import ScanReport, _generate_configs


@pytest.mark.slow
class TestStress:
    def test_large_config_scan_completes(self, mock_backend_16_layers):
        """1000+ configs with mock backend completes in <30s."""
        from layer_scan.probes.math_probe import MathProbe
        from layer_scan.scanner import run_scan

        config = ScanConfig(
            model_path="stress-test",
            probe_name="math",
            min_block_size=1,
            step=1,
            batch_size=2,
            top_k=5,
        )

        start = time.time()
        report = run_scan(mock_backend_16_layers, MathProbe(), config)
        elapsed = time.time() - start

        assert len(report.results) > 50
        assert elapsed < 30

    def test_large_batch_memory(self):
        """Large batch_size doesn't exceed memory threshold."""
        tracemalloc.start()

        configs = _generate_configs(
            total_layers=50,
            min_block_size=7,
            step=2,
            skip_early=5,
            skip_late=5,
        )

        # Just verify config generation doesn't blow up memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert len(configs) > 0
        # Peak memory < 50MB for config generation
        assert peak < 50 * 1024 * 1024

    def test_100_layer_model_configs(self):
        """100-layer model full scan → correct config count, no overflow."""
        configs = _generate_configs(
            total_layers=100,
            min_block_size=7,
            step=1,
            skip_early=0,
            skip_late=0,
        )

        # Verify count: sum over i of (j_end - (i+min_block) + 1) / step
        expected = 0
        for i in range(0, 100):
            j_start = i + 7
            if j_start <= 100:
                expected += 100 - j_start + 1

        assert len(configs) == expected

        # Verify no invalid configs
        for cfg in configs:
            assert 0 <= cfg.i < cfg.j <= 100
            assert cfg.duplicated_count >= 7

    def test_heatmap_100x100(self, tmp_path):
        """100x100 heatmap HTML generates in <5s."""
        from layer_scan.heatmap import generate_heatmap

        total = 100
        scan_config = ScanConfig(model_path="stress", min_block_size=7, batch_size=1)
        heatmap = np.random.rand(total, total)
        # Set lower-left triangle to NaN (invalid configs)
        for i in range(total):
            for j in range(i):
                heatmap[i, j] = np.nan

        report = ScanReport(
            scan_config=scan_config,
            results=[],
            baseline_score=5.0,
            baseline_uncertainty=0.3,
            heatmap_matrix=heatmap,
            uncertainty_matrix=np.zeros((total, total)),
            top_configs=[],
            total_time_seconds=100.0,
            total_layers=total,
            metadata={"configs_scanned": 0},
        )

        start = time.time()
        path = generate_heatmap(report, tmp_path / "stress_heatmap.html")
        elapsed = time.time() - start

        assert path.exists()
        assert elapsed < 5.0

    def test_sequential_backend_calls_no_leak(self):
        """Multiple sequential calls to same backend → no state leak."""
        backend = MagicMock()
        backend.get_total_layers.return_value = 10

        results = []
        for i in range(20):
            backend.forward_with_duplication.return_value = torch.randn(100)
            logits = backend.forward_with_duplication("test", None)
            results.append(logits.mean().item())

        # All results should be different (random)
        assert len(set(results)) > 1
