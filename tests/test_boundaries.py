"""Boundary and edge-case tests."""

import math

import pytest
import torch

from layer_scan.config import DuplicationConfig, ScanConfig
from layer_scan.scoring import score_from_logits


class TestDuplicationConfigBoundaries:
    def test_zero_duplication(self):
        """i=0, j=0 → no layers duplicated."""
        cfg = DuplicationConfig(i=0, j=0, total_layers=10)
        assert cfg.duplicated_count == 0
        assert cfg.effective_depth == 10
        assert cfg.execution_order() == list(range(10))

    def test_full_model_duplication(self):
        """i=0, j=total → entire model duplicated."""
        cfg = DuplicationConfig(i=0, j=10, total_layers=10)
        assert cfg.duplicated_count == 10
        assert cfg.effective_depth == 20
        order = cfg.execution_order()
        assert order == list(range(10)) + list(range(10))

    def test_single_layer_model(self):
        """total_layers=1 → valid configs possible."""
        cfg = DuplicationConfig(i=0, j=1, total_layers=1)
        assert cfg.duplicated_count == 1
        assert cfg.effective_depth == 2
        assert cfg.execution_order() == [0, 0]

    def test_single_layer_no_dup(self):
        """total_layers=1, i=j=0 → no dup."""
        cfg = DuplicationConfig(i=0, j=0, total_layers=1)
        assert cfg.execution_order() == [0]

    def test_single_layer_end_no_dup(self):
        """total_layers=1, i=j=1 → no dup."""
        cfg = DuplicationConfig(i=1, j=1, total_layers=1)
        assert cfg.execution_order() == [0]


class TestScoringBoundaries:
    def test_uniform_logits(self):
        """All-zero logits → uniform distribution → score ≈ 4.5."""
        logits = torch.zeros(100)
        token_ids = list(range(10))
        result = score_from_logits(logits, token_ids)
        assert abs(result.expected_score - 4.5) < 0.01

    def test_logits_with_nan(self):
        """NaN in logits → NaN in score (propagates)."""
        logits = torch.zeros(100)
        logits[0] = float("nan")
        token_ids = list(range(10))
        result = score_from_logits(logits, token_ids)
        assert math.isnan(result.expected_score)

    def test_logits_with_positive_inf(self):
        """+Inf at one position → all probability on that token."""
        logits = torch.zeros(100)
        logits[5] = float("inf")
        token_ids = list(range(10))
        result = score_from_logits(logits, token_ids)
        # Token 5 has value 5 with probability ~1.0
        assert abs(result.expected_score - 5.0) < 0.01

    def test_logits_with_negative_inf(self):
        """-Inf at one position → zero probability on that token."""
        logits = torch.zeros(100)
        logits[3] = float("-inf")
        token_ids = list(range(10))
        result = score_from_logits(logits, token_ids)
        # Token 3 has probability ~0, rest uniform
        # Expected = sum(i * 1/9 for i in range(10) if i != 3)
        expected = sum(i / 9 for i in range(10) if i != 3)
        assert abs(result.expected_score - expected) < 0.1

    def test_empty_score_token_ids(self):
        """Empty token ID list → error (softmax over empty)."""
        logits = torch.zeros(100)
        with pytest.raises((ValueError, RuntimeError)):
            score_from_logits(logits, [])

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_float_precision(self, dtype):
        """float16 vs float32 → results should be close."""
        torch.manual_seed(42)
        logits = torch.randn(100, dtype=dtype)
        token_ids = list(range(10))
        result = score_from_logits(logits, token_ids)
        assert 0.0 <= result.expected_score <= 9.0

    def test_very_large_logits(self):
        """Very large logits → doesn't overflow."""
        logits = torch.full((100,), 1000.0)
        logits[7] = 1001.0  # Slightly higher
        token_ids = list(range(10))
        result = score_from_logits(logits, token_ids)
        # Should be close to 7 since token 7 dominates
        assert not math.isnan(result.expected_score)


class TestScanConfigValidation:
    def test_batch_size_zero(self):
        with pytest.raises(ValueError, match="batch_size"):
            ScanConfig(model_path="m", batch_size=0)

    def test_batch_size_negative(self):
        with pytest.raises(ValueError, match="batch_size"):
            ScanConfig(model_path="m", batch_size=-1)

    def test_min_block_size_zero(self):
        with pytest.raises(ValueError, match="min_block_size"):
            ScanConfig(model_path="m", min_block_size=0)

    def test_step_zero(self):
        with pytest.raises(ValueError, match="step"):
            ScanConfig(model_path="m", step=0)

    def test_top_k_zero(self):
        with pytest.raises(ValueError, match="top_k"):
            ScanConfig(model_path="m", top_k=0)

    def test_valid_minimum_values(self):
        """All minimum valid values (1) should work."""
        cfg = ScanConfig(
            model_path="m",
            batch_size=1,
            min_block_size=1,
            step=1,
            top_k=1,
        )
        assert cfg.batch_size == 1


class TestGenerateConfigsBoundaries:
    def test_skip_early_late_exceeds_total(self):
        """skip_early + skip_late >= total → empty config list."""
        from layer_scan.scanner import _generate_configs
        configs = _generate_configs(
            total_layers=20,
            min_block_size=7,
            step=1,
            skip_early=10,
            skip_late=10,
        )
        assert len(configs) == 0

    def test_min_block_exceeds_available(self):
        """min_block > available range → empty config list."""
        from layer_scan.scanner import _generate_configs
        configs = _generate_configs(
            total_layers=10,
            min_block_size=11,
            step=1,
            skip_early=0,
            skip_late=0,
        )
        assert len(configs) == 0
