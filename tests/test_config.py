import pytest

from layer_scan.config import DuplicationConfig, ScanConfig, ScanResult


class TestDuplicationConfig:
    def test_valid_config(self):
        cfg = DuplicationConfig(i=10, j=20, total_layers=80)
        assert cfg.i == 10
        assert cfg.j == 20
        assert cfg.total_layers == 80

    def test_duplicated_count(self):
        cfg = DuplicationConfig(i=45, j=52, total_layers=80)
        assert cfg.duplicated_count == 7

    def test_effective_depth(self):
        cfg = DuplicationConfig(i=45, j=52, total_layers=80)
        assert cfg.effective_depth == 87  # 80 + 7

    def test_execution_order(self):
        cfg = DuplicationConfig(i=2, j=5, total_layers=8)
        order = cfg.execution_order()
        # First pass: [0, 1, 2, 3, 4]
        # Jump back: [2, 3, 4, 5, 6, 7]
        assert order == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7]

    def test_no_duplication(self):
        cfg = DuplicationConfig(i=5, j=5, total_layers=10)
        assert cfg.duplicated_count == 0
        assert cfg.effective_depth == 10
        order = cfg.execution_order()
        # [0..4] + [5..9] = [0..9]
        assert order == list(range(10))

    def test_invalid_i_greater_than_j(self):
        with pytest.raises(ValueError, match="Invalid config"):
            DuplicationConfig(i=20, j=10, total_layers=80)

    def test_invalid_j_greater_than_total(self):
        with pytest.raises(ValueError, match="Invalid config"):
            DuplicationConfig(i=10, j=90, total_layers=80)

    def test_invalid_negative_i(self):
        with pytest.raises(ValueError, match="Invalid config"):
            DuplicationConfig(i=-1, j=10, total_layers=80)

    def test_full_model_duplication(self):
        cfg = DuplicationConfig(i=0, j=80, total_layers=80)
        assert cfg.duplicated_count == 80
        assert cfg.effective_depth == 160

    def test_frozen_dataclass(self):
        cfg = DuplicationConfig(i=10, j=20, total_layers=80)
        with pytest.raises(AttributeError):
            cfg.i = 15


class TestScanConfig:
    def test_defaults(self):
        cfg = ScanConfig(model_path="/models/test")
        assert cfg.probe_name == "math"
        assert cfg.min_block_size == 7
        assert cfg.step == 1
        assert cfg.backend == "transformers"

    def test_custom_values(self):
        cfg = ScanConfig(
            model_path="/models/test",
            probe_name="json",
            min_block_size=5,
            step=2,
            skip_early=3,
            skip_late=5,
        )
        assert cfg.probe_name == "json"
        assert cfg.min_block_size == 5
        assert cfg.step == 2


class TestScanResult:
    def test_creation(self):
        cfg = DuplicationConfig(i=10, j=20, total_layers=80)
        result = ScanResult(
            config=cfg,
            score=7.5,
            uncertainty=0.3,
        )
        assert result.score == 7.5
        assert result.config.i == 10

    def test_with_per_sample_scores(self):
        cfg = DuplicationConfig(i=10, j=20, total_layers=80)
        result = ScanResult(
            config=cfg,
            score=7.5,
            uncertainty=0.3,
            per_sample_scores=[7.0, 8.0, 7.5],
        )
        assert len(result.per_sample_scores) == 3
