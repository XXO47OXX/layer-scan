from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Ensure a mock 'datasets' module exists so patch("datasets.load_dataset") works
if "datasets" not in sys.modules:
    _mock_datasets = types.ModuleType("datasets")
    _mock_datasets.load_dataset = MagicMock()
    sys.modules["datasets"] = _mock_datasets

from layer_scan.lookup import (
    DATASET_ID,
    _normalize_model_id,
    fetch_results,
    format_lookup_result,
)


class TestNormalizeModelId:
    def test_lowercase(self):
        assert _normalize_model_id("Qwen/Qwen2-7B") == "qwen/qwen2-7b"

    def test_strips_whitespace(self):
        assert _normalize_model_id("  Qwen/Qwen2-7B  ") == "qwen/qwen2-7b"

    def test_underscores_to_hyphens(self):
        assert _normalize_model_id("my_org/my_model") == "my-org/my-model"


class TestFetchResults:
    def test_import_error_without_datasets(self):
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                fetch_results("Qwen/Qwen2-7B")

    def test_exact_match(self):
        mock_row = {
            "model_id": "Qwen/Qwen2-7B",
            "probe": "math",
            "scan_date": "2026-03-15",
            "baseline_score": 5.2,
            "total_layers": 28,
            "top_configs": '[{"i": 12, "j": 20, "score": 6.1}]',
        }
        mock_ds = [mock_row]

        with patch("datasets.load_dataset", return_value=mock_ds) as mock_load:
            result = fetch_results("Qwen/Qwen2-7B", "math")

        mock_load.assert_called_once_with(DATASET_ID, split="train")
        assert result is not None
        assert result["model_id"] == "Qwen/Qwen2-7B"
        assert result["probe"] == "math"

    def test_case_insensitive_match(self):
        mock_row = {
            "model_id": "Qwen/Qwen2-7B",
            "probe": "math",
            "scan_date": "2026-03-15",
        }
        mock_ds = [mock_row]

        with patch("datasets.load_dataset", return_value=mock_ds):
            result = fetch_results("qwen/qwen2-7b", "MATH")

        assert result is not None

    def test_fuzzy_match_short_name(self):
        mock_row = {
            "model_id": "SomeOrg/Qwen2-7B",
            "probe": "math",
            "scan_date": "2026-03-15",
        }
        mock_ds = [mock_row]

        with patch("datasets.load_dataset", return_value=mock_ds):
            result = fetch_results("Qwen2-7B", "math")

        assert result is not None

    def test_no_match_returns_none(self):
        mock_row = {
            "model_id": "Qwen/Qwen2-7B",
            "probe": "math",
        }
        mock_ds = [mock_row]

        with patch("datasets.load_dataset", return_value=mock_ds):
            result = fetch_results("Llama/Llama-3-8B", "math")

        assert result is None

    def test_wrong_probe_returns_none(self):
        mock_row = {
            "model_id": "Qwen/Qwen2-7B",
            "probe": "math",
        }
        mock_ds = [mock_row]

        with patch("datasets.load_dataset", return_value=mock_ds):
            result = fetch_results("Qwen/Qwen2-7B", "json")

        assert result is None

    def test_dataset_load_failure_returns_none(self):
        with patch(
            "datasets.load_dataset",
            side_effect=Exception("Network error"),
        ):
            result = fetch_results("Qwen/Qwen2-7B")

        assert result is None


class TestFormatLookupResult:
    def test_basic_format(self):
        record = {
            "model_id": "Qwen/Qwen2-7B",
            "probe": "math",
            "scan_date": "2026-03-15",
            "layer_scan_version": "0.2.0",
            "total_layers": 28,
            "baseline_score": 5.2,
        }
        text = format_lookup_result(record)
        assert "Qwen/Qwen2-7B" in text
        assert "math" in text
        assert "2026-03-15" in text
        assert "28" in text

    def test_with_top_configs_dict(self):
        record = {
            "model_id": "Qwen/Qwen2-7B",
            "probe": "math",
            "scan_date": "2026-03-15",
            "layer_scan_version": "0.2.0",
            "total_layers": 28,
            "baseline_score": 5.2,
            "top_configs": [
                {"i": 12, "j": 20, "score": 6.1},
                {"i": 14, "j": 22, "score": 5.9},
            ],
        }
        text = format_lookup_result(record)
        assert "#1" in text
        assert "#2" in text
        assert "i=12" in text

    def test_with_top_configs_json_string(self):
        record = {
            "model_id": "test",
            "probe": "math",
            "scan_date": "2026-01-01",
            "layer_scan_version": "0.2.0",
            "total_layers": 10,
            "baseline_score": 3.0,
            "top_configs": '[{"i": 5, "j": 8, "score": 4.0}]',
        }
        text = format_lookup_result(record)
        assert "#1" in text
        assert "i=5" in text

    def test_missing_fields_graceful(self):
        record = {}
        text = format_lookup_result(record)
        assert "unknown" in text
