import json

import numpy as np

from layer_scan.config import ScanConfig
from layer_scan.heatmap import (
    _build_hover_text,
    generate_heatmap,
    generate_summary_text,
    save_results_json,
)
from layer_scan.scanner import ScanReport


class TestGenerateHeatmap:
    def test_output_file_exists(self, sample_scan_report, tmp_path):
        path = generate_heatmap(sample_scan_report, tmp_path / "test.html")
        assert path.exists()
        assert path.suffix == ".html"

    def test_output_contains_plotly(self, sample_scan_report, tmp_path):
        path = generate_heatmap(sample_scan_report, tmp_path / "test.html")
        content = path.read_text()
        assert "plotly" in content.lower()

    def test_custom_title(self, sample_scan_report, tmp_path):
        title = "My Custom Heatmap Title"
        path = generate_heatmap(sample_scan_report, tmp_path / "test.html", title=title)
        content = path.read_text()
        assert title in content

    def test_top_configs_marked(self, sample_scan_report, tmp_path):
        path = generate_heatmap(sample_scan_report, tmp_path / "test.html")
        content = path.read_text()
        # Should contain references to top configs
        assert "Top" in content or "#1" in content

    def test_output_dir_created(self, sample_scan_report, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        path = generate_heatmap(sample_scan_report, nested / "test.html")
        assert path.exists()

    def test_1x1_matrix(self, tmp_path):
        config = ScanConfig(model_path="tiny", batch_size=1)
        report = ScanReport(
            scan_config=config,
            results=[],
            baseline_score=5.0,
            baseline_uncertainty=0.3,
            heatmap_matrix=np.array([[5.0]]),
            uncertainty_matrix=np.array([[0.1]]),
            top_configs=[],
            total_time_seconds=1.0,
            total_layers=1,
            metadata={"configs_scanned": 0},
        )
        path = generate_heatmap(report, tmp_path / "tiny.html")
        assert path.exists()

    def test_all_nan_matrix(self, tmp_path):
        config = ScanConfig(model_path="nan", batch_size=1)
        size = 5
        report = ScanReport(
            scan_config=config,
            results=[],
            baseline_score=5.0,
            baseline_uncertainty=0.3,
            heatmap_matrix=np.full((size, size), np.nan),
            uncertainty_matrix=np.full((size, size), np.nan),
            top_configs=[],
            total_time_seconds=1.0,
            total_layers=size,
            metadata={"configs_scanned": 0},
        )
        path = generate_heatmap(report, tmp_path / "nan.html")
        assert path.exists()


class TestGenerateSummaryText:
    def test_contains_baseline(self, sample_scan_report):
        text = generate_summary_text(sample_scan_report)
        assert "6.0000" in text or "6.00" in text

    def test_contains_top_configs(self, sample_scan_report):
        text = generate_summary_text(sample_scan_report)
        assert "#1" in text
        assert "#2" in text

    def test_empty_results(self):
        config = ScanConfig(model_path="empty", batch_size=1)
        report = ScanReport(
            scan_config=config,
            results=[],
            baseline_score=5.0,
            baseline_uncertainty=0.3,
            heatmap_matrix=np.zeros((5, 5)),
            uncertainty_matrix=np.zeros((5, 5)),
            top_configs=[],
            total_time_seconds=1.0,
            total_layers=5,
            metadata={"configs_scanned": 0},
        )
        text = generate_summary_text(report)
        assert "LAYER-SCAN" in text
        assert "5.0000" in text or "5.00" in text


class TestSaveResultsJson:
    def test_valid_json(self, sample_scan_report, tmp_path):
        path = save_results_json(sample_scan_report, tmp_path / "results.json")
        data = json.loads(path.read_text())
        assert "model" in data
        assert "baseline_score" in data
        assert "top_configs" in data
        assert "all_results" in data

    def test_top_configs_ranked(self, sample_scan_report, tmp_path):
        path = save_results_json(sample_scan_report, tmp_path / "results.json")
        data = json.loads(path.read_text())
        scores = [c["score"] for c in data["top_configs"]]
        assert scores == sorted(scores, reverse=True)

    def test_required_fields(self, sample_scan_report, tmp_path):
        path = save_results_json(sample_scan_report, tmp_path / "results.json")
        data = json.loads(path.read_text())

        required = ["model", "probe", "total_layers", "baseline_score",
                     "baseline_uncertainty", "scan_time_seconds", "top_configs", "all_results"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_output_dir_created(self, sample_scan_report, tmp_path):
        path = save_results_json(sample_scan_report, tmp_path / "new_dir" / "results.json")
        assert path.exists()


class TestBuildHoverText:
    def test_nan_shows_not_scanned(self):
        matrix = np.array([[np.nan, 5.0], [np.nan, np.nan]])
        delta = np.array([[np.nan, 1.0], [np.nan, np.nan]])
        hover = _build_hover_text(matrix, delta, 4.0, 2)
        assert hover[0][0] == "Not scanned"
        assert "5.0000" in hover[0][1]

    def test_valid_values_formatted(self):
        matrix = np.array([[6.0]])
        delta = np.array([[1.0]])
        hover = _build_hover_text(matrix, delta, 5.0, 1)
        assert "6.0000" in hover[0][0]
        assert "+1.0000" in hover[0][0]
