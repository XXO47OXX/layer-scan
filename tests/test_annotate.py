import json

import pytest

from layer_scan.annotate import (
    count_reasoning_layers,
    generate_annotated_heatmap,
    generate_annotation_text,
    load_layer_scan_results,
    load_neuro_report,
)


@pytest.fixture
def sample_layer_results(tmp_path):
    data = {
        "model": "test-model",
        "probe": "math",
        "total_layers": 32,
        "baseline_score": 5.0,
        "top_configs": [
            {"rank": 1, "i": 10, "j": 20, "score": 7.5, "delta": 2.5, "uncertainty": 0.1},
            {"rank": 2, "i": 12, "j": 22, "score": 7.2, "delta": 2.2, "uncertainty": 0.1},
        ],
        "all_results": [
            {"i": 10, "j": 20, "score": 7.5},
            {"i": 12, "j": 22, "score": 7.2},
            {"i": 5, "j": 15, "score": 6.0},
        ],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def sample_neuro_report(tmp_path):
    labels = {}
    for i in range(32):
        if i < 3:
            labels[str(i)] = "early_processing"
        elif i < 10:
            labels[str(i)] = "syntax"
        elif i < 25:
            labels[str(i)] = "reasoning" if i in [14, 16, 18, 20] else "semantic_processing"
        else:
            labels[str(i)] = "output"
    data = {
        "model": "test-model",
        "total_layers": 32,
        "layer_labels": labels,
        "top_important_layers": [14, 16, 18, 20],
    }
    path = tmp_path / "neuro_report.json"
    path.write_text(json.dumps(data))
    return path


class TestCountReasoningLayers:
    def test_counts_labels(self):
        labels = {"0": "syntax", "1": "reasoning", "2": "reasoning", "3": "output"}
        counts = count_reasoning_layers(1, 3, 4, labels)
        assert counts == {"reasoning": 2}

    def test_empty_range(self):
        labels = {"0": "syntax"}
        counts = count_reasoning_layers(0, 0, 1, labels)
        assert counts == {}


class TestAnnotationText:
    def test_generates_text(self, sample_layer_results, sample_neuro_report):
        lr = json.loads(sample_layer_results.read_text())
        nr = json.loads(sample_neuro_report.read_text())
        text = generate_annotation_text(lr, nr)
        assert "CROSS-TOOL ANNOTATION" in text
        assert "#1" in text

    def test_mentions_reasoning(self, sample_layer_results, sample_neuro_report):
        lr = json.loads(sample_layer_results.read_text())
        nr = json.loads(sample_neuro_report.read_text())
        text = generate_annotation_text(lr, nr)
        assert "reasoning" in text.lower()


class TestAnnotatedHeatmap:
    def test_generates_html(self, sample_layer_results, sample_neuro_report, tmp_path):
        lr = json.loads(sample_layer_results.read_text())
        nr = json.loads(sample_neuro_report.read_text())
        output = tmp_path / "annotated.html"
        result = generate_annotated_heatmap(lr, nr, output)
        assert result.exists()
        content = result.read_text()
        assert "plotly" in content.lower()


class TestLoaders:
    def test_load_layer_results(self, sample_layer_results):
        data = load_layer_scan_results(sample_layer_results)
        assert data["model"] == "test-model"

    def test_load_neuro_report(self, sample_neuro_report):
        data = load_neuro_report(sample_neuro_report)
        assert data["total_layers"] == 32
