import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from layer_scan.config import DuplicationConfig, ScanConfig, ScanResult
from layer_scan.scanner import ScanReport


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer mapping digits 0-9 to token IDs 48-57."""
    tokenizer = MagicMock()
    def mock_encode(text, add_special_tokens=True):
        if len(text) == 1 and text.isdigit():
            return [48 + int(text)]
        return [100]
    tokenizer.encode = mock_encode
    return tokenizer


@pytest.fixture
def mock_backend(mock_tokenizer):
    """Mock backend returning random logits, records layer execution."""
    backend = MagicMock()
    backend.get_total_layers.return_value = 32
    backend.get_tokenizer.return_value = mock_tokenizer
    # Return logits with vocab_size=100
    backend.forward_with_duplication.return_value = torch.randn(100)
    return backend


@pytest.fixture
def mock_backend_16_layers(mock_tokenizer):
    """Mock backend with 16 layers for smaller test configs."""
    backend = MagicMock()
    backend.get_total_layers.return_value = 16
    backend.get_tokenizer.return_value = mock_tokenizer
    backend.forward_with_duplication.return_value = torch.randn(100)
    return backend


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test results."""
    output = tmp_path / "results"
    output.mkdir()
    return output


@pytest.fixture
def sample_probe_json(tmp_path):
    """Temporary custom probe JSON file."""
    probe_data = {
        "name": "test_probe",
        "description": "A test probe for unit tests",
        "scoring": "digits",
        "samples": [
            {
                "prompt": "What is 2+2? Answer with a single digit: ",
                "expected_score": 4.0,
                "metadata": {"category": "arithmetic"},
            },
            {
                "prompt": "What is 3+3? Answer with a single digit: ",
                "expected_score": 6.0,
                "metadata": {"category": "arithmetic"},
            },
        ],
    }
    path = tmp_path / "test_probe.json"
    path.write_text(json.dumps(probe_data))
    return path


@pytest.fixture
def sample_scan_report():
    """Pre-built ScanReport for testing output functions."""
    scan_config = ScanConfig(
        model_path="test-model",
        probe_name="math",
        min_block_size=7,
        step=1,
        batch_size=16,
        top_k=3,
    )

    total_layers = 20
    results = []
    heatmap = np.full((total_layers, total_layers), np.nan)
    uncertainty = np.full((total_layers, total_layers), np.nan)

    # Create some results
    configs_data = [
        (5, 15, 7.2),
        (8, 18, 7.8),
        (3, 12, 6.5),
        (10, 19, 7.5),
        (6, 16, 7.0),
    ]
    for i, j, score in configs_data:
        cfg = DuplicationConfig(i=i, j=j, total_layers=total_layers)
        result = ScanResult(
            config=cfg,
            score=score,
            uncertainty=0.5,
            per_sample_scores=[score - 0.1, score, score + 0.1],
        )
        results.append(result)
        heatmap[i, j] = score
        uncertainty[i, j] = 0.5

    top_configs = sorted(results, key=lambda r: r.score, reverse=True)[:3]

    return ScanReport(
        scan_config=scan_config,
        results=results,
        baseline_score=6.0,
        baseline_uncertainty=0.4,
        heatmap_matrix=heatmap,
        uncertainty_matrix=uncertainty,
        top_configs=top_configs,
        total_time_seconds=10.0,
        total_layers=total_layers,
        metadata={"configs_scanned": 5, "samples_per_config": 3, "probe": "math"},
    )
