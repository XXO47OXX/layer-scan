import json
import tempfile
from pathlib import Path

import pytest

from layer_scan.probes.base import ProbeSample
from layer_scan.probes.eq_probe import EqProbe
from layer_scan.probes.json_probe import JsonProbe
from layer_scan.probes.math_probe import MathProbe


class TestProbeSample:
    def test_basic_sample(self):
        sample = ProbeSample(prompt="Test prompt", expected_score=5.0)
        assert sample.prompt == "Test prompt"
        assert sample.expected_score == 5.0
        assert sample.scoring_suffix == ""

    def test_full_text(self):
        sample = ProbeSample(
            prompt="Question: ",
            scoring_suffix="Answer: ",
        )
        assert sample.full_text == "Question: Answer: "

    def test_frozen(self):
        sample = ProbeSample(prompt="test")
        with pytest.raises(AttributeError):
            sample.prompt = "changed"


class TestMathProbe:
    def test_name(self):
        probe = MathProbe()
        assert probe.name == "math"

    def test_description(self):
        probe = MathProbe()
        assert "math" in probe.description.lower()

    def test_samples_count(self):
        probe = MathProbe()
        samples = probe.get_samples()
        assert len(samples) >= 10

    def test_samples_limited(self):
        probe = MathProbe()
        samples = probe.get_samples(count=3)
        assert len(samples) == 3

    def test_samples_have_prompts(self):
        probe = MathProbe()
        for sample in probe.get_samples():
            assert len(sample.prompt) > 0

    def test_samples_have_categories(self):
        probe = MathProbe()
        for sample in probe.get_samples():
            assert sample.metadata is not None
            assert "category" in sample.metadata


class TestJsonProbe:
    def test_name(self):
        probe = JsonProbe()
        assert probe.name == "json"

    def test_samples_count(self):
        probe = JsonProbe()
        samples = probe.get_samples()
        assert len(samples) >= 8

    def test_categories(self):
        probe = JsonProbe()
        categories = {s.metadata["category"] for s in probe.get_samples()}
        assert "grounded_extraction" in categories
        assert "null_handling" in categories


class TestEqProbe:
    def test_name(self):
        probe = EqProbe()
        assert probe.name == "eq"

    def test_samples_count(self):
        probe = EqProbe()
        samples = probe.get_samples()
        assert len(samples) >= 10


class TestCustomProbe:
    def test_load_from_json(self):
        from layer_scan.probes.custom import CustomProbe

        data = {
            "name": "test_probe",
            "description": "Test probe",
            "scoring": "digits",
            "samples": [
                {"prompt": "Rate 0-9: ", "expected_score": 5.0},
                {"prompt": "Rate 0-9: ", "expected_score": 3.0},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            probe = CustomProbe(f.name)
            assert probe.name == "test_probe"
            samples = probe.get_samples()
            assert len(samples) == 2

        Path(f.name).unlink()

    def test_missing_file(self):
        from layer_scan.probes.custom import CustomProbe

        with pytest.raises(FileNotFoundError):
            CustomProbe("/nonexistent/probe.json")

    def test_empty_samples(self):
        from layer_scan.probes.custom import CustomProbe

        data = {
            "name": "empty",
            "samples": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            with pytest.raises(ValueError, match="no samples"):
                CustomProbe(f.name)

        Path(f.name).unlink()
