"""CLI black-box tests — test CLI behavior without caring about internals."""

import json

from typer.testing import CliRunner

from layer_scan.cli import app

runner = CliRunner()


class TestScanErrors:
    """Scan command error handling (no model needed)."""

    def test_scan_missing_model(self):
        """scan without --model → non-zero exit."""
        result = runner.invoke(app, ["scan"])
        assert result.exit_code != 0

    def test_scan_unknown_probe(self):
        """scan --probe unknown_name → error with available probes."""
        result = runner.invoke(app, ["scan", "--model", "dummy", "--probe", "nonexistent"])
        assert result.exit_code != 0
        assert "Unknown probe" in result.output or "nonexistent" in result.output

    def test_scan_unknown_backend(self):
        """scan --backend unknown → error with available backends."""
        result = runner.invoke(app, ["scan", "--model", "dummy", "--backend", "fake_backend"])
        assert result.exit_code != 0
        assert "Unknown backend" in result.output or "fake_backend" in result.output

    def test_scan_custom_probe_missing_path(self):
        """scan --probe custom without --custom-probe → error."""
        result = runner.invoke(app, ["scan", "--model", "dummy", "--probe", "custom"])
        assert result.exit_code != 0
        assert "custom-probe" in result.output.lower() or "required" in result.output.lower()

    def test_scan_custom_probe_file_not_found(self):
        """scan --custom-probe nonexistent.json → FileNotFoundError."""
        result = runner.invoke(
            app,
            ["scan", "--model", "dummy", "--probe", "custom",
             "--custom-probe", "/no/such/file.json"],
        )
        assert result.exit_code != 0

    def test_scan_custom_probe_valid_json(self, tmp_path):
        """scan --custom-probe with valid JSON loads without probe error.

        Will still fail at model loading, but should not fail at probe loading.
        """
        probe = {
            "name": "test",
            "description": "test probe",
            "scoring": "digits",
            "samples": [{"prompt": "Rate: ", "expected_score": 5.0}],
        }
        path = tmp_path / "probe.json"
        path.write_text(json.dumps(probe))
        result = runner.invoke(
            app,
            ["scan", "--model", "dummy", "--probe", "custom", "--custom-probe", str(path)],
        )
        # Should fail at model loading, not probe loading
        # So the error should NOT mention "probe"
        output_lower = result.output.lower()
        assert "unknown probe" not in output_lower
        assert "custom-probe required" not in output_lower


class TestGpuSplitParsing:
    """Test --gpu-split option parsing."""

    def test_gpu_split_invalid_backend_but_valid_parse(self):
        """--gpu-split with non-exllamav2 backend — backend error, not parse error."""
        result = runner.invoke(
            app,
            ["scan", "--model", "dummy", "--backend", "bad", "--gpu-split", "22000,22000"],
        )
        assert result.exit_code != 0
        # Should fail on backend, not gpu_split parsing
        assert "Unknown backend" in result.output or "bad" in result.output


class TestProbesCommand:
    def test_probes_lists_all(self):
        """probes command lists math, eq, json, custom."""
        result = runner.invoke(app, ["probes"])
        assert result.exit_code == 0
        for name in ["math", "eq", "json", "custom"]:
            assert name in result.output

    def test_probes_shows_sample_counts(self):
        """probes command shows sample counts for each probe."""
        result = runner.invoke(app, ["probes"])
        assert result.exit_code == 0
        assert "Samples:" in result.output


class TestVersionCommand:
    def test_version_output(self):
        """version command outputs version number."""
        from layer_scan import __version__
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_contains_prefix(self):
        """version output contains 'layer-scan'."""
        result = runner.invoke(app, ["version"])
        assert "layer-scan" in result.output


class TestSparseFirst:
    def test_sparse_first_flag_accepted(self):
        """--sparse-first is a valid flag (fails on model, not flag)."""
        result = runner.invoke(
            app,
            ["scan", "--model", "dummy", "--sparse-first"],
        )
        # Will fail at model loading or backend, not at flag parsing
        assert "no such option" not in result.output.lower()


class TestExportMergekit:
    def test_export_mergekit_flag_accepted(self):
        """--export-mergekit is a valid option (fails on model, not option)."""
        result = runner.invoke(
            app,
            ["scan", "--model", "dummy", "--export-mergekit", "/tmp/test.yaml"],
        )
        assert "no such option" not in result.output.lower()
