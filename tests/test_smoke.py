import pytest


@pytest.mark.smoke
class TestSmoke:
    def test_import(self):
        """Package imports without error."""
        import layer_scan
        assert layer_scan is not None

    def test_version(self):
        """Version string is valid semver-like format."""
        from layer_scan import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)

    def test_cli_help(self):
        """CLI --help returns 0 and contains 'Usage'."""
        from typer.testing import CliRunner

        from layer_scan.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Typer uses "Usage" in help output
        assert "Usage" in result.output or "usage" in result.output.lower()

    def test_cli_probes(self):
        """'probes' command lists the 3 built-in probes."""
        from typer.testing import CliRunner

        from layer_scan.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["probes"])
        assert result.exit_code == 0
        assert "math" in result.output
        assert "eq" in result.output
        assert "json" in result.output

    def test_cli_version(self):
        """'version' command outputs version string."""
        from typer.testing import CliRunner

        from layer_scan.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        from layer_scan import __version__
        assert __version__ in result.output
