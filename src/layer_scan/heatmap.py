"""Heatmap visualization for layer-scan results.

Generates interactive HTML heatmaps using Plotly, showing the
score landscape across (i, j) configurations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from layer_scan.scanner import ScanReport

logger = logging.getLogger(__name__)


def generate_heatmap(
    report: ScanReport,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Generate an interactive HTML heatmap from scan results.

    Args:
        report: The scan report containing the heatmap matrix.
        output_path: Path for the output HTML file.
        title: Custom title. Defaults to auto-generated.

    Returns:
        Path to the generated HTML file.
    """
    import plotly.graph_objects as go

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = report.heatmap_matrix
    total_layers = report.total_layers
    baseline = report.baseline_score

    if title is None:
        model_name = Path(report.scan_config.model_path).name
        probe_name = report.scan_config.probe_name
        title = f"Layer Duplication Heatmap — {model_name} / {probe_name}"

    # Compute delta from baseline
    delta_matrix = matrix - baseline

    # Create hover text with detailed info
    hover_text = _build_hover_text(matrix, delta_matrix, baseline, total_layers)

    # Main heatmap
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=delta_matrix,
            x=list(range(total_layers)),
            y=list(range(total_layers)),
            colorscale="RdYlGn",
            zmid=0,
            colorbar={"title": "Score Δ vs Baseline"},
            text=hover_text,
            hovertemplate=(
                "i=%{y}, j=%{x}<br>"
                "Score: %{text}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Mark top-k configurations
    for rank, result in enumerate(report.top_configs, 1):
        fig.add_trace(
            go.Scatter(
                x=[result.config.j],
                y=[result.config.i],
                mode="markers+text",
                marker={"size": 14, "color": "gold", "symbol": "star"},
                text=[f"#{rank}"],
                textposition="top center",
                name=f"Top {rank}: ({result.config.i},{result.config.j}) = {result.score:.3f}",
                hovertemplate=(
                    f"<b>Top {rank}</b><br>"
                    f"i={result.config.i}, j={result.config.j}<br>"
                    f"Score: {result.score:.3f} (Δ={result.score - baseline:+.3f})<br>"
                    f"Block size: {result.config.duplicated_count}<br>"
                    f"Effective depth: {result.config.effective_depth}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="j (end of duplicated block, exclusive)",
        yaxis_title="i (start of duplicated block, inclusive)",
        width=900,
        height=800,
        xaxis={"dtick": max(1, total_layers // 20)},
        yaxis={"dtick": max(1, total_layers // 20), "autorange": "reversed"},
    )

    # Add annotation for baseline
    fig.add_annotation(
        text=f"Baseline: {baseline:.3f}",
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font={"size": 12, "color": "gray"},
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("Heatmap saved to %s", output_path)

    return output_path


def generate_summary_text(report: ScanReport) -> str:
    """Generate a text summary of scan results.

    Args:
        report: The scan report.

    Returns:
        Formatted text summary.
    """
    lines = [
        "=" * 60,
        "LAYER-SCAN RESULTS",
        "=" * 60,
        f"Model: {report.scan_config.model_path}",
        f"Probe: {report.scan_config.probe_name}",
        f"Total layers: {report.total_layers}",
        f"Configs scanned: {report.metadata.get('configs_scanned', '?')}",
        f"Scan time: {report.total_time_seconds:.1f}s",
        "",
        f"Baseline score: {report.baseline_score:.4f} "
        f"(±{report.baseline_uncertainty:.4f})",
        "",
        "TOP CONFIGURATIONS:",
        "-" * 60,
    ]

    for rank, result in enumerate(report.top_configs, 1):
        delta = result.score - report.baseline_score
        line = (
            f"  #{rank}: i={result.config.i:3d}, j={result.config.j:3d} "
            f"(block={result.config.duplicated_count:2d} layers) "
            f"→ score={result.score:.4f} (Δ={delta:+.4f})"
        )
        if result.log_odds is not None:
            line += f" | log-odds={result.log_odds:+.3f}"
        if result.accuracy is not None:
            line += f" | acc={result.accuracy:.1%}"
        if result.mean_coverage is not None:
            line += f" | cov={result.mean_coverage:.3f}"
        lines.append(line)

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def save_results_json(report: ScanReport, output_path: str | Path) -> Path:
    """Save scan results as JSON for programmatic consumption.

    Args:
        report: The scan report.
        output_path: Path for the output JSON file.

    Returns:
        Path to the generated JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model": report.scan_config.model_path,
        "probe": report.scan_config.probe_name,
        "total_layers": report.total_layers,
        "baseline_score": report.baseline_score,
        "baseline_uncertainty": report.baseline_uncertainty,
        "scan_time_seconds": report.total_time_seconds,
        "configs_scanned": report.metadata.get("configs_scanned"),
        "top_configs": [
            {
                "rank": rank,
                "i": r.config.i,
                "j": r.config.j,
                "block_size": r.config.duplicated_count,
                "effective_depth": r.config.effective_depth,
                "score": r.score,
                "delta": r.score - report.baseline_score,
                "uncertainty": r.uncertainty,
                "log_odds": r.log_odds,
                "accuracy": r.accuracy,
                "mean_coverage": r.mean_coverage,
            }
            for rank, r in enumerate(report.top_configs, 1)
        ],
        "all_results": [
            {
                "i": r.config.i,
                "j": r.config.j,
                "score": r.score,
                "uncertainty": r.uncertainty,
            }
            for r in report.results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Results JSON saved to %s", output_path)
    return output_path


def _build_hover_text(
    matrix: np.ndarray,
    delta_matrix: np.ndarray,
    baseline: float,
    total_layers: int,
) -> list[list[str]]:
    """Build hover text matrix for the heatmap."""
    hover = []
    for i in range(total_layers):
        row = []
        for j in range(total_layers):
            val = matrix[i, j]
            if np.isnan(val):
                row.append("Not scanned")
            else:
                delta = delta_matrix[i, j]
                row.append(f"{val:.4f} (Δ={delta:+.4f})")
        hover.append(row)
    return hover
