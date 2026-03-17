"""Cross-tool annotation — overlay neuro-scan layer labels on layer-scan heatmaps.

Reads both a layer-scan results.json and a neuro-scan report.json,
then generates an annotated heatmap showing which layer ranges in
the top configs correspond to reasoning, syntax, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_layer_scan_results(path: str | Path) -> dict:
    """Load layer-scan results JSON."""
    with open(path) as f:
        return json.load(f)


def load_neuro_report(path: str | Path) -> dict:
    """Load neuro-scan report JSON."""
    with open(path) as f:
        return json.load(f)


def count_reasoning_layers(
    config_i: int,
    config_j: int,
    total_layers: int,
    layer_labels: dict[str, str],
) -> dict[str, int]:
    """Count how many layers of each type are in the duplicated range [i, j).

    Args:
        config_i: Start of duplicated block.
        config_j: End of duplicated block (exclusive).
        total_layers: Total layers in the model.
        layer_labels: layer_idx (str) -> label mapping from neuro-scan.

    Returns:
        Dict of label -> count for layers in [i, j).
    """
    counts: dict[str, int] = {}
    for layer_idx in range(config_i, config_j):
        label = layer_labels.get(str(layer_idx), "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def generate_annotation_text(
    layer_results: dict,
    neuro_report: dict,
) -> str:
    """Generate human-readable annotation of top configs.

    Returns:
        Multi-line text explaining each top config in terms of
        neuro-scan layer labels.
    """
    layer_labels = neuro_report.get("layer_labels", {})
    total_layers = neuro_report.get("total_layers", 0)
    top_important = neuro_report.get("top_important_layers", [])

    lines = [
        "CROSS-TOOL ANNOTATION",
        "=" * 60,
        f"Layer-scan model: {layer_results.get('model', '?')}",
        f"Neuro-scan model: {neuro_report.get('model', '?')}",
        "",
        "TOP CONFIGS ANNOTATED:",
        "-" * 60,
    ]

    for cfg in layer_results.get("top_configs", []):
        rank = cfg.get("rank", "?")
        i = cfg.get("i", 0)
        j = cfg.get("j", 0)
        score = cfg.get("score", 0.0)
        delta = cfg.get("delta", 0.0)

        label_counts = count_reasoning_layers(i, j, total_layers, layer_labels)
        reasoning_count = label_counts.get("reasoning", 0)

        # Check overlap with top important layers
        duplicated_range = set(range(i, j))
        important_overlap = [layer for layer in top_important if layer in duplicated_range]

        lines.append(
            f"  #{rank}: i={i}, j={j} "
            f"(score={score:.4f}, delta={delta:+.4f})"
        )

        # Layer composition
        if label_counts:
            composition = ", ".join(
                f"{count} {label}" for label, count in
                sorted(label_counts.items(), key=lambda x: -x[1])
            )
            lines.append(f"    Composition: {composition}")

        if important_overlap:
            lines.append(
                f"    Contains {len(important_overlap)} top-important layers: "
                f"{important_overlap}"
            )

        # Explanation
        if reasoning_count > 0:
            lines.append(
                f"    → Duplicates {reasoning_count} reasoning layer(s) "
                f"(core computation enhancement)"
            )

        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_annotated_heatmap(
    layer_results: dict,
    neuro_report: dict,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Generate a heatmap with neuro-scan layer labels overlaid.

    Creates an HTML file with:
    1. The layer-scan heatmap (reconstructed from top_configs)
    2. A color band along axes showing layer function
    3. Annotations for reasoning layers

    Args:
        layer_results: Loaded layer-scan results.json.
        neuro_report: Loaded neuro-scan report.json.
        output_path: Path for output HTML.
        title: Custom title.

    Returns:
        Path to generated HTML file.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layer_labels = neuro_report.get("layer_labels", {})
    total_layers = neuro_report.get("total_layers", 0)
    top_important = neuro_report.get("top_important_layers", [])
    baseline = layer_results.get("baseline_score", 0.0)

    if title is None:
        model_name = Path(layer_results.get("model", "unknown")).name
        title = f"Annotated Heatmap — {model_name}"

    # Label color mapping (matches neuro-scan's labeler.py)
    label_colors = {
        "early_processing": "#636EFA",
        "syntax": "#00CC96",
        "reasoning": "#EF553B",
        "semantic_processing": "#AB63FA",
        "formatting": "#FFA15A",
        "output": "#19D3F3",
    }

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.08, 0.92],
        vertical_spacing=0.02,
        subplot_titles=["Layer Function", ""],
    )

    # Top row: layer function color band
    layer_colors = []
    layer_texts = []
    for i in range(total_layers):
        label = layer_labels.get(str(i), "unknown")
        layer_colors.append(label_colors.get(label, "#B6B6B6"))
        layer_texts.append(f"Layer {i}: {label}")

    fig.add_trace(
        go.Bar(
            x=list(range(total_layers)),
            y=[1] * total_layers,
            marker_color=layer_colors,
            text=layer_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Mark reasoning layers
    for layer_idx in top_important:
        fig.add_trace(
            go.Scatter(
                x=[layer_idx],
                y=[1.2],
                mode="markers",
                marker={"size": 8, "color": "red", "symbol": "star"},
                showlegend=False,
                hovertemplate=f"Top important: layer {layer_idx}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Bottom row: scatter of top configs
    top_configs = layer_results.get("top_configs", [])
    if top_configs:
        # Also plot all results as background
        all_results = layer_results.get("all_results", [])
        if all_results:
            all_i = [r["i"] for r in all_results]
            all_j = [r["j"] for r in all_results]
            all_scores = [r.get("score", 0.0) for r in all_results]

            fig.add_trace(
                go.Scatter(
                    x=all_j,
                    y=all_i,
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": [s - baseline for s in all_scores],
                        "colorscale": "RdYlGn",
                        "showscale": True,
                        "colorbar": {"title": "Score Delta"},
                    },
                    hovertemplate=(
                        "i=%{y}, j=%{x}"
                        "<br>Score delta: %{marker.color:.4f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=2, col=1,
            )

        # Top configs with stars
        for rank, cfg in enumerate(top_configs, 1):
            label_counts = count_reasoning_layers(
                cfg["i"], cfg["j"], total_layers, layer_labels
            )
            reasoning = label_counts.get("reasoning", 0)
            annotation = f"#{rank} ({reasoning}R)"

            fig.add_trace(
                go.Scatter(
                    x=[cfg["j"]],
                    y=[cfg["i"]],
                    mode="markers+text",
                    marker={"size": 14, "color": "gold", "symbol": "star"},
                    text=[annotation],
                    textposition="top center",
                    name=f"Top {rank}: i={cfg['i']}, j={cfg['j']} ({reasoning} reasoning)",
                    showlegend=True,
                ),
                row=2, col=1,
            )

    fig.update_layout(
        title=title,
        width=1000,
        height=900,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Layer Index", row=2, col=1)
    fig.update_yaxes(title_text="i (start)", row=2, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("Annotated heatmap saved to %s", output_path)

    return output_path
