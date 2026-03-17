"""Pre-computed scan results lookup from HuggingFace Hub."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

DATASET_ID = "XXO47OXX/layer-scan-results"


def _normalize_model_id(model_id: str) -> str:
    """Normalize model ID for fuzzy matching."""
    return model_id.strip().lower().replace("_", "-")


def fetch_results(model_id: str, probe: str = "math") -> dict | None:
    """Fetch pre-computed scan results from HF Hub dataset.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen2-7B").
        probe: Probe name to look up (default: "math").

    Returns:
        Matching record dict, or None if not found.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
    """
    try:
        import datasets as _datasets_mod
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for lookup.\n"
            "Install it with: pip install layer-scan[lookup]"
        ) from None

    logger.info("Fetching results from %s ...", DATASET_ID)

    try:
        ds = _datasets_mod.load_dataset(DATASET_ID, split="train")
    except Exception as exc:
        logger.warning("Could not load dataset %s: %s", DATASET_ID, exc)
        return None

    needle = _normalize_model_id(model_id)

    for row in ds:
        row_id = _normalize_model_id(row.get("model_id", ""))
        row_probe = row.get("probe", "").strip().lower()
        if row_id == needle and row_probe == probe.strip().lower():
            return dict(row)

    # Fuzzy: try matching the last segment (e.g. "Qwen2-7B")
    short_needle = needle.rsplit("/", 1)[-1]
    for row in ds:
        row_id = _normalize_model_id(row.get("model_id", ""))
        row_short = row_id.rsplit("/", 1)[-1]
        row_probe = row.get("probe", "").strip().lower()
        if row_short == short_needle and row_probe == probe.strip().lower():
            return dict(row)

    return None


def format_lookup_result(record: dict) -> str:
    """Format a lookup result for console display.

    Args:
        record: A record dict from :func:`fetch_results`.

    Returns:
        Formatted multi-line string.
    """
    lines: list[str] = []
    lines.append(f"Model:    {record.get('model_id', 'unknown')}")
    lines.append(f"Probe:    {record.get('probe', 'unknown')}")
    lines.append(f"Scanned:  {record.get('scan_date', 'unknown')}")
    lines.append(f"Version:  {record.get('layer_scan_version', 'unknown')}")
    lines.append(f"Layers:   {record.get('total_layers', '?')}")
    lines.append(f"Baseline: {record.get('baseline_score', '?')}")

    top_configs = record.get("top_configs")
    if top_configs:
        if isinstance(top_configs, str):
            try:
                top_configs = json.loads(top_configs)
            except (json.JSONDecodeError, TypeError):
                top_configs = []

        lines.append("")
        lines.append("Top Configurations:")
        for rank, cfg in enumerate(top_configs[:5], 1):
            if isinstance(cfg, dict):
                i = cfg.get("i", "?")
                j = cfg.get("j", "?")
                score = cfg.get("score", cfg.get("delta", "?"))
                lines.append(f"  #{rank}: i={i}, j={j} -> score={score}")
            else:
                lines.append(f"  #{rank}: {cfg}")

    return "\n".join(lines)
