"""Multi-probe cross analysis — find Pareto-optimal (i, j) configs.

When different probes disagree on the best config, this module finds
configurations that balance performance across multiple tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from layer_scan.backends.base import Backend
    from layer_scan.config import DuplicationConfig, ScanConfig
    from layer_scan.probes.base import Probe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiProbeResult:
    """Result of multi-probe evaluation for a single (i, j) config."""

    config: DuplicationConfig
    probe_scores: dict[str, float]
    probe_log_odds: dict[str, float | None] = field(default_factory=dict)
    probe_accuracies: dict[str, float | None] = field(default_factory=dict)
    normalized_score: float = 0.0
    is_pareto_optimal: bool = False


@dataclass
class MultiProbeReport:
    """Complete multi-probe analysis report."""

    probe_names: list[str]
    all_results: list[MultiProbeResult]
    pareto_configs: list[MultiProbeResult]
    best_balanced: MultiProbeResult | None
    per_probe_best: dict[str, MultiProbeResult]


def run_multi_probe(
    backend: Backend,
    probes: list[Probe],
    scan_config: ScanConfig,
) -> MultiProbeReport:
    """Run multi-probe cross analysis.

    Evaluates each (i, j) config against all probes and identifies
    Pareto-optimal configurations.

    Args:
        backend: Loaded model backend.
        probes: List of probe instances to evaluate.
        scan_config: Scan configuration (step, skip, etc.).

    Returns:
        MultiProbeReport with Pareto frontier and balanced rankings.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    from layer_scan.scanner import (
        _evaluate_config,
        _generate_configs,
        _generate_sparse_configs,
    )
    from layer_scan.scoring import aggregate_scores_full

    total_layers = backend.get_total_layers()
    probe_names = [p.name for p in probes]

    # Prepare samples and token IDs for each probe
    probe_data = {}
    for probe in probes:
        samples = probe.get_samples(count=scan_config.batch_size)
        token_ids, score_values = probe.get_score_token_ids(backend.get_tokenizer())
        probe_data[probe.name] = {
            "samples": samples,
            "token_ids": token_ids,
            "score_values": score_values,
        }

    # Generate configs
    if scan_config.sparse_first:
        configs = _generate_sparse_configs(
            total_layers=total_layers,
            min_block_size=scan_config.min_block_size,
            sparse_step=scan_config.sparse_step,
            skip_early=scan_config.skip_early,
            skip_late=scan_config.skip_late,
        )
    else:
        configs = _generate_configs(
            total_layers=total_layers,
            min_block_size=scan_config.min_block_size,
            step=scan_config.step,
            skip_early=scan_config.skip_early,
            skip_late=scan_config.skip_late,
        )

    total_evals = len(configs) * len(probes)
    logger.info(
        "Multi-probe: %d configs × %d probes = %d evaluations",
        len(configs), len(probes), total_evals,
    )

    # Evaluate all configs against all probes
    all_results: list[MultiProbeResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Multi-probe ({len(configs)} configs × {len(probes)} probes)",
            total=total_evals,
        )

        for config in configs:
            probe_scores: dict[str, float] = {}
            probe_log_odds: dict[str, float | None] = {}
            probe_accuracies: dict[str, float | None] = {}

            for probe in probes:
                data = probe_data[probe.name]
                score_results = _evaluate_config(
                    backend=backend,
                    config=config,
                    samples=data["samples"],
                    score_token_ids=data["token_ids"],
                    score_values=data["score_values"],
                )
                agg = aggregate_scores_full(score_results)
                probe_scores[probe.name] = agg.mean_score
                probe_log_odds[probe.name] = agg.mean_log_odds
                probe_accuracies[probe.name] = agg.accuracy
                progress.update(task, advance=1)

            all_results.append(
                MultiProbeResult(
                    config=config,
                    probe_scores=probe_scores,
                    probe_log_odds=probe_log_odds,
                    probe_accuracies=probe_accuracies,
                )
            )

    # Normalize scores per probe (min-max → [0, 1])
    normalized = _normalize_scores(all_results, probe_names)

    # Find Pareto frontier
    pareto = _find_pareto_frontier(normalized, probe_names)

    # Best balanced = highest geometric mean of normalized scores
    best_balanced = max(pareto, key=lambda r: r.normalized_score) if pareto else None

    # Per-probe best
    per_probe_best: dict[str, MultiProbeResult] = {}
    for name in probe_names:
        per_probe_best[name] = max(
            normalized, key=lambda r, n=name: r.probe_scores[n]
        )

    return MultiProbeReport(
        probe_names=probe_names,
        all_results=normalized,
        pareto_configs=pareto,
        best_balanced=best_balanced,
        per_probe_best=per_probe_best,
    )


def _normalize_scores(
    results: list[MultiProbeResult],
    probe_names: list[str],
) -> list[MultiProbeResult]:
    """Normalize per-probe scores to [0, 1] and compute geometric mean."""
    if not results:
        return results

    # Find min/max per probe
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    for name in probe_names:
        values = [r.probe_scores[name] for r in results]
        mins[name] = min(values)
        maxs[name] = max(values)

    normalized: list[MultiProbeResult] = []
    for result in results:
        norm_scores = {}
        for name in probe_names:
            range_val = maxs[name] - mins[name]
            if range_val > 1e-10:
                norm_scores[name] = (result.probe_scores[name] - mins[name]) / range_val
            else:
                norm_scores[name] = 0.5

        # Geometric mean of normalized scores
        geo_mean = float(
            np.exp(np.mean(np.log(np.clip(list(norm_scores.values()), 1e-10, None))))
        )

        normalized.append(
            MultiProbeResult(
                config=result.config,
                probe_scores=result.probe_scores,
                probe_log_odds=result.probe_log_odds,
                probe_accuracies=result.probe_accuracies,
                normalized_score=geo_mean,
                is_pareto_optimal=False,
            )
        )

    return normalized


def _find_pareto_frontier(
    results: list[MultiProbeResult],
    probe_names: list[str],
) -> list[MultiProbeResult]:
    """Find Pareto-optimal configurations (no config dominates them in all probes)."""
    n = len(results)
    is_pareto = [True] * n

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # Check if j dominates i (j >= i in all probes, j > i in at least one)
            all_geq = all(
                results[j].probe_scores[name] >= results[i].probe_scores[name]
                for name in probe_names
            )
            any_gt = any(
                results[j].probe_scores[name] > results[i].probe_scores[name]
                for name in probe_names
            )
            if all_geq and any_gt:
                is_pareto[i] = False
                break

    pareto = []
    for idx, result in enumerate(results):
        if is_pareto[idx]:
            pareto.append(
                MultiProbeResult(
                    config=result.config,
                    probe_scores=result.probe_scores,
                    probe_log_odds=result.probe_log_odds,
                    probe_accuracies=result.probe_accuracies,
                    normalized_score=result.normalized_score,
                    is_pareto_optimal=True,
                )
            )

    return pareto
