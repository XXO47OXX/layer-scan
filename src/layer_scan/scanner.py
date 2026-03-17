"""Core scanning engine — orchestrates (i, j) configuration search.

The scanner iterates over valid (i, j) pairs, applies layer duplication
to the model's forward pass, evaluates using the selected probe, and
collects scores into a heatmap-ready matrix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from layer_scan.config import DuplicationConfig, ScanConfig, ScanResult
from layer_scan.scoring import ScoreResult, aggregate_scores, aggregate_scores_full

if TYPE_CHECKING:
    from layer_scan.backends.base import Backend
    from layer_scan.probes.base import Probe


@dataclass
class ScanReport:
    """Complete report from a scan run.

    Args:
        scan_config: The configuration used for this scan.
        results: All individual (i, j) results.
        baseline_score: Score of the unmodified model (i=j, no duplication).
        heatmap_matrix: 2D numpy array of scores indexed by (i, j).
        top_configs: Top-k best configurations.
        total_time_seconds: Wall-clock time for the full scan.
    """

    scan_config: ScanConfig
    results: list[ScanResult]
    baseline_score: float
    baseline_uncertainty: float
    heatmap_matrix: np.ndarray
    uncertainty_matrix: np.ndarray
    top_configs: list[ScanResult]
    total_time_seconds: float
    total_layers: int
    metadata: dict[str, object] = field(default_factory=dict)


def _generate_configs(
    total_layers: int,
    min_block_size: int,
    step: int,
    skip_early: int,
    skip_late: int,
) -> list[DuplicationConfig]:
    """Generate all valid (i, j) configurations to scan."""
    configs = []
    i_start = skip_early
    j_end = total_layers - skip_late

    for i in range(i_start, j_end, step):
        for j in range(i + min_block_size, j_end + 1, step):
            configs.append(
                DuplicationConfig(i=i, j=j, total_layers=total_layers)
            )

    return configs


def _generate_sparse_configs(
    total_layers: int,
    min_block_size: int,
    sparse_step: int,
    skip_early: int,
    skip_late: int,
) -> list[DuplicationConfig]:
    """Generate sparse (i, j) configurations for initial exploration."""
    configs = []
    i_start = skip_early
    j_end = total_layers - skip_late

    for i in range(i_start, j_end, sparse_step):
        for j in range(i + min_block_size, j_end + 1, sparse_step):
            configs.append(
                DuplicationConfig(i=i, j=j, total_layers=total_layers)
            )

    return configs


def run_scan(
    backend: Backend,
    probe: Probe,
    scan_config: ScanConfig,
) -> ScanReport:
    """Execute a full scan over (i, j) configurations.

    Args:
        backend: The model backend to use for inference.
        probe: The evaluation probe providing test samples.
        scan_config: Scan configuration parameters.

    Returns:
        ScanReport with all results, heatmap matrix, and top configs.
    """
    total_layers = backend.get_total_layers()
    tokenizer = backend.get_tokenizer()
    samples = probe.get_samples(count=scan_config.batch_size)
    token_ids, score_values = probe.get_score_token_ids(tokenizer)

    # Step 1: Baseline (no duplication)
    baseline_scores = _evaluate_config(
        backend=backend,
        config=None,
        samples=samples,
        score_token_ids=token_ids,
        score_values=score_values,
        tokenizer=tokenizer,
    )
    baseline_score, baseline_uncertainty = aggregate_scores(baseline_scores)

    # Step 2: Generate configs (true two-stage if sparse_first)
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

    # Step 3: Scan configs
    results: list[ScanResult] = []
    heatmap = np.full((total_layers, total_layers + 1), np.nan)
    uncertainty_map = np.full((total_layers, total_layers + 1), np.nan)

    start_time = time.time()

    results = _scan_configs(
        backend=backend,
        configs=configs,
        samples=samples,
        token_ids=token_ids,
        score_values=score_values,
        heatmap=heatmap,
        uncertainty_map=uncertainty_map,
        baseline_score=baseline_score,
        label="Scanning",
        tokenizer=tokenizer,
    )

    # Step 3b: If sparse_first, refine around top results
    if scan_config.sparse_first and results:
        sparse_sorted = sorted(results, key=lambda r: r.score, reverse=True)
        top_sparse = sparse_sorted[:3]

        refine_radius = scan_config.sparse_step - 1
        refine_configs = _generate_refinement_configs(
            top_results=top_sparse,
            total_layers=total_layers,
            min_block_size=scan_config.min_block_size,
            step=scan_config.step,
            radius=refine_radius,
            skip_early=scan_config.skip_early,
            skip_late=scan_config.skip_late,
            already_scanned={(r.config.i, r.config.j) for r in results},
        )

        if refine_configs:
            refine_results = _scan_configs(
                backend=backend,
                configs=refine_configs,
                samples=samples,
                token_ids=token_ids,
                score_values=score_values,
                heatmap=heatmap,
                uncertainty_map=uncertainty_map,
                baseline_score=baseline_score,
                label="Refining",
                tokenizer=tokenizer,
            )
            results.extend(refine_results)

    total_time = time.time() - start_time

    # Step 4: Rank and select top-k
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    top_configs = sorted_results[: scan_config.top_k]

    total_configs = len(results)

    return ScanReport(
        scan_config=scan_config,
        results=results,
        baseline_score=baseline_score,
        baseline_uncertainty=baseline_uncertainty,
        heatmap_matrix=heatmap,
        uncertainty_matrix=uncertainty_map,
        top_configs=top_configs,
        total_time_seconds=total_time,
        total_layers=total_layers,
        metadata={
            "configs_scanned": total_configs,
            "samples_per_config": len(samples),
            "probe": probe.name,
        },
    )


def _evaluate_config(
    backend: Backend,
    config: DuplicationConfig | None,
    samples: list,
    score_token_ids: list[int],
    score_values: list[float],
    tokenizer=None,
) -> list[ScoreResult]:
    """Evaluate a single (i, j) configuration on all probe samples.

    Args:
        backend: Model backend.
        config: Duplication config, or None for baseline (no duplication).
        samples: Probe samples to evaluate.
        score_token_ids: Token IDs for scoring.
        score_values: Numeric values for each score token.
        tokenizer: Optional tokenizer for full-vocab diagnostics.

    Returns:
        List of ScoreResult for each sample.
    """
    # Try batched forward for better GPU utilization
    # Only use batch path for real Backend subclasses (not mocks)
    from layer_scan.backends.base import Backend as _Backend
    from layer_scan.scoring import score_from_logits

    if isinstance(backend, _Backend) and config is None:
        texts = [s.full_text for s in samples]
        all_logits = backend.forward_batch(texts, duplication_config=config)
        results = []
        for logits, sample in zip(all_logits, samples):
            result = score_from_logits(
                logits=logits,
                score_token_ids=score_token_ids,
                score_values=score_values,
                correct_answer=sample.correct_answer,
                tokenizer=tokenizer,
            )
            results.append(result)
        return results

    # Sequential forward (used for duplication configs and non-Backend mocks)
    results = []
    for sample in samples:
        logits = backend.forward_with_duplication(
            text=sample.full_text,
            duplication_config=config,
        )
        result = score_from_logits(
            logits=logits,
            score_token_ids=score_token_ids,
            score_values=score_values,
            correct_answer=sample.correct_answer,
            tokenizer=tokenizer,
        )
        results.append(result)

    return results


def _scan_configs(
    backend: Backend,
    configs: list[DuplicationConfig],
    samples: list,
    token_ids: list[int],
    score_values: list[float],
    heatmap: np.ndarray,
    uncertainty_map: np.ndarray,
    baseline_score: float,
    label: str = "Scanning",
    tokenizer=None,
) -> list[ScanResult]:
    """Scan a batch of configs and populate heatmap/uncertainty matrices.

    Returns list of ScanResult for all configs.
    """
    results: list[ScanResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"{label} {len(configs)} configs (baseline={baseline_score:.3f})",
            total=len(configs),
        )

        for config in configs:
            t0 = time.time()
            score_results = _evaluate_config(
                backend=backend,
                config=config,
                samples=samples,
                score_token_ids=token_ids,
                score_values=score_values,
                tokenizer=tokenizer,
            )
            elapsed = time.time() - t0

            agg = aggregate_scores_full(score_results)

            result = ScanResult(
                config=config,
                score=agg.mean_score,
                uncertainty=agg.mean_uncertainty,
                per_sample_scores=[r.expected_score for r in score_results],
                log_odds=agg.mean_log_odds,
                accuracy=agg.accuracy,
                mean_coverage=agg.mean_coverage,
                metadata={"eval_time_seconds": elapsed},
            )
            results.append(result)
            heatmap[config.i, config.j] = agg.mean_score
            uncertainty_map[config.i, config.j] = agg.mean_uncertainty

            progress.update(task, advance=1)

    return results


def _generate_refinement_configs(
    top_results: list[ScanResult],
    total_layers: int,
    min_block_size: int,
    step: int,
    radius: int,
    skip_early: int,
    skip_late: int,
    already_scanned: set[tuple[int, int]],
) -> list[DuplicationConfig]:
    """Generate fine-grained configs around top sparse scan results.

    For each top config (i, j), generates configs in the neighborhood
    [i-radius..i+radius] × [j-radius..j+radius] with fine step size.
    Skips configs that were already evaluated in the sparse phase.
    """
    i_start = skip_early
    j_end = total_layers - skip_late
    configs: list[DuplicationConfig] = []
    seen: set[tuple[int, int]] = set(already_scanned)

    for result in top_results:
        center_i = result.config.i
        center_j = result.config.j

        for i in range(
            max(i_start, center_i - radius),
            min(j_end, center_i + radius + 1),
            step,
        ):
            for j in range(
                max(i + min_block_size, center_j - radius),
                min(j_end + 1, center_j + radius + 1),
                step,
            ):
                if (i, j) not in seen:
                    seen.add((i, j))
                    configs.append(
                        DuplicationConfig(i=i, j=j, total_layers=total_layers)
                    )

    return configs
