"""Model configuration and layer duplication parameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DuplicationConfig:
    """A single layer duplication configuration (i, j).

    Execution order: [0...j-1, i...N-1]
    where layers [i...j-1] are executed twice.

    Args:
        i: Start of duplicated block (inclusive). Must be <= j.
        j: End of duplicated block (exclusive). The forward pass runs
           layers [0..j-1] then jumps back to layer i and continues [i..N-1].
        total_layers: Total number of layers in the base model.
    """

    i: int
    j: int
    total_layers: int

    def __post_init__(self) -> None:
        if not (0 <= self.i <= self.j <= self.total_layers):
            raise ValueError(
                f"Invalid config: need 0 <= i({self.i}) <= j({self.j}) "
                f"<= total_layers({self.total_layers})"
            )

    @property
    def duplicated_count(self) -> int:
        """Number of layers that are executed twice."""
        return self.j - self.i

    @property
    def effective_depth(self) -> int:
        """Total effective layer count after duplication."""
        return self.total_layers + self.duplicated_count

    def execution_order(self) -> list[int]:
        """Return the layer indices in execution order."""
        first_pass = list(range(self.j))
        second_pass = list(range(self.i, self.total_layers))
        return first_pass + second_pass


@dataclass(frozen=True)
class ScanConfig:
    """Configuration for a full scan run.

    Args:
        model_path: Path to the model directory or HuggingFace model ID.
        probe_name: Name of the evaluation probe to use.
        min_block_size: Minimum duplicated block size (layers). Default 7
            based on RYS finding that <7 layers is ineffective.
        step: Step size for scanning i and j. Default 1 for exhaustive.
        skip_early: Number of early layers to skip (input translation region).
        skip_late: Number of late layers to skip (output formatting region).
        gpu_memory_limit: Max GPU memory in GB. 0 = no limit.
        batch_size: Number of probe samples per evaluation.
        output_dir: Directory for results output.
    """

    model_path: str
    probe_name: str = "math"
    min_block_size: int = 7
    step: int = 1
    skip_early: int = 0
    skip_late: int = 0
    gpu_memory_limit: float = 0.0
    batch_size: int = 32
    output_dir: str = "./results"
    custom_probe_path: str | None = None
    backend: str = "transformers"
    dtype: str = "float16"
    device: str = "auto"
    top_k: int = 5
    sparse_first: bool = False
    sparse_step: int = 4
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.min_block_size < 1:
            raise ValueError(f"min_block_size must be >= 1, got {self.min_block_size}")
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


@dataclass(frozen=True)
class ScanResult:
    """Result of evaluating a single (i, j) configuration.

    Args:
        config: The duplication configuration tested.
        score: Aggregate score from the probe (higher is better).
        uncertainty: Variance of the score estimate.
        per_sample_scores: Individual scores for each probe sample.
        log_odds: Mean log-odds of correct answer (None if no annotations).
        accuracy: Fraction of samples where argmax == correct (None if N/A).
        metadata: Additional info (latency, memory usage, etc.).
    """

    config: DuplicationConfig
    score: float
    uncertainty: float
    per_sample_scores: list[float] = field(default_factory=list)
    log_odds: float | None = None
    accuracy: float | None = None
    mean_coverage: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)
