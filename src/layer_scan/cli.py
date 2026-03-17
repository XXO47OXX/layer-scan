"""CLI entry point for layer-scan.

Usage:
    layer-scan scan --model <path> --probe math
    layer-scan scan --model <path> --probe json --backend exllamav2
    layer-scan scan --model <path> --probe custom --custom-probe my_probe.json
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from layer_scan import __version__

app = typer.Typer(
    name="layer-scan",
    help="Automated LLM layer duplication configuration scanner",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_probe(probe_name: str, custom_probe_path: str | None = None):
    """Load the specified probe by name."""
    from layer_scan.probes.custom import CustomProbe
    from layer_scan.probes.eq_probe import EqProbe
    from layer_scan.probes.json_probe import JsonProbe
    from layer_scan.probes.math_probe import MathProbe

    builtin_probes = {
        "math": MathProbe,
        "eq": EqProbe,
        "json": JsonProbe,
    }

    if probe_name == "custom":
        if not custom_probe_path:
            console.print("[red]--custom-probe required when --probe=custom[/red]")
            raise typer.Exit(1)
        return CustomProbe(custom_probe_path)

    if probe_name in builtin_probes:
        return builtin_probes[probe_name]()

    console.print(f"[red]Unknown probe: {probe_name}[/red]")
    console.print(f"Available probes: {', '.join(builtin_probes.keys())}, custom")
    raise typer.Exit(1)


def _load_backend(backend_name: str):
    """Load the specified inference backend."""
    if backend_name == "transformers":
        from layer_scan.backends.transformers_backend import TransformersBackend

        return TransformersBackend()
    elif backend_name == "exllamav2":
        from layer_scan.backends.exllamav2 import ExLlamaV2Backend

        return ExLlamaV2Backend()
    elif backend_name == "vllm":
        from layer_scan.backends.vllm_backend import VLLMBackend

        return VLLMBackend()
    else:
        console.print(f"[red]Unknown backend: {backend_name}[/red]")
        console.print("Available backends: transformers, exllamav2, vllm")
        raise typer.Exit(1)


@app.command()
def scan(
    model: str = typer.Option(..., "--model", "-m", help="Model path or HuggingFace ID"),
    probe: str = typer.Option("math", "--probe", "-p", help="Probe name: math, eq, json, custom"),
    backend: str = typer.Option(
        "transformers", "--backend", "-b", help="Backend: transformers, exllamav2",
    ),
    min_block: int = typer.Option(7, "--min-block", help="Minimum duplicated block size"),
    step: int = typer.Option(1, "--step", "-s", help="Step size for scanning i and j"),
    skip_early: int = typer.Option(0, "--skip-early", help="Skip N early layers"),
    skip_late: int = typer.Option(0, "--skip-late", help="Skip N late layers"),
    batch_size: int = typer.Option(16, "--batch-size", help="Samples per evaluation"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top configs to report"),
    output: str = typer.Option("./results", "--output", "-o", help="Output directory"),
    sparse_first: bool = typer.Option(
        False, "--sparse-first", help="Do sparse scan first, then refine",
    ),
    sparse_step: int = typer.Option(4, "--sparse-step", help="Step size for sparse scanning"),
    custom_probe: str = typer.Option(None, "--custom-probe", help="Path to custom probe JSON file"),
    dtype: str = typer.Option("float16", "--dtype", help="Model dtype: float16, bfloat16, float32"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    gpu_split: str = typer.Option(
        None, "--gpu-split", help="GPU memory split in MB, e.g. '22000,22000'",
    ),
    export_mergekit: str = typer.Option(
        None, "--export-mergekit", help="Export top config as mergekit YAML",
    ),
) -> None:
    """Scan layer duplication configurations and generate heatmap."""
    _setup_logging(verbose)

    from layer_scan.config import ScanConfig
    from layer_scan.heatmap import generate_heatmap, generate_summary_text, save_results_json
    from layer_scan.scanner import run_scan

    console.print(
        Panel(
            f"[bold]layer-scan v{__version__}[/bold]\n"
            f"Model: {model}\n"
            f"Probe: {probe}\n"
            f"Backend: {backend}\n"
            f"Min block: {min_block} | Step: {step}",
            title="Configuration",
        )
    )

    # Load probe and backend
    probe_instance = _load_probe(probe, custom_probe)
    backend_instance = _load_backend(backend)

    # Parse GPU split
    backend_kwargs = {"dtype": dtype}
    if gpu_split:
        backend_kwargs["gpu_split"] = [int(x.strip()) for x in gpu_split.split(",")]

    console.print("[bold cyan]Loading model...[/bold cyan]")
    backend_instance.load(model, **backend_kwargs)

    total_layers = backend_instance.get_total_layers()
    console.print(f"[green]Model loaded: {total_layers} layers[/green]")

    # Validate probe
    probe_instance.validate(backend_instance.get_tokenizer())
    console.print(f"[green]Probe '{probe}' validated[/green]")

    # Build scan config
    scan_config = ScanConfig(
        model_path=model,
        probe_name=probe,
        min_block_size=min_block,
        step=step,
        skip_early=skip_early,
        skip_late=skip_late,
        batch_size=batch_size,
        output_dir=output,
        backend=backend,
        dtype=dtype,
        top_k=top_k,
        sparse_first=sparse_first,
        sparse_step=sparse_step,
        custom_probe_path=custom_probe,
    )

    # Run scan
    console.print("[bold cyan]Starting scan...[/bold cyan]")
    report = run_scan(
        backend=backend_instance,
        probe=probe_instance,
        scan_config=scan_config,
    )

    # Output results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = generate_summary_text(report)
    console.print(summary)

    heatmap_path = generate_heatmap(
        report,
        output_dir / "heatmap.html",
    )
    console.print(f"\n[green]Heatmap: {heatmap_path}[/green]")

    json_path = save_results_json(report, output_dir / "results.json")
    console.print(f"[green]Results: {json_path}[/green]")

    # Export mergekit YAML if requested
    if export_mergekit:
        from layer_scan.export import export_mergekit_yaml

        yaml_content = export_mergekit_yaml(report, model)
        mergekit_path = Path(export_mergekit)
        mergekit_path.parent.mkdir(parents=True, exist_ok=True)
        mergekit_path.write_text(yaml_content)
        console.print(f"[green]mergekit YAML: {mergekit_path}[/green]")

    # Cleanup
    backend_instance.cleanup()
    console.print("[dim]Done.[/dim]")


@app.command(name="multi-probe")
def multi_probe_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Model path or HuggingFace ID"),
    probe_list: str = typer.Option(
        "math,eq,json", "--probes", help="Comma-separated probe names",
    ),
    backend: str = typer.Option(
        "transformers", "--backend", "-b", help="Backend: transformers, exllamav2",
    ),
    min_block: int = typer.Option(7, "--min-block", help="Minimum duplicated block size"),
    step: int = typer.Option(1, "--step", "-s", help="Step size"),
    skip_early: int = typer.Option(0, "--skip-early", help="Skip N early layers"),
    skip_late: int = typer.Option(0, "--skip-late", help="Skip N late layers"),
    batch_size: int = typer.Option(16, "--batch-size", help="Samples per evaluation"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Top configs to report"),
    output: str = typer.Option("./results", "--output", "-o", help="Output directory"),
    sparse_first: bool = typer.Option(True, "--sparse-first", help="Use sparse-then-dense"),
    sparse_step: int = typer.Option(4, "--sparse-step", help="Sparse step size"),
    dtype: str = typer.Option("float16", "--dtype", help="Model dtype"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Cross-analyze multiple probes to find Pareto-optimal configs."""
    _setup_logging(verbose)

    import json as json_mod
    from pathlib import Path as PathLib

    from layer_scan.config import ScanConfig
    from layer_scan.multi_probe import run_multi_probe

    probe_names = [p.strip() for p in probe_list.split(",")]
    probe_instances = [_load_probe(name) for name in probe_names]

    backend_instance = _load_backend(backend)

    console.print(
        Panel(
            f"[bold]layer-scan multi-probe[/bold]\n"
            f"Model: {model}\n"
            f"Probes: {', '.join(probe_names)}\n"
            f"Backend: {backend}",
            title="Multi-Probe Analysis",
        )
    )

    console.print("[bold cyan]Loading model...[/bold cyan]")
    backend_instance.load(model, dtype=dtype)
    total_layers = backend_instance.get_total_layers()
    console.print(f"[green]Model loaded: {total_layers} layers[/green]")

    for p in probe_instances:
        p.validate(backend_instance.get_tokenizer())

    scan_config = ScanConfig(
        model_path=model,
        probe_name=",".join(probe_names),
        min_block_size=min_block,
        step=step,
        skip_early=skip_early,
        skip_late=skip_late,
        batch_size=batch_size,
        output_dir=output,
        backend=backend,
        dtype=dtype,
        top_k=top_k,
        sparse_first=sparse_first,
        sparse_step=sparse_step,
    )

    report = run_multi_probe(backend_instance, probe_instances, scan_config)

    # Print results
    console.print("\n[bold]PARETO-OPTIMAL CONFIGURATIONS:[/bold]")
    console.print("-" * 70)
    for rank, result in enumerate(
        sorted(report.pareto_configs, key=lambda r: r.normalized_score, reverse=True)[:top_k],
        1,
    ):
        scores_str = " | ".join(
            f"{name}={result.probe_scores[name]:.3f}" for name in report.probe_names
        )
        console.print(
            f"  #{rank}: i={result.config.i}, j={result.config.j} "
            f"(balanced={result.normalized_score:.3f}) — {scores_str}"
        )

    if report.best_balanced:
        b = report.best_balanced
        console.print(
            f"\n[bold green]Best balanced:[/bold green] "
            f"i={b.config.i}, j={b.config.j} (score={b.normalized_score:.3f})"
        )

    console.print("\n[bold]PER-PROBE BEST:[/bold]")
    for name, result in report.per_probe_best.items():
        console.print(
            f"  {name}: i={result.config.i}, j={result.config.j} "
            f"(score={result.probe_scores[name]:.3f})"
        )

    # Save results
    output_dir = PathLib(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "model": model,
        "probes": probe_names,
        "pareto_configs": [
            {
                "i": r.config.i,
                "j": r.config.j,
                "normalized_score": r.normalized_score,
                "probe_scores": r.probe_scores,
                "probe_log_odds": r.probe_log_odds,
                "probe_accuracies": r.probe_accuracies,
            }
            for r in sorted(
                report.pareto_configs,
                key=lambda r: r.normalized_score,
                reverse=True,
            )
        ],
        "per_probe_best": {
            name: {
                "i": r.config.i,
                "j": r.config.j,
                "score": r.probe_scores[name],
            }
            for name, r in report.per_probe_best.items()
        },
    }

    results_path = output_dir / "multi_probe.json"
    with open(results_path, "w") as f:
        json_mod.dump(results_data, f, indent=2)
    console.print(f"\n[green]Results: {results_path}[/green]")

    backend_instance.cleanup()
    console.print("[dim]Done.[/dim]")


@app.command()
def annotate(
    results: str = typer.Option(
        ..., "--results", "-r", help="Path to layer-scan results.json"
    ),
    neuro_report: str = typer.Option(
        ..., "--neuro-report", "-n", help="Path to neuro-scan report.json"
    ),
    output: str = typer.Option(
        "./results/annotated_heatmap.html", "--output", "-o", help="Output HTML path"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Annotate layer-scan results with neuro-scan layer labels."""
    _setup_logging(verbose)

    from layer_scan.annotate import (
        generate_annotated_heatmap,
        generate_annotation_text,
        load_layer_scan_results,
        load_neuro_report,
    )

    console.print(Panel("[bold]Cross-Tool Annotation[/bold]", title="layer-scan"))

    layer_results = load_layer_scan_results(results)
    neuro_data = load_neuro_report(neuro_report)

    console.print(f"[green]Loaded layer-scan results: {results}[/green]")
    console.print(f"[green]Loaded neuro-scan report: {neuro_report}[/green]")

    # Generate annotation text
    annotation = generate_annotation_text(layer_results, neuro_data)
    console.print(annotation)

    # Generate annotated heatmap
    heatmap_path = generate_annotated_heatmap(layer_results, neuro_data, output)
    console.print(f"\n[green]Annotated heatmap: {heatmap_path}[/green]")


@app.command()
def lookup(
    model: str = typer.Option(..., "--model", "-m", help="Model path or HuggingFace ID"),
    probe: str = typer.Option("math", "--probe", "-p", help="Probe name"),
    download: bool = typer.Option(False, "--download", help="Save results.json locally"),
    output: str = typer.Option("./results.json", "--output", "-o", help="Output path for download"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Fetch pre-computed scan results from HuggingFace Hub."""
    _setup_logging(verbose)

    from layer_scan.lookup import fetch_results, format_lookup_result

    console.print(
        Panel(
            f"[bold]Pre-computed Lookup[/bold]\n"
            f"Model: {model}\n"
            f"Probe: {probe}",
            title="layer-scan",
        )
    )

    console.print("[bold cyan]Searching HuggingFace Hub...[/bold cyan]")
    record = fetch_results(model, probe)

    if record is None:
        console.print(f"[yellow]No pre-computed results found for {model} / {probe}[/yellow]")
        console.print("[dim]Tip: run 'layer-scan scan' to generate your own results[/dim]")
        raise typer.Exit(1)

    console.print(format_lookup_result(record))

    if download:
        import json as json_mod

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json_mod.dump(record, f, indent=2)
        console.print(f"\n[green]Results saved to {out_path}[/green]")


@app.command()
def probes() -> None:
    """List available evaluation probes."""
    from layer_scan.probes.eq_probe import EqProbe
    from layer_scan.probes.json_probe import JsonProbe
    from layer_scan.probes.math_probe import MathProbe

    console.print("[bold]Available Probes:[/bold]\n")

    for probe_cls in [MathProbe, EqProbe, JsonProbe]:
        p = probe_cls()
        samples = p.get_samples()
        console.print(f"  [cyan]{p.name}[/cyan]")
        console.print(f"    {p.description}")
        console.print(f"    Samples: {len(samples)}")
        console.print()

    console.print("  [cyan]custom[/cyan]")
    console.print("    Load from JSON file with --custom-probe <path>")


@app.command()
def version() -> None:
    """Show version."""
    from layer_scan import __version__

    console.print(f"layer-scan v{__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
