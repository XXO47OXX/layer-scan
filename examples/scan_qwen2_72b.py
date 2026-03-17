"""Example: Scan Qwen2-72B for optimal layer duplication configuration.

This reproduces the RYS methodology on Qwen2-72B, which found
the optimal configuration at (i=45, j=52) — a 7-layer duplication
of the middle "reasoning cortex".

Requirements:
    - 2×RTX 4090 (48GB total VRAM)
    - EXL2 quantized Qwen2-72B model
    - pip install layer-scan[exllamav2]

Usage:
    python examples/scan_qwen2_72b.py /path/to/qwen2-72b-exl2
"""

import sys
from pathlib import Path

from layer_scan.backends.exllamav2 import ExLlamaV2Backend
from layer_scan.config import ScanConfig
from layer_scan.heatmap import generate_heatmap, generate_summary_text, save_results_json
from layer_scan.probes.math_probe import MathProbe
from layer_scan.scanner import run_scan


def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_qwen2_72b.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_dir = Path("./results/qwen2-72b")

    # Step 1: Load model
    backend = ExLlamaV2Backend()
    backend.load(
        model_path,
        gpu_split=[22000, 22000],  # 2×RTX 4090
        max_seq_len=4096,
    )

    print(f"Model loaded: {backend.get_total_layers()} layers")

    # Step 2: Configure scan
    # Qwen2-72B has 80 layers. Based on RYS findings:
    # - Skip first ~10 layers (input translation)
    # - Skip last ~10 layers (output formatting)
    # - Focus on middle "reasoning cortex"
    config = ScanConfig(
        model_path=model_path,
        probe_name="math",
        min_block_size=7,
        step=1,
        skip_early=10,
        skip_late=10,
        batch_size=16,
        top_k=10,
        output_dir=str(output_dir),
        backend="exllamav2",
    )

    # Step 3: Scan with math probe
    probe = MathProbe()
    report = run_scan(backend, probe, config)

    # Step 4: Output results
    print(generate_summary_text(report))

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_heatmap(report, output_dir / "math_heatmap.html")
    save_results_json(report, output_dir / "math_results.json")

    # Step 5: Also scan with JSON probe to check IFEval regression
    from layer_scan.probes.json_probe import JsonProbe

    json_probe = JsonProbe()
    json_config = ScanConfig(
        model_path=model_path,
        probe_name="json",
        min_block_size=7,
        step=1,
        skip_early=10,
        skip_late=10,
        batch_size=16,
        top_k=10,
        output_dir=str(output_dir),
        backend="exllamav2",
    )

    json_report = run_scan(backend, json_probe, json_config)
    print(generate_summary_text(json_report))
    generate_heatmap(json_report, output_dir / "json_heatmap.html")

    backend.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
