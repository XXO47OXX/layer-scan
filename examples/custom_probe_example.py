"""Example: Create and use a custom probe for domain-specific evaluation.

This example shows how to create a custom probe JSON file and use it
with layer-scan. Useful for finding optimal layer duplication configs
for specific downstream tasks (e.g., fingerprint extraction, code gen).

Usage:
    python examples/custom_probe_example.py
"""

import json
import tempfile
from pathlib import Path


def create_fingerprint_probe() -> Path:
    """Create a fingerprint extraction probe for testing M6 quality.

    This probe evaluates the model's ability to:
    1. Extract structured fields from network banners
    2. Follow grounding rules (only extract what's in the banner)
    3. Handle ambiguous/missing data correctly
    """
    probe_data = {
        "name": "fingerprint",
        "description": (
            "Fingerprint extraction probe for network banner analysis. "
            "Tests grounded extraction, brand/product identification, "
            "and JSON format compliance."
        ),
        "scoring": "digits",
        "samples": [
            {
                "prompt": (
                    "Banner: <title>D-Link DCS-930L Network Camera</title>\n"
                    "Extract brand and product. Rate 0-9 your confidence "
                    "in grounded extraction (only from banner text).\n"
                    "Answer: "
                ),
                "expected_score": 8.0,
                "metadata": {"category": "simple_extraction", "banner_type": "html_title"},
            },
            {
                "prompt": (
                    "Banner: SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.4\n"
                    "Extract all identifiable software and versions. "
                    "Rate 0-9 your confidence in complete extraction.\n"
                    "Answer: "
                ),
                "expected_score": 7.0,
                "metadata": {"category": "compound_extraction", "banner_type": "ssh"},
            },
            {
                "prompt": (
                    "Banner: HTTP/1.1 200 OK\\r\\nServer: nginx\\r\\n\\r\\n"
                    "<html><head><title>Welcome to 太达办公</title></head></html>\n\n"
                    "Extract ONLY technical identifiers (server software), "
                    "NOT business names. Rate 0-9 your ability to filter "
                    "non-technical content.\n"
                    "Answer: "
                ),
                "expected_score": 6.0,
                "metadata": {"category": "contamination_resistance", "banner_type": "http"},
            },
            {
                "prompt": (
                    "Banner: \\xff\\xfb\\x01\\xff\\xfb\\x03\\xff\\xfd\\x18\n"
                    "This is a raw Telnet negotiation. Rate 0-9 how well "
                    "you can extract useful identifiers from binary protocols.\n"
                    "Answer: "
                ),
                "expected_score": 2.0,
                "metadata": {"category": "binary_protocol", "banner_type": "telnet"},
            },
            {
                "prompt": (
                    "Banner: 220 mail.example.com ESMTP Postfix (Ubuntu)\n"
                    "Extract: server software, version if available, OS. "
                    "Rate 0-9 your confidence that all extracted fields "
                    "are directly present in the banner.\n"
                    "Answer: "
                ),
                "expected_score": 7.0,
                "metadata": {"category": "smtp_extraction", "banner_type": "smtp"},
            },
        ],
    }

    # Save to temp file
    probe_path = Path(tempfile.gettempdir()) / "fingerprint_probe.json"
    with open(probe_path, "w") as f:
        json.dump(probe_data, f, indent=2)

    print(f"Probe saved to: {probe_path}")
    print(f"Samples: {len(probe_data['samples'])}")
    print()
    print("Usage with layer-scan CLI:")
    print(f"  layer-scan scan --model <path> --probe custom --custom-probe {probe_path}")

    return probe_path


if __name__ == "__main__":
    create_fingerprint_probe()
