"""Math reasoning probe based on RYS evaluation methodology.

Tests the model's ability to perform mathematical estimation and reasoning,
which is the capability most improved by layer duplication.

Scoring: Model rates its confidence/answer on a 0-9 scale using logit
distribution over digit tokens.
"""

from __future__ import annotations

from layer_scan.probes.base import Probe, ProbeSample
from layer_scan.scoring import get_digit_token_ids

# Math estimation problems with known difficulty levels.
# The model sees the problem and rates on 0-9 scale.
# We use logit distribution to get fine-grained scores.
_MATH_SAMPLES = [
    {
        "prompt": (
            "Estimate the result of 347 × 28. "
            "Rate your confidence from 0-9 that the answer is between 9700 and 9750.\n"
            "Answer: "
        ),
        "expected": 7.0,  # 347*28 = 9716, so high confidence
        "category": "arithmetic",
    },
    {
        "prompt": (
            "Without calculating exactly, is 2^17 closer to 100000 or 150000? "
            "Rate from 0 (definitely 100000) to 9 (definitely 150000).\n"
            "Answer: "
        ),
        "expected": 3.0,  # 2^17 = 131072, closer to 150000 but not extreme
        "category": "powers",
    },
    {
        "prompt": (
            "A triangle has sides of length 5, 12, and 13. "
            "Rate from 0-9 how likely this is a right triangle.\n"
            "Answer: "
        ),
        "expected": 9.0,  # 5-12-13 is a Pythagorean triple
        "category": "geometry",
    },
    {
        "prompt": (
            "If log₂(x) = 10, rate from 0-9 how close x is to 1000.\n"
            "Answer: "
        ),
        "expected": 7.0,  # 2^10 = 1024, very close to 1000
        "category": "logarithm",
    },
    {
        "prompt": (
            "Rate from 0-9 how confident you are that √(144) + √(169) > 25.\n"
            "Answer: "
        ),
        "expected": 1.0,  # 12 + 13 = 25, not > 25
        "category": "roots",
    },
    {
        "prompt": (
            "A store offers 30% off, then an additional 20% off the reduced price. "
            "Rate from 0-9 how close the total discount is to 50%.\n"
            "Answer: "
        ),
        "expected": 4.0,  # 1 - 0.7*0.8 = 0.44, somewhat close to 0.5
        "category": "percentage",
    },
    {
        "prompt": (
            "How many prime numbers are there between 1 and 20? "
            "Rate from 0 (fewer than 6) to 9 (more than 10).\n"
            "Answer: "
        ),
        "expected": 5.0,  # 8 primes: 2,3,5,7,11,13,17,19 — middle range
        "category": "primes",
    },
    {
        "prompt": (
            "If you flip a fair coin 10 times, rate from 0-9 how likely "
            "you are to get exactly 5 heads.\n"
            "Answer: "
        ),
        "expected": 3.0,  # ~24.6% probability, moderate
        "category": "probability",
    },
    {
        "prompt": (
            "The sum of interior angles of a hexagon is ___. "
            "Rate from 0-9 how confident you are that it's 720 degrees.\n"
            "Answer: "
        ),
        "expected": 9.0,  # (6-2)*180 = 720, correct
        "category": "geometry",
    },
    {
        "prompt": (
            "Rate from 0-9: The derivative of x³ + 2x² at x=1 equals 7.\n"
            "Answer: "
        ),
        "expected": 9.0,  # 3x² + 4x at x=1 = 3+4 = 7, correct
        "category": "calculus",
    },
    {
        "prompt": (
            "Estimate: 999 × 1001 is closest to which value? "
            "Rate from 0 (around 990000) to 9 (around 1000000).\n"
            "Answer: "
        ),
        "expected": 9.0,  # 999*1001 = 999999 ≈ 1000000
        "category": "arithmetic",
    },
    {
        "prompt": (
            "Rate from 0-9: In the Fibonacci sequence (1,1,2,3,5,8,13,...), "
            "the ratio of consecutive terms approaches 1.618.\n"
            "Answer: "
        ),
        "expected": 9.0,  # Golden ratio, correct
        "category": "sequences",
    },
    {
        "prompt": (
            "Rate from 0-9 your confidence that the integral of 1/x from 1 to e equals 1.\n"
            "Answer: "
        ),
        "expected": 9.0,  # ln(e) - ln(1) = 1 - 0 = 1
        "category": "calculus",
    },
    {
        "prompt": (
            "A group of 23 people is in a room. "
            "Rate from 0-9 how likely at least two share a birthday.\n"
            "Answer: "
        ),
        "expected": 5.0,  # ~50.7%, moderate
        "category": "probability",
    },
    {
        "prompt": (
            "Rate from 0-9: The series 1 + 1/2 + 1/4 + 1/8 + ... converges to 2.\n"
            "Answer: "
        ),
        "expected": 9.0,  # Geometric series sum = 1/(1-1/2) = 2
        "category": "series",
    },
    {
        "prompt": (
            "Rate from 0-9 how confident you are that "
            "the number 91 is prime.\n"
            "Answer: "
        ),
        "expected": 1.0,  # 91 = 7*13, not prime
        "category": "primes",
    },
]


class MathProbe(Probe):
    """Math reasoning probe using 0-9 digit scoring."""

    @property
    def name(self) -> str:
        return "math"

    @property
    def description(self) -> str:
        return (
            "Mathematical estimation and reasoning probe. Tests arithmetic, "
            "geometry, calculus, probability, and number theory. Scored on "
            "a 0-9 scale via logit distribution over digit tokens."
        )

    def get_samples(self, count: int | None = None) -> list[ProbeSample]:
        samples = [
            ProbeSample(
                prompt=s["prompt"],
                expected_score=s["expected"],
                correct_answer=int(s["expected"]),
                metadata={"category": s["category"]},
            )
            for s in _MATH_SAMPLES
        ]
        if count is not None:
            samples = samples[:count]
        return samples

    def get_score_token_ids(self, tokenizer) -> tuple[list[int], list[float]]:
        return get_digit_token_ids(tokenizer)
