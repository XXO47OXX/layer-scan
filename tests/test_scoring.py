import numpy as np
import pytest
import torch

from layer_scan.scoring import ScoreResult, aggregate_scores, score_from_logits


class TestScoreFromLogits:
    def test_uniform_distribution(self):
        logits = torch.zeros(100)  # vocab size 100
        token_ids = list(range(10))  # tokens 0-9
        values = list(range(10))  # values 0-9

        result = score_from_logits(logits, token_ids, values)

        # Uniform over 0-9: mean = 4.5
        assert abs(result.expected_score - 4.5) < 0.01
        assert result.uncertainty > 0  # non-zero variance
        assert len(result.probabilities) == 10
        # All probabilities should be ~0.1
        for p in result.probabilities:
            assert abs(p - 0.1) < 0.01

    def test_peaked_distribution(self):
        logits = torch.zeros(100)
        token_ids = list(range(10))
        values = list(range(10))

        # Make token 7 dominant
        logits[7] = 100.0

        result = score_from_logits(logits, token_ids, values)

        assert result.expected_score > 6.9
        assert result.uncertainty < 0.1

    def test_bimodal_distribution(self):
        logits = torch.zeros(100)
        token_ids = list(range(10))
        values = list(range(10))

        logits[0] = 10.0  # value 0
        logits[9] = 10.0  # value 9

        result = score_from_logits(logits, token_ids, values)

        assert abs(result.expected_score - 4.5) < 0.5
        assert result.uncertainty > 10  # high variance

    def test_custom_values(self):
        logits = torch.zeros(50)
        token_ids = [10, 20, 30]
        values = [1.0, 5.0, 10.0]

        # Make middle value dominant
        logits[20] = 100.0

        result = score_from_logits(logits, token_ids, values)

        assert abs(result.expected_score - 5.0) < 0.1

    def test_mismatched_lengths_raises(self):
        logits = torch.zeros(100)
        with pytest.raises(ValueError, match="must have equal length"):
            score_from_logits(logits, [0, 1, 2], [0.0, 1.0])

    def test_result_types(self):
        logits = torch.zeros(100)
        result = score_from_logits(logits, list(range(10)))

        assert isinstance(result, ScoreResult)
        assert isinstance(result.expected_score, float)
        assert isinstance(result.uncertainty, float)
        assert isinstance(result.probabilities, list)
        assert isinstance(result.raw_logits, list)

    def test_default_values(self):
        logits = torch.zeros(100)
        token_ids = list(range(5))

        result = score_from_logits(logits, token_ids)

        # Uniform over [0,1,2,3,4] → mean = 2.0
        assert abs(result.expected_score - 2.0) < 0.01


class TestAggregateScores:
    def test_single_result(self):
        results = [ScoreResult(5.0, 1.0, [], [])]
        score, unc = aggregate_scores(results)
        assert score == 5.0
        assert unc == 1.0

    def test_multiple_results(self):
        results = [
            ScoreResult(4.0, 1.0, [], []),
            ScoreResult(6.0, 2.0, [], []),
            ScoreResult(8.0, 0.5, [], []),
        ]
        score, unc = aggregate_scores(results)
        assert abs(score - 6.0) < 0.01
        assert abs(unc - np.mean([1.0, 2.0, 0.5])) < 0.01

    def test_empty_results(self):
        score, unc = aggregate_scores([])
        assert score == 0.0
        assert unc == 0.0
