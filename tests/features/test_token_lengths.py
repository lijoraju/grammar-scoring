import numpy as np
import pandas as pd
import pytest

from grammar_scoring.features.token_lengths import audit_token_lengths


class Tokenizer:
    model_max_length = 999

    def __call__(self, text, **kwargs):
        assert kwargs == {
            "add_special_tokens": True,
            "truncation": False,
            "padding": False,
        }
        return {"input_ids": [0] * (len(text) + 2)}


def test_counts_limits_order_summary_determinism():
    frame = pd.DataFrame(
        {
            "split": ["train"] * 3,
            "filename": ["c", "b", "a"],
            "text": [" x ", "four", ""],
        }
    )
    result = audit_token_lengths(frame, Tokenizer(), 5)
    assert result.samples.filename.tolist() == ["c", "b", "a"]
    assert result.samples.token_count.tolist() == [5, 6, 2]
    assert result.samples.exceeds_limit.tolist() == [False, True, False]
    assert result.summary == {
        "n_samples": 3,
        "min": 2,
        "mean": 13 / 3,
        "median": 5.0,
        "p95": float(np.percentile([5, 6, 2], 95)),
        "max": 6,
        "n_exceeding_limit": 1,
        "pct_exceeding_limit": 100 / 3,
    }
    assert result.configured_limit == 5
    assert result.tokenizer_limit == 999
    pd.testing.assert_frame_equal(
        result.samples, audit_token_lengths(frame, Tokenizer(), 5).samples
    )


def test_empty():
    result = audit_token_lengths(
        pd.DataFrame(columns=["split", "filename", "text"]), Tokenizer(), 256
    )
    assert result.samples.empty
    assert result.summary["n_samples"] == 0
    assert result.summary["mean"] is None
    assert result.summary["pct_exceeding_limit"] == 0


@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
def test_bad_limit(limit):
    with pytest.raises(ValueError):
        audit_token_lengths(
            pd.DataFrame(columns=["split", "filename", "text"]), Tokenizer(), limit
        )
