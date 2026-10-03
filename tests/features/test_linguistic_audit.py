import numpy as np
import pandas as pd
import pytest

from grammar_scoring.features.linguistic_audit import (
    align_training_labels,
    distribution_summary,
    high_correlation_pairs,
    ks_statistic,
    label_correlations,
    standardized_mean_difference,
    variance_and_sparsity,
)


def test_distribution():
    summary = distribution_summary(pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})).loc["x"]
    assert summary["mean"] == 1.5
    assert summary["median"] == 1.5
    assert summary["std"] == pytest.approx(np.sqrt(5 / 3))
    assert summary["p01"] == pytest.approx(0.03)
    assert summary["p99"] == pytest.approx(2.97)
    assert summary["zero_fraction"] == 0.25


def test_variance_sparsity_strict_thresholds():
    frame = pd.DataFrame(
        {
            "constant": [2] * 100,
            "zero": [0] * 100,
            "rare": [0] * 99 + [1],
            "boundary": [0] * 95 + [1] * 5,
        }
    )
    flags = variance_and_sparsity(frame)
    assert flags["zero_variance"] == ["constant", "zero"]
    assert flags["near_zero_variance"] == ["rare", "boundary"]
    assert flags["zero_over_95_percent"] == ["zero", "rare"]
    assert flags["zero_over_99_percent"] == ["zero"]


def test_smd_and_ks():
    train, test = pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 3.0, 4.0])
    assert standardized_mean_difference(train, test) == -1
    assert ks_statistic(train, test) == pytest.approx(1 / 3)
    assert ks_statistic(train, train) == 0
    assert standardized_mean_difference(pd.Series([1, 1]), pd.Series([1, 1])) == 0
    assert standardized_mean_difference(pd.Series([2, 2]), pd.Series([1, 1])) == np.inf
    assert standardized_mean_difference(pd.Series([1, 1]), pd.Series([2, 2])) == -np.inf


def test_unique_correlation_pairs():
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
            "c": [-1, -2, -3, -4],
            "constant": [1] * 4,
        }
    )
    pairs = high_correlation_pairs(frame)
    assert {(p["left"], p["right"]) for p in pairs} == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    }
    assert len(pairs) == 3
    assert all(p["near_exact_affine"] for p in pairs)
    assert pairs[0]["affine_slope"] == 2


def test_label_correlations():
    frame = pd.DataFrame({"x": [1, 2, 3, 10], "constant": [1] * 4})
    result = label_correlations(frame, pd.Series([1, 2, 3, 4]))
    assert result.loc["x", "pearson"] == pytest.approx(0.8854377448471462)
    assert result.loc["x", "spearman"] == 1
    assert result.loc["constant"].isna().all()
    assert label_correlations(frame, pd.Series([1] * 4)).isna().all().all()


def test_explicit_filename_alignment():
    features = pd.DataFrame(
        {"split": ["train", "train"], "filename": ["b", "a"]}, index=[7, 2]
    )
    labels = pd.DataFrame({"filename": ["a", "b"], "label": [1.0, 4.0]})
    aligned = align_training_labels(features, labels)
    assert aligned.tolist() == [4.0, 1.0]
    assert aligned.index.tolist() == [7, 2]
    with pytest.raises(ValueError, match="training"):
        align_training_labels(features.assign(split="test"), labels)
    with pytest.raises(ValueError, match="duplicate"):
        align_training_labels(features, pd.concat([labels, labels]))
    with pytest.raises(ValueError, match="match"):
        align_training_labels(features, labels.iloc[:1])
