"""Descriptive E004 audit; no feature selection, transformations, or modeling."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from grammar_scoring.config.paths import FEATURES_DIR, TRANSCRIPTS_DIR
from grammar_scoring.data.dataset import load_train_dataframe
from grammar_scoring.data.transcripts import load_transcript_jsonl
from grammar_scoring.features.linguistic import FEATURE_COLUMNS, FEATURE_FAMILIES
from grammar_scoring.features.linguistic_artifacts import load_features

EXTREME_FEATURES = (
    "max_dependency_distance",
    "max_dependency_tree_depth",
    "repeated_bigram_count",
    "repeated_trigram_count",
    "words_per_minute",
)


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric columns using sample standard deviation (ddof=1)."""
    result = frame.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    result = result.rename(
        columns={
            "1%": "p01",
            "5%": "p05",
            "25%": "p25",
            "50%": "median",
            "75%": "p75",
            "95%": "p95",
            "99%": "p99",
        }
    ).drop(columns="count")
    result["zero_fraction"] = frame.eq(0).mean()
    return result


def variance_and_sparsity(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Flag constants, dominant values, and strict >95%/>99% zero fractions.

    Near-zero variance means nonconstant, at most 10% unique values, and a
    most-common/second-most-common frequency ratio >=19 (caret-style).
    """
    constant, near = [], []
    for name in frame:
        counts = frame[name].value_counts()
        if len(counts) <= 1:
            constant.append(name)
        elif len(counts) / len(frame) <= 0.1 and counts.iloc[0] / counts.iloc[1] >= 19:
            near.append(name)
    zeros = frame.eq(0).mean()
    return {
        "zero_variance": constant,
        "near_zero_variance": near,
        "zero_over_95_percent": zeros.index[zeros > 0.95].tolist(),
        "zero_over_99_percent": zeros.index[zeros > 0.99].tolist(),
    }


def standardized_mean_difference(train: pd.Series, test: pd.Series) -> float:
    """Return signed SMD using sample variances; equal constants return zero.

    Unequal constants return signed infinity, serialized as an explicit string.
    """
    delta = float(train.mean() - test.mean())
    pooled = float(np.sqrt((train.var(ddof=1) + test.var(ddof=1)) / 2))
    if pooled == 0:
        return 0.0 if delta == 0 else float(np.copysign(np.inf, delta))
    return delta / pooled


def ks_statistic(train: pd.Series, test: pd.Series) -> float:
    """Return the descriptive two-sample KS statistic without a p-value decision."""
    return float(ks_2samp(train, test).statistic)


def shift_metrics(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Calculate signed/absolute SMD and KS for each feature."""
    records = []
    for name in train:
        smd = standardized_mean_difference(train[name], test[name])
        records.append(
            {
                "feature": name,
                "smd": smd,
                "abs_smd": abs(smd),
                "ks": ks_statistic(train[name], test[name]),
            }
        )
    return pd.DataFrame(records).set_index("feature")


def high_correlation_pairs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return unique pairs with |Pearson| >=.95 and affine-relation diagnostics.

    Near-exact affine relations have maximum residual <=1e-10 times the larger
    of one and the maximum absolute response. These are diagnostics, not proofs.
    """
    correlations = frame.corr()
    pairs = []
    for i, left in enumerate(frame.columns):
        for right in frame.columns[i + 1 :]:
            value = float(correlations.loc[left, right])
            if abs(value) >= 0.95:
                x, y = frame[left].to_numpy(), frame[right].to_numpy()
                slope = float(
                    np.sum((x - x.mean()) * (y - y.mean()))
                    / np.sum((x - x.mean()) ** 2)
                )
                intercept = float(y.mean() - slope * x.mean())
                residual = float(np.max(np.abs(y - (slope * x + intercept))))
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "pearson": value,
                        "abs_pearson": abs(value),
                        "affine_slope": slope,
                        "affine_intercept": intercept,
                        "max_affine_residual": residual,
                        "near_exact_affine": residual
                        <= 1e-10 * max(1, float(np.max(np.abs(y)))),
                    }
                )
    return sorted(pairs, key=lambda p: (-p["abs_pearson"], p["left"], p["right"]))


def align_training_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Align canonical training labels by filename with one-to-one validation."""
    if not features["split"].eq("train").all():
        raise ValueError("Labels may only be aligned to training features")
    for frame in (features, labels):
        names = frame["filename"]
        if (
            names.isna().any()
            or names.str.strip().eq("").any()
            or names.duplicated().any()
        ):
            raise ValueError("Missing or duplicate filenames")
    if set(features.filename) != set(labels.filename):
        raise ValueError("Training filenames do not match canonical labels")
    aligned = pd.to_numeric(
        features[["filename"]].merge(
            labels[["filename", "label"]],
            on="filename",
            validate="one_to_one",
            how="left",
            sort=False,
        )["label"],
        errors="raise",
    )
    if not np.isfinite(aligned).all() or not aligned.between(0, 5).all():
        raise ValueError("Training labels must be finite and within [0, 5]")
    aligned.index = features.index
    return aligned


def label_correlations(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Calculate descriptive Pearson/Spearman; constants produce NaN."""
    result = pd.DataFrame(
        index=frame.columns, columns=["pearson", "spearman"], dtype=float
    )
    if labels.nunique() > 1:
        for name in frame:
            if frame[name].nunique() > 1:
                for method in result.columns:
                    result.loc[name, method] = frame[name].corr(labels, method=method)
    return result


def _clean(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def audit_main(argv: list[str] | None = None) -> int:
    """Validate artifacts, print diagnostics, and save a label-row-free JSON audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPTS_DIR)
    options = parser.parse_args(argv)
    frames = {}
    inputs = {}
    for split in ("train", "test"):
        path = options.feature_dir / f"linguistic_{split}.csv"
        expected = load_transcript_jsonl(options.transcript_dir / f"{split}.jsonl")
        frames[split] = load_features(path, expected, split)
        if len(frames[split]) < 2:
            raise ValueError("Each split requires at least two rows")
        inputs[split] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    numeric = {s: f[list(FEATURE_COLUMNS)] for s, f in frames.items()}
    labels = align_training_labels(frames["train"], load_train_dataframe())
    correlations = label_correlations(numeric["train"], labels)
    families = {}
    for family, names in FEATURE_FAMILIES.items():
        values = correlations.loc[list(names)].abs()
        families[family] = {
            "feature_count": len(names),
            **{
                f"{stat}_absolute_{method}": float(getattr(values[method], stat)())
                for method in values
                for stat in ("median", "max")
            },
        }
    shift = shift_metrics(numeric["train"], numeric["test"])
    pairs = high_correlation_pairs(numeric["train"])
    audit = {
        "purpose": "Descriptive EDA only; no selection, tuning, or modeling",
        "inputs": inputs,
        "definitions": {
            "std_variance_ddof": 1,
            "near_zero_variance": (
                "Nonconstant; unique fraction <=0.10; frequency ratio >=19"
            ),
            "zero_thresholds": "Strictly greater than 0.95 and 0.99",
            "zero_pooled_variance_smd": "0 for equal means; signed Infinity otherwise",
            "undefined_correlations": "null",
            "near_exact_affine": "max residual <=1e-10 * max(1, max(abs(response)))",
        },
        "integrity": {
            s: {
                "row_count": len(f),
                "feature_count": len(numeric[s].columns),
                "duplicate_identities": int(f.duplicated(["split", "filename"]).sum()),
                "missing_values": int(f.isna().sum().sum()),
                "non_finite_values": int((~np.isfinite(numeric[s])).sum().sum()),
                "canonical_schema_equal": set(numeric[s]) == set(FEATURE_COLUMNS),
                "canonical_order_equal": tuple(numeric[s]) == FEATURE_COLUMNS,
            }
            for s, f in frames.items()
        },
        "train_test_schema_equal": set(numeric["train"]) == set(numeric["test"]),
        "train_test_order_equal": list(numeric["train"]) == list(numeric["test"]),
        "variance_and_sparsity": {
            s: variance_and_sparsity(f) for s, f in numeric.items()
        },
        "distributions": {
            s: distribution_summary(f).to_dict("index") for s, f in numeric.items()
        },
        "shift_metrics": shift.to_dict("index"),
        "high_correlation_pairs": pairs,
        "train_label_correlations": correlations.to_dict("index"),
        "family_correlations": families,
    }
    output = options.feature_dir / "linguistic_audit.json"
    output.write_text(json.dumps(_clean(audit), indent=2, allow_nan=False) + "\n")
    for section in ("integrity", "variance_and_sparsity", "family_correlations"):
        print(f"\n{section}:\n{json.dumps(_clean(audit[section]), indent=2)}")
    for metric in ("abs_smd", "ks"):
        top = shift.sort_values(metric, ascending=False, kind="stable").head(15)
        print(f"\nTop 15 shift by {metric}:\n{top.to_string()}")
    print("\nAll high-correlation pairs:")
    print(pd.DataFrame(pairs).to_string(index=False))
    for method in correlations:
        order = correlations[method].abs().sort_values(ascending=False, kind="stable")
        top = correlations.loc[order.index].head(15)
        print(f"\nTop 15 label {method}:\n{top.to_string()}")
    noteworthy = list(FEATURE_FAMILIES["spoken"]) + [
        "duration_seconds",
        "word_count",
        "sentence_count",
        "words_per_minute",
        "type_token_ratio",
        "mean_dependency_distance",
        "max_dependency_distance",
        "mean_dependency_tree_depth",
        "max_dependency_tree_depth",
        "sentence_completeness_rate",
    ]
    for split, frame in frames.items():
        summary = distribution_summary(numeric[split]).loc[noteworthy]
        print(f"\n{split} noteworthy distributions:\n{summary.to_string()}")
        for name in EXTREME_FEATURES:
            display = frame[["filename", name]].copy()
            if split == "train":
                display["label"] = labels
            top = display.sort_values(name, ascending=False, kind="stable").head(10)
            print(f"\n{split} top 10 {name}:\n{top.to_string(index=False)}")
    print(f"\nSaved {output}")
    return 0
