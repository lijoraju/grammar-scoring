"""Parity-gated E006a fold ensemble and fixed E005 fusion, entirely offline."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grammar_scoring.config.paths import ARTIFACT_DIR, FEATURES_DIR, OOF_DIR
from grammar_scoring.data.dataset import load_test_dataframe, load_train_dataframe
from grammar_scoring.experiments.acoustic_baseline import (
    align_inputs,
    align_oof,
    evaluate_acoustic_baseline,
)
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.features.acoustic import FEATURE_COLUMNS
from grammar_scoring.features.acoustic_artifacts import load_features, validate_features

E005_WEIGHT = 0.55
E006A_WEIGHT = 0.45
PARITY_ATOL = 1e-10
N_FOLDS = 5


def align_filenames(frame: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Validate exact identities and return rows in canonical filename order."""
    for source in (frame, expected):
        _validate_filenames(source)
    if set(frame.filename) != set(expected.filename):
        raise ValueError("Filename identities mismatch")
    return (
        frame.set_index("filename", drop=False)
        .loc[expected.filename]
        .reset_index(drop=True)
    )


def load_aligned_features(
    path: Path, expected: pd.DataFrame, split: str
) -> pd.DataFrame:
    """Reuse artifact validation before aligning to the canonical dataset order."""
    identities = pd.read_csv(path, usecols=["filename"], dtype={"filename": str})
    frame = load_features(path, identities, split)
    aligned = align_filenames(frame, expected)
    validate_features(aligned, expected, split)
    return aligned


def validate_predictions(frame: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Require exact filename/label schema, identities and finite predictions."""
    if list(frame.columns) != ["filename", "label"]:
        raise ValueError("Prediction schema must be exactly filename,label")
    aligned = align_filenames(frame, test)
    if not np.isfinite(aligned.label.to_numpy(dtype=float)).all():
        raise ValueError("Predictions must be finite")
    return aligned


def check_oof_parity(
    reconstructed: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, object]:
    """Reject OOF label/fold/identity differences and material numerical drift."""
    aligned = align_oof(reconstructed, reference)
    difference = float(
        np.max(np.abs(aligned.acoustic_prediction - aligned.e005_prediction))
    )
    if difference > PARITY_ATOL:
        raise ValueError(f"OOF parity failed: max absolute difference {difference}")
    return {"passed": True, "atol": PARITY_ATOL, "max_absolute_difference": difference}


def predict_fold_ensemble(
    train: pd.DataFrame,
    frozen_folds: pd.DataFrame,
    train_features: pd.DataFrame,
    test: pd.DataFrame,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    """Fit five isolated canonical pipelines; each predicts every test example."""
    aligned, folds = align_inputs(train, frozen_folds, train_features)
    ordered = align_filenames(test_features, test)
    validate_features(ordered, test, "test")
    if "duration_seconds" in FEATURE_COLUMNS:
        raise ValueError("Raw duration must be excluded from model features")
    x = aligned[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    x_test = ordered[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    result = test[["filename"]].reset_index(drop=True).copy()
    for fold in range(N_FOLDS):
        mask = folds != fold
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, solver="lsqr")),
            ]
        )
        model.fit(x[mask], train.label.to_numpy(dtype=float)[mask])
        result[f"fold_{fold}"] = model.predict(x_test)
    if not np.isfinite(result.iloc[:, 1:].to_numpy()).all():
        raise ValueError("Fold predictions must be finite")
    return result


def make_candidates(
    test: pd.DataFrame, per_fold: pd.DataFrame, e005: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply equal fold averaging and the exact fixed 0.55/0.45 equation."""
    columns = [f"fold_{fold}" for fold in range(N_FOLDS)]
    if list(per_fold.columns) != ["filename", *columns]:
        raise ValueError("Five-fold prediction schema mismatch")
    ordered = align_filenames(per_fold, test)
    values = ordered[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Fold predictions must be finite")
    acoustic = test[["filename"]].reset_index(drop=True).copy()
    acoustic["label"] = values.mean(axis=1)
    e005 = validate_predictions(e005, test)
    blended = acoustic.copy()
    blended["label"] = (
        E005_WEIGHT * e005.label.to_numpy() + E006A_WEIGHT * acoustic.label.to_numpy()
    )
    return validate_predictions(acoustic, test), validate_predictions(blended, test)


def distribution(values: pd.Series) -> dict[str, float | int]:
    """Summarize unmodified predictions with population standard deviation."""
    x = values.to_numpy(dtype=float)
    return {
        "count": len(x),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "below_0": int((x < 0).sum()),
        "above_5": int((x > 5).sum()),
    }


def run_acoustic_inference() -> dict[str, object]:
    """Verify canonical OOF parity before test prediction or candidate writes."""
    train = load_train_dataframe()
    test = load_test_dataframe()[["filename"]]
    if len(train) != 769 or len(test) != 216:
        raise ValueError("Expected 769 train and 216 test rows")
    paths = {
        "train_features": FEATURES_DIR / "acoustic_train.csv",
        "test_features": FEATURES_DIR / "acoustic_test.csv",
        "folds": FEATURES_DIR / "train_folds.csv",
        "reference_oof": OOF_DIR / "E006a_acoustic_ridge.csv",
        "e005_oof": OOF_DIR / "E005_deberta_finetuned.csv",
        "e005_test": ARTIFACT_DIR / "submissions" / "E005_test_predictions.csv",
        "metadata": FEATURES_DIR / "acoustic_metadata.json",
    }
    metadata = json.loads(paths["metadata"].read_text())
    if tuple(metadata["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Canonical feature metadata mismatch")
    train_features = load_aligned_features(paths["train_features"], train, "train")
    test_features = load_aligned_features(paths["test_features"], test, "test")
    folds = pd.read_csv(paths["folds"])
    reconstructed = evaluate_acoustic_baseline(train, folds, train_features)
    parity = check_oof_parity(reconstructed.oof, pd.read_csv(paths["reference_oof"]))
    metrics = reconstructed.oof_metrics
    if not (
        abs(metrics["rmse"] - 0.868546) <= 1e-6
        and abs(metrics["pearson_correlation"] - 0.714764) <= 1e-6
    ):
        raise ValueError("Canonical OOF metric verification failed")
    align_oof(reconstructed.oof, pd.read_csv(paths["e005_oof"]))
    e005 = validate_predictions(pd.read_csv(paths["e005_test"]), test)
    per_fold = predict_fold_ensemble(train, folds, train_features, test, test_features)
    acoustic, blended = make_candidates(test, per_fold, e005)
    output_dir = ARTIFACT_DIR / "submissions"
    diagnostics_dir = ARTIFACT_DIR / "experiments" / "E005_E006a_w055"
    summary = {
        "weights": {"E005": E005_WEIGHT, "E006a": E006A_WEIGHT},
        "model": {"scaler": "StandardScaler", "alpha": 1.0, "solver": "lsqr"},
        "seed": 42,
        "folds": "existing frozen train_folds.csv",
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "oof_parity": parity,
        "reconstructed_oof_metrics": metrics,
        "E006a_test": distribution(acoustic.label),
        "E005_test": distribution(e005.label),
        "blend_test": distribution(blended.label),
        "per_fold": {
            f"fold_{fold}": distribution(per_fold[f"fold_{fold}"])
            for fold in range(N_FOLDS)
        },
        "input_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in paths.items()
        },
        "output_paths": [
            str(output_dir / "E006a_test_predictions.csv"),
            str(output_dir / "E005_E006a_w055_test_predictions.csv"),
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    acoustic.to_csv(summary["output_paths"][0], index=False)
    blended.to_csv(summary["output_paths"][1], index=False)
    per_fold.to_csv(diagnostics_dir / "fold_test_predictions.csv", index=False)
    (diagnostics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary
