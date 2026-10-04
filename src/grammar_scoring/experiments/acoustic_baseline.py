"""E006a fixed acoustic Ridge baseline and offline E005 fusion diagnostics."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grammar_scoring.config.paths import ARTIFACT_DIR, FEATURES_DIR, OOF_DIR
from grammar_scoring.data.dataset import load_test_dataframe, load_train_dataframe
from grammar_scoring.evaluation.metrics import (
    pearson_correlation,
    regression_metrics,
    rmse,
)
from grammar_scoring.evaluation.validation import validate_cv_folds
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.features.acoustic import FEATURE_COLUMNS, extraction_config
from grammar_scoring.features.acoustic_artifacts import (
    load_features,
    shift_audit,
    validate_features,
)

EXPERIMENT_DIR = ARTIFACT_DIR / "experiments" / "E006a"


@dataclass
class AcousticBaselineResult:
    """Canonical acoustic OOF rows, fold scores and pooled training diagnostic."""

    oof: pd.DataFrame
    per_fold: pd.DataFrame
    oof_metrics: dict[str, float]
    training_rmse: float


def align_inputs(
    train: pd.DataFrame,
    frozen_folds: pd.DataFrame,
    features: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Align features and frozen folds by filename, checking labels and identity."""
    for frame in (train, frozen_folds, features):
        _validate_filenames(frame)
    if "label" not in train or "fold" not in frozen_folds:
        raise ValueError("Training label and frozen fold columns are required")
    if "split" in train and not train["split"].eq("train").all():
        raise ValueError("Only training rows can enter OOF fitting")
    labels = train.label.to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not ((labels >= 0) & (labels <= 5)).all():
        raise ValueError("Training labels must be finite in [0, 5]")
    expected = set(train.filename)
    if any(set(frame.filename) != expected for frame in (frozen_folds, features)):
        raise ValueError("Frozen fold or acoustic feature identities mismatch")
    ordered = (
        features.set_index("filename", drop=False)
        .loc[train.filename]
        .reset_index(drop=True)
    )
    validate_features(ordered, train, "train")
    assignments = frozen_folds.set_index("filename").loc[train.filename]
    if "label" in assignments and not np.array_equal(
        labels, assignments.label.to_numpy()
    ):
        raise ValueError("Frozen fold labels mismatch")
    folds = assignments.fold.to_numpy()
    validate_cv_folds(labels, folds, n_splits)
    return ordered, folds


def evaluate_acoustic_baseline(
    train: pd.DataFrame,
    frozen_folds: pd.DataFrame,
    artifact: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> AcousticBaselineResult:
    """Fit scaler and fixed Ridge only inside frozen fold training partitions.

    Training RMSE pools predictions across all fold training partitions, matching
    E002/E003/E004 diagnostics. Validation and training predictions are unclipped.
    """
    aligned, folds = align_inputs(train, frozen_folds, artifact, n_splits=n_splits)
    x = aligned[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = train.label.to_numpy(dtype=float)
    predictions = np.full(len(y), np.nan)
    coverage = np.zeros(len(y), dtype=int)
    diagnostics, train_actual, train_predicted = [], [], []
    for fold in range(n_splits):
        valid = folds == fold
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, solver="lsqr")),
            ]
        )
        model.fit(x[~valid], y[~valid])
        predictions[valid] = model.predict(x[valid])
        coverage[valid] += 1
        train_actual.append(y[~valid])
        train_predicted.append(model.predict(x[~valid]))
        diagnostics.append(
            {
                "fold": fold,
                "n_train": int((~valid).sum()),
                "n_valid": int(valid.sum()),
                **regression_metrics(y[valid], predictions[valid]),
            }
        )
    if not np.all(coverage == 1) or not np.isfinite(predictions).all():
        raise RuntimeError("Every row must have exactly one finite OOF prediction")
    oof = train[["filename", "label"]].copy()
    oof["fold"] = folds
    oof["prediction"] = predictions
    oof["residual"] = y - predictions
    oof["abs_error"] = np.abs(oof.residual)
    return AcousticBaselineResult(
        oof,
        pd.DataFrame(diagnostics),
        regression_metrics(y, predictions),
        rmse(np.concatenate(train_actual), np.concatenate(train_predicted)),
    )


def align_oof(acoustic: pd.DataFrame, e005: pd.DataFrame) -> pd.DataFrame:
    """Strictly align finite OOF predictions, labels and folds by filename."""
    for frame in (acoustic, e005):
        _validate_filenames(frame)
        if not {"label", "fold", "prediction"}.issubset(frame.columns):
            raise ValueError("OOF requires labels, folds and predictions")
        if not np.isfinite(
            frame[["label", "fold", "prediction"]].to_numpy(dtype=float)
        ).all():
            raise ValueError("OOF values must be finite")
    if set(acoustic.filename) != set(e005.filename):
        raise ValueError("E005 OOF filenames mismatch")
    ordered = e005.set_index("filename").loc[acoustic.filename]
    for column in ("label", "fold"):
        if not np.array_equal(acoustic[column].to_numpy(), ordered[column].to_numpy()):
            raise ValueError(f"E005 OOF {column} mismatch")
    result = acoustic[["filename", "label", "fold"]].copy()
    result["acoustic_prediction"] = acoustic.prediction.to_numpy()
    result["e005_prediction"] = ordered.prediction.to_numpy()
    for name in ("acoustic", "e005"):
        result[f"{name}_residual"] = result.label - result[f"{name}_prediction"]
    return result


def complementarity(aligned: pd.DataFrame) -> dict[str, object]:
    """Calculate pooled and per-fold prediction/residual Pearson correlations."""

    def correlations(frame: pd.DataFrame) -> dict[str, float]:
        return {
            f"{name}_correlation": pearson_correlation(
                frame[f"acoustic_{name}"], frame[f"e005_{name}"]
            )
            for name in ("prediction", "residual")
        }

    return {
        "pooled": correlations(aligned),
        "per_fold": [
            {"fold": int(fold), **correlations(frame)}
            for fold, frame in aligned.groupby("fold", sort=True)
        ],
    }


def blend_grid(aligned: pd.DataFrame) -> pd.DataFrame:
    """Evaluate fixed E005 weights 0.50..1.00 and per-fold stability diagnostics.

    RMSE delta is blend minus E005 (positive is worse). Pearson delta has the
    same subtraction order (negative is worse); worst is its minimum. Undefined
    Pearson has no improvement and propagates to the worst-fold diagnostic.
    """
    rows = []
    for weight in np.arange(50, 101, 5) / 100:
        predictions = (
            weight * aligned.e005_prediction
            + (1 - weight) * aligned.acoustic_prediction
        )
        row = {
            "e005_weight": float(weight),
            **regression_metrics(aligned.label, predictions),
        }
        rmse_deltas, pearson_deltas = [], []
        for fold, frame in aligned.groupby("fold", sort=True):
            blended = (
                weight * frame.e005_prediction
                + (1 - weight) * frame.acoustic_prediction
            )
            baseline = regression_metrics(frame.label, frame.e005_prediction)
            metrics = regression_metrics(frame.label, blended)
            for metric, value in metrics.items():
                row[f"fold_{int(fold)}_{metric}"] = value
            rmse_deltas.append(metrics["rmse"] - baseline["rmse"])
            pearson_deltas.append(
                metrics["pearson_correlation"] - baseline["pearson_correlation"]
            )
            row[f"fold_{int(fold)}_rmse_delta"] = rmse_deltas[-1]
            row[f"fold_{int(fold)}_pearson_delta"] = pearson_deltas[-1]
        row.update(
            {
                "folds_improving_rmse": int(np.sum(np.array(rmse_deltas) < 0)),
                "folds_improving_pearson": int(np.sum(np.array(pearson_deltas) > 0)),
                "worst_fold_rmse_delta": float(np.max(rmse_deltas)),
                "worst_fold_pearson_delta": float(np.min(pearson_deltas)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def run_acoustic_experiment(*, overwrite: bool = False) -> dict[str, object]:
    """Run canonical E006a and save offline diagnostic artifacts, never test scores."""
    output_oof = OOF_DIR / "E006a_acoustic_ridge.csv"
    names = (
        "config.json",
        "summary.json",
        "fold_metrics.csv",
        "acoustic_shift_audit.csv",
        "complementarity.json",
        "blend_grid.csv",
        "oof_comparison.csv",
    )
    if not overwrite and any(
        path.exists()
        for path in [output_oof, *(EXPERIMENT_DIR / name for name in names)]
    ):
        raise FileExistsError("E006a artifacts exist; pass --overwrite to rerun")
    train = load_train_dataframe()
    test = load_test_dataframe()[["filename"]]
    if len(train) != 769 or len(test) != 216:
        raise ValueError("E006a requires canonical 769 train and 216 test rows")
    train_features = load_features(FEATURES_DIR / "acoustic_train.csv", train, "train")
    test_features = load_features(FEATURES_DIR / "acoustic_test.csv", test, "test")
    audit = shift_audit(train_features, test_features)
    folds_path = FEATURES_DIR / "train_folds.csv"
    e005_path = OOF_DIR / "E005_deberta_finetuned.csv"
    result = evaluate_acoustic_baseline(train, pd.read_csv(folds_path), train_features)
    aligned = align_oof(result.oof, pd.read_csv(e005_path))
    correlations = complementarity(aligned)
    grid = blend_grid(aligned)
    summary = {
        "experiment": "E006a",
        "model_feature_count": len(FEATURE_COLUMNS),
        "oof_metrics": result.oof_metrics,
        "training_rmse": result.training_rmse,
        "e005_oof_metrics": regression_metrics(aligned.label, aligned.e005_prediction),
        "best_blend_rmse": grid.loc[grid.rmse.idxmin()].to_dict(),
        "best_blend_pearson": grid.loc[grid.pearson_correlation.idxmax()].to_dict(),
        "moderate_or_large_shift": audit.loc[
            audit.smd.abs() >= 0.5, ["feature", "smd"]
        ].to_dict("records"),
        "large_shift": audit.loc[audit.smd.abs() >= 0.8, ["feature", "smd"]].to_dict(
            "records"
        ),
        "limitations": [
            "Energy activity is not phonetic voicing; noise can be active",
            "No pitch; raw duration excluded; no label-driven feature selection",
            "Pause count/segment maxima may retain indirect length dependence",
            "Blend winners are exploratory OOF selections, not nested-CV estimates",
            "Train/test shift flags are diagnostics; predefined features retained",
        ],
    }
    configuration = {
        "seed": 42,
        "folds": "existing train_folds.csv; never regenerated",
        "estimator": {"scaler": "StandardScaler", "ridge_alpha": 1.0, "solver": "lsqr"},
        "features": FEATURE_COLUMNS,
        "extraction": extraction_config(),
        "smd_definition": "(test_mean-train_mean)/sqrt((train_var+test_var)/2); ddof=0",
        "blend_e005_weights": (np.arange(50, 101, 5) / 100).tolist(),
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                folds_path,
                e005_path,
                FEATURES_DIR / "acoustic_train.csv",
                FEATURES_DIR / "acoustic_test.csv",
                FEATURES_DIR / "acoustic_metadata.json",
            )
        },
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    result.oof.to_csv(output_oof, index=False)
    for filename, frame in (
        ("fold_metrics.csv", result.per_fold),
        ("acoustic_shift_audit.csv", audit),
        ("blend_grid.csv", grid),
        ("oof_comparison.csv", aligned),
    ):
        frame.to_csv(EXPERIMENT_DIR / filename, index=False)
    for filename, value in (
        ("summary.json", summary),
        ("config.json", configuration),
        ("complementarity.json", correlations),
    ):
        (EXPERIMENT_DIR / filename).write_text(
            json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n"
        )
    return summary
