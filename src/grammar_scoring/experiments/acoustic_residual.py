"""E012a: fixed acoustic correction and linear stacking of existing E008 OOF."""

import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from grammar_scoring.config.paths import ARTIFACT_DIR
from grammar_scoring.evaluation.metrics import pearson_correlation, regression_metrics
from grammar_scoring.experiments.acoustic_baseline import _json_safe
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.features.acoustic import (
    ACTIVITY_COLUMNS,
    ARTIFACT_COLUMNS,
    FEATURE_COLUMNS,
)


def align_inputs(baseline: pd.DataFrame, acoustic: pd.DataFrame) -> pd.DataFrame:
    """Validate the retained E008 population and align canonical features by name.

    The E008 OOF artifact is authoritative for labels and frozen folds. Acoustic
    rows outside that population are allowed, since E006a contains all 769 rows.
    """
    for frame in (baseline, acoustic):
        _validate_filenames(frame)
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate columns")
    if not {"label", "fold", "prediction"}.issubset(baseline):
        raise ValueError("E008 OOF requires label, fold, prediction")
    if len(baseline) != 732:
        raise ValueError("E008 retained population must contain exactly 732 rows")
    values = baseline[["label", "fold", "prediction"]]
    if any(dtype.kind not in "iuf" for dtype in values.dtypes):
        raise ValueError("E008 values must be numeric")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("E008 labels, folds and predictions must be finite")
    if not baseline.label.gt(0).all() or not baseline.label.le(5).all():
        raise ValueError("E008 labels must be positive and <= 5; zero rows forbidden")
    folds = baseline.fold.to_numpy()
    if not np.equal(folds, np.floor(folds)).all() or set(folds) != set(range(5)):
        raise ValueError("E008 folds must be integer IDs 0..4, all present")
    if len(FEATURE_COLUMNS) != 51 or tuple(acoustic.columns) != ARTIFACT_COLUMNS:
        raise ValueError("Acoustic schema must match exact 51 FEATURE_COLUMNS")
    if not acoustic.split.eq("train").all():
        raise ValueError("Only train acoustic features are allowed")
    numeric = acoustic[["duration_seconds", *FEATURE_COLUMNS]]
    if any(dtype.kind not in "iuf" for dtype in numeric.dtypes):
        raise ValueError("Acoustic features must be numeric")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Acoustic features must be finite")
    if not acoustic.duration_seconds.gt(0).all():
        raise ValueError("Acoustic duration must be positive")
    if not set(baseline.filename).issubset(acoustic.filename):
        raise ValueError("Missing acoustic row for E008 filename")
    ordered = acoustic.set_index("filename").loc[baseline.filename]
    result = baseline[["filename", "label", "fold", "prediction"]].copy()
    result = result.rename(columns={"prediction": "e008_prediction"})
    result["fold"] = result.fold.astype(int)
    result = result.reset_index(drop=True)
    result[list(FEATURE_COLUMNS)] = ordered[list(FEATURE_COLUMNS)].to_numpy()
    return result


def evaluate(aligned: pd.DataFrame, *, stack: bool = False) -> pd.DataFrame:
    """Fit each scaler and Ridge on fold != k and predict fold k exactly once."""
    x = aligned[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = aligned.label.to_numpy(dtype=float)
    base = aligned.e008_prediction.to_numpy(dtype=float)
    predictions = np.full(len(y), np.nan)
    corrections = np.full(len(y), np.nan)
    coverage = np.zeros(len(y), dtype=int)
    for fold in range(5):
        valid = aligned.fold.to_numpy() == fold
        scaler = StandardScaler().fit(x[~valid])
        train_x, valid_x = scaler.transform(x[~valid]), scaler.transform(x[valid])
        target = y[~valid] - base[~valid]
        if stack:
            train_x = np.column_stack((base[~valid], train_x))
            valid_x = np.column_stack((base[valid], valid_x))
            target = y[~valid]
        model = Ridge(alpha=1.0).fit(train_x, target)
        output = model.predict(valid_x)
        predictions[valid] = output if stack else base[valid] + output
        corrections[valid] = output
        coverage[valid] += 1
    if not (coverage == 1).all() or not np.isfinite(predictions).all():
        raise ValueError("OOF requires exactly one finite prediction per row")
    result = aligned[["filename", "label", "fold", "e008_prediction"]].copy()
    _validate_filenames(result)
    result["prediction"] = predictions
    result["residual"] = y - predictions
    result["abs_error"] = np.abs(result.residual)
    if not stack:
        result["predicted_correction"] = corrections
    return result


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Report pooled and per-fold RMSE and Pearson on raw OOF predictions."""
    return {
        "pooled": regression_metrics(frame.label, frame.prediction),
        "per_fold": [
            {"fold": int(k), "n": len(g), **regression_metrics(g.label, g.prediction)}
            for k, g in frame.groupby("fold", sort=True)
        ],
    }


def comparison(candidate: pd.DataFrame) -> dict[str, Any]:
    """Report paired error deltas, correlations and fixed true-label bands."""

    def paired(group: pd.DataFrame) -> dict[str, Any]:
        if group.empty:
            return {
                "n": 0,
                "e008": None,
                "candidate": None,
                "mean_e008_prediction": None,
                "mean_candidate_prediction": None,
            }
        old = regression_metrics(group.label, group.e008_prediction)
        new = regression_metrics(group.label, group.prediction)
        return {
            "n": len(group),
            "e008": old,
            "candidate": new,
            "rmse_delta": new["rmse"] - old["rmse"],
            "pearson_delta": new["pearson_correlation"] - old["pearson_correlation"],
            "mean_e008_prediction": float(group.e008_prediction.mean()),
            "mean_candidate_prediction": float(group.prediction.mean()),
        }

    folds = [
        {"fold": int(k), **paired(g)} for k, g in candidate.groupby("fold", sort=True)
    ]
    better = int(
        (
            candidate.abs_error < (candidate.label - candidate.e008_prediction).abs()
        ).sum()
    )
    return {
        **paired(candidate),
        "per_fold": folds,
        "delta_direction": "candidate minus E008",
        "folds_improving_rmse": sum(g["rmse_delta"] < 0 for g in folds),
        "folds_improving_pearson": sum(g["pearson_delta"] > 0 for g in folds),
        "prediction_correlation": pearson_correlation(
            candidate.prediction, candidate.e008_prediction
        ),
        "residual_correlation": pearson_correlation(
            candidate.residual, candidate.label - candidate.e008_prediction
        ),
        "samples_lower_absolute_error": better,
        "percent_lower_absolute_error": 100 * better / len(candidate),
        "score_bands": {
            "low": paired(candidate[candidate.label < 3]),
            "mid": paired(candidate[(candidate.label >= 3) & (candidate.label < 4)]),
            "high": paired(candidate[candidate.label >= 4]),
        },
    }


def _read_csv(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate CSV headers: {path}")
    return pd.read_csv(path)


def run_experiment(
    e008_oof: Path, acoustic_path: Path, artifact_dir: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Run fixed CPU models and persist OOF metrics, diagnostics and provenance."""
    directory = artifact_dir / "experiments" / "E012a"
    outputs = [
        artifact_dir / "oof" / f"E012a_{name}.csv" for name in ("residual", "stack")
    ]
    names = (
        "config.json",
        "metrics.json",
        "diagnostics.json",
        "residual_feature_correlations.csv",
    )
    if not overwrite and any(
        p.exists() for p in [*outputs, *(directory / n for n in names)]
    ):
        raise FileExistsError("E012a artifacts exist; pass --overwrite to rerun")
    aligned = align_inputs(_read_csv(e008_oof), _read_csv(acoustic_path))
    baseline = aligned.rename(columns={"e008_prediction": "prediction"})
    scores = {"E008": metrics(baseline)}
    reference = {"rmse": 0.619789, "pearson_correlation": 0.794231}
    discrepancy = {
        key: scores["E008"]["pooled"][key] - value for key, value in reference.items()
    }
    material = any(abs(value) > 0.01 for value in discrepancy.values())
    if material:
        warnings.warn(
            "Loaded E008 metrics differ >0.01 from known reference",
            UserWarning,
            stacklevel=2,
        )
    provenance = {
        "experiment": "E012a",
        "population_rule": "label > 0.0",
        "population_size": len(aligned),
        "seed": 42,
        "ridge_alpha": 1.0,
        "ridge_solver": "auto",
        "feature_columns": list(FEATURE_COLUMNS),
        "scaling": (
            "StandardScaler fit on fold != k acoustic features only; E008 unscaled"
        ),
        "fold_identity": aligned[["filename", "fold"]].to_dict("records"),
        "inputs": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in (("e008_oof", e008_oof), ("acoustic", acoustic_path))
        },
        "residual_target": "label - existing E008 OOF prediction",
        "prediction_behavior": "raw; no clipping, calibration, weighting or tuning",
    }
    residual = aligned.label - aligned.e008_prediction
    correlations = pd.DataFrame(
        [
            {
                "feature": name,
                "residual_correlation": pearson_correlation(aligned[name], residual),
            }
            for name in FEATURE_COLUMNS
        ]
    )
    ranked = (
        correlations.assign(absolute=correlations.residual_correlation.abs())
        .sort_values("absolute", ascending=False, kind="stable")
        .drop(columns="absolute")
    )
    diagnostics = {
        "provenance": provenance,
        "e008_reference": {
            "reference": reference,
            "deltas": discrepancy,
            "material_difference": material,
            "tolerance": 0.01,
        },
        "top_residual_features": ranked.head(10).to_dict("records"),
        "temporal_activity_correlations": correlations[
            correlations.feature.isin(ACTIVITY_COLUMNS)
        ].to_dict("records"),
        "feature_selection": "none; all 51 fixed features used",
        "comparisons": {},
    }
    candidates = {
        name: evaluate(aligned, stack=name == "stack") for name in ("residual", "stack")
    }
    for name, candidate in candidates.items():
        scores[f"E012a-{name}"] = metrics(candidate)
        diagnostics["comparisons"][f"E012a-{name}"] = comparison(candidate)
    directory.mkdir(parents=True, exist_ok=True)
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    for path, candidate in zip(outputs, candidates.values(), strict=True):
        candidate.to_csv(path, index=False)
    correlations.to_csv(directory / names[3], index=False)
    for name, value in zip(names[:3], (provenance, scores, diagnostics), strict=True):
        (directory / name).write_text(
            json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n"
        )
    return _json_safe(scores)


def main() -> None:
    """Parse artifact paths and run the fixed E012a protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e008-oof",
        type=Path,
        default=ARTIFACT_DIR / "oof" / "E008_deberta_no_zero_block.csv",
    )
    parser.add_argument(
        "--acoustic-path",
        type=Path,
        default=ARTIFACT_DIR / "features" / "acoustic_train.csv",
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_experiment(
                args.e008_oof,
                args.acoustic_path,
                args.artifact_dir,
                overwrite=args.overwrite,
            ),
            indent=2,
        )
    )
