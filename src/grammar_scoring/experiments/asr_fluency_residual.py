"""E013: fixed ASR fluency correction and linear stacking of existing E008 OOF."""

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
from grammar_scoring.data.transcripts import load_transcript_jsonl
from grammar_scoring.evaluation.metrics import pearson_correlation, regression_metrics
from grammar_scoring.experiments.acoustic_baseline import _json_safe
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.features.asr_fluency import FEATURE_COLUMNS, extract_features


def align_inputs(baseline: pd.DataFrame, asr_fluency: pd.DataFrame) -> pd.DataFrame:
    """Validate the retained E008 population and align canonical features by name.

    The E008 OOF artifact is authoritative for labels and frozen folds. Transcript
    rows outside that population are allowed and never enter fitting.
    """
    for frame in (baseline, asr_fluency):
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
    required = {"text", "duration_seconds", "segments"}
    if not required.issubset(asr_fluency):
        raise ValueError("Missing required transcript fields")
    if "split" in asr_fluency and not asr_fluency.split.eq("train").all():
        raise ValueError("Only train transcripts are allowed")
    if not set(baseline.filename).issubset(asr_fluency.filename):
        raise ValueError("Missing transcript record for E008 filename")
    ordered = asr_fluency.set_index("filename").loc[baseline.filename]
    features = pd.DataFrame(
        [extract_features(record) for record in ordered.to_dict("records")]
    )
    result = baseline[["filename", "label", "fold", "prediction"]].copy()
    result = result.rename(columns={"prediction": "e008_prediction"})
    result["fold"] = result.fold.astype(int)
    result = result.reset_index(drop=True)
    result[list(FEATURE_COLUMNS)] = features[list(FEATURE_COLUMNS)].to_numpy()
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
                "rmse_delta": None,
                "pearson_delta": None,
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
    e008_oof: Path, transcripts: Path, artifact_dir: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Run fixed CPU models and persist OOF metrics, diagnostics and provenance."""
    directory = artifact_dir / "experiments" / "E013"
    outputs = [
        artifact_dir / "oof" / f"E013_{name}.csv" for name in ("residual", "stack")
    ]
    names = (
        "config.json",
        "metrics.json",
        "diagnostics.json",
    )
    if not overwrite and any(
        p.exists() for p in [*outputs, *(directory / n for n in names)]
    ):
        raise FileExistsError("E013 artifacts exist; pass --overwrite to rerun")
    aligned = align_inputs(_read_csv(e008_oof), load_transcript_jsonl(transcripts))
    baseline = aligned.rename(columns={"e008_prediction": "prediction"})
    scores = {"E008": metrics(baseline)}
    reference = {"rmse": 0.6197891638194158, "pearson_correlation": 0.7942308488890498}
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
        "experiment": "E013",
        "population_rule": "label > 0.0",
        "population_size": len(aligned),
        "seed": 42,
        "ridge_alpha": 1.0,
        "ridge_solver": "auto",
        "feature_columns": list(FEATURE_COLUMNS),
        "scaling": (
            "StandardScaler fit on fold != k asr_fluency features only; E008 unscaled"
        ),
        "fold_identity": aligned[["filename", "fold"]].to_dict("records"),
        "inputs": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in (("e008_oof", e008_oof), ("transcripts", transcripts))
        },
        "residual_target": "label - existing E008 OOF prediction",
        "prediction_behavior": "raw; no clipping, calibration, weighting or tuning",
    }
    provenance["feature_definitions"] = {
        "tokenizer": "lowercase ASCII alphabetic [a-z]+",
        "fillers": ["erm", "hmm", "uh", "um"],
        "std_ddof": 0,
        "short_segment_seconds_exclusive": 2.0,
        "gap_thresholds_inclusive": [0.5, 1.0, 2.0],
        "segment_order": "artifact order; no sorting or overlap merging",
        "zero_denominators": 0.0,
    }
    diagnostics = {
        "provenance": provenance,
        "e008_reference": {
            "reference": reference,
            "deltas": discrepancy,
            "material_difference": material,
        },
        "continuation_gate": {
            "rmse_max": 0.58,
            "pearson_min": 0.7942308488890498,
            "improved_rmse_folds_min": 4,
        },
        "comparisons": {},
    }
    candidates = {
        name: evaluate(aligned, stack=name == "stack") for name in ("residual", "stack")
    }
    for name, candidate in candidates.items():
        scores[f"E013-{name}"] = metrics(candidate)
        report = comparison(candidate)
        report["compelling"] = bool(
            report["candidate"]["rmse"] <= 0.58
            and report["candidate"]["pearson_correlation"] >= 0.7942308488890498
            and report["folds_improving_rmse"] >= 4
        )
        report["absolute_error_win_rate"] = report[
            "samples_lower_absolute_error"
        ] / len(candidate)
        diagnostics["comparisons"][f"E013-{name}"] = report
    directory.mkdir(parents=True, exist_ok=True)
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    for path, candidate in zip(outputs, candidates.values(), strict=True):
        candidate.to_csv(path, index=False)
    for name, value in zip(names[:3], (provenance, scores, diagnostics), strict=True):
        (directory / name).write_text(
            json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n"
        )
    return _json_safe(scores)


def main() -> None:
    """Parse artifact paths and run the fixed E013 protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e008-oof",
        type=Path,
        default=ARTIFACT_DIR / "oof" / "E008_deberta_no_zero_block.csv",
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=ARTIFACT_DIR / "transcripts" / "train.jsonl",
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_experiment(
                args.e008_oof,
                args.transcripts,
                args.artifact_dir,
                overwrite=args.overwrite,
            ),
            indent=2,
        )
    )
