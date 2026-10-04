"""Deterministic diagnostics of frozen OOF predictions; no base model training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from grammar_scoring.evaluation.metrics import pearson_correlation, regression_metrics

MODELS = {
    "E005": "E005_deberta_finetuned.csv",
    "E003b": "E003b_deberta_ridge.csv",
    "E002b": "E002b_tfidf_word_char_ridge.csv",
    "E004b": "E004b_linguistic_surface_pos_ridge.csv",
    "E004d": "E004d_linguistic_all_ridge.csv",
}
FIELDS = ["filename", "label", "fold", "prediction"]
TOLERANCE = 1e-10


def align_oof(
    frames: dict[str, pd.DataFrame], expected_rows: int = 769
) -> dict[str, pd.DataFrame]:
    """Validate artifact identities and align all rows to E005 filename order.

    Args:
        frames: Named OOF tables including E005, with canonical column names.
        expected_rows: Required number of examples.

    Returns:
        Validated canonical tables in reference order.

    Raises:
        ValueError: If any schema, identity, value, label or fold is invalid.
    """
    if "E005" not in frames:
        raise ValueError("E005 reference is required")
    validated = {}
    for name, frame in frames.items():
        if not frame.columns.is_unique or not set(FIELDS) <= set(frame.columns):
            raise ValueError(f"{name}: required fields are ambiguous or missing")
        if len(frame) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} rows, got {len(frame)}")
        table = frame[FIELDS].copy()
        filenames = table.filename
        if not filenames.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{name}: missing or empty filename")
        if filenames.duplicated().any():
            raise ValueError(f"{name}: duplicate filename")
        for column in FIELDS[1:]:
            try:
                table[column] = pd.to_numeric(table[column], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{name}: invalid numeric {column}") from exc
            if not np.isfinite(table[column].to_numpy(dtype=float)).all():
                raise ValueError(f"{name}: non-finite {column}")
        if not table.fold.isin(range(5)).all():
            raise ValueError(f"{name}: folds must be integers 0..4")
        if set(table.fold) != set(range(5)):
            raise ValueError(f"{name}: all five folds are required")
        table["fold"] = table.fold.astype(int)
        validated[name] = table
    reference = validated["E005"].reset_index(drop=True)
    result = {}
    for name, table in validated.items():
        if set(table.filename) != set(reference.filename):
            raise ValueError(f"{name}: filename-set mismatch")
        aligned = table.set_index("filename").loc[reference.filename].reset_index()
        if not np.allclose(aligned.label, reference.label, rtol=0, atol=TOLERANCE):
            raise ValueError(f"{name}: label mismatch")
        if not np.array_equal(aligned.fold, reference.fold):
            raise ValueError(f"{name}: fold mismatch")
        result[name] = aligned
    return result


def crossfit_ols(
    features: NDArray[np.float64],
    labels: NDArray[np.float64],
    folds: NDArray[np.int64],
) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
    """Fit OLS on other folds and predict each held-out fold.

    Args:
        features: Finite matrix of frozen OOF model predictions.
        labels: Finite target vector.
        folds: Integer fold assignments, containing folds 0 through 4.

    Returns:
        Cross-fitted predictions and fold parameters including matrix rank.

    Raises:
        ValueError: If arrays are malformed or non-finite.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    folds = np.asarray(folds)
    if (
        features.ndim != 2
        or features.shape[1] == 0
        or labels.ndim != 1
        or folds.shape != labels.shape
        or len(features) != len(labels)
        or not np.isfinite(features).all()
        or not np.isfinite(labels).all()
        or set(folds) != set(range(5))
    ):
        raise ValueError("Invalid cross-fitting arrays or folds")
    design = np.column_stack([np.ones(len(labels)), features])
    predictions = np.empty(len(labels))
    parameters = []
    for fold in range(5):
        held_out = folds == fold
        coefficients, _, rank, _ = np.linalg.lstsq(
            design[~held_out], labels[~held_out], rcond=None
        )
        predictions[held_out] = design[held_out] @ coefficients
        parameters.append(
            {
                "fold": fold,
                "training_rows": int((~held_out).sum()),
                "intercept": float(coefficients[0]),
                "coefficients": coefficients[1:].tolist(),
                "rank": int(rank),
            }
        )
    return predictions, parameters


def fixed_blend(
    primary: NDArray[np.float64], secondary: NDArray[np.float64], weight: float
) -> NDArray[np.float64]:
    """Return weight * primary + (1 - weight) * secondary."""
    return weight * primary + (1 - weight) * secondary


def classify_candidate(
    metrics: dict[str, float], reference: dict[str, float], tolerance: float = TOLERANCE
) -> str:
    """Classify strict metric improvements relative to raw E005."""
    if not all(
        np.isfinite(value) for value in [*metrics.values(), *reference.values()]
    ):
        raise ValueError("Cannot classify undefined metrics")
    lower = metrics["rmse"] < reference["rmse"] - tolerance
    higher = (
        metrics["pearson_correlation"] > reference["pearson_correlation"] + tolerance
    )
    return (
        "dominates_e005"
        if lower and higher
        else "tradeoff"
        if lower or higher
        else "worse"
    )


def run_analysis(oof_dir: Path, output_dir: Path) -> dict[str, object]:
    """Analyze frozen artifacts and write reproducible diagnostic reports.

    Args:
        oof_dir: Directory containing the five canonical OOF artifacts.
        output_dir: Separate directory for analysis artifacts.

    Returns:
        JSON-compatible report metadata and candidate metrics.

    Raises:
        ValueError: If artifacts or reference metrics fail validation.
    """
    paths = {name: oof_dir / filename for name, filename in MODELS.items()}
    frames = align_oof({name: pd.read_csv(path) for name, path in paths.items()})
    reference = frames["E005"]
    labels = reference.label.to_numpy(dtype=float)
    folds = reference.fold.to_numpy(dtype=np.int64)
    raw = reference.prediction.to_numpy(dtype=float)
    baseline = regression_metrics(labels, raw)
    if not (
        abs(baseline["rmse"] - 0.835248) < 5e-6
        and abs(baseline["pearson_correlation"] - 0.741944) < 5e-6
    ):
        raise ValueError(f"E005 reference metrics differ materially: {baseline}")
    if output_dir.resolve() == oof_dir.resolve() or any(
        output_dir.resolve() in path.resolve().parents for path in paths.values()
    ):
        raise ValueError("Analysis output must be separate from source OOF artifacts")
    candidates = []
    predictions = {}

    def add(name: str, values: NDArray[np.float64], kind: str) -> None:
        metrics = regression_metrics(labels, values)
        candidates.append(
            {
                "candidate": name,
                "kind": kind,
                **metrics,
                "classification": classify_candidate(metrics, baseline),
            }
        )
        if kind == "crossfit" or name == "E005_clipped":
            predictions[name] = values

    for name, frame in frames.items():
        add(name, frame.prediction.to_numpy(dtype=float), "raw")
    add("E005_clipped", np.clip(raw, 0, 5), "clipping")
    calibrated, calibration = crossfit_ols(raw[:, None], labels, folds)
    for parameter in calibration:
        parameter["slope"] = parameter["coefficients"][0]
    add("E005_calibrated", calibrated, "crossfit")
    add("E005_calibrated_clipped", np.clip(calibrated, 0, 5), "crossfit")
    grid = []
    complementarity = []
    learned = {}
    for name in list(MODELS)[1:]:
        secondary = frames[name].prediction.to_numpy(dtype=float)
        complementarity.append(
            {
                "secondary": name,
                "prediction_correlation": pearson_correlation(raw, secondary),
                "residual_correlation": pearson_correlation(
                    labels - raw, labels - secondary
                ),
            }
        )
        pair_rows = []
        for step in range(21):
            weight = step / 20
            values = fixed_blend(raw, secondary, weight)
            candidate = f"E005_{name}_w{weight:.2f}"
            add(candidate, values, "fixed_grid_diagnostic")
            pair_rows.append(
                {"secondary": name, "weight_e005": weight, **candidates[-1]}
            )
        best_rmse = min(pair_rows, key=lambda row: row["rmse"])
        best_pearson = max(pair_rows, key=lambda row: row["pearson_correlation"])
        for row in pair_rows:
            row["best_by_rmse"] = row is best_rmse
            row["best_by_pearson"] = row is best_pearson
        grid.extend(pair_rows)
        blended, parameters = crossfit_ols(
            np.column_stack([raw, secondary]), labels, folds
        )
        for parameter in parameters:
            parameter["beta_e005"], parameter["beta_secondary"] = parameter[
                "coefficients"
            ]
        learned[name] = parameters
        add(f"E005_{name}_learned", blended, "crossfit")
        add(f"E005_{name}_learned_clipped", np.clip(blended, 0, 5), "crossfit")
    report = {
        "analyzed_models": list(MODELS),
        "row_count": len(reference),
        "fold_ids": sorted(set(folds.tolist())),
        "random_seed": None,
        "determinism": "No stochastic operations; numpy.linalg.lstsq OLS",
        "inputs": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
        "fixed_grid_e005_weights": [step / 20 for step in range(21)],
        "clipping_bounds": [0, 5],
        "ols_solver": "numpy.linalg.lstsq, rcond=None, intercept included",
        "comparison_tolerance": TOLERANCE,
        "label_tolerance": TOLERANCE,
        "reference_metric_tolerance": 5e-6,
        "raw_e005_reference_metrics": baseline,
        "candidate_metrics": candidates,
        "clipping_counts": {
            "below_zero": int((raw < 0).sum()),
            "above_five": int((raw > 5).sum()),
            "changed": int(((raw < 0) | (raw > 5)).sum()),
        },
        "calibration_parameters": calibration,
        "learned_blend_parameters": learned,
        "interpretation": (
            "Grid optima are in-sample diagnostic selections, not unbiased estimates. "
            "Cross-fitted OLS excludes held-out labels, but base OOF models were not "
            "nested: training-fold predictions may depend on held-out-fold labels. "
            "These results are offline diagnostics, not a fully nested stacking "
            "evaluation or a Kaggle winner selection. No combined metric is used."
        ),
        "software_versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(grid).to_csv(output_dir / "blend_grid.csv", index=False)
    pd.DataFrame(complementarity).to_csv(
        output_dir / "complementarity.csv", index=False
    )
    for name, values in predictions.items():
        reference.assign(prediction=values).to_csv(
            output_dir / f"{name}.csv", index=False
        )
    for filename, value in [
        ("report.json", report),
        ("crossfit_parameters.json", {"calibration": calibration, "blends": learned}),
    ]:
        (output_dir / filename).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return report


def print_summary(output_dir: Path) -> None:
    """Print concise candidate, blend-optimum and complementarity tables."""
    report = json.loads((output_dir / "report.json").read_text())
    print(f"E005 clipping counts: {report['clipping_counts']}")
    print(report["interpretation"])
    summary = pd.read_csv(output_dir / "summary.csv")
    grid = pd.read_csv(output_dir / "blend_grid.csv")
    print(summary[summary.kind != "fixed_grid_diagnostic"].to_string(index=False))
    print("\nBest fixed blends (diagnostic; ties use lowest E005 weight):")
    print(grid[grid.best_by_rmse | grid.best_by_pearson].to_string(index=False))
    print("\nComplementarity:")
    print(pd.read_csv(output_dir / "complementarity.csv").to_string(index=False))
    print("\nCandidates dominating raw E005:")
    print(summary[summary.classification == "dominates_e005"].to_string(index=False))
    print(f"\nArtifacts: {output_dir}")
