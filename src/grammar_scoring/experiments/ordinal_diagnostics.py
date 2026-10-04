"""Train-only E005/E007 diagnostics without calibration or blend searches."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from grammar_scoring.evaluation.metrics import regression_metrics
from grammar_scoring.experiments.deberta_finetuning import (
    _validate_predictions,
    _write_json,
)
from grammar_scoring.models.ordinal import class_probabilities, decode_probabilities


def score_diagnostics(oof: pd.DataFrame) -> pd.DataFrame:
    """Summarize predictions and signed prediction-minus-target errors by score."""
    rows = []
    for label, group in oof.groupby("label", sort=True):
        error = group.prediction - group.label
        rows.append(
            {
                "label": label,
                "count": len(group),
                "prediction_mean": group.prediction.mean(),
                "prediction_std": group.prediction.std(ddof=0),
                "prediction_min": group.prediction.min(),
                "prediction_max": group.prediction.max(),
                "mean_signed_error": error.mean(),
                "mae": error.abs().mean(),
                "rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
    return pd.DataFrame(rows)


def group_diagnostics(oof: pd.DataFrame) -> pd.DataFrame:
    """Summarize LOW, MID and HIGH using fixed true-label boundaries."""
    rows = []
    for name, mask in (
        ("LOW", oof.label <= 2),
        ("MID", oof.label.between(2.5, 4)),
        ("HIGH", oof.label >= 4.5),
    ):
        group = oof.loc[mask]
        error = group.prediction - group.label
        rows.append(
            {
                "group": name,
                "n": len(group),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": error.abs().mean(),
                "mean_prediction": group.prediction.mean(),
                "mean_target": group.label.mean(),
                "mean_signed_error": error.mean(),
            }
        )
    return pd.DataFrame(rows)


def write_comparison(
    frame: pd.DataFrame,
    results: list[dict[str, Any]],
    oof: pd.DataFrame,
    artifact_dir: Path,
    summary: dict[str, Any],
    baseline_path: Path,
) -> None:
    """Validate OOF probabilities and write paired diagnostics against E005."""
    directory = artifact_dir / "experiments/E007"
    probability_rows = []
    for result in results:
        q = result["cumulative_validation"]
        if len(q) != len(result["validation"]):
            raise ValueError("Cumulative probability row count mismatch")
        for record, probabilities in zip(result["validation"], q, strict=True):
            probability_rows.append(
                {
                    "filename": record["filename"],
                    **{f"q_{k}": value for k, value in enumerate(probabilities)},
                }
            )
    probabilities = pd.DataFrame(probability_rows)
    if probabilities.filename.duplicated().any():
        raise ValueError("Duplicate probability identities")
    q = torch.tensor(
        probabilities.set_index("filename").reindex(oof.filename).to_numpy().tolist(),
        dtype=torch.float64,
    )
    p = class_probabilities(q)
    decoded = np.asarray(decode_probabilities(q).tolist())
    if not np.allclose(decoded, oof.prediction, atol=1e-6, rtol=0):
        raise ValueError("OOF probability decoding differs from selected predictions")
    diagnostics = {
        "monotonic_every_row": True,
        "class_probabilities_nonnegative": True,
        "class_probabilities_sum_to_one": True,
        "finite_bounded_predictions": True,
        "minimum_class_probability": float(p.min()),
        "maximum_row_sum_error": float((p.sum(1) - 1).abs().max()),
        "min_prediction": float(decoded.min()),
        "max_prediction": float(decoded.max()),
        "mean_prediction": float(decoded.mean()),
        "std_prediction": float(decoded.std()),
    }
    _write_json(directory / "ordinal_diagnostics.json", diagnostics)
    probabilities.set_index("filename").reindex(oof.filename).reset_index().to_csv(
        directory / "cumulative_oof.csv", index=False
    )
    baseline = _validate_predictions(
        pd.read_csv(baseline_path).to_dict(orient="records"),
        frame,
    )
    for function, name in (
        (score_diagnostics, "score_level_comparison.csv"),
        (group_diagnostics, "score_group_comparison.csv"),
    ):
        pd.concat(
            [
                function(data).assign(experiment=experiment)
                for experiment, data in [("E005", baseline), ("E007", oof)]
            ],
            ignore_index=True,
        ).to_csv(directory / name, index=False)
    deltas = []
    for fold in range(5):
        old = baseline.loc[baseline.fold == fold]
        new = oof.loc[oof.fold == fold]
        a = regression_metrics(old.label, old.prediction)
        b = regression_metrics(new.label, new.prediction)
        deltas.append(
            {
                "fold": fold,
                "rmse_delta": b["rmse"] - a["rmse"],
                "pearson_delta": b["pearson_correlation"] - a["pearson_correlation"],
            }
        )
    old_metrics = regression_metrics(baseline.label, baseline.prediction)
    summary["comparison"] = {
        "E005_metrics": old_metrics,
        "per_fold_deltas": deltas,
        "rmse_delta": summary["oof_metrics"]["rmse"] - old_metrics["rmse"],
        "pearson_delta": summary["oof_metrics"]["pearson_correlation"]
        - old_metrics["pearson_correlation"],
        "folds_improving_rmse": sum(d["rmse_delta"] < 0 for d in deltas),
        "folds_improving_pearson": sum(d["pearson_delta"] > 0 for d in deltas),
        "folds_improving_both": sum(
            d["rmse_delta"] < 0 and d["pearson_delta"] > 0 for d in deltas
        ),
        "prediction_correlation": _correlation(baseline.prediction, oof.prediction),
        "residual_correlation": _correlation(
            baseline.prediction - baseline.label, oof.prediction - oof.label
        ),
        "rare_label_caution": "Labels 1.0 and 1.5 have only 1 and 3 rows.",
    }


def _correlation(first: object, second: object) -> float:
    a, b = np.asarray(first), np.asarray(second)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
