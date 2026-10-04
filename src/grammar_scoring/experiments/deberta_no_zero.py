"""E008 controlled population ablation using the frozen E005 protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grammar_scoring.config.paths import ARTIFACT_DIR, FEATURES_DIR, TRANSCRIPTS_DIR
from grammar_scoring.evaluation.metrics import pearson_correlation, regression_metrics
from grammar_scoring.experiments import deberta_finetuning as e005
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames

OOF_NAME = "E008_deberta_no_zero_block"


def filter_population(canonical: pd.DataFrame) -> pd.DataFrame:
    """Retain positive labels and preserve original order, text and frozen folds."""
    aligned = e005.align_inputs(
        canonical[["filename", "label"]],
        canonical[["filename", "text"]].assign(split="train"),
        canonical[["filename", "label", "fold"]],
    )
    if "split" in canonical and not canonical.split.eq("train").all():
        raise ValueError("E008 accepts only train records")
    retained = aligned.loc[aligned.label > 0.0].reset_index(drop=True)
    if len(retained) != 732 or len(aligned) - len(retained) != 37:
        raise ValueError("E008 requires canonical population 769 -> 732 (37 removed)")
    if set(retained.fold) != set(range(5)):
        raise ValueError("All frozen folds 0..4 must remain present")
    original = canonical.set_index("filename").loc[retained.filename, "fold"]
    if not np.array_equal(original, retained.fold):
        raise ValueError("Retained folds differ from canonical E005 folds")
    return retained


def validate_oof(oof: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Require exactly 732 positive-label, finite predictions on frozen identities."""
    if len(oof) != 732 or not oof.label.gt(0).all():
        raise ValueError("E008 OOF requires 732 rows with positive labels")
    if set(oof.fold) != set(range(5)):
        raise ValueError("OOF requires all folds 0..4")
    return e005._validate_predictions(oof.to_dict("records"), expected)


def align_baseline(baseline: pd.DataFrame, retained: pd.DataFrame) -> pd.DataFrame:
    """Restrict canonical E005 OOF by label and align by filename, checking targets."""
    _validate_filenames(baseline)
    if len(baseline) != 769:
        raise ValueError("Canonical E005 OOF requires 769 rows")
    if baseline.label.dtype.kind not in "iuf" or not np.isfinite(baseline.label).all():
        raise ValueError("E005 labels must be finite numeric values")
    return validate_oof(baseline.loc[baseline.label > 0.0], retained)


def _comparison(frame: pd.DataFrame) -> dict[str, Any]:
    old = regression_metrics(frame.label, frame.e005_prediction)
    new = regression_metrics(frame.label, frame.prediction)
    delta = new["rmse"] - old["rmse"]
    return {
        "n": len(frame),
        "e005_rmse": old["rmse"],
        "e008_rmse": new["rmse"],
        "rmse_delta": delta,
        "relative_rmse_change_percent": (
            100 * delta / old["rmse"] if old["rmse"] else None
        ),
        "e005_pearson": old["pearson_correlation"],
        "e008_pearson": new["pearson_correlation"],
        "pearson_delta": new["pearson_correlation"] - old["pearson_correlation"],
    }


def score_groups(joined: pd.DataFrame) -> pd.DataFrame:
    """Report disjoint low (<3), mid (3..<4), high (>=4) positive target groups."""
    rows = []
    for name, mask in (
        ("low", joined.label < 3),
        ("mid", (joined.label >= 3) & (joined.label < 4)),
        ("high", joined.label >= 4),
    ):
        group = joined.loc[mask]
        row: dict[str, Any] = {"group": name, "n": len(group)}
        if len(group):
            row.update(_comparison(group))
            row["e005_mean_prediction"] = float(group.e005_prediction.mean())
            row["e008_mean_prediction"] = float(group.prediction.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics(
    oof: pd.DataFrame,
    baseline: pd.DataFrame,
    retained: pd.DataFrame,
    selected: list[dict[str, Any]],
    directory: Path,
) -> dict[str, Any]:
    """Persist paired pooled/fold comparisons, correlations and ranked errors."""
    oof = validate_oof(oof, retained)
    baseline = align_baseline(baseline, retained)
    joined = oof.merge(
        baseline[["filename", "prediction"]].rename(
            columns={"prediction": "e005_prediction"}
        ),
        on="filename",
        validate="one_to_one",
    )
    if len(joined) != 732:
        raise ValueError("Comparison must match exactly 732 filenames")
    joined["e005_residual"] = joined.label - joined.e005_prediction
    joined["residual"] = joined.label - joined.prediction
    joined["e005_abs_error"] = joined.e005_residual.abs()
    joined["e008_abs_error"] = joined.residual.abs()
    joined["abs_error_improvement"] = joined.e005_abs_error - joined.e008_abs_error
    epochs = {row["fold"]: row["selected_epoch"] for row in selected}
    per_fold = [
        {"fold": fold, **_comparison(group), "selected_epoch": epochs[fold]}
        for fold, group in joined.groupby("fold", sort=True)
    ]
    report = {
        **_comparison(joined),
        "delta_direction": "E008 minus E005; negative RMSE / positive Pearson improves",
        "per_fold": per_fold,
        "folds_improving_rmse": sum(row["rmse_delta"] < 0 for row in per_fold),
        "folds_improving_pearson": sum(row["pearson_delta"] > 0 for row in per_fold),
        "folds_improving_both": sum(
            row["rmse_delta"] < 0 and row["pearson_delta"] > 0 for row in per_fold
        ),
        "prediction_correlation": pearson_correlation(
            joined.e005_prediction, joined.prediction
        ),
        "residual_correlation": pearson_correlation(
            joined.e005_residual, joined.residual
        ),
    }
    directory.mkdir(parents=True, exist_ok=True)
    e005._write_json(directory / "e005_comparison.json", report)
    pd.DataFrame(per_fold).to_csv(directory / "fold_comparison.csv", index=False)
    score_groups(joined).to_csv(directory / "score_groups.csv", index=False)
    joined.to_csv(directory / "paired_errors.csv", index=False)
    for name, ascending in (("improvements", False), ("deteriorations", True)):
        joined.sort_values("abs_error_improvement", ascending=ascending).head(
            20
        ).to_csv(directory / f"largest_{name}.csv", index=False)
    return report


def run_experiment(
    canonical: pd.DataFrame,
    baseline: pd.DataFrame,
    artifact_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Train CUDA E008 on positive labels, then stop after paired OOF diagnostics."""
    retained = filter_population(canonical)
    e005._validate_predictions(baseline.to_dict("records"), canonical)
    align_baseline(baseline, retained)  # Fail before spending GPU time.
    directory = artifact_dir / "experiments/E008"
    e005._write_json(
        directory / "resolved_config.json",
        {
            "experiment": "E008",
            "population_criterion": "label > 0.0",
            "training_protocol": asdict(e005.CONFIG),
            "prediction_gate": "OOF diagnostics only",
        },
    )
    e005._write_json(
        directory / "population.json",
        {
            "experiment": "E008",
            "criterion": "label > 0.0",
            "canonical_n": 769,
            "retained_n": 732,
            "removed_n": 37,
            "canonical_input_sha256": e005.experiment_identity(canonical),
            "e005_oof_sha256": e005._digest(baseline.to_dict("records")),
            "frozen_e005_protocol": asdict(e005.CONFIG),
        },
    )
    summary = e005._run_aligned_experiment(
        retained,
        artifact_dir,
        experiment="E008",
        oof_name=OOF_NAME,
        device="cuda",
        resume=resume,
    )
    oof = pd.read_csv(artifact_dir / "oof" / f"{OOF_NAME}.csv")
    summary["e005_comparison"] = diagnostics(
        oof, baseline, retained, summary["per_fold"], directory
    )
    e005._write_json(directory / "summary.json", summary)
    return summary


def main() -> None:
    """Run full E008 or isolated bounded CUDA smoke checks; never infer test data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument(
        "--fold-path", type=Path, default=FEATURES_DIR / "train_folds.csv"
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--e005-oof", type=Path, default=ARTIFACT_DIR / "oof/E005_deberta_finetuned.csv"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    canonical = e005.load_inputs(args.transcript_dir, args.fold_path)
    baseline = pd.read_csv(args.e005_oof)
    retained = filter_population(canonical)
    align_baseline(baseline, retained)
    if args.smoke:
        from grammar_scoring.experiments.deberta_smoke import run_smoke

        print("SMOKE TEST ONLY — NOT E008 RESULT", flush=True)
        run_smoke(
            args.transcript_dir,
            args.fold_path,
            args.artifact_dir / "smoke/E008",
            frame=retained,
            experiment="E008",
        )
    else:
        result = run_experiment(
            canonical, baseline, args.artifact_dir, resume=args.resume
        )
        print(e005._json_safe(result))
