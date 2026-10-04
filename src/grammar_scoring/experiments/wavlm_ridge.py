"""E009 train-only frozen audio representation and fixed Ridge OOF diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grammar_scoring.config.paths import ARTIFACT_DIR, FEATURES_DIR
from grammar_scoring.data.audio import read_audio_metadata
from grammar_scoring.data.dataset import (
    TRAIN_AUDIO_DIR,
    _audio_path,
    load_train_dataframe,
)
from grammar_scoring.evaluation.metrics import pearson_correlation, regression_metrics
from grammar_scoring.evaluation.validation import validate_cv_folds
from grammar_scoring.experiments.deberta_finetuning import _write_json
from grammar_scoring.experiments.deberta_no_zero import validate_oof
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.features.wavlm import (
    CHUNK_SAMPLES,
    MODEL,
    SAMPLE_RATE,
    FrozenWavLM,
    chunk_sample_counts,
    extract_embeddings,
    validate_embeddings,
)


def retained_population(train: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Filter exclusively by positive labels, preserving canonical E005/E008 folds."""
    for frame in (train, folds):
        _validate_filenames(frame)
        if "split" in frame and not frame.split.eq("train").all():
            raise ValueError("E009 accepts only train rows")
    if len(train) != 769 or set(train.filename) != set(folds.filename):
        raise ValueError("Canonical train and folds must match all 769 filenames")
    labels = train.label.to_numpy()
    if labels.dtype.kind not in "iuf" or not np.isfinite(labels).all():
        raise ValueError("Labels must be finite numeric values")
    if not ((labels >= 0) & (labels <= 5)).all():
        raise ValueError("Labels must be within [0, 5]")
    assigned = folds.set_index("filename").loc[train.filename]
    if "label" in assigned and not np.array_equal(assigned.label, labels):
        raise ValueError("Frozen fold labels differ from canonical train")
    validate_cv_folds(labels, assigned.fold, 5)
    if not np.array_equal(np.bincount(assigned.fold), [154, 154, 154, 154, 153]):
        raise ValueError("Unexpected canonical E005 fold sizes")
    result = train[["filename", "label"]].reset_index(drop=True).copy()
    result["fold"] = assigned.fold.to_numpy()
    result = result.loc[result.label > 0].reset_index(drop=True)
    if len(result) != 732 or set(result.fold) != set(range(5)):
        raise ValueError("E009 requires 732 positive rows and all frozen folds")
    return result


def evaluate_ridge(
    embeddings: NDArray, metadata: pd.DataFrame, retained: pd.DataFrame
) -> pd.DataFrame:
    """Fit fresh training-only scalers and Ridge(alpha=1.0) in each frozen fold."""
    validate_embeddings(embeddings, metadata, retained)
    predictions = np.full(len(retained), np.nan)
    coverage = np.zeros(len(retained), dtype=int)
    for fold in range(5):
        valid = retained.fold.to_numpy() == fold
        pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipeline.fit(embeddings[~valid], retained.label.to_numpy()[~valid])
        predictions[valid] = pipeline.predict(embeddings[valid])
        coverage[valid] += 1
    if not (coverage == 1).all() or not np.isfinite(predictions).all():
        raise RuntimeError("Expected exactly one finite OOF prediction per row")
    oof = retained.copy()
    oof["prediction"] = predictions
    oof["residual"] = oof.label - oof.prediction
    oof["abs_error"] = oof.residual.abs()
    return validate_oof(oof, retained)


def compare_oof(
    oof: pd.DataFrame, baseline: pd.DataFrame, retained: pd.DataFrame
) -> dict[str, Any]:
    """Compare aligned E008/E009 OOF using label-minus-prediction residuals."""
    current = validate_oof(oof, retained)
    previous = validate_oof(baseline, retained)

    def score(valid: NDArray) -> dict[str, Any]:
        actual = retained.label.to_numpy()[valid]
        old = previous.prediction.to_numpy()[valid]
        new = current.prediction.to_numpy()[valid]
        a, b = regression_metrics(actual, old), regression_metrics(actual, new)
        return {
            "n": int(valid.sum()),
            "e008": a,
            "e009": b,
            "rmse_delta": b["rmse"] - a["rmse"],
            "pearson_delta": b["pearson_correlation"] - a["pearson_correlation"],
            "prediction_correlation": pearson_correlation(old, new),
            "residual_correlation": pearson_correlation(actual - old, actual - new),
        }

    return {
        "experiment": "E009",
        "alpha": 1.0,
        "seed": 42,
        "delta_direction": "E009 minus E008",
        **score(np.ones(len(retained), dtype=bool)),
        "per_fold": [
            {"fold": fold, **score(retained.fold.to_numpy() == fold)}
            for fold in range(5)
        ],
    }


def e009_diagnostics(oof: pd.DataFrame, retained: pd.DataFrame) -> dict[str, Any]:
    """Report standalone E009 pooled/fold metrics without baseline statistics."""
    aligned = validate_oof(oof, retained)
    return {
        "experiment": "E009",
        "alpha": 1.0,
        "seed": 42,
        "n": len(aligned),
        "e009": regression_metrics(aligned.label, aligned.prediction),
        "per_fold": [
            {
                "fold": int(fold),
                "n": len(group),
                "e009": regression_metrics(group.label, group.prediction),
            }
            for fold, group in aligned.groupby("fold", sort=True)
        ],
        "e008_comparison_status": "not_run",
        "e008_oof_path": None,
    }


def select_smoke_recordings(
    retained: pd.DataFrame, audio_dir: Path
) -> tuple[list[str], list[str]]:
    """Choose duration coverage deterministically, reporting missing buckets."""
    durations = {
        name: read_audio_metadata(_audio_path(audio_dir, name)).duration_seconds
        for name in retained.filename
    }
    selected, missing = [], []
    for bucket, lower in (
        ("<20 seconds", 0),
        ("20-40 seconds", 20),
        (">40 seconds", 40),
    ):
        candidates = [
            name
            for name, duration in durations.items()
            if (
                0 < duration < 20
                if lower == 0
                else 20 <= duration <= 40
                if lower == 20
                else duration > 40
            )
        ]
        if candidates:
            # Longest in each bucket exercises as much chunking as possible.
            selected.append(max(candidates, key=lambda name: durations[name]))
        else:
            missing.append(bucket)
            print(f"E009 smoke: missing duration bucket {bucket}", flush=True)
    while len(selected) < min(3, len(durations)):
        candidates = [name for name in durations if name not in selected]
        # Maximize distance from covered durations, including boundary recordings.
        chosen = max(
            candidates,
            key=lambda name: min(
                (abs(durations[name] - durations[other]) for other in selected),
                default=durations[name],
            ),
        )
        selected.append(chosen)
        print(f"E009 smoke: fallback {chosen} ({durations[chosen]:.6f}s)", flush=True)
    return selected, missing


def run_smoke(
    retained: pd.DataFrame, audio_dir: Path, encoder: FrozenWavLM, output: Path
) -> dict[str, Any]:
    """Persist duration coverage, chunk details, runtime metadata and invariance."""
    import torch

    from grammar_scoring.features.acoustic import load_waveform

    filenames, missing = select_smoke_recordings(retained, audio_dir)
    cuda = torch.device(encoder.device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(encoder.device)
    rows = []
    reference_encoder = FrozenWavLM(
        encoder.device, 1, model=encoder.model, processor=encoder.processor
    )
    for filename in filenames:
        waveform = load_waveform(_audio_path(audio_dir, filename))
        vector, count = encoder.encode(waveform)
        if vector.shape != (768,) or not np.isfinite(vector).all():
            raise ValueError("Invalid smoke embedding")
        reference, _ = reference_encoder.encode(waveform)
        if not np.allclose(vector, reference, atol=1e-5, rtol=1e-4):
            raise ValueError("Batch-size invariance smoke check failed")
        counts = chunk_sample_counts(len(waveform), encoder.minimum_samples)
        frames = encoder.model._get_feat_extract_output_lengths(torch.tensor(counts))
        rows.append(
            {
                "filename": filename,
                "duration_seconds": len(waveform) / SAMPLE_RATE,
                "waveform_samples": len(waveform),
                "num_chunks": count,
                "chunk_sample_counts": counts,
                "valid_hidden_frame_counts": frames.tolist(),
                "embedding_dimension": len(vector),
                "finite": bool(np.isfinite(vector).all()),
                "batch_size_invariance_passed": True,
            }
        )
    report = {
        "smoke_only": True,
        "missing_duration_buckets": missing,
        "model": MODEL,
        "resolved_model_revision": encoder.revision,
        "sample_rate": SAMPLE_RATE,
        "chunk_duration_seconds": CHUNK_SAMPLES / SAMPLE_RATE,
        "minimum_chunk_samples": encoder.minimum_samples,
        "device": encoder.device,
        "dtype": "float32",
        "chunk_batch_size": encoder.batch_size,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(encoder.device)) if cuda else None
        ),
        "recordings": rows,
    }
    _write_json(output, report)
    print(report, flush=True)
    return report


def main() -> None:
    """Run bounded extraction smoke or full cached extraction and fixed Ridge OOF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-path", type=Path, default=FEATURES_DIR / "train_folds.csv"
    )
    parser.add_argument("--audio-dir", type=Path, default=TRAIN_AUDIO_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--e008-oof",
        type=Path,
        default=None,
        help="Optional E008 OOF for strict paired comparison; ignored by --smoke",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-batch-size", type=int, default=4)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    retained = retained_population(load_train_dataframe(), pd.read_csv(args.fold_path))
    if args.smoke:
        encoder = FrozenWavLM(
            args.device, args.chunk_batch_size, revision=args.revision
        )
        run_smoke(
            retained,
            args.audio_dir,
            encoder,
            args.artifact_dir / "smoke/E009/summary.json",
        )
        return
    baseline = None
    if args.e008_oof is not None:
        if not args.e008_oof.is_file():
            raise FileNotFoundError(f"E008 OOF file does not exist: {args.e008_oof}")
        baseline = pd.read_csv(args.e008_oof)
        validate_oof(baseline, retained)
    encoder = FrozenWavLM(args.device, args.chunk_batch_size, revision=args.revision)
    matrix, metadata = extract_embeddings(
        retained, args.audio_dir, args.artifact_dir / "embeddings/E009", encoder
    )
    oof = evaluate_ridge(matrix, metadata, retained)
    if baseline is None:
        report = e009_diagnostics(oof, retained)
    else:
        report = compare_oof(oof, baseline, retained)
        report["e008_comparison_status"] = "completed"
        report["e008_oof_path"] = str(args.e008_oof)
    output = args.artifact_dir / "oof/E009_wavlm_ridge.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output, index=False)
    _write_json(args.artifact_dir / "experiments/E009/diagnostics.json", report)
    print(report)
