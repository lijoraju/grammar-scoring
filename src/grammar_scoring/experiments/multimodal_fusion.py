"""E011a frozen cross-fitted text/audio representation fusion."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from grammar_scoring.config.paths import ARTIFACT_DIR, FEATURES_DIR, TRANSCRIPTS_DIR
from grammar_scoring.evaluation.metrics import pearson_correlation, regression_metrics
from grammar_scoring.experiments import deberta_finetuning as e005
from grammar_scoring.experiments.deberta_no_zero import filter_population, validate_oof
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames

DIMENSION = 768


def align_metadata(metadata: pd.DataFrame, expected: pd.DataFrame) -> NDArray:
    """Return filename-based indices after checking exact population identity."""
    _validate_filenames(metadata)
    if len(metadata) != len(expected) or set(metadata.filename) != set(
        expected.filename
    ):
        raise ValueError("Embedding filenames must match exactly; missing/extra rows")
    positions = pd.Series(np.arange(len(metadata)), index=metadata.filename)
    indices = positions.loc[expected.filename].to_numpy()
    aligned = metadata.iloc[indices]
    for key in ("label", "fold"):
        if not np.array_equal(aligned[key], expected[key]):
            raise ValueError(f"Embedding {key} mismatch")
    return indices


def validate_embeddings(
    matrix: NDArray, metadata: pd.DataFrame, expected: pd.DataFrame, *, text: bool
) -> NDArray:
    """Validate finite 768-dimensional features and align audio by filename."""
    if matrix.shape != (len(expected), DIMENSION):
        raise ValueError("Expected embedding shape (population size, 768)")
    if matrix.dtype.kind not in "iuf" or not np.isfinite(matrix).all():
        raise ValueError("Embeddings must contain finite real values")
    indices = align_metadata(metadata, expected)
    if text:
        if not np.array_equal(indices, np.arange(len(expected))):
            raise ValueError("Text cache ordering mismatch")
        if "checkpoint_fold" not in metadata or not np.array_equal(
            metadata.fold, metadata.checkpoint_fold
        ):
            raise ValueError("Leakage guard: fold != checkpoint_fold")
    return matrix[indices]


def checkpoint_provenance(frame: pd.DataFrame, model_dir: Path) -> dict[str, Any]:
    """Verify selected E008 checkpoints and their original training partitions."""
    mapping = {}
    identity = e005.experiment_identity(frame)
    for fold in range(5):
        directory = model_dir / f"fold_{fold}"
        result = e005.validate_completed_fold(directory, frame, fold, identity)
        mapping[str(fold)] = {
            "path": str(directory / "best.pt"),
            "sha256": result["checkpoint_sha256"],
            "model": result["model"],
            "identity": result["identity"],
        }
    return mapping


def extraction_config(frame: pd.DataFrame, mapping: dict[str, Any]) -> dict[str, Any]:
    """Describe all semantic inputs that invalidate an embedding cache."""
    return {
        "experiment": "E011a",
        "population_rule": "label > 0.0",
        "population_size": len(frame),
        "inputs_sha256": e005._digest(frame.to_dict("records")),
        "model": e005.CONFIG.model_name,
        "tokenizer": e005.CONFIG.model_name,
        "max_length": 256,
        "pooling": "final hidden state attention-mask-aware mean; no normalization",
        "dimension": DIMENSION,
        "checkpoint_mapping": mapping,
        "ordering": "canonical training input order after positive-label filtering",
        "seed": 42,
        "dtype": "float32",
    }


def extract_text(
    frame: pd.DataFrame,
    directory: Path,
    config: dict[str, Any],
    *,
    device: str = "cuda",
    batch_size: int = 8,
    cache_only: bool = False,
) -> tuple[NDArray, pd.DataFrame, dict[str, Any]]:
    """Extract only fold-k rows with checkpoint k, or reuse a verified cache."""
    paths = [
        directory / name
        for name in ("text_embeddings.npy", "text_metadata.csv", "extraction.json")
    ]
    if any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths):
            raise ValueError("Incomplete E011a text cache")
        report = json.loads(paths[2].read_text())
        if report.get("configuration") != config or report.get("fingerprint") != (
            e005._digest(config)
        ):
            raise ValueError("Incompatible E011a cache provenance")
        for path in paths[:2]:
            if report.get("checksums", {}).get(path.name) != e005._file_digest(path):
                raise ValueError("Text cache checksum mismatch")
        matrix = np.load(paths[0], allow_pickle=False)
        metadata = pd.read_csv(paths[1])
        validate_embeddings(matrix, metadata, frame, text=True)
        return matrix, metadata, report
    if cache_only:
        raise FileNotFoundError("Validated E011a text cache required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from grammar_scoring.inference.deberta import load_checkpoint
    from grammar_scoring.models.deberta_regressor import (
        DebertaRegressor,
        masked_mean_pool,
    )

    e005.seed_everything()
    torch.use_deterministic_algorithms(True)
    matrix = np.full((len(frame), DIMENSION), np.nan, dtype=np.float32)
    metadata = frame[["filename", "label", "fold"]].copy()
    metadata["checkpoint_fold"] = -1
    for fold in range(5):
        provenance = config["checkpoint_mapping"][str(fold)]
        saved = provenance["model"]
        # Construct the exact saved architecture; no pretrained encoder weights.
        backbone_config = AutoConfig.for_model(**saved["configuration"])
        model = DebertaRegressor(AutoModel.from_config(backbone_config))
        load_checkpoint(model, Path(provenance["path"]), torch.device("cpu"))
        model.to(device).float().eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer"],
            revision=saved.get("revision"),
        )
        indices = np.flatnonzero(frame.fold.to_numpy() == fold)
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                rows = indices[start : start + batch_size]
                inputs = tokenizer(
                    frame.iloc[rows].text.tolist(),
                    max_length=256,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                hidden = model.backbone(**inputs).last_hidden_state
                matrix[rows] = (
                    masked_mean_pool(hidden, inputs["attention_mask"]).cpu().numpy()
                )
                metadata.loc[rows, "checkpoint_fold"] = fold
        del model
        print(f"E011a extracted text fold {fold}: {len(indices)} rows", flush=True)
    validate_embeddings(matrix, metadata, frame, text=True)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(paths[0], matrix)
    metadata.to_csv(paths[1], index=False)
    report = {
        "configuration": config,
        "fingerprint": e005._digest(config),
        "checksums": {p.name: e005._file_digest(p) for p in paths[:2]},
        "runtime": {"device": device, "batch_size": batch_size},
        "validation": "shape, finite, identities, checkpoint folds, provenance, hashes",
    }
    e005._write_json(paths[2], report)
    return matrix, metadata, report


def evaluate(frame: pd.DataFrame, modalities: list[NDArray]) -> pd.DataFrame:
    """Fit independent modality scalers and fixed Ridge within each frozen fold."""
    for matrix in modalities:
        if matrix.shape != (len(frame), DIMENSION) or not np.isfinite(matrix).all():
            raise ValueError("Invalid modality dimensions or finite values")
    predictions = np.full(len(frame), np.nan)
    coverage = np.zeros(len(frame), dtype=int)
    for fold in range(5):
        valid = frame.fold.to_numpy() == fold
        training, validation = [], []
        for matrix in modalities:
            scaler = StandardScaler().fit(matrix[~valid])
            training.append(scaler.transform(matrix[~valid]))
            validation.append(scaler.transform(matrix[valid]))
        model = Ridge(alpha=1.0).fit(
            np.concatenate(training, axis=1), frame.label.to_numpy()[~valid]
        )
        predictions[valid] = model.predict(np.concatenate(validation, axis=1))
        coverage[valid] += 1
    if not (coverage == 1).all() or not np.isfinite(predictions).all():
        raise ValueError("OOF requires exactly one finite prediction per retained row")
    result = frame[["filename", "label", "fold"]].assign(prediction=predictions)
    result["residual"] = result.label - result.prediction
    result["abs_error"] = result.residual.abs()
    return validate_oof(result, frame)


def metrics(oof: pd.DataFrame) -> dict[str, Any]:
    """Compute pooled and per-fold regression metrics from raw predictions."""
    return {
        "pooled": regression_metrics(oof.label, oof.prediction),
        "per_fold": [
            {
                "fold": int(fold),
                "n": len(group),
                **regression_metrics(group.label, group.prediction),
            }
            for fold, group in oof.groupby("fold", sort=True)
        ],
    }


def comparison(fusion: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    """Report raw paired E008 comparisons and fixed true-label score bands."""
    baseline = validate_oof(baseline, fusion)

    def paired(mask: NDArray) -> dict[str, Any]:
        old, new = baseline.loc[mask], fusion.loc[mask]
        if not len(new):
            return {
                "n": 0,
                "e008": None,
                "fusion": None,
                "mean_e008_prediction": None,
                "mean_fusion_prediction": None,
            }
        a = regression_metrics(new.label, old.prediction)
        b = regression_metrics(new.label, new.prediction)
        return {
            "n": len(new),
            "e008": a,
            "fusion": b,
            "rmse_delta": b["rmse"] - a["rmse"],
            "pearson_delta": b["pearson_correlation"] - a["pearson_correlation"],
            "mean_e008_prediction": float(old.prediction.mean()),
            "mean_fusion_prediction": float(new.prediction.mean()),
        }

    folds = [
        {"fold": fold, **paired(fusion.fold.to_numpy() == fold)} for fold in range(5)
    ]
    better = int(
        (
            (fusion.label - fusion.prediction).abs()
            < (baseline.label - baseline.prediction).abs()
        ).sum()
    )
    return {
        **paired(np.ones(len(fusion), dtype=bool)),
        "per_fold": folds,
        "delta_direction": "fusion minus E008",
        "folds_improving_rmse": sum(r["rmse_delta"] < 0 for r in folds),
        "folds_improving_pearson": sum(r["pearson_delta"] > 0 for r in folds),
        "prediction_correlation": pearson_correlation(
            baseline.prediction, fusion.prediction
        ),
        "residual_correlation": pearson_correlation(
            baseline.label - baseline.prediction, fusion.label - fusion.prediction
        ),
        "samples_lower_absolute_error": better,
        "percent_lower_absolute_error": 100 * better / len(fusion),
        "score_bands": {
            "low": paired((fusion.label < 3).to_numpy()),
            "mid": paired(((fusion.label >= 3) & (fusion.label < 4)).to_numpy()),
            "high": paired((fusion.label >= 4).to_numpy()),
        },
    }


def run_experiment(
    frame: pd.DataFrame,
    text: NDArray,
    audio: NDArray,
    baseline: pd.DataFrame,
    artifact_dir: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Persist three fixed OOF models, metrics and uninterpreted diagnostics."""
    baseline = validate_oof(baseline, frame)
    outputs = {
        name: evaluate(frame, arrays)
        for name, arrays in (
            ("text", [text]),
            ("audio", [audio]),
            ("fusion", [text, audio]),
        )
    }
    scores = {f"E011a-{name}": metrics(oof) for name, oof in outputs.items()}
    audio_score = scores["E011a-audio"]["pooled"]
    discrepancy = {
        "reference_rmse": 0.899787,
        "reference_pearson": 0.642864,
        "rmse_delta": audio_score["rmse"] - 0.899787,
        "pearson_delta": audio_score["pearson_correlation"] - 0.642864,
        "material_threshold_absolute": 0.01,
    }
    discrepancy["warning"] = bool(
        abs(discrepancy["rmse_delta"]) > 0.01
        or abs(discrepancy["pearson_delta"]) > 0.01
    )
    if discrepancy["warning"]:
        warnings.warn("E011a audio control differs materially from E009", stacklevel=2)
    config = {
        "experiment": "E011a",
        "population_size": len(frame),
        "population_rule": "label > 0.0",
        "seed": 42,
        "text_dimension": 768,
        "audio_dimension": 768,
        "fusion_dimension": 1536,
        "scaler": "independent StandardScaler per modality, training fold rows only",
        "ridge_alpha": 1.0,
        "provenance": provenance,
    }
    report = {
        **config,
        "metrics": scores,
        "e008_comparison": comparison(outputs["fusion"], baseline),
        "audio_sanity": discrepancy,
    }
    for name, oof in outputs.items():
        path = artifact_dir / "oof" / f"E011a_{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        oof.to_csv(path, index=False)
    for name, value in (
        ("config", config),
        ("metrics", scores),
        ("diagnostics", report),
    ):
        e005._write_json(artifact_dir / "experiments/E011a" / f"{name}.json", value)
    return report


def main() -> None:
    """Run E011a extraction and/or fixed OOF evaluation using existing artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument(
        "--fold-path", type=Path, default=FEATURES_DIR / "train_folds.csv"
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--e008-model-dir", type=Path)
    parser.add_argument("--e008-oof", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--text-cache-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    frame = filter_population(e005.load_inputs(args.transcript_dir, args.fold_path))
    mapping = checkpoint_provenance(
        frame, args.e008_model_dir or args.artifact_dir / "models/E008"
    )
    config = extraction_config(frame, mapping)
    audio_dir = args.audio_dir or args.artifact_dir / "embeddings/E009"
    if not args.extract_only:
        baseline_path = args.e008_oof or (
            args.artifact_dir / "oof/E008_deberta_no_zero_block.csv"
        )
        baseline = validate_oof(pd.read_csv(baseline_path), frame)
        audio_metadata = pd.read_csv(audio_dir / "train_metadata.csv")
        audio = validate_embeddings(
            np.load(audio_dir / "train_embeddings.npy", allow_pickle=False),
            audio_metadata,
            frame,
            text=False,
        )
    text, _, extraction = extract_text(
        frame,
        args.text_cache_dir or args.artifact_dir / "embeddings/E011a",
        config,
        device=args.device,
        batch_size=args.batch_size,
        cache_only=args.cache_only,
    )
    if not args.extract_only:
        audio_provenance_path = audio_dir / "extraction.json"
        audio_provenance = (
            json.loads(audio_provenance_path.read_text())
            if audio_provenance_path.exists()
            else None
        )
        report = run_experiment(
            frame,
            text,
            audio,
            baseline,
            args.artifact_dir,
            {
                "fold_source": str(args.fold_path),
                "fold_source_sha256": e005._file_digest(args.fold_path),
                "text_extraction": extraction,
                "audio_source": str(audio_dir),
                "audio_extraction": audio_provenance,
                "audio_checksums": {
                    name: e005._file_digest(audio_dir / name)
                    for name in ("train_embeddings.npy", "train_metadata.csv")
                },
                "e008_oof": str(baseline_path),
                "e008_oof_sha256": e005._file_digest(baseline_path),
            },
        )
        print(json.dumps(e005._json_safe(report), allow_nan=False), flush=True)
