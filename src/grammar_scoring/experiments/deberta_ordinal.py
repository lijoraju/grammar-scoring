"""Canonical E007 supervised fine-tuning with fold-level recovery.

Heavy runtime imports are deferred so input/artifact validation works without
PyTorch or Transformers. No test data or additional features are consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch import device as TorchDevice
    from torch.amp import GradScaler
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerBase

import numpy as np
import pandas as pd

from grammar_scoring.data.dataset import load_train_dataframe
from grammar_scoring.data.transcripts import load_transcript_jsonl
from grammar_scoring.evaluation.metrics import regression_metrics, rmse
from grammar_scoring.evaluation.validation import validate_cv_folds
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.models.ordinal import (
    SCORE_VALUES,
    decode_probabilities,
    encode_targets,
    ordinal_loss,
    validate_scores,
)


@dataclass(frozen=True)
class E007Config:
    """Immutable canonical protocol; the CLI exposes no training overrides."""

    experiment: str = "E007"
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 256
    truncation: bool = True
    padding: str = "dynamic"
    pooling: str = "attention_mask_mean"
    dropout: float = 0.1
    head: str = "shared_latent_minus_ordered_softplus_thresholds"
    score_values: tuple[float, ...] = SCORE_VALUES
    threshold_initialization: str = "theta_0=-2; softplus(delta_i)=0.5"
    latent_bias: bool = False
    loss: str = "mean_BCEWithLogits_9_boundaries"
    optimizer: str = "AdamW"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_epochs: int = 5
    train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    effective_batch_size: int = 16
    scheduler: str = "linear"
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    precision: str = "fp16_on_cuda_float32_on_cpu"
    seed: int = 42
    n_splits: int = 5
    selection: str = "minimum_validation_rmse_first_epoch_on_ties"
    postprocessing: str = "none"


CONFIG = E007Config()


def align_inputs(
    train: pd.DataFrame, transcripts: pd.DataFrame, folds: pd.DataFrame
) -> pd.DataFrame:
    """Align train-only raw text and frozen assignments by canonical filename.

    Args:
        train: Canonical filenames and labels in competition order.
        transcripts: Train transcript records, with split and unchanged text.
        folds: Existing filename/fold assignments and optional labels.

    Returns:
        Validated filename, label, text, fold rows in canonical order.

    Raises:
        ValueError: If identities, text, labels, or five-fold sizes are invalid.
    """
    for frame in (train, transcripts, folds):
        _validate_filenames(frame)
        if "split" in frame and not frame["split"].eq("train").all():
            raise ValueError("E007 accepts only train records")
        if set(frame.filename) != set(train.filename):
            raise ValueError("Missing or unexpected filenames")
    if len(train) != 769:
        raise ValueError("E007 requires exactly 769 canonical train rows")
    if "split" not in transcripts or "text" not in transcripts:
        raise ValueError("Transcripts require split and text")
    labels = train.label.to_numpy()
    validate_scores(labels)
    if labels.dtype.kind not in "iuf" or not np.isfinite(labels).all():
        raise ValueError("Labels must be finite numeric values")
    if not ((labels >= 0) & (labels <= 5)).all():
        raise ValueError("Labels must be in [0, 5]")
    text = transcripts.set_index("filename").reindex(train.filename).text
    if any(not isinstance(value, str) or not value.strip() for value in text):
        raise ValueError("Empty or invalid transcripts")
    assigned = folds.set_index("filename").reindex(train.filename)
    if "label" in assigned and not np.array_equal(assigned.label, labels):
        raise ValueError("Frozen fold labels differ from canonical labels")
    validate_cv_folds(labels, assigned.fold, 5)
    if not np.array_equal(np.bincount(assigned.fold), [154, 154, 154, 154, 153]):
        raise ValueError("Canonical fold sizes must be 154/154/154/154/153")
    result = train[["filename", "label"]].reset_index(drop=True).copy()
    result["text"] = text.to_numpy()
    result["fold"] = assigned.fold.to_numpy()
    return result


def load_inputs(transcript_dir: Path, fold_path: Path) -> pd.DataFrame:
    """Load canonical train labels, raw train JSONL, and frozen CSV assignments."""
    return align_inputs(
        load_train_dataframe(),
        load_transcript_jsonl(transcript_dir / "train.jsonl"),
        pd.read_csv(fold_path),
    )


def split_fold(frame: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return disjoint training and validation partitions for a frozen fold."""
    if (
        isinstance(fold, bool)
        or not isinstance(fold, (int, np.integer))
        or fold not in range(5)
    ):
        raise ValueError("Fold must be 0..4")
    return frame.loc[frame.fold != fold], frame.loc[frame.fold == fold]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_identity(frame: pd.DataFrame) -> str:
    """Hash the exact protocol and aligned raw inputs for safe fold recovery."""
    return _digest(
        {"config": asdict(CONFIG), "inputs": frame.to_dict(orient="records")}
    )


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def seed_everything() -> None:
    """Reapply seed 42 to Python, NumPy, Torch, CUDA and deterministic kernels."""
    import torch

    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def optimizer_step_count(minibatches: int) -> int:
    """Count updates including the final partial accumulation group."""
    return math.ceil(minibatches / CONFIG.gradient_accumulation_steps)


def create_scaler(device: TorchDevice) -> GradScaler:
    """Use modern AMP with a compatibility fallback for older CPU test runtimes."""
    import torch

    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler,
    device: TorchDevice,
    *,
    dtype_observer: Callable[[str, nn.Module, Tensor], None] | None = None,
    update_observer: Callable[[str, nn.Module, dict[str, Any]], None] | None = None,
) -> float:
    """Train once and return sample-weighted mean ordinal BCE before each update.

    Accumulated loss is weighted by sample count within each group, including
    the final partial group. Every group's gradients are stepped, clipped after
    AMP unscale, and cleared. Scheduler advances only on successful AMP updates.
    Optional observers report dtypes and smoke-only update diagnostics.
    Update observers must inspect tensors without changing training state.
    """
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    samples = 0
    iterator = iter(loader)
    update_index = 0
    while group := list(_take(iterator, CONFIG.gradient_accumulation_steps)):
        update_index += 1
        update_report: dict[str, Any] = {
            "update_index": update_index,
            "microbatches": [],
        }
        group_samples = sum(len(batch["labels"]) for batch in group)
        for batch in group:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                predicted = model(**inputs)
                loss = ordinal_loss(predicted, encode_targets(labels))
            count = len(labels)
            weighted_loss += loss.detach().item() * count
            samples += count
            if dtype_observer is not None:
                dtype_observer("before_backward", model, loss)
            if update_observer is not None:
                update_report["microbatches"].append(
                    {
                        "raw_loss": loss.detach().item(),
                        "scale_before_backward": scaler.get_scale(),
                        "loss_weight": count / group_samples,
                    }
                )
            scaler.scale(loss * count / group_samples).backward()
            if update_observer is not None:
                update_observer("after_backward", model, update_report)
        if dtype_observer is not None:
            dtype_observer("before_unscale", model, loss)
        if update_observer is not None:
            update_observer("before_unscale", model, update_report)
        scaler.unscale_(optimizer)
        if update_observer is not None:
            update_observer("after_unscale", model, update_report)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CONFIG.max_grad_norm
        )
        if update_observer is not None:
            update_report["unclipped_gradient_norm"] = float(grad_norm)
            update_report["gradient_norm_is_finite"] = math.isfinite(float(grad_norm))
            update_observer("after_clip", model, update_report)
        old_scale = scaler.get_scale()
        if update_observer is not None:
            update_report["scale_before_step"] = old_scale
            update_observer("before_step", model, update_report)
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        if update_observer is not None:
            update_report["scale_after_update"] = new_scale
            update_report["scale_change"] = (
                "decreased"
                if new_scale < old_scale
                else "increased"
                if new_scale > old_scale
                else "equal"
            )
            update_report["update_not_skipped_by_scale"] = new_scale >= old_scale
            update_observer("after_step", model, update_report)
        if new_scale >= old_scale:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return weighted_loss / samples


def _take(iterator: Iterable[Any], count: int) -> Iterable[Any]:
    from itertools import islice

    return islice(iterator, count)


def predict(model: nn.Module, loader: DataLoader, device: TorchDevice) -> np.ndarray:
    """Decode continuous expected scores in loader order."""
    import torch

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                predictions.extend(
                    decode_probabilities(model(**inputs).float().sigmoid())
                    .cpu()
                    .tolist()
                )
    result = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite predictions")
    return result


def _loader(
    frame: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, *, shuffle: bool
) -> tuple[DataLoader, int]:
    import torch
    from transformers import DataCollatorWithPadding

    texts = frame.text.tolist()
    untruncated = tokenizer(texts, truncation=False, padding=False)
    exceeded = sum(len(ids) > CONFIG.max_length for ids in untruncated["input_ids"])
    encoded = tokenizer(
        texts, max_length=CONFIG.max_length, truncation=True, padding=False
    )
    records = [
        {**{key: value[i] for key, value in encoded.items()}, "labels": float(label)}
        for i, label in enumerate(frame.label)
    ]
    generator = torch.Generator().manual_seed(CONFIG.seed)
    return (
        torch.utils.data.DataLoader(
            records,
            batch_size=CONFIG.train_batch_size,
            shuffle=shuffle,
            collate_fn=DataCollatorWithPadding(tokenizer, padding=True),
            generator=generator,
            num_workers=0,
        ),
        exceeded,
    )


def select_best_epoch(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the first epoch attaining minimum finite validation RMSE."""
    if not history or any(not math.isfinite(row["validation_rmse"]) for row in history):
        raise ValueError("Epoch history requires finite validation RMSE")
    return min(history, key=lambda row: row["validation_rmse"])


def runtime_info(device: TorchDevice) -> dict[str, Any]:
    """Describe the actual training runtime without promising bitwise portability."""
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
    }


def train_fold(
    frame: pd.DataFrame,
    fold: int,
    directory: Path,
    identity: str,
    device: TorchDevice,
    *,
    model_factory: Callable[[], nn.Module],
    tokenizer_factory: Callable[[], PreTrainedTokenizerBase],
) -> dict[str, Any]:
    """Train all five epochs from original weights and restore the best state."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    seed_everything()
    tokenizer = tokenizer_factory()
    model = model_factory().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training, validation = split_fold(frame, fold)
    train_loader, train_exceeded = _loader(training, tokenizer, shuffle=True)
    valid_loader, valid_exceeded = _loader(validation, tokenizer, shuffle=False)
    diagnostic_loader, _ = _loader(training, tokenizer, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
    )
    steps = CONFIG.max_epochs * optimizer_step_count(len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(steps * CONFIG.warmup_ratio),
        num_training_steps=steps,
    )
    scaler = create_scaler(device)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "best.pt"
    history = []
    best_rmse = math.inf
    for epoch in range(1, CONFIG.max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        predicted = predict(model, valid_loader, device)
        metrics = regression_metrics(validation.label, predicted)
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": loss,
            "validation_rmse": metrics["rmse"],
            "validation_pearson": metrics["pearson_correlation"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        _write_json(directory / "history.json", history)
        print(json.dumps(_json_safe(row)), flush=True)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            temporary = directory / "best.tmp"
            torch.save(model.state_dict(), temporary)
            temporary.replace(checkpoint)
    best = select_best_epoch(history)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    valid_pred = predict(model, valid_loader, device)
    train_pred = predict(model, diagnostic_loader, device)
    metadata = {
        "fold": fold,
        "identity": identity,
        "config": asdict(CONFIG),
        "selected_epoch": best["epoch"],
        "validation_rmse": best["validation_rmse"],
        "validation_pearson": best["validation_pearson"],
        "n_train": len(training),
        "n_valid": len(validation),
        "history": history,
        "validation": _prediction_records(validation, valid_pred),
        "training": _prediction_records(training, train_pred),
        "checkpoint_sha256": _file_digest(checkpoint),
        "token_count_exceeds_max_length": train_exceeded + valid_exceeded,
        "runtime": runtime_info(device),
        "gpu_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cumulative_validation": predict_cumulative(
            model, valid_loader, device
        ).tolist(),
        "model": {
            "name": CONFIG.model_name,
            "revision": getattr(model.backbone.config, "_commit_hash", None),
            "configuration": model.backbone.config.to_dict(),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_configuration": json.loads(
                json.dumps(tokenizer.init_kwargs, default=str)
            ),
        },
    }
    _write_json(directory / "result.json", metadata)
    return validate_completed_fold(directory, frame, fold, identity)


def _prediction_records(frame: pd.DataFrame, predictions: np.ndarray) -> list[dict]:
    result = frame[["filename", "label", "fold"]].copy()
    result["prediction"] = predictions
    return result.to_dict(orient="records")


def _validate_predictions(records: list[dict], expected: pd.DataFrame) -> pd.DataFrame:
    actual = pd.DataFrame(records)
    _validate_filenames(actual)
    if set(actual.filename) != set(expected.filename):
        raise ValueError("Prediction filenames differ from frozen partition")
    actual = actual.set_index("filename").reindex(expected.filename).reset_index()
    for key in ("label", "fold"):
        if not np.array_equal(actual[key], expected[key]):
            raise ValueError(f"Prediction {key} differs from canonical inputs")
    if (
        actual.prediction.dtype.kind not in "iuf"
        or not np.isfinite(actual.prediction).all()
    ):
        raise ValueError("Predictions must be finite numeric values")
    return actual


def validate_completed_fold(
    directory: Path, frame: pd.DataFrame, fold: int, identity: str
) -> dict[str, Any]:
    """Recognize completion only after hash, history and prediction validation.

    Missing/corrupt artifacts and mismatched input/configuration fail clearly.
    A directory alone never indicates completion. The checkpoint checksum binds
    cached selected-checkpoint predictions to the retained weight file.
    """
    try:
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if result["identity"] != identity or result["config"] != json.loads(
            json.dumps(asdict(CONFIG))
        ):
            raise ValueError("Resume configuration/input mismatch")
        if result["fold"] != fold:
            raise ValueError("Resume fold mismatch")
        if _file_digest(directory / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        history = result["history"]
        if [row["epoch"] for row in history] != list(range(1, 6)) or any(
            row["fold"] != fold for row in history
        ):
            raise ValueError("Incomplete epoch history")
        if json.loads((directory / "history.json").read_text()) != history:
            raise ValueError("History artifact mismatch")
        for row in history:
            if not math.isfinite(row["train_loss"]) or not math.isfinite(
                row["learning_rate"]
            ):
                raise ValueError("Invalid epoch history")
        best = select_best_epoch(history)
        for field, source in (
            ("selected_epoch", "epoch"),
            ("validation_rmse", "validation_rmse"),
            ("validation_pearson", "validation_pearson"),
        ):
            if result[field] != best[source]:
                raise ValueError("Selected epoch metadata mismatch")
        training, validation = split_fold(frame, fold)
        if result["n_train"] != len(training) or result["n_valid"] != len(validation):
            raise ValueError("Partition sizes differ")
        actual = _validate_predictions(result["validation"], validation)
        _validate_predictions(result["training"], training)
        import torch

        cumulative = torch.tensor(result["cumulative_validation"], dtype=torch.float64)
        decoded = np.asarray(decode_probabilities(cumulative).tolist())
        if not np.allclose(decoded, actual.prediction, atol=1e-6, rtol=0):
            raise ValueError("Cached cumulative probabilities differ from predictions")
        metrics = regression_metrics(actual.label, actual.prediction)
        if not math.isclose(metrics["rmse"], result["validation_rmse"], rel_tol=1e-6):
            raise ValueError("Selected checkpoint RMSE mismatch")
        pearson = _json_safe(metrics["pearson_correlation"])
        saved = result["validation_pearson"]
        if (pearson is None) != (saved is None) or (
            pearson is not None and not math.isclose(pearson, saved, abs_tol=1e-6)
        ):
            raise ValueError("Selected checkpoint Pearson mismatch")
        if not result["runtime"] or result["model"]["name"] != CONFIG.model_name:
            raise ValueError("Missing or invalid runtime/model metadata")
        return result
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Incomplete/corrupt fold {fold}: {exc}") from exc


def assemble_oof(frame: pd.DataFrame, results: list[dict[str, Any]]) -> pd.DataFrame:
    """Validate exactly-once OOF coverage and restore canonical filename order."""
    if sorted(result["fold"] for result in results) != list(range(5)):
        raise ValueError("Exactly one completed result for each fold is required")
    records = []
    for result in results:
        _, validation = split_fold(frame, result["fold"])
        _validate_predictions(result["validation"], validation)
        records.extend(result["validation"])
    oof = _validate_predictions(records, frame)
    oof["residual"] = oof.label - oof.prediction
    oof["abs_error"] = oof.residual.abs()
    return oof


def run_experiment(
    frame: pd.DataFrame,
    artifact_dir: Path,
    *,
    device: str = "auto",
    resume: bool = False,
    fold: int | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    tokenizer_factory: Callable[[], PreTrainedTokenizerBase] | None = None,
    e005_oof_path: Path | None = None,
) -> dict[str, Any]:
    """Execute canonical folds; partial runs never emit overall OOF metrics.

    Factories are explicit offline test seams, not experimental CLI overrides.
    Every newly trained fold calls each factory independently after reseeding.
    CUDA OOM propagates; batch size and protocol are never adjusted.
    """
    import torch
    from transformers import AutoTokenizer

    from grammar_scoring.models.deberta_ordinal import DebertaOrdinal

    if "split" in frame and not frame["split"].eq("train").all():
        raise ValueError("E007 accepts only train records")
    # Revalidate public entry-point inputs instead of trusting a caller's frame.
    frame = align_inputs(
        frame[["filename", "label"]],
        frame[["filename", "text"]].assign(split="train"),
        frame[["filename", "label", "fold"]],
    )
    baseline_path = e005_oof_path or (
        artifact_dir / "oof" / "E005_deberta_finetuned.csv"
    )
    if fold is None:
        _validate_predictions(
            pd.read_csv(baseline_path).to_dict(orient="records"), frame
        )
    identity = experiment_identity(frame)
    if fold is not None:
        split_fold(frame, fold)
    chosen = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if chosen.type not in ("cpu", "cuda"):
        raise ValueError("E007 supports CPU or CUDA only")
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model_factory = model_factory or DebertaOrdinal.from_pretrained
    tokenizer_factory = tokenizer_factory or (
        lambda: AutoTokenizer.from_pretrained(CONFIG.model_name)
    )
    experiment_dir = artifact_dir / "experiments" / "E007"
    config_path = experiment_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(
        json.dumps(asdict(CONFIG))
    ):
        raise ValueError("Existing E007 configuration mismatch")
    _write_json(config_path, asdict(CONFIG))
    results = []
    for current in range(5) if fold is None else [fold]:
        directory = artifact_dir / "models" / "E007" / f"fold_{current}"
        if resume and directory.exists():
            result = validate_completed_fold(directory, frame, current, identity)
        else:
            # Invalidate any previous completion marker before rewriting weights.
            if (directory / "result.json").exists():
                (directory / "result.json").unlink()
            result = train_fold(
                frame,
                current,
                directory,
                identity,
                chosen,
                model_factory=model_factory,
                tokenizer_factory=tokenizer_factory,
            )
        results.append(result)
    history = [row for result in results for row in result["history"]]
    selected = [
        {
            key: result[key]
            for key in (
                "fold",
                "n_train",
                "n_valid",
                "selected_epoch",
                "validation_rmse",
                "validation_pearson",
            )
        }
        for result in results
    ]
    # A single-fold run preserves full experiment reports from earlier runs.
    if fold is not None:
        return {"complete": False, "per_fold": selected, "history": history}
    oof = assemble_oof(frame, results)
    training = [row for result in results for row in result["training"]]
    summary = {
        "experiment": "E007",
        "identity": identity,
        "E005_oof_sha256": _file_digest(baseline_path),
        "oof_metrics": regression_metrics(oof.label, oof.prediction),
        "diagnostic_training_rmse": rmse(
            [row["label"] for row in training],
            [row["prediction"] for row in training],
        ),
        "per_fold": selected,
        "runtime_by_fold": [result["runtime"] for result in results],
        "model_by_fold": [result["model"] for result in results],
        "gpu_peak_memory_bytes_by_fold": [
            result["gpu_peak_memory_bytes"] for result in results
        ],
        "train_loss_semantics": (
            "sample-weighted mean pre-update BCE across nine thresholds"
        ),
        "token_count_exceeds_max_length": [
            result["token_count_exceeds_max_length"] for result in results
        ],
    }
    output = artifact_dir / "oof" / "E007_deberta_ordinal.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output, index=False)
    _write_json(experiment_dir / "training_history.json", history)
    _write_json(experiment_dir / "fold_metrics.json", selected)
    _write_json(experiment_dir / "summary.json", summary)
    from grammar_scoring.experiments.ordinal_diagnostics import write_comparison

    write_comparison(frame, results, oof, artifact_dir, summary, baseline_path)
    _write_json(experiment_dir / "summary.json", summary)
    return _json_safe(summary)


def predict_cumulative(
    model: nn.Module, loader: DataLoader, device: TorchDevice
) -> np.ndarray:
    """Return validated cumulative probabilities in loader order."""
    import torch

    model.eval()
    batches = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                q = model(**inputs).float().sigmoid()
            decode_probabilities(q)
            batches.append(np.asarray(q.cpu().tolist(), dtype=np.float64))
    return np.concatenate(batches)
