"""Shared inference utilities for E005 and E008 DeBERTa ensembles."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from grammar_scoring.experiments.deberta_finetuning import CONFIG

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerBase


def validate_test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate test identities and raw transcript text.

    Args:
        frame: Test dataframe containing filename and text columns.

    Returns:
        A copy containing filename and text in the original row order.

    Raises:
        ValueError: If required columns, filenames, or transcript text are invalid.
    """
    required = {"filename", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test frame missing required columns: {sorted(missing)}")

    filenames = frame["filename"]
    if any(not isinstance(value, str) or not value.strip() for value in filenames):
        raise ValueError("Test filenames must be nonempty strings")
    if filenames.duplicated().any():
        raise ValueError("Test filenames must be unique")

    texts = frame["text"]
    if any(not isinstance(value, str) or not value.strip() for value in texts):
        raise ValueError("Test transcripts must be nonempty strings")

    return frame[["filename", "text"]].reset_index(drop=True).copy()


def discover_checkpoints(model_dir: Path, *, experiment: str = "E005") -> list[Path]:
    """Discover exactly one checkpoint for every canonical fold.

    Args:
        model_dir: Directory containing ``fold_0`` through ``fold_4``.
        experiment: E005 or E008 identifier used in validation messages.

    Returns:
        Checkpoint paths ordered by fold number.

    Raises:
        FileNotFoundError: If a canonical fold checkpoint is missing.
        ValueError: If unexpected fold directories are present.
    """
    if experiment not in ("E005", "E008"):
        raise ValueError(f"Unsupported DeBERTa experiment: {experiment}")
    expected = [model_dir / f"fold_{fold}" / "best.pt" for fold in range(5)]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing {experiment} checkpoints: {formatted}")

    fold_directories = {path.name for path in model_dir.glob("fold_*") if path.is_dir()}
    expected_directories = {f"fold_{fold}" for fold in range(5)}
    unexpected = fold_directories - expected_directories
    if unexpected:
        raise ValueError(
            f"Unexpected {experiment} fold directories: {sorted(unexpected)}"
        )

    return expected


def create_inference_loader(
    frame: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    *,
    collator_factory: Callable[..., Any] | None = None,
) -> tuple[DataLoader, int]:
    """Create deterministic dynamically padded E005 inference batches.

    The tokenizer semantics intentionally match canonical E005 training:
    maximum length 256, truncation enabled, dynamic padding, batch size 8,
    and no shuffled rows.

    Args:
        frame: Validated dataframe containing raw transcript text.
        tokenizer: Tokenizer associated with the E005 DeBERTa backbone.
        collator_factory: Optional offline test seam that builds the batch
            collator from the tokenizer. Defaults to
            ``DataCollatorWithPadding``.

    Returns:
        DataLoader preserving frame order and the number of transcripts whose
        untruncated token count exceeds the canonical maximum length.
    """
    import torch

    if collator_factory is None:
        from transformers import DataCollatorWithPadding

        collator_factory = DataCollatorWithPadding

    texts = frame["text"].tolist()
    untruncated = tokenizer(texts, truncation=False, padding=False)
    exceeded = sum(
        len(input_ids) > CONFIG.max_length for input_ids in untruncated["input_ids"]
    )

    encoded = tokenizer(
        texts,
        max_length=CONFIG.max_length,
        truncation=CONFIG.truncation,
        padding=False,
    )
    records = [
        {key: values[index] for key, values in encoded.items()}
        for index in range(len(frame))
    ]

    loader = torch.utils.data.DataLoader(
        records,
        batch_size=CONFIG.train_batch_size,
        shuffle=False,
        collate_fn=collator_factory(tokenizer, padding=True),
        num_workers=0,
    )
    return loader, exceeded


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Predict raw continuous E005 values in loader order.

    Args:
        model: Loaded E005 regression model.
        loader: Deterministic inference loader.
        device: Torch CPU or CUDA device.

    Returns:
        Finite raw predictions as float64 values.

    Raises:
        RuntimeError: If any prediction is non-finite.
    """
    import torch

    model.eval()
    predictions: list[float] = []

    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                predicted = model(**inputs)

            predictions.extend(predicted.float().cpu().tolist())

    result = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite E005 predictions")

    return result


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a canonical E005 state dictionary into a model.

    Args:
        model: Fresh E005 model initialized from the pretrained backbone.
        checkpoint_path: Selected fold ``best.pt`` checkpoint.
        device: Torch device used for checkpoint mapping.

    Returns:
        The model containing the selected fold weights.
    """
    import torch

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model


def predict_deberta_ensemble(
    frame: pd.DataFrame,
    model_dir: Path,
    *,
    device: str = "auto",
    experiment: str = "E005",
    model_factory: Callable[[], nn.Module] | None = None,
    tokenizer_factory: Callable[[], PreTrainedTokenizerBase] | None = None,
) -> pd.DataFrame:
    """Generate equal-weight predictions from five E005 or E008 checkpoints.

    No clipping, rounding, calibration, or other postprocessing is applied.

    Args:
        frame: Test dataframe containing canonical filename and transcript text.
        model_dir: Model directory containing the five fold directories.
        experiment: E005 or E008 identifier used in validation messages.
        device: ``auto``, ``cpu``, or ``cuda``.
        model_factory: Optional offline test seam for model construction.
        tokenizer_factory: Optional offline test seam for tokenizer construction.

    Returns:
        Dataframe containing filename and raw ensemble label prediction.

    Raises:
        ValueError: If the device or input frame is invalid.
        RuntimeError: If predictions are malformed or non-finite.
    """
    import torch

    validated = validate_test_frame(frame)
    checkpoints = discover_checkpoints(model_dir, experiment=experiment)

    chosen = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if chosen.type not in ("cpu", "cuda"):
        raise ValueError(f"{experiment} inference supports CPU or CUDA only")
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    if model_factory is None:
        from grammar_scoring.models.deberta_regressor import DebertaRegressor

        model_factory = DebertaRegressor.from_pretrained

    if tokenizer_factory is None:
        from transformers import AutoTokenizer

        def load_tokenizer() -> PreTrainedTokenizerBase:
            return AutoTokenizer.from_pretrained(CONFIG.model_name)

        tokenizer_factory = load_tokenizer

    tokenizer = tokenizer_factory()
    loader, _ = create_inference_loader(validated, tokenizer)

    fold_predictions = []

    for checkpoint in checkpoints:
        model = model_factory().to(chosen)
        load_checkpoint(model, checkpoint, chosen)

        predicted = predict_loader(model, loader, chosen)
        if len(predicted) != len(validated):
            raise RuntimeError(
                f"{experiment} prediction count differs from test row count"
            )

        fold_predictions.append(predicted)

        del model
        if chosen.type == "cuda":
            torch.cuda.empty_cache()

    stacked = np.vstack(fold_predictions)
    ensemble = stacked.mean(axis=0)

    if not np.isfinite(ensemble).all():
        raise RuntimeError(f"Nonfinite {experiment} ensemble predictions")

    result = validated[["filename"]].copy()
    result["label"] = ensemble
    return result


def predict_e005_ensemble(
    frame: pd.DataFrame,
    model_dir: Path,
    *,
    device: str = "auto",
    model_factory: Callable[[], nn.Module] | None = None,
    tokenizer_factory: Callable[[], PreTrainedTokenizerBase] | None = None,
) -> pd.DataFrame:
    """Generate canonical E005 predictions with the shared implementation.

    Args:
        frame: Test dataframe containing filename and raw transcript text.
        model_dir: E005 directory containing the five selected checkpoints.
        device: ``auto``, ``cpu``, or ``cuda``.
        model_factory: Optional offline model construction seam.
        tokenizer_factory: Optional offline tokenizer construction seam.

    Returns:
        Filename and raw ensemble label predictions in input order.
    """
    return predict_deberta_ensemble(
        frame,
        model_dir,
        device=device,
        model_factory=model_factory,
        tokenizer_factory=tokenizer_factory,
    )
