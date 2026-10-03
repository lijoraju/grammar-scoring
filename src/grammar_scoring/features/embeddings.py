"""Frozen text encoders and safe, ordered embedding artifacts."""

from __future__ import annotations

import importlib.metadata
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from grammar_scoring.features.interfaces import EncoderModel, Tokenizer, TorchRuntime

if TYPE_CHECKING:
    from torch import Tensor

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384, 256, "minilm"),
    "deberta": ("microsoft/deberta-v3-base", 768, 512, "deberta_v3_base"),
}


def validate_transcripts(frame: pd.DataFrame) -> None:
    """Validate raw text and unique split/filename identities in input order."""
    for column in ("split", "filename", "text"):
        if column not in frame:
            raise ValueError(f"Missing column: {column}")
        if any(not isinstance(value, str) for value in frame[column]):
            raise ValueError(f"{column} must contain strings")
        if column != "text" and any(not value.strip() for value in frame[column]):
            raise ValueError(f"{column} must contain nonempty strings")
    if frame.duplicated(["split", "filename"]).any():
        raise ValueError("Duplicate split/filename identity")


@dataclass
class EmbeddingResult:
    """An ordered matrix associated with explicit split/filename identities."""

    split: np.ndarray
    filenames: np.ndarray
    embeddings: np.ndarray
    dimension: int

    def __post_init__(self) -> None:
        """Reject malformed identities, dimensions, dtypes, and values."""
        for values in (self.split, self.filenames):
            if values.ndim != 1 or values.dtype.kind not in "US":
                raise ValueError("Identities must be one-dimensional string arrays")
            if any(not value.strip() for value in values):
                raise ValueError("Identities must be nonempty")
        if len(self.split) != len(self.filenames):
            raise ValueError("Identity lengths differ")
        if len(set(zip(self.split, self.filenames, strict=True))) != len(self.split):
            raise ValueError("Duplicate split/filename identity")
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension <= 0
        ):
            raise ValueError("Expected dimension must be a positive integer")
        if self.embeddings.shape != (
            len(self.split),
            self.dimension,
        ):
            raise ValueError("Unexpected embedding shape/dimension")
        if self.embeddings.dtype != np.float32:
            raise ValueError("Embeddings must be float32")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("Embeddings must be finite")


def save_embeddings(
    result: EmbeddingResult, path: Path, *, overwrite: bool = False
) -> None:
    """Save a validated numeric NPZ, refusing replacement unless requested."""
    result.__post_init__()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb" if overwrite else "xb") as stream:
        np.savez_compressed(
            stream,
            split=result.split,
            filenames=result.filenames,
            embeddings=result.embeddings,
        )


def load_embeddings(path: Path, expected_dimension: int) -> EmbeddingResult:
    """Load and validate an artifact without permitting pickle deserialization."""
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"split", "filenames", "embeddings"}:
            raise ValueError("Unexpected NPZ fields")
        return EmbeddingResult(
            artifact["split"],
            artifact["filenames"],
            artifact["embeddings"],
            expected_dimension,
        )


def masked_mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool final hidden states while excluding padding, on the hidden device."""
    mask = attention_mask.unsqueeze(-1).to(device=hidden.device, dtype=hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class FrozenEmbedder:
    """Common inference interface with lazy optional ML dependencies.

    Args:
        variant: Canonical minilm or deberta variant.
        device: Torch device for inference.
        batch_size: Positive number of texts per batch.
        model: Optional injected model for offline tests.
        tokenizer: Optional injected DeBERTa tokenizer.
        torch_module: Optional injected torch-compatible runtime.
    """

    def __init__(
        self,
        variant: str,
        device: str = "cpu",
        batch_size: int = 32,
        *,
        model: EncoderModel | None = None,
        tokenizer: Tokenizer | None = None,
        torch_module: TorchRuntime | None = None,
    ) -> None:
        """Initialize one canonical frozen encoder; load dependencies only as needed."""
        if variant not in MODELS:
            raise ValueError(f"Unknown model: {variant}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.variant, self.device, self.batch_size = variant, device, batch_size
        self.model_name, self.dimension, self.max_length, _ = MODELS[variant]
        self.torch = torch_module
        if model is None:
            import torch

            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            self.torch = torch
            if variant == "minilm":
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.model_name, device=device)
            else:
                from transformers import AutoModel, AutoTokenizer

                model = AutoModel.from_pretrained(self.model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model, self.tokenizer = model, tokenizer
        if variant == "minilm":
            self.model.max_seq_length = self.max_length
        else:
            if tokenizer is None:
                raise ValueError("DeBERTa requires a tokenizer")
            if self.torch is None:
                import torch

                self.torch = torch
            self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    def encode(self, frame: pd.DataFrame) -> EmbeddingResult:
        """Encode unchanged raw text, preserving input identities and ordering."""
        validate_transcripts(frame)
        texts = frame["text"].tolist()
        if not texts:
            matrix = np.empty((0, self.dimension), dtype=np.float32)
        elif self.variant == "minilm":
            matrix = np.asarray(
                self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    device=self.device,
                    normalize_embeddings=False,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        else:
            batches = []
            with self.torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    inputs = self.tokenizer(
                        texts[start : start + self.batch_size],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
                    inputs = {
                        key: value.to(self.device) for key, value in inputs.items()
                    }
                    hidden = self.model(**inputs).last_hidden_state
                    pooled = masked_mean_pool(hidden, inputs["attention_mask"])
                    batches.append(pooled.float().cpu().numpy())
            matrix = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        return EmbeddingResult(
            np.asarray(frame["split"].tolist(), dtype=str),
            np.asarray(frame["filename"].tolist(), dtype=str),
            matrix,
            self.dimension,
        )

    def metadata(self) -> dict[str, Any]:
        """Describe model configuration independently of the generated split."""
        versions = {}
        for package in ("numpy", "torch", "transformers", "sentence-transformers"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        return {
            "model_name": self.model_name,
            "model_family": self.variant,
            "embedding_dimension": self.dimension,
            "pooling": "native SentenceTransformers"
            if self.variant == "minilm"
            else "final hidden layer attention-mask-aware mean",
            "max_length": self.max_length,
            "normalize_embeddings": False,
            "batch_size": self.batch_size,
            "device": self.device,
            "generation_configuration": {
                "frozen": True,
                "dtype": "float32",
                "raw_text": True,
                "truncation": True,
                "add_special_tokens": True,
                "random_seed": 42,
            },
            "package_versions": versions,
        }
