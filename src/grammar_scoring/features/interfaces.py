"""Structural interfaces for optional transformer dependencies and test fakes."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


class Tokenizer(Protocol):
    """Tokenizer boundary accepting raw strings or batches."""

    def __call__(self, text: str | list[str], **kwargs: object) -> dict[str, Any]:
        """Return token IDs or tensor inputs according to inference options."""
        ...


class ModelOutput(Protocol):
    """Final hidden states returned by a transformer."""

    last_hidden_state: Tensor


class EncoderModel(Protocol):
    """Frozen model boundary supporting native or transformer inference."""

    max_seq_length: int

    def eval(self) -> object:
        """Disable training behavior."""
        ...

    def requires_grad_(self, value: bool) -> object:
        """Enable or disable parameter gradients."""
        ...

    def to(self, device: str) -> object:
        """Move the model to the selected device."""
        ...

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        """Run native SentenceTransformers encoding."""
        ...

    def __call__(self, **kwargs: Tensor) -> ModelOutput:
        """Run transformer inference."""
        ...


class TorchRuntime(Protocol):
    """Minimal runtime boundary for disabling autograd during inference."""

    def inference_mode(self) -> AbstractContextManager[object]:
        """Enter inference-only execution."""
        ...
