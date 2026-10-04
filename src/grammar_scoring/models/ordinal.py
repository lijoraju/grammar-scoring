"""Exact score vocabulary, cumulative BCE targets and expectation decoding."""

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

SCORE_VALUES = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def validate_scores(labels: object) -> None:
    """Reject every label outside the exact observed vocabulary."""
    values = np.asarray(labels)
    if values.dtype.kind not in "iuf" or not np.isin(values, SCORE_VALUES).all():
        raise ValueError("Labels must belong exactly to SCORE_VALUES")


def encode_targets(labels: Tensor) -> Tensor:
    """Encode t_k = 1[y > SCORE_VALUES[k]] for nine boundaries."""
    values = labels.new_tensor(SCORE_VALUES)
    if labels.ndim != 1 or not (labels[:, None] == values).any(1).all():
        raise ValueError("Labels must be a vector in SCORE_VALUES")
    return (labels[:, None] > values[:-1]).float()


def class_probabilities(q: Tensor, tolerance: float = 1e-6) -> Tensor:
    """Validate cumulative probabilities and difference them into ten classes."""
    if q.ndim != 2 or q.shape[1] != 9 or not torch.isfinite(q).all():
        raise ValueError("Expected finite cumulative probabilities [batch, 9]")
    if (q < 0).any() or (q > 1).any() or (q[:, 1:] > q[:, :-1]).any():
        raise ValueError("Cumulative probabilities must decrease within [0, 1]")
    p = torch.cat((1 - q[:, :1], q[:, :-1] - q[:, 1:], q[:, -1:]), 1)
    if (p < -tolerance).any() or not torch.allclose(
        p.sum(1), torch.ones_like(p[:, 0]), atol=tolerance, rtol=0
    ):
        raise ValueError("Invalid categorical distribution")
    return p


def decode_probabilities(q: Tensor) -> Tensor:
    """Return the continuous expectation without clipping or calibration."""
    p = class_probabilities(q)
    result = (p * p.new_tensor(SCORE_VALUES)).sum(1)
    if not torch.isfinite(result).all() or (result < 0).any() or (result > 5).any():
        raise ValueError("Invalid expected score")
    return result


def ordinal_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Average unweighted BCE over samples and all nine thresholds."""
    if logits.ndim != 2 or logits.shape[1] != 9 or targets.shape != logits.shape:
        raise ValueError("Logits and cumulative targets must have shape [batch, 9]")
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float())
