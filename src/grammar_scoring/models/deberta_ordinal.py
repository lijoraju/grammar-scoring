"""Rank-consistent ordinal DeBERTa with a shared latent score.

Logit k is s(x) - theta_k. theta_0 is freely learned and theta_k equals
 theta_0 + sum_{i=1}^k softplus(delta_i). Positive increments order thresholds,
so sigmoid logits are non-increasing by construction. Initial thresholds are
-2,-1.5,...,2; the latent layer has no redundant intercept.
"""

import math

import torch
from torch import Tensor, float32, nn
from torch.nn import functional as F

from grammar_scoring.models.deberta_regressor import MODEL_NAME, masked_mean_pool


class DebertaOrdinal(nn.Module):
    """Fully trainable backbone, masked pooling and nine ordered logits."""

    def __init__(self, backbone: nn.Module) -> None:
        """Initialize shared latent weights and equally spaced thresholds."""
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(backbone.config.hidden_size, 1, bias=False)
        self.threshold_start = nn.Parameter(torch.tensor(-2.0))
        self.threshold_increments = nn.Parameter(
            torch.full((8,), math.log(math.expm1(0.5)))
        )
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)

    def thresholds(self) -> Tensor:
        """Return nine ordered thresholds in FP32, including under autocast."""
        increments = F.softplus(self.threshold_increments.float()).cumsum(0)
        return self.threshold_start.float() + torch.cat(
            (increments.new_zeros(1), increments)
        )

    @classmethod
    def from_pretrained(cls) -> "DebertaOrdinal":
        """Load original pretrained FP32 weights independently for each fold."""
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(MODEL_NAME, dtype=float32))

    def forward(self, **inputs: Tensor) -> Tensor:
        """Return [batch, 9] shared latent minus ordered threshold logits."""
        states = self.backbone(**inputs).last_hidden_state
        pooled = masked_mean_pool(states, inputs["attention_mask"])
        latent = self.head(self.dropout(pooled)).float()
        return latent - self.thresholds()[None, :]
