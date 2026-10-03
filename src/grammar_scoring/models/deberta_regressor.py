"""Fully trainable E005 backbone and masked-mean regression head.

PyTorch is a runtime dependency for this module; Transformers is imported only
when constructing the real pretrained backbone, allowing offline fake models.
"""

from torch import Tensor, float32, nn

MODEL_NAME = "microsoft/deberta-v3-base"


def masked_mean_pool(hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool final token states while excluding padded positions.

    Args:
        hidden_state: Tensor of shape [batch, tokens, hidden].
        attention_mask: Tensor of shape [batch, tokens].

    Returns:
        Tensor of shape [batch, hidden], zero for fully masked examples.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    return (hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)


class DebertaRegressor(nn.Module):
    """AutoModel, masked mean pooling, dropout 0.1, and one linear output."""

    def __init__(self, backbone: nn.Module) -> None:
        """Attach the canonical head to an unfrozen pretrained backbone."""
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(backbone.config.hidden_size, 1)
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)

    @classmethod
    def from_pretrained(cls) -> "DebertaRegressor":
        """Initialize from original pretrained weights, never a fold checkpoint."""
        from transformers import AutoModel

        # Transformers v5 infers storage dtype by default. AMP needs FP32
        # trainable weights; autocast alone controls forward compute precision.
        return cls(AutoModel.from_pretrained(MODEL_NAME, dtype=float32))

    def forward(self, **inputs: Tensor) -> Tensor:
        """Return raw continuous predictions of shape [batch]."""
        states = self.backbone(**inputs).last_hidden_state
        pooled = masked_mean_pool(states, inputs["attention_mask"])
        return self.head(self.dropout(pooled)).squeeze(-1)
