from types import SimpleNamespace

import pytest

# Heavy runtime is optional locally; the full E005 CPU suite uses PyTorch only.
torch = pytest.importorskip("torch")

from grammar_scoring.models.deberta_regressor import (  # noqa: E402
    DebertaRegressor,
    masked_mean_pool,
)


def test_pooling_mask_shape_and_zero_denominator():
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [999.0, 999.0]]])
    pooled = masked_mean_pool(hidden, torch.tensor([[1, 1, 0]]))
    assert pooled.shape == (1, 2)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(
        masked_mean_pool(hidden, torch.zeros(1, 3)), torch.zeros(1, 2)
    )


def test_regression_head_and_trainable_backbone():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)
            self.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(
                last_hidden_state=input_ids.unsqueeze(-1).float() * self.weight
            )

    model = DebertaRegressor(Backbone()).eval()
    assert model.dropout.p == 0.1
    assert model.backbone.weight.requires_grad
    with torch.no_grad():
        model.head.weight.fill_(1)
        model.head.bias.zero_()
    result = model(
        input_ids=torch.tensor([[1, 3, 99], [2, 99, 99]]),
        attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
    )
    assert result.shape == (2,)
    torch.testing.assert_close(result, torch.tensor([4.0, 4.0]))
    result.sum().backward()
    assert model.backbone.weight.grad is not None


def test_pretrained_loading_keeps_fp32_storage_with_amp(monkeypatch):
    import sys
    from unittest.mock import Mock

    def load_backbone(name, **kwargs):
        # Model Transformers v5's inferred FP16 checkpoint storage offline.
        backbone = torch.nn.Linear(2, 2).to(dtype=kwargs.get("dtype", torch.float16))
        backbone.config = SimpleNamespace(hidden_size=2)
        return backbone

    load = Mock(side_effect=load_backbone)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load)),
    )
    model = DebertaRegressor.from_pretrained().to(torch.device("cpu"))
    load.assert_called_once_with("microsoft/deberta-v3-base", dtype=torch.float32)
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    # CPU autocast checks storage preservation only, not real CUDA FP16 AMP.
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        model.head(torch.ones(1, 2)).float().sum().backward()
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert model.head.weight.grad.dtype == torch.float32
