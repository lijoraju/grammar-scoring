from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from grammar_scoring.models.deberta_ordinal import DebertaOrdinal  # noqa: E402
from grammar_scoring.models.deberta_regressor import masked_mean_pool  # noqa: E402
from grammar_scoring.models.ordinal import (  # noqa: E402
    SCORE_VALUES,
    class_probabilities,
    decode_probabilities,
    encode_targets,
    ordinal_loss,
    validate_scores,
)


def test_exact_vocabulary_and_encoding():
    assert SCORE_VALUES == (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
    targets = encode_targets(torch.tensor([0.0, 1.0, 2.0, 5.0]))
    assert targets.tolist() == [
        [0] * 9,
        [1] + [0] * 8,
        [1] * 3 + [0] * 6,
        [1] * 9,
    ]
    validate_scores(SCORE_VALUES)


@pytest.mark.parametrize("value", [0.5, 1.25, -1, 6, float("nan"), float("inf")])
def test_invalid_scores(value):
    with pytest.raises(ValueError):
        encode_targets(torch.tensor([value]))
    with pytest.raises(ValueError):
        validate_scores([value])


def test_probability_conversion_and_expectation():
    q = encode_targets(torch.tensor(SCORE_VALUES))
    p = class_probabilities(q)
    torch.testing.assert_close(p, torch.eye(10))
    torch.testing.assert_close(decode_probabilities(q), torch.tensor(SCORE_VALUES))
    q = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]])
    torch.testing.assert_close(class_probabilities(q), torch.full((1, 10), 0.1))
    assert decode_probabilities(q).item() == pytest.approx(sum(SCORE_VALUES) / 10)
    for bad in [q.flip(1), q * 2, torch.full((1, 9), float("nan"))]:
        with pytest.raises(ValueError):
            class_probabilities(bad)


def test_rank_consistent_model_pooling_and_backbone_gradients():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)
            self.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(
                last_hidden_state=input_ids.unsqueeze(-1).float() * self.weight
            )

    model = DebertaOrdinal(Backbone()).eval()
    assert model.dropout.p == 0.1
    assert model.backbone.weight.requires_grad
    with torch.no_grad():
        model.head.weight.fill_(1)
    inputs = dict(
        input_ids=torch.tensor([[1, 3, 99], [2, 99, 99]]),
        attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
    )
    logits = model(**inputs)
    assert logits.shape == (2, 9)
    assert (model.thresholds()[1:] > model.thresholds()[:-1]).all()
    assert (logits.sigmoid()[:, 1:] <= logits.sigmoid()[:, :-1]).all()
    torch.testing.assert_close(logits[0], logits[1])
    predictions = decode_probabilities(logits.sigmoid())
    assert ((predictions >= 0) & (predictions <= 5)).all()
    ordinal_loss(logits, encode_targets(torch.tensor([0.0, 5.0]))).backward()
    assert model.backbone.weight.grad is not None
    assert model.threshold_increments.grad is not None
    hidden = torch.tensor([[[1.0, 3.0], [999.0, 999.0]]])
    torch.testing.assert_close(
        masked_mean_pool(hidden, torch.tensor([[1, 0]])), torch.tensor([[1.0, 3.0]])
    )


def test_mean_bce_loss():
    logits = torch.zeros(4, 9, requires_grad=True)
    targets = encode_targets(torch.tensor([0.0, 1.0, 2.0, 5.0]))
    loss = ordinal_loss(logits, targets)
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.69314718)
    loss.backward()
    torch.testing.assert_close(logits.grad, (0.5 - targets) / 36)
    with pytest.raises(ValueError):
        ordinal_loss(logits[:, :8], targets)


def test_pretrained_fp32_storage(monkeypatch):
    import sys
    from unittest.mock import Mock

    def load(name, **kwargs):
        backbone = torch.nn.Linear(2, 2).to(dtype=kwargs["dtype"])
        backbone.config = SimpleNamespace(hidden_size=2)
        return backbone

    loader = Mock(side_effect=load)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=loader)),
    )
    model = DebertaOrdinal.from_pretrained()
    loader.assert_called_once_with("microsoft/deberta-v3-base", dtype=torch.float32)
    assert {p.dtype for p in model.parameters()} == {torch.float32}
