"""Offline contracts for the frozen E011a protocol."""

import json

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments import multimodal_fusion as e


@pytest.fixture
def inputs():
    frame = pd.DataFrame(
        {
            "filename": [f"sample_{i}.wav" for i in range(732)],
            "label": np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 147)[:732],
            "fold": np.arange(732) % 5,
            "text": ["canonical transcript"] * 732,
        }
    )
    matrix = np.random.default_rng(42).normal(size=(732, 768))
    metadata = frame.drop(columns="text").assign(checkpoint_fold=frame.fold)
    return frame, matrix, metadata


def test_audio_alignment_by_filename(inputs):
    frame, matrix, metadata = inputs
    actual = e.validate_embeddings(matrix[::-1], metadata.iloc[::-1], frame, text=False)
    np.testing.assert_array_equal(actual, matrix)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "label",
        "fold",
        "missing",
        "nan",
        "inf",
        "dimension",
        "checkpoint_fold",
        "order",
    ],
)
def test_validation_rejects_damage(inputs, damage):
    frame, matrix, metadata = inputs
    if damage == "duplicate":
        metadata.loc[1, "filename"] = metadata.loc[0, "filename"]
    elif damage in {"label", "fold", "checkpoint_fold"}:
        metadata.loc[0, damage] += 1
    elif damage == "missing":
        metadata = metadata.iloc[:-1]
    elif damage in {"nan", "inf"}:
        matrix[0, 0] = float(damage)
    elif damage == "dimension":
        matrix = matrix[:, :-1]
    else:
        metadata = metadata.iloc[::-1]
    with pytest.raises(ValueError):
        e.validate_embeddings(matrix, metadata, frame, text=True)


def test_checkpoint_provenance_selects_exact_fold(inputs, tmp_path, monkeypatch):
    frame, _, _ = inputs
    calls = []

    def verify(directory, actual, fold, identity):
        assert directory == tmp_path / f"fold_{fold}"
        assert actual is frame
        assert identity == e.e005.experiment_identity(frame)
        calls.append(fold)
        return {"checkpoint_sha256": str(fold), "model": {}, "identity": identity}

    monkeypatch.setattr(e.e005, "validate_completed_fold", verify)
    mapping = e.checkpoint_provenance(frame, tmp_path)
    assert calls == list(range(5))
    for fold in range(5):
        assert mapping[str(fold)]["path"] == str(tmp_path / f"fold_{fold}/best.pt")


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_incomplete_cache(inputs, tmp_path, missing):
    frame, _, _ = inputs
    names = ["text_embeddings.npy", "text_metadata.csv", "extraction.json"]
    for i, name in enumerate(names):
        if i != missing:
            (tmp_path / name).touch()
    with pytest.raises(ValueError, match="Incomplete"):
        e.extract_text(frame, tmp_path, {}, cache_only=True)


def save_cache(directory, matrix, metadata, config):
    np.save(directory / "text_embeddings.npy", matrix)
    metadata.to_csv(directory / "text_metadata.csv", index=False)
    report = {
        "configuration": config,
        "fingerprint": e.e005._digest(config),
        "checksums": {
            name: e.e005._file_digest(directory / name)
            for name in ("text_embeddings.npy", "text_metadata.csv")
        },
    }
    (directory / "extraction.json").write_text(json.dumps(report))


@pytest.mark.parametrize("damage", [None, "provenance", "checksum", "mapping"])
def test_cache_fail_closed(inputs, tmp_path, damage):
    frame, matrix, metadata = inputs
    config = e.extraction_config(frame, {})
    if damage == "mapping":
        metadata.loc[0, "checkpoint_fold"] = 4
    save_cache(tmp_path, matrix, metadata, config)
    if damage == "provenance":
        config = {**config, "max_length": 128}
    if damage == "checksum":
        (tmp_path / "text_metadata.csv").write_text("modified")
    if damage:
        with pytest.raises(ValueError):
            e.extract_text(frame, tmp_path, config, cache_only=True)
    else:
        loaded, _, _ = e.extract_text(frame, tmp_path, config, cache_only=True)
        np.testing.assert_array_equal(loaded, matrix)


def test_fold_local_scaling_fusion_and_coverage(inputs, monkeypatch):
    frame, text, _ = inputs
    audio = text + 10
    fitted, dimensions = [], []
    scaler_type = e.StandardScaler

    class Scaler(scaler_type):
        def fit(self, x, y=None, sample_weight=None):
            fitted.append(x.copy())
            return super().fit(x, y, sample_weight=sample_weight)

    class FixedRidge:
        def __init__(self, *, alpha):
            assert alpha == 1.0

        def fit(self, x, y):
            dimensions.append(x.shape[1])
            self.mean = y.mean()
            return self

        def predict(self, x):
            assert x.shape[1] == 1536
            return np.full(len(x), self.mean)

    monkeypatch.setattr(e, "StandardScaler", Scaler)
    monkeypatch.setattr(e, "Ridge", FixedRidge)
    result = e.evaluate(frame, [text, audio])
    for fold in range(5):
        np.testing.assert_array_equal(fitted[fold * 2], text[frame.fold != fold])
        np.testing.assert_array_equal(fitted[fold * 2 + 1], audio[frame.fold != fold])
    assert dimensions == [1536] * 5
    assert len(result) == 732 and result.filename.is_unique
    assert np.isfinite(result.prediction).all()
    np.testing.assert_allclose(result.residual, result.label - result.prediction)
    np.testing.assert_allclose(result.abs_error, abs(result.residual))
    frame.loc[0, "fold"] = 99
    with pytest.raises(ValueError, match="exactly one"):
        e.evaluate(frame, [text, audio])


def test_metrics_deterministic():
    report = e.metrics(
        pd.DataFrame(
            {
                "label": [1.0, 2.0, 3.0, 4.0],
                "prediction": [2.0, 3.0, 4.0, 5.0],
                "fold": [0, 0, 1, 1],
            }
        )
    )
    assert report["pooled"]["rmse"] == 1
    assert report["pooled"]["pearson_correlation"] == pytest.approx(1)
    assert [row["rmse"] for row in report["per_fold"]] == [1, 1]


def test_artifact_schema_and_comparison(inputs, tmp_path, monkeypatch):
    frame, matrix, _ = inputs
    baseline = frame.drop(columns="text").assign(prediction=frame.label + 1)

    def evaluate(actual, modalities):
        return actual.drop(columns="text").assign(
            prediction=actual.label + 0.5,
            residual=-0.5,
            abs_error=0.5,
        )

    monkeypatch.setattr(e, "evaluate", evaluate)
    with pytest.warns(UserWarning, match="audio control"):
        report = e.run_experiment(
            frame,
            matrix,
            matrix,
            baseline.iloc[::-1],
            tmp_path,
            {"fold_source": "frozen"},
        )
    paired = report["e008_comparison"]
    assert paired["rmse_delta"] == -0.5
    assert paired["samples_lower_absolute_error"] == 732
    assert paired["percent_lower_absolute_error"] == 100
    assert paired["folds_improving_rmse"] == 5
    assert set(paired["score_bands"]) == {"low", "mid", "high"}
    for name in ("text", "audio", "fusion"):
        oof = pd.read_csv(tmp_path / f"oof/E011a_{name}.csv")
        assert list(oof) == [
            "filename",
            "label",
            "fold",
            "prediction",
            "residual",
            "abs_error",
        ]
    for name in ("config", "metrics", "diagnostics"):
        json.loads((tmp_path / f"experiments/E011a/{name}.json").read_text())
    assert not list(tmp_path.rglob("*submission*"))


def test_actual_extraction_cross_fitted_encoder_before_head(
    inputs, tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    import torch

    from grammar_scoring.inference import deberta
    from grammar_scoring.models.deberta_regressor import DebertaRegressor

    frame, _, _ = inputs
    # Local Torch predates NumPy 2; keep this offline test independent of its ABI.
    monkeypatch.setattr(torch.Tensor, "numpy", lambda self: np.array(self.tolist()))
    current = {"fold": None}
    seen = []

    class Backbone(torch.nn.Module):
        config = SimpleNamespace(hidden_size=768)

        def forward(self, **inputs):
            assert not self.training and not torch.is_grad_enabled()
            folds = inputs["input_ids"][:, 0]
            assert (folds == current["fold"]).all()
            seen.extend(folds.tolist())
            hidden = torch.ones(len(folds), 2, 768) * (current["fold"] + 1)
            hidden[:, 1] = 999
            return SimpleNamespace(last_hidden_state=hidden)

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == {
                "max_length": 256,
                "truncation": True,
                "padding": True,
                "return_tensors": "pt",
            }
            folds = [int(text) for text in texts]
            return {
                "input_ids": torch.tensor([[f, 0] for f in folds]),
                "attention_mask": torch.tensor([[1, 0]] * len(folds)),
            }

    def load(model, path, device):
        assert isinstance(model, DebertaRegressor)
        current["fold"] = int(path.parent.name.removeprefix("fold_"))
        return model

    monkeypatch.setattr(deberta, "load_checkpoint", load)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(for_model=lambda **kwargs: None),
            AutoModel=SimpleNamespace(from_config=lambda config: Backbone()),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        ),
    )
    frame["text"] = frame.fold.astype(str)
    mapping = {
        str(f): {
            "path": str(tmp_path / f"fold_{f}/best.pt"),
            "model": {"configuration": {}, "revision": "fake"},
        }
        for f in range(5)
    }
    matrix, metadata, report = e.extract_text(
        frame,
        tmp_path / "cache",
        e.extraction_config(frame, mapping),
        device="cpu",
        batch_size=19,
    )
    assert len(seen) == 732
    np.testing.assert_array_equal(metadata.checkpoint_fold, frame.fold)
    np.testing.assert_array_equal(matrix[:, 0], frame.fold + 1)
    assert matrix.shape == (732, 768)
    assert report["runtime"]["batch_size"] == 19


def test_real_fixed_ridge_controls(inputs):
    frame, matrix, _ = inputs
    for modalities in ([matrix], [matrix, matrix + 0.1]):
        result = e.evaluate(frame, modalities)
        assert len(result) == 732 and np.isfinite(result.prediction).all()
        assert len(e.metrics(result)["per_fold"]) == 5
