"""CPU tests for frozen-fold inference and fixed blending."""

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.evaluation.metrics import regression_metrics
from grammar_scoring.experiments.embedding_baseline import evaluate_embedding_baseline
from grammar_scoring.features.embeddings import EmbeddingResult
from grammar_scoring.inference import embedding_ridge as module


@pytest.fixture
def inputs(monkeypatch):
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {"filename": [f"train{i}" for i in range(769)], "label": rng.normal(size=769)}
    )
    folds = train.copy()
    folds["fold"] = np.arange(769) % 5
    test = pd.DataFrame({"filename": [f"test{i}" for i in range(216)]})

    def artifact(frame, split):
        values = np.zeros((len(frame), 768), dtype=np.float32)
        values[:, :3] = rng.normal(size=(len(frame), 3))
        return EmbeddingResult(
            np.full(len(frame), split), frame.filename.to_numpy(dtype=str), values, 768
        )

    train_emb = artifact(train, "train")
    test_emb = artifact(test, "test")
    oof = evaluate_embedding_baseline(train, folds, train_emb).oof
    monkeypatch.setattr(
        module, "EXPECTED_METRICS", regression_metrics(train.label, oof.prediction)
    )
    return train, folds, train_emb, test, test_emb, oof


def infer(inputs):
    return module.infer_embedding_ridge(*inputs)


def test_alignment_ensemble_and_determinism(inputs):
    original = infer(inputs)
    train, folds, train_emb, test, test_emb, oof = inputs
    for artifact in (train_emb, test_emb):
        artifact.filenames = artifact.filenames[::-1]
        artifact.split = artifact.split[::-1]
        artifact.embeddings = artifact.embeddings[::-1]
    shuffled = module.infer_embedding_ridge(
        train, folds.iloc[::-1], train_emb, test, test_emb, oof.iloc[::-1]
    )
    np.testing.assert_array_equal(
        original.fold_test_predictions, shuffled.fold_test_predictions
    )
    np.testing.assert_array_equal(original.oof_predictions, shuffled.oof_predictions)
    np.testing.assert_array_equal(
        shuffled.predictions.label, shuffled.fold_test_predictions.mean(axis=0)
    )
    assert shuffled.fold_ids == (0, 1, 2, 3, 4)
    assert shuffled.fold_test_predictions.shape == (5, 216)
    assert shuffled.predictions.filename.tolist() == test.filename.tolist()


def test_ridge_configuration_and_fold_isolation(inputs, monkeypatch):
    real = module.Ridge
    calls = []

    class Spy:
        def __init__(self, **kwargs):
            assert kwargs == {"alpha": 1.0, "solver": "lsqr"}
            self.model = real(**kwargs)
            self.fold = len(calls)
            calls.append(self)
            self.predict_sizes = []

        def fit(self, x, y):
            train, folds, emb, *_ = inputs
            valid = folds.fold.to_numpy() == self.fold
            np.testing.assert_array_equal(x, emb.embeddings[~valid])
            np.testing.assert_array_equal(y, train.label.to_numpy()[~valid])
            self.model.fit(x, y)

        def predict(self, x):
            self.predict_sizes.append(len(x))
            return self.model.predict(x)

    monkeypatch.setattr(module, "Ridge", Spy)
    infer(inputs)
    assert len(calls) == 5
    assert all(call.predict_sizes[-1] == 216 for call in calls)


@pytest.mark.parametrize(
    "case",
    [
        "train_identity",
        "test_identity",
        "duplicate",
        "fold",
        "nonfinite",
        "split",
        "dimension",
        "train_count",
        "test_count",
        "parity",
        "label",
    ],
)
def test_rejects_invalid_inputs(inputs, case):
    train, folds, train_emb, test, test_emb, oof = inputs
    if case == "train_identity":
        train_emb.filenames[0] = "missing"
    elif case == "test_identity":
        test_emb.filenames[0] = "missing"
    elif case == "duplicate":
        test.loc[0, "filename"] = test.loc[1, "filename"]
    elif case == "fold":
        folds.loc[0, "fold"] = 5
    elif case == "nonfinite":
        test_emb.embeddings[0, 0] = np.inf
    elif case == "split":
        test_emb.split[0] = "train"
    elif case == "dimension":
        test_emb.dimension = 767
    elif case == "train_count":
        inputs = (train.iloc[:-1], *inputs[1:])
    elif case == "test_count":
        inputs = (*inputs[:3], test.iloc[:-1], *inputs[4:])
    elif case == "parity":
        oof.loc[0, "prediction"] += 0.01
    elif case == "label":
        train.loc[0, "label"] = np.nan
    with pytest.raises(ValueError):
        infer(inputs)


def test_nonfinite_model_prediction(inputs, monkeypatch):
    class BadRidge:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            pass

        def predict(self, x):
            return np.full(len(x), np.inf)

    monkeypatch.setattr(module, "Ridge", BadRidge)
    with pytest.raises(ValueError, match="finite"):
        infer(inputs)


def test_metric_gate(inputs, monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_METRICS", {"rmse": 123.0})
    with pytest.raises(ValueError, match="metric"):
        infer(inputs)


def test_fixed_blend_and_output(tmp_path):
    e003b = pd.DataFrame({"filename": ["b", "a"], "label": [-2.0, 8.0]})
    e005 = pd.DataFrame({"filename": ["a", "b"], "label": [7.0, -1.0]})
    result = module.blend_predictions(e005, e003b)
    np.testing.assert_array_equal(
        result.label, 0.65 * np.array([-1.0, 7.0]) + 0.35 * np.array([-2.0, 8.0])
    )
    output = tmp_path / "predictions.csv"
    module.write_predictions(result, e003b, output)
    written = pd.read_csv(output)
    assert written.columns.tolist() == ["filename", "label"]
    assert written.filename.tolist() == ["b", "a"]
    np.testing.assert_allclose(written.label, result.label, rtol=1e-15)


@pytest.mark.parametrize(
    "case", ["identity", "duplicate", "empty", "nan", "string", "schema"]
)
def test_e005_validation(case):
    canonical = pd.DataFrame({"filename": ["a", "b"], "label": [1.0, 2.0]})
    frame = canonical.copy()
    if case == "identity":
        frame.loc[0, "filename"] = "c"
    elif case == "duplicate":
        frame.loc[0, "filename"] = "b"
    elif case == "empty":
        frame.loc[0, "filename"] = " "
    elif case == "nan":
        frame.loc[0, "label"] = np.nan
    elif case == "string":
        frame["label"] = ["one", "two"]
    elif case == "schema":
        frame["extra"] = 0
    with pytest.raises(ValueError):
        module.blend_predictions(frame, canonical)


def test_cli_explicit_paths(inputs, tmp_path):
    from grammar_scoring.features.embeddings import save_embeddings
    from grammar_scoring.inference.embedding_ridge_cli import main

    train, folds, train_emb, test, test_emb, oof = inputs
    frames = {
        "train-csv": train,
        "folds": folds,
        "test-csv": test,
        "e003b-oof": oof,
        "e005-predictions": test.assign(label=2.5).iloc[::-1],
    }
    args = []
    for name, frame in frames.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        args.extend([f"--{name}", str(path)])
    for name, artifact in (
        ("train-embeddings", train_emb),
        ("test-embeddings", test_emb),
    ):
        path = tmp_path / f"{name}.npz"
        save_embeddings(artifact, path)
        args.extend([f"--{name}", str(path)])
    for name in ("output", "e003b-output", "diagnostics"):
        args.extend([f"--{name}", str(tmp_path / name)])
    assert main(args) == 0
    first = (tmp_path / "output").read_bytes()
    assert main(args) == 0
    assert (tmp_path / "output").read_bytes() == first
    written = pd.read_csv(tmp_path / "output")
    assert written.filename.tolist() == test.filename.tolist()
    assert written.columns.tolist() == ["filename", "label"]


def test_reversed_prediction_schema_rejected():
    frame = pd.DataFrame({"filename": ["a"], "label": [1.0]})
    with pytest.raises(ValueError, match="columns"):
        module.align_prediction_frame(frame[["label", "filename"]], frame)
