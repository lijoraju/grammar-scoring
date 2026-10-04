"""Offline E009 population, cache, leakage and comparison contracts."""

import inspect

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.evaluation.metrics import pearson_correlation
from grammar_scoring.experiments import wavlm_ridge as experiment
from grammar_scoring.features.wavlm import (
    CHUNK_SAMPLES,
    chunk_audio,
    minimum_input_samples,
    validate_embeddings,
    weighted_embedding,
)


@pytest.fixture
def inputs():
    n = 769
    train = pd.DataFrame(
        {
            "filename": [f"arbitrary_{i * 17}.wav" for i in range(n)],
            "label": np.r_[np.zeros(37), np.linspace(1, 5, 732)],
        }
    )
    folds = train.copy()
    folds["fold"] = np.repeat(np.arange(5), [154, 154, 154, 154, 153])
    retained = experiment.retained_population(train, folds.iloc[::-1])
    metadata = retained.assign(duration_seconds=41.0, num_chunks=3)
    matrix = np.random.default_rng(42).normal(size=(732, 768)).astype(np.float32)
    return train, folds, retained, metadata, matrix


def test_population_and_folds(inputs):
    train, folds, retained, _, _ = inputs
    assert len(retained) == 732
    assert retained.filename.tolist() == train.filename.iloc[37:].tolist()
    assert retained.fold.tolist() == folds.fold.iloc[37:].tolist()
    train.loc[0, "label"], train.loc[50, "label"] = 3, 0
    changed = experiment.retained_population(train, folds.drop(columns="label"))
    assert train.filename[0] in set(changed.filename)
    assert train.filename[50] not in set(changed.filename)


@pytest.mark.parametrize(
    "size,counts",
    [
        (16000, [16000]),
        (CHUNK_SAMPLES, [CHUNK_SAMPLES]),
        (CHUNK_SAMPLES + 16000, [CHUNK_SAMPLES, 16000]),
        (2 * CHUNK_SAMPLES + 13, [CHUNK_SAMPLES, CHUNK_SAMPLES + 13]),
        (3 * CHUNK_SAMPLES, [CHUNK_SAMPLES] * 3),
        (CHUNK_SAMPLES + 399, [CHUNK_SAMPLES + 399]),
        (CHUNK_SAMPLES + 400, [CHUNK_SAMPLES, 400]),
    ],
)
def test_chunking_no_truncation(size, counts):
    audio = np.arange(size, dtype=np.float32)
    chunks = chunk_audio(audio)
    assert [len(chunk) for chunk in chunks] == counts
    np.testing.assert_array_equal(np.concatenate(chunks), audio)
    np.testing.assert_array_equal(np.concatenate(chunk_audio(audio)), audio)


def test_receptive_field_derived_from_convolutions():
    assert minimum_input_samples() == 400
    assert minimum_input_samples((3, 2), (2, 3)) == 5


def test_duration_weighting():
    np.testing.assert_allclose(
        weighted_embedding(np.array([[1, 2], [4, 8]]), np.array([20, 10])),
        [2, 4],
    )


@pytest.mark.parametrize(
    "damage", ["nan", "inf", "dimension", "row", "order", "label", "fold", "duplicate"]
)
def test_invalid_embeddings_and_alignment(inputs, damage):
    _, _, retained, metadata, matrix = inputs
    if damage in {"nan", "inf"}:
        matrix[0, 0] = np.nan if damage == "nan" else np.inf
    elif damage == "dimension":
        matrix = matrix[:, :-1]
    elif damage == "row":
        matrix = matrix[:-1]
    elif damage == "order":
        metadata = metadata.iloc[::-1]
    elif damage == "duplicate":
        metadata.loc[1, "filename"] = metadata.loc[0, "filename"]
    else:
        metadata.loc[0, damage] += 1
    with pytest.raises(ValueError):
        validate_embeddings(matrix, metadata, retained)


def test_fold_local_scaler_fixed_ridge_and_coverage(inputs, monkeypatch):
    _, _, retained, metadata, matrix = inputs
    fitted, alphas = [], []
    original_scaler = experiment.StandardScaler
    original_ridge = experiment.Ridge

    class SpyScaler(original_scaler):
        def fit(self, x, y=None, sample_weight=None):
            fitted.append(x.copy())
            return super().fit(x, y, sample_weight=sample_weight)

    def ridge(*, alpha):
        alphas.append(alpha)
        return original_ridge(alpha=alpha)

    monkeypatch.setattr(experiment, "StandardScaler", SpyScaler)
    monkeypatch.setattr(experiment, "Ridge", ridge)
    oof = experiment.evaluate_ridge(matrix, metadata, retained)
    assert alphas == [1.0] * 5
    for fold, partition in enumerate(fitted):
        np.testing.assert_array_equal(partition, matrix[retained.fold != fold])
    assert len(oof) == 732 and oof.filename.is_unique
    assert np.isfinite(oof.prediction).all()
    np.testing.assert_allclose(oof.residual, oof.label - oof.prediction)
    np.testing.assert_allclose(oof.abs_error, abs(oof.residual))


@pytest.mark.parametrize("column", ["filename", "label", "fold"])
def test_comparison_rejects_alignment(inputs, column):
    _, _, retained, _, _ = inputs
    oof = retained.assign(prediction=np.linspace(0, 4, 732))
    baseline = oof.copy()
    baseline.loc[0, column] = "other.wav" if column == "filename" else 99
    with pytest.raises(ValueError):
        experiment.compare_oof(oof, baseline, retained)


def test_residual_comparison_recomputes_convention(inputs):
    _, _, retained, _, _ = inputs
    random = np.random.default_rng(42)
    oof = retained.assign(prediction=random.normal(size=732), residual=999)
    baseline = retained.assign(prediction=random.normal(size=732), residual=-999)
    report = experiment.compare_oof(oof, baseline.iloc[::-1], retained)
    assert report["residual_correlation"] == pytest.approx(
        pearson_correlation(
            retained.label - baseline.prediction, retained.label - oof.prediction
        )
    )
    assert len(report["per_fold"]) == 5


def test_cli_train_only():
    source = inspect.getsource(experiment)
    for prohibited in ("load_test", "get_test", "submission", "blend", "GridSearch"):
        assert prohibited not in source


def test_frozen_encoder_batch_invariance_and_masks():
    from types import SimpleNamespace

    import torch

    from grammar_scoring.features.wavlm import FrozenWavLM

    class Model:
        config = SimpleNamespace(
            hidden_size=768,
            _commit_hash="offline",
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
        )

        def to(self, device):
            return self

        def float(self):
            return self

        def eval(self):
            self.evaluating = True
            return self

        def requires_grad_(self, value):
            self.frozen = not value
            return self

        def _get_feat_extract_output_lengths(self, lengths):
            return torch.ones(len(lengths), dtype=torch.int64)

        def __call__(self, **inputs):
            assert not torch.is_grad_enabled()
            assert self.evaluating and self.frozen
            means = inputs["input_values"].mean(dim=1)
            valid = means[:, None, None].expand(-1, 1, 768)
            padding = torch.full_like(valid, 999)
            return SimpleNamespace(last_hidden_state=torch.cat([valid, padding], dim=1))

    def processor(chunks, **options):
        assert options["sampling_rate"] == 16000
        assert options["padding"] is False
        return {"input_values": torch.tensor([c.tolist() for c in chunks])}

    audio = np.r_[np.ones(CHUNK_SAMPLES * 2), np.full(1000, 4)]
    encoder = FrozenWavLM("cpu", 4, model=Model(), processor=processor)
    vector, chunks = encoder.encode(audio)
    reference, _ = FrozenWavLM("cpu", 1, model=Model(), processor=processor).encode(
        audio
    )
    assert chunks == 3
    np.testing.assert_allclose(vector, audio.mean(), rtol=1e-6)
    np.testing.assert_allclose(vector, reference)


def test_cache_incompatibility_rejected_before_encoding(inputs, tmp_path):
    import json
    from types import SimpleNamespace

    from grammar_scoring.features.wavlm import extract_embeddings

    _, _, retained, _, _ = inputs
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for name in retained.filename:
        (audio_dir / name).write_bytes(b"source identity only")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "extraction.json").write_text(json.dumps({"configuration": {}}))
    (cache / "train_embeddings.npy").touch()
    (cache / "train_metadata.csv").touch()
    # The seam bypasses optional runtime version discovery while checking rejection.
    from unittest.mock import patch

    with patch("importlib.metadata.version", return_value="offline"):
        with pytest.raises(ValueError, match="Incompatible"):
            extract_embeddings(
                retained,
                audio_dir,
                cache,
                SimpleNamespace(revision="old", minimum_samples=400),
            )


def test_merged_chunk_metadata(inputs):
    _, _, retained, metadata, matrix = inputs
    metadata.loc[0, "duration_seconds"] = (2 * CHUNK_SAMPLES + 13) / 16000
    metadata.loc[0, "num_chunks"] = 2
    validate_embeddings(matrix, metadata, retained)
    metadata.loc[0, "num_chunks"] = 3
    with pytest.raises(ValueError, match="Chunk counts"):
        validate_embeddings(matrix, metadata, retained)


@pytest.mark.parametrize(
    "artifact", ["train_embeddings.npy", "train_metadata.csv", "extraction.json"]
)
def test_incomplete_cache(inputs, tmp_path, artifact):
    from grammar_scoring.features.wavlm import extract_embeddings

    _, _, retained, _, _ = inputs
    (tmp_path / artifact).touch()
    with pytest.raises(ValueError, match="Incomplete E009 extraction cache"):
        extract_embeddings(retained, tmp_path / "absent", tmp_path, None)


@pytest.mark.parametrize(
    "durations,missing",
    [([10, 30, 50, 40], []), ([25, 30, 40], ["<20 seconds", ">40 seconds"])],
)
def test_smoke_duration_selection(monkeypatch, tmp_path, durations, missing):
    from types import SimpleNamespace

    names = [f"recording_{i}.wav" for i in range(len(durations))]
    lookup = dict(zip(names, durations, strict=True))
    monkeypatch.setattr(
        experiment,
        "read_audio_metadata",
        lambda path: SimpleNamespace(duration_seconds=lookup[path.name]),
    )
    selected, absent = experiment.select_smoke_recordings(
        pd.DataFrame({"filename": names}), tmp_path
    )
    assert absent == missing
    assert len(selected) == len(set(selected)) == 3
    if not missing:
        assert sorted(lookup[name] for name in selected) == [10, 40, 50]


def test_smoke_report_offline(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from grammar_scoring.features import acoustic

    waveforms = {
        "short.wav": np.ones(16000),
        "medium.wav": np.ones(CHUNK_SAMPLES + 16000),
        "long.wav": np.ones(2 * CHUNK_SAMPLES + 13),
    }
    monkeypatch.setattr(
        experiment, "select_smoke_recordings", lambda *_: (list(waveforms), [])
    )
    monkeypatch.setattr(acoustic, "load_waveform", lambda path: waveforms[path.name])

    class Encoder:
        device = "cpu"
        batch_size = 4
        revision = "offline-commit"
        minimum_samples = 400
        processor = None
        model = SimpleNamespace(
            _get_feat_extract_output_lengths=lambda counts: (counts - 400) // 320 + 1
        )

        def encode(self, waveform):
            return np.ones(768, dtype=np.float32), len(chunk_audio(waveform))

    monkeypatch.setattr(experiment, "FrozenWavLM", lambda *args, **kwargs: Encoder())
    report = experiment.run_smoke(
        pd.DataFrame(), tmp_path, Encoder(), tmp_path / "summary.json"
    )
    assert report["resolved_model_revision"] == "offline-commit"
    assert report["dtype"] == "float32"
    assert report["peak_cuda_memory_bytes"] is None
    assert report["chunk_batch_size"] == 4
    assert (tmp_path / "summary.json").is_file()
    for row in report["recordings"]:
        assert row["waveform_samples"] == sum(row["chunk_sample_counts"])
        assert row["duration_seconds"] == row["waveform_samples"] / 16000
        assert row["embedding_dimension"] == 768 and row["finite"]
        assert len(row["valid_hidden_frame_counts"]) == row["num_chunks"]
        assert row["batch_size_invariance_passed"]


@pytest.fixture
def cli_inputs(inputs, tmp_path, monkeypatch):
    import sys

    train, folds, retained, metadata, matrix = inputs
    # Use canonical half-point targets so CSV round trips preserve exact identity.
    train["label"] = np.round(train.label * 2) / 2
    folds["label"] = train.label
    retained = experiment.retained_population(train, folds)
    metadata["label"] = retained.label
    fold_path = tmp_path / "folds.csv"
    folds.to_csv(fold_path, index=False)
    monkeypatch.setattr(experiment, "load_train_dataframe", lambda: train)
    monkeypatch.setattr(experiment, "FrozenWavLM", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        experiment, "extract_embeddings", lambda *args: (matrix, metadata)
    )
    argv = [
        "run_wavlm_ridge.py",
        "--fold-path",
        str(fold_path),
        "--artifact-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return retained, tmp_path, argv


@pytest.mark.parametrize("supplied", [False, True])
def test_smoke_cli_never_reads_baseline(cli_inputs, monkeypatch, supplied):
    _, directory, argv = cli_inputs
    argv.append("--smoke")
    if supplied:
        argv.extend(["--e008-oof", str(directory / "nonexistent.csv")])
    calls = []
    monkeypatch.setattr(experiment, "run_smoke", lambda *args: calls.append(args))
    monkeypatch.setattr(
        experiment,
        "extract_embeddings",
        lambda *args: pytest.fail("Full extraction invoked by smoke"),
    )
    experiment.main()
    assert len(calls) == 1
    assert not (directory / "oof/E009_wavlm_ridge.csv").exists()


def test_full_cli_without_baseline(cli_inputs):
    import json

    retained, directory, _ = cli_inputs
    experiment.main()
    oof = pd.read_csv(directory / "oof/E009_wavlm_ridge.csv")
    report = json.loads((directory / "experiments/E009/diagnostics.json").read_text())
    assert len(oof) == 732 and oof.filename.is_unique
    assert oof.filename.tolist() == retained.filename.tolist()
    assert np.isfinite(oof.prediction).all()
    assert report["e009"] == pytest.approx(
        experiment.regression_metrics(oof.label, oof.prediction)
    )
    assert len(report["per_fold"]) == 5
    for row in report["per_fold"]:
        group = oof.loc[oof.fold == row["fold"]]
        assert row["e009"] == pytest.approx(
            experiment.regression_metrics(group.label, group.prediction)
        )
    assert report["e008_comparison_status"] == "not_run"
    assert report["e008_oof_path"] is None
    assert "e008" not in report and "rmse_delta" not in report
    assert all("e008" not in row for row in report["per_fold"])


def test_full_cli_with_baseline_preserves_comparison(cli_inputs):
    import json

    retained, directory, argv = cli_inputs
    baseline = retained.assign(prediction=np.linspace(1.2, 4.2, 732))
    path = directory / "baseline.csv"
    baseline.iloc[::-1].to_csv(path, index=False)
    argv.extend(["--e008-oof", str(path)])
    experiment.main()
    oof = pd.read_csv(directory / "oof/E009_wavlm_ridge.csv")
    report = json.loads((directory / "experiments/E009/diagnostics.json").read_text())
    expected = experiment.compare_oof(oof, pd.read_csv(path), retained)
    assert report["e008_comparison_status"] == "completed"
    assert report["e008_oof_path"] == str(path)
    for key in (
        "e008",
        "e009",
        "prediction_correlation",
        "residual_correlation",
        "rmse_delta",
        "pearson_delta",
    ):
        assert report[key] == pytest.approx(expected[key])
    for actual, comparison in zip(
        report["per_fold"], expected["per_fold"], strict=True
    ):
        for key in ("e008", "e009", "rmse_delta", "pearson_delta"):
            assert actual[key] == pytest.approx(comparison[key])


@pytest.mark.parametrize("column", ["filename", "label", "fold"])
def test_supplied_misaligned_baseline_fails_before_encoder(
    cli_inputs, monkeypatch, column
):
    retained, directory, argv = cli_inputs
    baseline = retained.assign(prediction=2.0)
    baseline.loc[0, column] = "other.wav" if column == "filename" else 99
    path = directory / "misaligned.csv"
    baseline.to_csv(path, index=False)
    argv.extend(["--e008-oof", str(path)])
    monkeypatch.setattr(
        experiment,
        "FrozenWavLM",
        lambda *args, **kwargs: pytest.fail(
            "Encoder constructed before alignment validation"
        ),
    )
    with pytest.raises(ValueError):
        experiment.main()


def test_supplied_missing_baseline_fails_clearly(cli_inputs, monkeypatch):
    _, directory, argv = cli_inputs
    argv.extend(["--e008-oof", str(directory / "missing.csv")])
    monkeypatch.setattr(
        experiment,
        "FrozenWavLM",
        lambda *args, **kwargs: pytest.fail(
            "Encoder constructed before path validation"
        ),
    )
    with pytest.raises(FileNotFoundError, match="E008 OOF file does not exist"):
        experiment.main()
