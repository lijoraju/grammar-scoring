from pathlib import Path

import pandas as pd
import pytest

from grammar_scoring.data import dataset


@pytest.fixture
def synthetic_dataset(tmp_path, monkeypatch):
    root = tmp_path / "Dataset_Final"
    root.mkdir()
    for name in ("train", "test"):
        directory = root / name
        directory.mkdir()
        for filename in ("audio_0.wav", "audio_1.wav"):
            (directory / filename).touch()
    frames = {
        "train.csv": pd.DataFrame(
            {"filename": ["audio_0.wav", "audio_1.wav"], "label": [0, 5]}
        ),
        "test.csv": pd.DataFrame(
            {"filename": ["audio_0.wav", "audio_1.wav"], "label": [99, -10]}
        ),
        "sample_submission.csv": pd.DataFrame(
            {"filename": ["audio_0.wav", "audio_1.wav"], "label": [0, 0]}
        ),
    }
    for name, frame in frames.items():
        frame.to_csv(root / name, index=False)
    for constant, relative in {
        "DATASET_DIR": ".",
        "TRAIN_CSV": "train.csv",
        "TEST_CSV": "test.csv",
        "SAMPLE_SUBMISSION_CSV": "sample_submission.csv",
        "TRAIN_AUDIO_DIR": "train",
        "TEST_AUDIO_DIR": "test",
    }.items():
        monkeypatch.setattr(dataset, constant, root / relative)
    return root


@pytest.mark.parametrize(
    ("loader", "name"),
    [
        (dataset.load_train_dataframe, "train.csv"),
        (dataset.load_test_dataframe, "test.csv"),
        (dataset.load_sample_submission, "sample_submission.csv"),
    ],
)
def test_loading(synthetic_dataset, loader, name):
    expected = pd.read_csv(synthetic_dataset / name, dtype={"filename": "string"})
    pd.testing.assert_frame_equal(loader(), expected)


def test_split_paths(synthetic_dataset):
    assert dataset.get_train_audio_path("audio_0.wav") == (
        synthetic_dataset / "train" / "audio_0.wav"
    )
    assert dataset.get_test_audio_path("audio_0.wav") == (
        synthetic_dataset / "test" / "audio_0.wav"
    )


@pytest.mark.parametrize(
    "filename", ["", " ", "../audio.wav", "/audio.wav", "a/b.wav", "a\\b.wav"]
)
@pytest.mark.parametrize(
    "resolver", [dataset.get_train_audio_path, dataset.get_test_audio_path]
)
def test_invalid_audio_basename(filename, resolver):
    with pytest.raises(ValueError, match="basename"):
        resolver(filename)


def test_valid_dataset(synthetic_dataset):
    assert dataset.validate_dataset() == {}


@pytest.mark.parametrize(
    "name", ["train.csv", "test.csv", "sample_submission.csv", "train", "test"]
)
def test_missing_required_path(synthetic_dataset, name):
    path = synthetic_dataset / name
    if path.is_dir():
        path.rename(synthetic_dataset / f"removed_{name}")
    else:
        path.unlink()
    with pytest.raises(FileNotFoundError, match="Required"):
        dataset.validate_dataset()


@pytest.mark.parametrize("split", ["train", "test"])
def test_missing_audio(synthetic_dataset, split):
    (synthetic_dataset / split / "audio_0.wav").unlink()
    with pytest.raises(FileNotFoundError, match="CSV-referenced"):
        dataset.validate_dataset()


@pytest.mark.parametrize("name", ["train.csv", "test.csv", "sample_submission.csv"])
@pytest.mark.parametrize("column", ["filename", "label"])
def test_missing_column(synthetic_dataset, name, column):
    path = synthetic_dataset / name
    pd.read_csv(path).drop(columns=column).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        dataset.validate_dataset()


@pytest.mark.parametrize("name", ["train.csv", "test.csv", "sample_submission.csv"])
@pytest.mark.parametrize(
    "value, message",
    [
        (None, "missing filenames"),
        (" ", "missing filenames"),
        ("audio_1.wav", "duplicate filenames"),
    ],
)
def test_invalid_filenames(synthetic_dataset, name, value, message):
    path = synthetic_dataset / name
    frame = pd.read_csv(path)
    frame.loc[0, "filename"] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        dataset.validate_dataset()


@pytest.mark.parametrize(
    "value, message",
    [
        (None, "missing values"),
        ("bad", "numeric"),
        (-0.1, "within"),
        (5.1, "within"),
        (float("inf"), "within"),
    ],
)
def test_invalid_training_label(synthetic_dataset, value, message):
    pd.DataFrame(
        {"filename": ["audio_0.wav", "audio_1.wav"], "label": [value, 3]}
    ).to_csv(synthetic_dataset / "train.csv", index=False)
    with pytest.raises(ValueError, match=message):
        dataset.validate_dataset()


def test_unreferenced_audio(synthetic_dataset):
    extra = synthetic_dataset / "train" / "extra.WAV"
    extra.touch()
    assert dataset.validate_dataset() == {}
    assert dataset.validate_dataset(identify_unreferenced=True) == {
        "train": [extra],
        "test": [],
    }


def test_loading_does_not_validate(synthetic_dataset):
    (synthetic_dataset / "train" / "audio_0.wav").unlink()
    assert len(dataset.load_train_dataframe()) == 2


def test_configured_paths():
    assert dataset.DATASET_DIR == dataset.DATA_DIR / "Dataset_Final"
    for name in (
        "TRAIN_CSV",
        "TEST_CSV",
        "SAMPLE_SUBMISSION_CSV",
        "TRAIN_AUDIO_DIR",
        "TEST_AUDIO_DIR",
    ):
        assert isinstance(getattr(dataset, name), Path)
