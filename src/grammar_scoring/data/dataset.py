"""Load competition tables and explicitly validate their filesystem integrity.

Test labels are competition-supplied random placeholders, never ground truth.
"""

from pathlib import Path

import pandas as pd

from grammar_scoring.config.paths import DATA_DIR

DATASET_DIR: Path = DATA_DIR / "Dataset_Final"
TRAIN_CSV: Path = DATASET_DIR / "train.csv"
TEST_CSV: Path = DATASET_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV: Path = DATASET_DIR / "sample_submission.csv"
TRAIN_AUDIO_DIR: Path = DATASET_DIR / "train"
TEST_AUDIO_DIR: Path = DATASET_DIR / "test"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file does not exist: {path}")
    try:
        return pd.read_csv(path, dtype={"filename": "string"})
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot parse dataset CSV {path}: {exc}") from exc


def load_train_dataframe() -> pd.DataFrame:
    """Load train.csv without running dataset validation.

    Returns:
        Training filenames and grammar labels.
    """
    return _load_csv(TRAIN_CSV)


def load_test_dataframe() -> pd.DataFrame:
    """Load test.csv, preserving its random placeholder labels.

    Returns:
        Test table; its label column must never be used as ground truth.
    """
    return _load_csv(TEST_CSV)


def load_sample_submission() -> pd.DataFrame:
    """Load sample_submission.csv without running dataset validation.

    Returns:
        Submission template with filename and label columns.
    """
    return _load_csv(SAMPLE_SUBMISSION_CSV)


def _audio_path(directory: Path, filename: str) -> Path:
    if (
        not isinstance(filename, str)
        or not filename.strip()
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
    ):
        raise ValueError(f"Expected a nonempty audio basename, got {filename!r}")
    return directory / filename


def get_train_audio_path(filename: str) -> Path:
    """Resolve a basename in the training split without checking existence.

    Args:
        filename: Audio basename without directory components.

    Returns:
        Path inside the training audio directory.

    Raises:
        ValueError: If filename is empty or contains directory components.
    """
    return _audio_path(TRAIN_AUDIO_DIR, filename)


def get_test_audio_path(filename: str) -> Path:
    """Resolve a basename in the test split without checking existence.

    Args:
        filename: Audio basename without directory components.

    Returns:
        Path inside the test audio directory.

    Raises:
        ValueError: If filename is empty or contains directory components.
    """
    return _audio_path(TEST_AUDIO_DIR, filename)


def _validate_table(frame: pd.DataFrame, path: Path) -> None:
    missing = {"filename", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    filenames = frame["filename"]
    if filenames.isna().any() or filenames.str.strip().eq("").any():
        raise ValueError(f"{path}: missing filenames")
    if filenames.duplicated().any():
        raise ValueError(f"{path}: duplicate filenames")
    for filename in filenames:
        _audio_path(path.parent, filename)


def _validate_train_labels(frame: pd.DataFrame) -> None:
    labels = frame["label"]
    if labels.isna().any():
        raise ValueError(f"{TRAIN_CSV}: training labels contain missing values")
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{TRAIN_CSV}: training labels must be numeric")
    if not numeric.between(0, 5, inclusive="both").all():
        raise ValueError(f"{TRAIN_CSV}: training labels must be within [0, 5]")


def _validate_audio(
    frame: pd.DataFrame, directory: Path, identify_unreferenced: bool
) -> list[Path]:
    referenced = {_audio_path(directory, filename) for filename in frame["filename"]}
    for path in sorted(referenced):
        if not path.is_file():
            raise FileNotFoundError(f"CSV-referenced audio file does not exist: {path}")
    if not identify_unreferenced:
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav" and path not in referenced
    )


def validate_dataset(*, identify_unreferenced: bool = False) -> dict[str, list[Path]]:
    """Validate tables, split audio files, labels, and submission alignment.

    Training scores must be numeric and within the inclusive range 0–5.
    Test and submission label values are deliberately not validated.
    No directories are created and no audio contents are read.

    Args:
        identify_unreferenced: Also list WAV files absent from each split CSV.
            Extra files are reported rather than treated as validation failures.

    Returns:
        Mapping of train/test to unreferenced WAV paths when requested;
        otherwise an empty dictionary.

    Raises:
        FileNotFoundError: If a required CSV, directory, or audio file is missing.
        ValueError: If a table, training label, or submission alignment is invalid.
    """
    train = load_train_dataframe()
    test = load_test_dataframe()
    submission = load_sample_submission()
    for directory in (TRAIN_AUDIO_DIR, TEST_AUDIO_DIR):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required audio directory does not exist: {directory}"
            )
    for frame, path in (
        (train, TRAIN_CSV),
        (test, TEST_CSV),
        (submission, SAMPLE_SUBMISSION_CSV),
    ):
        _validate_table(frame, path)
    _validate_train_labels(train)
    if len(submission) != len(test):
        raise ValueError("sample_submission.csv row count must match test.csv")
    if not submission["filename"].equals(test["filename"]):
        raise ValueError("sample_submission.csv filenames/order must match test.csv")
    extras = {
        "train": _validate_audio(train, TRAIN_AUDIO_DIR, identify_unreferenced),
        "test": _validate_audio(test, TEST_AUDIO_DIR, identify_unreferenced),
    }
    return extras if identify_unreferenced else {}
