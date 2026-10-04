"""Generate canonical E005 test predictions without submission packaging."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from grammar_scoring.config.paths import ARTIFACT_DIR, MODELS_DIR, TRANSCRIPTS_DIR
from grammar_scoring.data.dataset import TEST_CSV
from grammar_scoring.data.transcripts import load_transcript_jsonl
from grammar_scoring.inference.deberta import predict_e005_ensemble, validate_test_frame


def load_test_inputs(test_csv: Path, transcripts: Path) -> pd.DataFrame:
    """Align raw test transcripts by filename in competition CSV order.

    Args:
        test_csv: Competition test table; placeholder labels are ignored.
        transcripts: Canonical raw Whisper JSONL artifact.

    Returns:
        Validated filename/text dataframe in test CSV order.

    Raises:
        ValueError: If inputs are empty, invalid, duplicated, or misaligned.
        FileNotFoundError: If either input is absent.
    """
    try:
        test = pd.read_csv(test_csv, dtype={"filename": "string"})
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot parse test CSV {test_csv}: {exc}") from exc
    if "filename" not in test.columns:
        raise ValueError("Test CSV missing filename column")
    filenames = test["filename"]
    if test.empty:
        raise ValueError("Test CSV must contain at least one example")
    if any(not isinstance(name, str) or not name.strip() for name in filenames):
        raise ValueError("Test filenames must be nonempty strings")
    if filenames.duplicated().any():
        raise ValueError("Test filenames must be unique")

    raw = load_transcript_jsonl(transcripts)
    if raw["filename"].duplicated().any():
        raise ValueError("Transcript filenames must be unique")
    if not raw["split"].eq("test").all():
        raise ValueError("Every transcript must have split 'test'")
    expected, actual = set(filenames), set(raw["filename"])
    issues = []
    if missing := expected - actual:
        issues.append(f"missing transcripts: {sorted(missing)}")
    if unexpected := actual - expected:
        issues.append(f"unexpected transcripts: {sorted(unexpected)}")
    if issues:
        raise ValueError("; ".join(issues))
    aligned = raw.set_index("filename").reindex(filenames)
    return validate_test_frame(aligned.reset_index()[["filename", "text"]])


def validate_predictions(predictions: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Check prediction identity and numeric integrity without changing values.

    Args:
        predictions: Ensemble result to write unchanged.
        frame: Validated input in canonical row order.

    Raises:
        ValueError: If columns, count, identities, or labels are invalid.
    """
    if predictions.columns.tolist() != ["filename", "label"]:
        raise ValueError("Prediction columns must be exactly filename,label in order")
    if len(predictions) != len(frame):
        raise ValueError("Prediction count differs from test row count")
    if predictions["filename"].duplicated().any():
        raise ValueError("Prediction filenames must be unique")
    if predictions["filename"].tolist() != frame["filename"].tolist():
        raise ValueError("Prediction filename order differs from test CSV")
    labels = predictions["label"]
    if (
        not is_numeric_dtype(labels.dtype)
        or is_bool_dtype(labels.dtype)
        or np.iscomplexobj(labels.to_numpy())
    ):
        raise ValueError("Prediction labels must be real numeric values")
    if not np.isfinite(labels.to_numpy(dtype=np.float64, na_value=np.nan)).all():
        raise ValueError("Prediction labels must be finite")


def main(argv: list[str] | None = None) -> int:
    """Generate E005 predictions and print their path and descriptive statistics.

    Args:
        argv: CLI arguments, or None to read process arguments.

    Returns:
        Zero after successfully writing the validated prediction CSV.
    """
    parser = argparse.ArgumentParser(description="Generate raw E005 test predictions")
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--transcripts", type=Path, default=TRANSCRIPTS_DIR / "test.jsonl"
    )
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR / "E005")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "submissions" / "E005_test_predictions.csv",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    options = parser.parse_args(argv)
    frame = load_test_inputs(options.test_csv, options.transcripts)
    predictions = predict_e005_ensemble(frame, options.model_dir, device=options.device)
    validate_predictions(predictions, frame)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(options.output, index=False)
    labels = predictions["label"]
    print(
        f"Saved {len(predictions)} predictions to {options.output}: "
        f"min={labels.min()}, max={labels.max()}, mean={labels.mean()}"
    )
    return 0
