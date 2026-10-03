"""Load cached ASR text without normalization or label interpretation."""

import json
import math
from pathlib import Path
from typing import Literal

import pandas as pd

from grammar_scoring.config.paths import TRANSCRIPTS_DIR
from grammar_scoring.data import dataset

_REQUIRED_FIELDS = (
    "split",
    "filename",
    "text",
    "language",
    "duration_seconds",
    "segments",
)


def load_transcript_jsonl(path: Path) -> pd.DataFrame:
    """Load and validate a UTF-8 transcript JSONL artifact.

    Blank lines are ignored. Each nonblank line must contain a JSON object.
    Additional fields are retained; text and segment contents are unchanged.
    Identities are unique pairs of split and filename, not filenames alone.

    Args:
        path: Path to the cached transcript artifact.

    Returns:
        Records in artifact order, with required columns even for an empty file.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If encoding, JSON, required fields, or identities are invalid.
    """
    records: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                context = f"{path}: line {line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{context}: malformed JSON: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{context}: expected a JSON object")
                missing = set(_REQUIRED_FIELDS) - record.keys()
                if missing:
                    raise ValueError(
                        f"{context}: missing required fields: {sorted(missing)}"
                    )
                for field in ("split", "filename"):
                    value = record[field]
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"{context}: {field} must be a nonempty string"
                        )
                if not isinstance(record["text"], str):
                    raise ValueError(f"{context}: text must be a string")
                duration = record["duration_seconds"]
                try:
                    valid_duration = (
                        isinstance(duration, (int, float))
                        and not isinstance(duration, bool)
                        and duration >= 0
                        and math.isfinite(duration)
                    )
                except OverflowError:
                    valid_duration = False
                if not valid_duration:
                    raise ValueError(
                        f"{context}: duration_seconds must be finite and non-negative"
                    )
                if not isinstance(record["segments"], list):
                    raise ValueError(f"{context}: segments must be a list")
                identity = (record["split"], record["filename"])
                if identity in identities:
                    raise ValueError(f"{context}: duplicate identity {identity!r}")
                identities.add(identity)
                records.append(record)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: artifact must be valid UTF-8") from exc
    if not records:
        return pd.DataFrame(columns=list(_REQUIRED_FIELDS))
    return pd.DataFrame.from_records(records)


def load_competition_transcripts(split: Literal["train", "test"]) -> pd.DataFrame:
    """Join canonical transcripts to competition rows in competition order.

    Labels are preserved without inspection. Test labels are placeholders and
    must never be interpreted as ground truth. Only required transcript fields
    are joined, so arbitrary artifact fields cannot overwrite competition data.

    Args:
        split: Competition split, either train or test.

    Returns:
        Competition dataframe with split, raw text, language, duration_seconds,
        and segments, preserving its row order and index.

    Raises:
        FileNotFoundError: If a required artifact or competition CSV is absent.
        ValueError: If the split, records, or filename alignment are invalid.
    """
    if split not in ("train", "test"):
        raise ValueError(f"Expected split 'train' or 'test', got {split!r}")
    path = TRANSCRIPTS_DIR / f"{split}.jsonl"
    transcripts = load_transcript_jsonl(path)
    if not transcripts["split"].eq(split).all():
        raise ValueError(f"{path}: every record must have requested split {split!r}")
    competition = (
        dataset.load_train_dataframe()
        if split == "train"
        else dataset.load_test_dataframe()
    )
    if "filename" not in competition.columns:
        raise ValueError(f"{split}: competition dataframe is missing filename column")
    filenames = competition["filename"]
    if any(not isinstance(name, str) or not name.strip() for name in filenames):
        raise ValueError(f"{split}: competition filenames must be nonempty strings")
    if filenames.duplicated().any():
        raise ValueError(f"{split}: duplicate competition filenames")
    expected = set(filenames)
    actual = set(transcripts["filename"])
    issues = []
    if missing := expected - actual:
        issues.append(f"missing transcripts: {sorted(missing)}")
    if extra := actual - expected:
        issues.append(f"unexpected extra transcripts: {sorted(extra)}")
    if issues:
        raise ValueError(f"{path}: {'; '.join(issues)}")
    metadata = [field for field in _REQUIRED_FIELDS if field != "filename"]
    if collisions := set(metadata) & set(competition.columns):
        raise ValueError(f"{split}: transcript column conflicts: {sorted(collisions)}")
    aligned = transcripts.set_index("filename").reindex(filenames)
    result = competition.copy()
    for column in metadata:
        result[column] = aligned[column].to_numpy()
    return result
