"""Strict CSV persistence for linguistic features, without labels or text."""

from pathlib import Path

import numpy as np
import pandas as pd

from grammar_scoring.features.linguistic import FEATURE_COLUMNS, IDENTITY_COLUMNS


def validate_features(
    frame: pd.DataFrame,
    expected: pd.DataFrame,
    split: str,
) -> None:
    """Validate exact schema, finite numeric values, and canonical identity order."""
    if split not in ("train", "test"):
        raise ValueError("Invalid split")
    if frame.columns.has_duplicates or len(set(FEATURE_COLUMNS)) != len(
        FEATURE_COLUMNS
    ):
        raise ValueError("Duplicate feature names")
    if tuple(frame.columns) != IDENTITY_COLUMNS + FEATURE_COLUMNS:
        raise ValueError("Feature schema mismatch")
    for source in (frame, expected):
        for name in IDENTITY_COLUMNS:
            if any(not isinstance(v, str) or not v.strip() for v in source[name]):
                raise ValueError("Missing identity")
        if not source["split"].eq(split).all():
            raise ValueError("Unexpected split")
        if source.duplicated(list(IDENTITY_COLUMNS)).any():
            raise ValueError("Duplicate identity")
    actual_ids = list(frame[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None))
    expected_ids = list(
        expected[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None)
    )
    if actual_ids != expected_ids:
        raise ValueError("Missing/unexpected identities or canonical ordering mismatch")
    numeric = frame[list(FEATURE_COLUMNS)]
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in numeric.dtypes):
        raise ValueError("Features must be numeric")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite")


def save_features(
    frame: pd.DataFrame,
    path: Path,
    expected: pd.DataFrame,
    split: str,
    *,
    overwrite: bool = False,
) -> None:
    """Validate and save an inspectable CSV, protecting existing artifacts."""
    validate_features(frame, expected, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def load_features(path: Path, expected: pd.DataFrame, split: str) -> pd.DataFrame:
    """Load CSV and validate against canonical transcripts without accessing labels."""
    # Check the header before pandas can silently rename duplicate columns.
    import csv

    with path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), [])
    if tuple(header) != IDENTITY_COLUMNS + FEATURE_COLUMNS:
        raise ValueError("Feature schema mismatch or duplicate feature names")
    frame = pd.read_csv(
        path, dtype={"split": str, "filename": str}, keep_default_na=False
    )
    if frame.empty:
        frame = frame.astype(dict.fromkeys(FEATURE_COLUMNS, float))
    validate_features(frame, expected, split)
    return frame
