"""Canonical label-free acoustic artifacts and distribution shift diagnostics."""

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from grammar_scoring.config.paths import FEATURES_DIR
from grammar_scoring.data.dataset import (
    get_test_audio_path,
    get_train_audio_path,
    load_test_dataframe,
    load_train_dataframe,
)
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.features.acoustic import (
    ARTIFACT_COLUMNS,
    CONFIG,
    FEATURE_COLUMNS,
    extract_features,
    extraction_config,
    load_waveform,
)


def validate_features(frame: pd.DataFrame, expected: pd.DataFrame, split: str) -> None:
    """Validate ordered label-free schema, identities, split and finite values."""
    if split not in ("train", "test"):
        raise ValueError("Invalid split")
    if tuple(frame.columns) != ARTIFACT_COLUMNS:
        raise ValueError("Acoustic feature schema mismatch")
    for source in (frame, expected):
        _validate_filenames(source)
    if list(frame.filename) != list(expected.filename):
        raise ValueError("Acoustic identities or canonical ordering mismatch")
    if not frame["split"].eq(split).all():
        raise ValueError("Unexpected acoustic split")
    numeric = frame[["duration_seconds", *FEATURE_COLUMNS]]
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in numeric.dtypes):
        raise ValueError("Acoustic features must be numeric")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Acoustic features must be finite")
    if not frame.duration_seconds.gt(0).all():
        raise ValueError("Audio duration must be positive")


def load_features(path: Path, expected: pd.DataFrame, split: str) -> pd.DataFrame:
    """Load features while rejecting duplicate CSV headers and identity mismatch."""
    with path.open(encoding="utf-8", newline="") as stream:
        if tuple(next(csv.reader(stream), [])) != ARTIFACT_COLUMNS:
            raise ValueError("Acoustic CSV schema mismatch")
    frame = pd.read_csv(path, dtype={"filename": str, "split": str})
    validate_features(frame, expected, split)
    return frame


def generate_acoustic_features(*, overwrite: bool = False) -> dict[str, object]:
    """Extract canonical 769/216 WAVs and persist configuration/input fingerprints.

    Test CSV reads filenames only; neither train nor test labels enter extraction.
    Existing outputs require explicit overwrite; metadata fingerprints support
    cache auditing rather than silently accepting stale feature files.
    """
    started = perf_counter()
    tables = {
        "train": load_train_dataframe()[["filename"]],
        "test": load_test_dataframe()[["filename"]],
    }
    outputs = [FEATURES_DIR / f"acoustic_{split}.csv" for split in tables]
    metadata_path = FEATURES_DIR / "acoustic_metadata.json"
    if not overwrite and any(path.exists() for path in [*outputs, metadata_path]):
        raise FileExistsError(
            "Acoustic artifacts exist; pass --overwrite to regenerate"
        )
    extracted = {}
    inputs = {}
    for split, expected in tables.items():
        _validate_filenames(expected)
        if len(expected) != {"train": 769, "test": 216}[split]:
            raise ValueError(f"Unexpected canonical {split} row count")
        resolve = get_train_audio_path if split == "train" else get_test_audio_path
        rows = []
        records = []
        for filename in expected.filename:
            path = resolve(filename)
            waveform = load_waveform(path)
            rows.append(
                {
                    "filename": filename,
                    "split": split,
                    "duration_seconds": waveform.size / CONFIG.sample_rate,
                    **extract_features(waveform),
                }
            )
            records.append(
                {
                    "filename": filename,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sample_rate": CONFIG.sample_rate,
                    "channel_count": 1,
                    "sample_count": int(waveform.size),
                    "finite": True,
                    "readable": True,
                    "sample_width_bytes": 2,
                }
            )
        frame = pd.DataFrame(rows, columns=ARTIFACT_COLUMNS)
        validate_features(frame, expected, split)
        extracted[split] = frame
        inputs[split] = records
    metadata = {
        "experiment": "E006a",
        "configuration": extraction_config(),
        "feature_columns": FEATURE_COLUMNS,
        "artifact_columns": ARTIFACT_COLUMNS,
        "input_files": inputs,
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("acoustic.py", "acoustic_artifacts.py")
        },
        "extraction_seconds": perf_counter() - started,
    }
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for split, frame in extracted.items():
        with (FEATURES_DIR / f"acoustic_{split}.csv").open(
            "w" if overwrite else "x", encoding="utf-8", newline=""
        ) as stream:
            frame.to_csv(stream, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def shift_audit(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Audit duration and every feature with signed pooled-population-SD SMD.

    SMD = (test_mean - train_mean) / sqrt((train_var + test_var) / 2).
    Equal constant distributions yield zero; unequal constants yield signed
    infinity and a large-shift flag. Thresholds are diagnostic, never selection.
    """
    rows = []
    for column in ("duration_seconds", *FEATURE_COLUMNS):
        a = train[column].to_numpy(dtype=float)
        b = test[column].to_numpy(dtype=float)
        if not a.size or not b.size or not np.isfinite(np.r_[a, b]).all():
            raise ValueError("Shift audit requires nonempty finite feature values")
        scale = float(np.sqrt((a.var() + b.var()) / 2))
        delta = float(b.mean() - a.mean())
        smd = (
            delta / scale
            if scale
            else (0.0 if delta == 0 else np.copysign(np.inf, delta))
        )
        rows.append(
            {
                "feature": column,
                "model_feature": column in FEATURE_COLUMNS,
                "train_mean": a.mean(),
                "test_mean": b.mean(),
                "train_std": a.std(),
                "test_std": b.std(),
                "train_min": a.min(),
                "train_max": a.max(),
                "test_min": b.min(),
                "test_max": b.max(),
                "smd": smd,
                "shift": "large"
                if abs(smd) >= 0.8
                else "moderate"
                if abs(smd) >= 0.5
                else "low",
            }
        )
    return pd.DataFrame(rows)
