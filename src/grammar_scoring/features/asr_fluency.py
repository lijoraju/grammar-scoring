"""Fixed E013 temporal and lexical features from cached Whisper records."""

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

FEATURE_COLUMNS = (
    "segment_count",
    "words_per_segment_mean",
    "words_per_segment_std",
    "segment_duration_mean",
    "segment_duration_std",
    "short_segment_ratio",
    "recognized_speech_seconds",
    "speech_coverage_ratio",
    "words_per_recognized_speech_second",
    "gap_count",
    "gap_mean",
    "gap_std",
    "gap_max",
    "total_gap_seconds",
    "long_gap_0_5_count",
    "long_gap_1_0_count",
    "long_gap_2_0_count",
    "filler_rate",
    "immediate_repetition_rate",
)
FILLERS = frozenset({"um", "uh", "erm", "hmm"})


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def extract_features(record: Mapping[str, Any]) -> dict[str, float]:
    """Extract 19 finite features in fixed order from one canonical record.

    Args:
        record: Final text, duration_seconds, and ordered Whisper segments.

    Returns:
        Float features using population standard deviations and safe zero ratios.

    Raises:
        ValueError: If required fields or segment values are invalid or nonfinite.
    """
    if not {"text", "duration_seconds", "segments"}.issubset(record):
        raise ValueError("Missing required transcript fields")
    if not isinstance(record["text"], str) or not isinstance(record["segments"], list):
        raise ValueError("Transcript text must be string and segments must be list")
    duration = _number(record["duration_seconds"], "duration_seconds")
    if duration < 0:
        raise ValueError("duration_seconds must be nonnegative")
    starts, ends, words = [], [], []
    for segment in record["segments"]:
        if not isinstance(segment, dict) or not {"start", "end", "text"}.issubset(
            segment
        ):
            raise ValueError("Missing required segment fields")
        if not isinstance(segment["text"], str):
            raise ValueError("Segment text must be string")
        starts.append(_number(segment["start"], "start"))
        ends.append(_number(segment["end"], "end"))
        words.append(len(_tokens(segment["text"])))
    lengths = np.maximum(0.0, np.asarray(ends) - np.asarray(starts))
    gaps = np.maximum(0.0, np.asarray(starts[1:]) - np.asarray(ends[:-1]))
    tokens = _tokens(record["text"])
    speech = float(lengths.sum())

    def mean(values: list[int] | np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else 0.0

    def std(values: list[int] | np.ndarray) -> float:
        return float(np.std(values)) if len(values) >= 2 else 0.0

    values = (
        len(lengths),
        mean(words),
        std(words),
        mean(lengths),
        std(lengths),
        mean(lengths < 2.0),
        speech,
        speech / duration if duration else 0.0,
        len(tokens) / speech if speech else 0.0,
        len(gaps),
        mean(gaps),
        std(gaps),
        float(gaps.max()) if len(gaps) else 0.0,
        float(gaps.sum()),
        int((gaps >= 0.5).sum()),
        int((gaps >= 1.0).sum()),
        int((gaps >= 2.0).sum()),
        sum(token in FILLERS for token in tokens) / len(tokens) if tokens else 0.0,
        sum(a == b for a, b in zip(tokens, tokens[1:], strict=False))
        / max(len(tokens) - 1, 1),
    )
    if not np.isfinite(values).all():
        raise ValueError("Extracted features must be finite")
    return dict(zip(FEATURE_COLUMNS, map(float, values), strict=True))
