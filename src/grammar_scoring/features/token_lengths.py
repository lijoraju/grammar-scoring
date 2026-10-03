"""Audit untruncated tokenizer sequences against frozen experiment limits."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from grammar_scoring.features.embeddings import MODELS, validate_transcripts
from grammar_scoring.features.interfaces import Tokenizer


@dataclass
class TokenLengthAudit:
    """Ordered sample counts, summary statistics, and visible tokenizer limits."""

    samples: pd.DataFrame
    summary: dict[str, int | float | None]
    configured_limit: int
    tokenizer_limit: Any


def audit_token_lengths(
    frame: pd.DataFrame, tokenizer: Tokenizer, limit: int
) -> TokenLengthAudit:
    """Count raw-text tokens including special tokens without truncation.

    Args:
        frame: Ordered transcripts with split, filename, and raw text.
        tokenizer: Hugging Face-compatible tokenizer.
        limit: Authoritative configured experiment sequence limit.

    Returns:
        Per-sample counts and summaries; empty numeric summaries are None.
    """
    validate_transcripts(frame)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    counts = np.asarray(
        [
            len(
                tokenizer(
                    text, add_special_tokens=True, truncation=False, padding=False
                )["input_ids"]
            )
            for text in frame["text"]
        ],
        dtype=np.int64,
    )
    samples = frame[["split", "filename"]].copy()
    samples["token_count"] = counts
    samples["exceeds_limit"] = counts > limit
    n = len(counts)
    exceeding = int((counts > limit).sum())
    summary = {
        "n_samples": n,
        "min": int(counts.min()) if n else None,
        "mean": float(counts.mean()) if n else None,
        "median": float(np.median(counts)) if n else None,
        "p95": float(np.percentile(counts, 95)) if n else None,
        "max": int(counts.max()) if n else None,
        "n_exceeding_limit": exceeding,
        "pct_exceeding_limit": 100 * exceeding / n if n else 0.0,
    }
    return TokenLengthAudit(
        samples, summary, limit, getattr(tokenizer, "model_max_length", None)
    )


def load_audit_tokenizer(variant: str) -> Tokenizer:
    """Load the canonical tokenizer lazily without modifying experiment limits."""
    if variant not in MODELS:
        raise ValueError(f"Unknown model: {variant}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODELS[variant][0])
