"""E019: label-free disfluency and fluency features from verbatim transcripts.

Verbatim (E015) transcripts keep fillers, false starts and repetitions, and
their Whisper segments carry timestamps. Comparing them with the canonical
transcripts also measures how much Whisper's language model "repaired" the
speech: content words that differ once fillers are removed.
"""

from __future__ import annotations

import difflib
import re

FILLERS = re.compile(r"\b(?:u+m+|u+h+|h+m+|e+r+|a+h+|mm+)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z']+")
_FALSE_START = re.compile(r"\b\w+-(?=\s|$)")
_REPEAT = re.compile(r"\b(\w+)\b[,.]?\s+\1\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[.!?]+")


def content_words(text: str) -> list[str]:
    """Return lower-cased words without fillers or punctuation."""
    return _WORD.findall(FILLERS.sub(" ", text.lower()))


def repair_rate(canonical: str, verbatim: str) -> float:
    """Fraction of verbatim content words not matched in the canonical text.

    Args:
        canonical: Default Whisper transcript.
        verbatim: Disfluency-preserving transcript of the same audio.

    Returns:
        ``1 - matched / len(verbatim words)``; 0.0 for an empty verbatim text.
    """
    target = content_words(verbatim)
    if not target:
        return 0.0
    matcher = difflib.SequenceMatcher(None, content_words(canonical), target)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return 1.0 - matched / len(target)


def segment_features(segments: list[dict], duration: float) -> dict[str, float]:
    """Summarize timing from Whisper segments.

    Args:
        segments: Records with ``start``, ``end`` and ``text``.
        duration: Audio duration in seconds.

    Returns:
        Speech span, words per second over that span, and the count and total
        length of inter-segment pauses of at least 0.5 seconds.
    """
    if not segments:
        return {
            "speech_seconds": 0.0,
            "words_per_second": 0.0,
            "long_pauses": 0.0,
            "pause_seconds": 0.0,
        }
    span = max(segments[-1]["end"] - segments[0]["start"], 1e-6)
    n_words = sum(len(s["text"].split()) for s in segments)
    gaps = [b["start"] - a["end"] for a, b in zip(segments, segments[1:], strict=False)]
    long_gaps = [gap for gap in gaps if gap >= 0.5]
    return {
        "speech_seconds": min(span, duration) if duration > 0 else span,
        "words_per_second": n_words / span,
        "long_pauses": float(len(long_gaps)),
        "pause_seconds": float(sum(long_gaps)),
    }


def text_features(canonical: str, verbatim: str) -> dict[str, float]:
    """Compute per-word disfluency rates and simple complexity measures.

    Args:
        canonical: Default Whisper transcript.
        verbatim: Disfluency-preserving transcript.

    Returns:
        Feature dictionary with ``disf_`` prefixed keys.
    """
    tokens = verbatim.split()
    n = max(len(tokens), 1)
    words = content_words(verbatim)
    sentences = [s for s in _SENTENCE.split(canonical) if s.strip()]
    return {
        "disf_n_words": float(len(words)),
        "disf_filler_rate": len(FILLERS.findall(verbatim)) / n,
        "disf_false_start_rate": len(_FALSE_START.findall(verbatim)) / n,
        "disf_repeat_rate": len(_REPEAT.findall(verbatim)) / n,
        "disf_repair_rate": repair_rate(canonical, verbatim),
        "disf_type_token_ratio": len(set(words)) / max(len(words), 1),
        "disf_mean_word_length": sum(map(len, words)) / max(len(words), 1),
        "disf_mean_sentence_words": len(content_words(canonical))
        / max(len(sentences), 1),
    }
