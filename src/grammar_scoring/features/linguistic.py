"""Deterministic, interpretable features from raw annotated transcripts."""

import math
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token

SCHEMA_VERSION = "1"
VERY_SHORT_THRESHOLD = 3
FILLERS = (("uh",), ("um",), ("hmm",), ("er",), ("ah",), ("you", "know"), ("i", "mean"))
POS_TAGS = (
    "NOUN",
    "PROPN",
    "VERB",
    "AUX",
    "ADJ",
    "ADV",
    "PRON",
    "DET",
    "ADP",
    "CCONJ",
    "SCONJ",
    "PART",
    "NUM",
    "INTJ",
)
DEPENDENCIES = {
    "nominal_subject": ("nsubj",),
    "passive_subject": ("nsubjpass", "csubjpass"),
    "object": ("dobj", "obj", "iobj", "pobj", "dative"),
    "clausal_subject": ("csubj", "csubjpass"),
    "clausal_complement": ("ccomp",),
    "open_clausal_complement": ("xcomp",),
    "adverbial_clause": ("advcl",),
    "relative_clause": ("relcl",),
    "coordination": ("conj",),
}
SUBJECT_LABELS = frozenset(("nsubj", "nsubjpass", "csubj", "csubjpass"))
SURFACE = (
    "word_count",
    "unique_word_count",
    "type_token_ratio",
    "root_type_token_ratio",
    "sentence_count",
    "mean_sentence_length",
    "median_sentence_length",
    "std_sentence_length",
    "min_sentence_length",
    "max_sentence_length",
    "character_count",
    "mean_word_length",
    "punctuation_count",
    "punctuation_per_100_words",
    "question_mark_count",
    "exclamation_mark_count",
    "duration_seconds",
    "words_per_second",
    "words_per_minute",
)
POS = tuple(
    name
    for tag in POS_TAGS
    for name in (f"pos_{tag.lower()}_count", f"pos_{tag.lower()}_per_100_words")
) + (
    "finite_verb_count",
    "finite_verbs_per_100_words",
    "past_verb_count",
    "present_verb_count",
    "modal_auxiliary_count",
    "modal_auxiliaries_per_100_words",
)
SYNTAX = (
    (
        "mean_dependency_distance",
        "max_dependency_distance",
        "mean_dependency_tree_depth",
        "max_dependency_tree_depth",
    )
    + tuple(
        name
        for label in DEPENDENCIES
        for name in (f"{label}_count", f"{label}_per_sentence")
    )
    + (
        "sentences_with_subject_count",
        "sentences_with_finite_verb_count",
        "sentences_with_subject_and_finite_verb_count",
        "sentences_with_subject_rate",
        "sentences_with_finite_verb_rate",
        "sentence_completeness_rate",
    )
)
SPOKEN = tuple(
    name
    for metric in (
        "very_short_sentence",
        "adjacent_repeated_word",
        "repeated_bigram",
        "repeated_trigram",
        "repeated_sentence",
    )
    for name in (f"{metric}_count", f"{metric}_rate")
) + (
    "filler_count",
    "fillers_per_100_words",
)
FEATURE_FAMILIES = {"surface": SURFACE, "pos": POS, "syntax": SYNTAX, "spoken": SPOKEN}
FEATURE_COLUMNS = tuple(name for family in FEATURE_FAMILIES.values() for name in family)
IDENTITY_COLUMNS = ("split", "filename")


def _ratio(value: float, denominator: float) -> float:
    return value / denominator if denominator else 0.0


def _words(tokens: Sequence["Token"]) -> list["Token"]:
    return [token for token in tokens if not token.is_space and not token.is_punct]


def _finite(token: "Token") -> bool:
    return token.pos_ in ("VERB", "AUX") and "Fin" in token.morph.get("VerbForm")


def surface_features(doc: "Doc", duration_seconds: float) -> dict[str, float]:
    """Count non-space, non-punctuation tokens and population sentence statistics."""
    words = _words(list(doc))
    count = len(words)
    unique = len({token.text.casefold() for token in words})
    lengths = [len(_words(list(sentence))) for sentence in doc.sents]
    punctuation = sum(token.is_punct for token in doc)
    return dict(
        zip(
            SURFACE,
            (
                count,
                unique,
                _ratio(unique, count),
                _ratio(unique, math.sqrt(count)),
                len(lengths),
                np.mean(lengths) if lengths else 0,
                np.median(lengths) if lengths else 0,
                np.std(lengths) if lengths else 0,
                min(lengths, default=0),
                max(lengths, default=0),
                len(doc.text),
                _ratio(sum(len(token.text) for token in words), count),
                punctuation,
                100 * _ratio(punctuation, count),
                doc.text.count("?"),
                doc.text.count("!"),
                duration_seconds,
                _ratio(count, duration_seconds),
                60 * _ratio(count, duration_seconds),
            ),
            strict=True,
        )
    )


def pos_features(doc: "Doc") -> dict[str, float]:
    """Count explicit POS tags; verbs use morphology and modals use AUX/MD tags."""
    words = _words(list(doc))
    counts = Counter(token.pos_ for token in words)
    result = {}
    for tag in POS_TAGS:
        result[f"pos_{tag.lower()}_count"] = counts[tag]
        result[f"pos_{tag.lower()}_per_100_words"] = 100 * _ratio(
            counts[tag], len(words)
        )
    verbs = [token for token in words if token.pos_ in ("VERB", "AUX")]
    finite = sum(_finite(token) for token in verbs)
    modal = sum(token.pos_ == "AUX" and token.tag_ == "MD" for token in words)
    result.update(
        finite_verb_count=finite,
        finite_verbs_per_100_words=100 * _ratio(finite, len(words)),
        past_verb_count=sum("Past" in token.morph.get("Tense") for token in verbs),
        present_verb_count=sum("Pres" in token.morph.get("Tense") for token in verbs),
        modal_auxiliary_count=modal,
        modal_auxiliaries_per_100_words=100 * _ratio(modal, len(words)),
    )
    return result


def syntax_features(doc: "Doc") -> dict[str, float]:
    """Measure linguistic-token arcs and root-zero depths, plus sentence proxies."""
    words = _words(list(doc))
    distances = [
        abs(token.i - token.head.i)
        for token in words
        if token.head != token and not token.head.is_punct and not token.head.is_space
    ]
    depths = [len(list(token.ancestors)) for token in words]
    sentences = list(doc.sents)
    result = dict(
        zip(
            SYNTAX[:4],
            (
                np.mean(distances) if distances else 0,
                max(distances, default=0),
                np.mean(depths) if depths else 0,
                max(depths, default=0),
            ),
            strict=True,
        )
    )
    for name, labels in DEPENDENCIES.items():
        count = sum(token.dep_ in labels for token in words)
        result[f"{name}_count"] = count
        result[f"{name}_per_sentence"] = _ratio(count, len(sentences))
    subject = [
        any(token.dep_ in SUBJECT_LABELS for token in _words(list(s)))
        for s in sentences
    ]
    finite = [any(_finite(token) for token in _words(list(s))) for s in sentences]
    both = sum(a and b for a, b in zip(subject, finite, strict=True))
    result.update(
        sentences_with_subject_count=sum(subject),
        sentences_with_finite_verb_count=sum(finite),
        sentences_with_subject_and_finite_verb_count=both,
        sentences_with_subject_rate=_ratio(sum(subject), len(sentences)),
        sentences_with_finite_verb_rate=_ratio(sum(finite), len(sentences)),
        sentence_completeness_rate=_ratio(both, len(sentences)),
    )
    return result


def _repeats(items: list[tuple[str, ...]]) -> int:
    return sum(count - 1 for count in Counter(items).values())


def spoken_features(doc: "Doc") -> dict[str, float]:
    """Count excess repeated occurrences and left-to-right nonoverlapping fillers."""
    sentences = [tuple(t.text.casefold() for t in _words(list(s))) for s in doc.sents]
    words = tuple(t.text.casefold() for t in _words(list(doc)))
    short = sum(len(sentence) <= VERY_SHORT_THRESHOLD for sentence in sentences)
    adjacent = sum(a == b for a, b in zip(words, words[1:], strict=False))
    result = {
        "very_short_sentence_count": short,
        "very_short_sentence_rate": _ratio(short, len(sentences)),
        "adjacent_repeated_word_count": adjacent,
        "adjacent_repeated_word_rate": _ratio(adjacent, max(len(words) - 1, 0)),
    }
    for size, name in ((2, "bigram"), (3, "trigram")):
        grams = [words[i : i + size] for i in range(max(len(words) - size + 1, 0))]
        repeated = _repeats(grams)
        result[f"repeated_{name}_count"] = repeated
        result[f"repeated_{name}_rate"] = _ratio(repeated, len(grams))
    repeated = _repeats(sentences)
    result["repeated_sentence_count"] = repeated
    result["repeated_sentence_rate"] = _ratio(repeated, len(sentences))
    fillers = 0
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            match = next((f for f in FILLERS if sentence[i : i + len(f)] == f), None)
            if match:
                fillers += 1
                i += len(match)
            else:
                i += 1
    result["filler_count"] = fillers
    result["fillers_per_100_words"] = 100 * _ratio(fillers, len(words))
    return result


def extract_document(
    split: str,
    filename: str,
    text: str,
    duration_seconds: float,
    doc: "Doc",
) -> dict[str, str | float]:
    """Extract one raw document with identity, rejecting changed text or duration."""
    if split not in ("train", "test") or not filename.strip():
        raise ValueError("Invalid transcript identity")
    if doc.text != text:
        raise ValueError("Parsed document must preserve raw transcript text")
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("Duration must be finite and non-negative")
    if len(doc) and not all(doc.has_annotation(name) for name in ("POS", "DEP")):
        raise ValueError("Document requires POS and dependency annotations")
    features = (
        surface_features(doc, duration_seconds)
        | pos_features(doc)
        | syntax_features(doc)
        | spoken_features(doc)
    )
    return {
        "split": split,
        "filename": filename,
        **{name: float(features[name]) for name in FEATURE_COLUMNS},
    }


def extract_frame(
    transcripts: pd.DataFrame,
    nlp: "Language",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Batch raw transcripts through one injected pipeline, preserving input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    docs = nlp.pipe(transcripts["text"].tolist(), batch_size=batch_size)
    records = [
        extract_document(row.split, row.filename, row.text, row.duration_seconds, doc)
        for row, doc in zip(transcripts.itertuples(index=False), docs, strict=True)
    ]
    return pd.DataFrame(records, columns=IDENTITY_COLUMNS + FEATURE_COLUMNS).astype(
        dict.fromkeys(FEATURE_COLUMNS, float)
    )
