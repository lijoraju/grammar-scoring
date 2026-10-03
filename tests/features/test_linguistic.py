"""Offline tests with explicit annotations, independent of model installation."""

import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import spacy
from spacy.tokens import Doc

from grammar_scoring.features.linguistic import (
    FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    extract_document,
    extract_frame,
    pos_features,
    spoken_features,
    surface_features,
    syntax_features,
)
from grammar_scoring.features.linguistic_artifacts import (
    load_features,
    save_features,
    validate_features,
)
from grammar_scoring.features.linguistic_cli import generate_main


@pytest.fixture
def vocab():
    return spacy.blank("en").vocab


def sentence(vocab):
    return Doc(
        vocab,
        words=["I", "can", "like", "cats", "."],
        spaces=[True, True, True, False, False],
        heads=[2, 2, 2, 2, 2],
        deps=["nsubj", "aux", "ROOT", "dobj", "punct"],
        pos=["PRON", "AUX", "VERB", "NOUN", "PUNCT"],
        tags=["PRP", "MD", "VB", "NNS", "."],
        morphs=["", "VerbForm=Fin|Tense=Pres", "VerbForm=Inf", "", ""],
    )


def segmented(vocab, words, starts):
    return Doc(vocab, words=words, sent_starts=starts)


def test_surface(vocab):
    doc = segmented(
        vocab,
        ["Hi", "hi", "!", "Long", "words", "here", "?"],
        [True, False, False, True, False, False, False],
    )
    result = surface_features(doc, 2)
    assert result["word_count"] == 5
    assert result["unique_word_count"] == 4
    assert result["type_token_ratio"] == 0.8
    assert result["root_type_token_ratio"] == pytest.approx(4 / np.sqrt(5))
    assert result["sentence_count"] == 2
    assert result["mean_sentence_length"] == 2.5
    assert result["median_sentence_length"] == 2.5
    assert result["std_sentence_length"] == 0.5
    assert result["min_sentence_length"] == 2
    assert result["max_sentence_length"] == 3
    assert result["punctuation_count"] == 2
    assert result["punctuation_per_100_words"] == 40
    assert result["question_mark_count"] == result["exclamation_mark_count"] == 1
    assert result["character_count"] == len(doc.text)
    assert result["mean_word_length"] == 3.4
    assert result["words_per_second"] == 2.5
    assert result["words_per_minute"] == 150
    assert surface_features(doc, 0)["words_per_second"] == 0


def test_empty(vocab):
    doc = Doc(vocab)
    result = extract_document("train", "empty.wav", "", 0, doc)
    assert all(result[name] == 0 for name in FEATURE_COLUMNS)


def test_pos_and_morphology(vocab):
    result = pos_features(sentence(vocab))
    assert result["pos_verb_count"] == result["pos_aux_count"] == 1
    assert result["pos_verb_per_100_words"] == 25
    assert result["finite_verb_count"] == 1
    assert result["finite_verbs_per_100_words"] == 25
    assert result["present_verb_count"] == 1
    assert result["past_verb_count"] == 0
    assert result["modal_auxiliary_count"] == 1
    assert result["modal_auxiliaries_per_100_words"] == 25
    doc = Doc(
        vocab,
        words=["went", "go", "may"],
        pos=["VERB", "VERB", "VERB"],
        tags=["VBD", "VB", "MD"],
        morphs=["Tense=Past|VerbForm=Fin", "", ""],
    )
    result = pos_features(doc)
    assert result["past_verb_count"] == 1
    assert result["finite_verb_count"] == 1
    assert result["modal_auxiliary_count"] == 0


def test_syntax(vocab):
    result = syntax_features(sentence(vocab))
    assert result["mean_dependency_distance"] == pytest.approx(4 / 3)
    assert result["max_dependency_distance"] == 2
    assert result["mean_dependency_tree_depth"] == 0.75
    assert result["max_dependency_tree_depth"] == 1
    assert result["nominal_subject_count"] == 1
    assert result["object_count"] == 1
    assert result["object_per_sentence"] == 1
    assert result["sentence_completeness_rate"] == 1
    assert result["sentences_with_subject_rate"] == 1
    assert result["sentences_with_finite_verb_rate"] == 1
    doc = Doc(
        vocab,
        words=["They", "said", "he", "went", "home"],
        heads=[1, 1, 3, 1, 3],
        deps=["nsubj", "ROOT", "nsubj", "ccomp", "advmod"],
        pos=["PRON", "VERB", "PRON", "VERB", "ADV"],
        morphs=["", "VerbForm=Fin", "", "VerbForm=Fin", ""],
    )
    result = syntax_features(doc)
    assert result["clausal_complement_count"] == 1
    assert result["max_dependency_tree_depth"] == 2
    assert result["sentences_with_subject_and_finite_verb_count"] == 1


@pytest.mark.parametrize(
    "dep,feature",
    [
        ("nsubjpass", "passive_subject"),
        ("csubj", "clausal_subject"),
        ("csubjpass", "clausal_subject"),
        ("xcomp", "open_clausal_complement"),
        ("advcl", "adverbial_clause"),
        ("relcl", "relative_clause"),
        ("conj", "coordination"),
        ("pobj", "object"),
    ],
)
def test_dependency_indicators(vocab, dep, feature):
    doc = Doc(
        vocab,
        words=["root", "child"],
        heads=[0, 0],
        deps=["ROOT", dep],
        pos=["NOUN", "NOUN"],
    )
    result = syntax_features(doc)
    assert result[f"{feature}_count"] == 1
    assert result[f"{feature}_per_sentence"] == 1
    assert result["sentence_completeness_rate"] == 0


def test_repetition(vocab):
    doc = segmented(
        vocab,
        ["A", "a", "a", "a", ".", "A", "a", "a", "a", "!"],
        [True, False, False, False, False, True, False, False, False, False],
    )
    result = spoken_features(doc)
    assert result["adjacent_repeated_word_count"] == 7
    assert result["adjacent_repeated_word_rate"] == 1
    assert result["repeated_bigram_count"] == 6
    assert result["repeated_bigram_rate"] == pytest.approx(6 / 7)
    assert result["repeated_trigram_count"] == 5
    assert result["repeated_trigram_rate"] == pytest.approx(5 / 6)
    assert result["repeated_sentence_count"] == 1
    assert result["repeated_sentence_rate"] == 0.5
    assert result["very_short_sentence_count"] == 0


def test_fillers_and_short_sentences(vocab):
    words = ["UH", "um", "hmm", "er", "ah", "you", ",", "know", "I", "mean", "like"]
    doc = segmented(vocab, words, [True] + [False] * (len(words) - 1))
    assert spoken_features(doc)["filler_count"] == 7
    assert spoken_features(doc)["fillers_per_100_words"] == 70
    doc = segmented(
        vocab,
        ["you", "know", "you", "know", "I", "mean"],
        [True, False, False, False, False, False],
    )
    assert spoken_features(doc)["filler_count"] == 3
    doc = segmented(vocab, ["you", "know", "like"], [True, True, False])
    result = spoken_features(doc)
    assert result["filler_count"] == 0
    assert result["very_short_sentence_count"] == 2
    assert result["very_short_sentence_rate"] == 1


def test_registry():
    assert tuple(FEATURE_FAMILIES) == ("surface", "pos", "syntax", "spoken")
    assert FEATURE_COLUMNS == tuple(n for f in FEATURE_FAMILIES.values() for n in f)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 93
    assert not {"split", "filename", "label", "fold", "text"} & set(FEATURE_COLUMNS)


def frames(vocab, split="train"):
    doc = sentence(vocab)
    source = pd.DataFrame(
        [dict(split=split, filename="NA", text=doc.text, duration_seconds=2)]
    )
    nlp = Mock()
    nlp.pipe.return_value = iter([doc])
    return source, extract_frame(source, nlp), nlp


def test_artifact_roundtrip(vocab, tmp_path):
    source, frame, nlp = frames(vocab)
    nlp.pipe.assert_called_once_with(source.text.tolist(), batch_size=32)
    path = tmp_path / "features.csv"
    save_features(frame, path, source, "train")
    pd.testing.assert_frame_equal(frame, load_features(path, source, "train"))
    with pytest.raises(FileExistsError):
        save_features(frame, path, source, "train")
    save_features(frame, path, source, "train", overwrite=True)
    test_source, test_frame, _ = frames(vocab, "test")
    assert list(frame.columns) == list(test_frame.columns)
    validate_features(test_frame, test_source, "test")


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate",
        "nan",
        "inf",
        "text",
        "split",
        "missing",
        "unexpected",
        "order",
        "numeric",
    ],
)
def test_artifact_validation(vocab, problem):
    source, frame, _ = frames(vocab)
    if problem == "duplicate":
        frame = pd.concat([frame, frame])
    elif problem in ("nan", "inf"):
        frame.loc[0, "word_count"] = float(problem)
    elif problem == "text":
        frame["text"] = "forbidden"
    elif problem == "split":
        frame.loc[0, "split"] = "test"
    elif problem == "missing":
        frame = frame.iloc[:0]
    elif problem == "unexpected":
        frame.loc[0, "filename"] = "extra"
    elif problem == "order":
        frame = pd.concat([frame, frame.assign(filename="second")], ignore_index=True)
        source = pd.concat(
            [source.assign(filename="second"), source], ignore_index=True
        )
    else:
        frame["word_count"] = "bad"
    with pytest.raises(ValueError):
        validate_features(frame, source, "train")


def test_invalid_document(vocab):
    doc = sentence(vocab)
    for text, duration in (("changed", 2), (doc.text, -1), (doc.text, np.inf)):
        with pytest.raises(ValueError):
            extract_document("train", "a", text, duration, doc)
    with pytest.raises(ValueError):
        extract_frame(pd.DataFrame(), Mock(), 0)


def test_cli_reuses_pipeline(vocab, tmp_path, monkeypatch):
    source_dir = tmp_path / "transcripts"
    source_dir.mkdir()
    doc = sentence(vocab)
    for split in ("train", "test"):
        record = dict(
            split=split,
            filename="a.wav",
            text=doc.text,
            duration_seconds=2,
            language="en",
            segments=[],
        )
        (source_dir / f"{split}.jsonl").write_text(json.dumps(record) + "\n")
    nlp = Mock(meta={"version": "test"})
    nlp.pipe.side_effect = lambda texts, batch_size: iter([doc for _ in texts])
    loader = Mock(return_value=nlp)
    monkeypatch.setattr(spacy, "load", loader)
    output = tmp_path / "features"
    args = [
        "--transcript-dir",
        str(source_dir),
        "--output-dir",
        str(output),
        "--batch-size",
        "2",
    ]
    assert generate_main(args) == 0
    loader.assert_called_once_with("en_core_web_sm")
    assert nlp.pipe.call_count == 2
    metadata = json.loads((output / "linguistic_features.meta.json").read_text())
    assert set(metadata["artifacts"]) == {"train", "test"}
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert all(len(a["artifact_sha256"]) == 64 for a in metadata["artifacts"].values())
    with pytest.raises(FileExistsError):
        generate_main(args)
    with pytest.raises(ValueError, match="Configuration differs"):
        generate_main(args + ["--split", "train", "--overwrite", "--batch-size", "3"])


def test_spaces_punctuation_and_empty_artifact(vocab, tmp_path):
    doc = segmented(vocab, ["Hello", " ", ",", "world"], [True, False, False, False])
    assert surface_features(doc, 1)["word_count"] == 2
    assert spoken_features(doc)["very_short_sentence_count"] == 1
    source = pd.DataFrame(columns=["split", "filename", "text", "duration_seconds"])
    nlp = Mock()
    nlp.pipe.return_value = iter([])
    frame = extract_frame(source, nlp)
    path = tmp_path / "empty.csv"
    save_features(frame, path, source, "train")
    assert load_features(path, source, "train").empty


def test_duplicate_headers_and_missing_identity(vocab, tmp_path):
    source, frame, _ = frames(vocab)
    path = tmp_path / "duplicate.csv"
    save_features(frame, path, source, "train")
    path.write_text(path.read_text().replace("word_count,", "filename,", 1))
    with pytest.raises(ValueError):
        load_features(path, source, "train")
    frame.loc[0, "filename"] = ""
    with pytest.raises(ValueError):
        validate_features(frame, source, "train")


def test_no_annotation_guessing(vocab):
    doc = segmented(vocab, ["Hello"], [True])
    with pytest.raises(ValueError, match="annotations"):
        extract_document("train", "a", doc.text, 1, doc)
