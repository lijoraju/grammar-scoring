import numpy as np
import pytest
from sklearn.pipeline import FeatureUnion

from grammar_scoring.models.tfidf import build_tfidf_ridge


@pytest.mark.parametrize("include_char", [False, True])
def test_construction_and_determinism(include_char):
    model = build_tfidf_ridge(include_char=include_char)
    features = model.named_steps["tfidf"]
    if include_char:
        assert isinstance(features, FeatureUnion)
        word, char = [item[1] for item in features.transformer_list]
        assert char.analyzer == "char_wb"
        assert char.ngram_range == (3, 5)
        assert char.min_df == 2
        assert char.lowercase and char.sublinear_tf
    else:
        word = features
    assert word.ngram_range == (1, 2)
    assert word.min_df == 2
    assert word.lowercase and word.sublinear_tf
    assert model.named_steps["ridge"].alpha == 1.0
    texts = ["common good speech", "common good speech", "common poor speech"]
    labels = [4.0, 3.0, 1.0]
    first = model.fit(texts, labels).predict(texts)
    second = build_tfidf_ridge(include_char=include_char).fit(texts, labels)
    np.testing.assert_array_equal(first, second.predict(texts))


@pytest.mark.parametrize("include_char", [False, True])
def test_validation_cannot_change_vocabulary(include_char):
    model = build_tfidf_ridge(include_char=include_char)
    model.fit(["common training words", "common training words"], [1.0, 2.0])
    features = model.named_steps["tfidf"]
    vectorizers = (
        [item[1] for item in features.transformer_list] if include_char else [features]
    )
    before = [vectorizer.vocabulary_.copy() for vectorizer in vectorizers]
    model.predict(["validationexclusive zzzzz", "validationexclusive zzzzz"])
    for vectorizer, vocabulary in zip(vectorizers, before, strict=True):
        assert vectorizer.vocabulary_ == vocabulary
        assert "validationexclusive" not in vocabulary
        assert "zzz" not in vocabulary
