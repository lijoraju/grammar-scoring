"""Fixed TF-IDF representations for transcript regression."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline


def build_tfidf_ridge(*, include_char: bool = False) -> Pipeline:
    """Build an unfitted E002 pipeline with deterministic Ridge regression.

    Args:
        include_char: Combine word features with character boundary n-grams.

    Returns:
        Pipeline that fits vocabulary, IDF, and Ridge only on supplied rows.
    """
    word = TfidfVectorizer(
        ngram_range=(1, 2), lowercase=True, min_df=2, sublinear_tf=True
    )
    features = word
    if include_char:
        features = FeatureUnion(
            [
                ("word", word),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        lowercase=True,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    return Pipeline([("tfidf", features), ("ridge", Ridge(alpha=1.0, solver="lsqr"))])
