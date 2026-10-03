"""E002 evaluation using frozen folds and raw training transcripts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grammar_scoring.config.paths import FEATURES_DIR, OOF_DIR
from grammar_scoring.data.transcripts import load_competition_transcripts
from grammar_scoring.evaluation.metrics import regression_metrics, rmse
from grammar_scoring.evaluation.validation import validate_cv_folds
from grammar_scoring.models.tfidf import build_tfidf_ridge


@dataclass
class TfidfBaselineResult:
    """Ordered OOF rows and separate validation and training diagnostics."""

    oof: pd.DataFrame
    per_fold: pd.DataFrame
    oof_metrics: dict[str, float]
    training_rmse: float


def _validate_filenames(frame: pd.DataFrame) -> None:
    if "filename" not in frame:
        raise ValueError("Missing filename column")
    names = frame["filename"]
    if (
        any(not isinstance(name, str) or not name.strip() for name in names)
        or names.duplicated().any()
    ):
        raise ValueError("Filenames must be nonempty unique strings")


def evaluate_tfidf_baseline(
    train: pd.DataFrame,
    frozen_folds: pd.DataFrame,
    *,
    include_char: bool = False,
    n_splits: int = 5,
) -> TfidfBaselineResult:
    """Fit fresh pipelines within supplied folds without regenerating assignments.

    Args:
        train: Unique filenames, finite labels, and raw text in desired OOF order.
            If split is present, every row must belong to train.
        frozen_folds: Filename and integer fold columns; optional labels must match.
        include_char: Enable E002b's additional character features.
        n_splits: Expected number of frozen validation folds.

    Returns:
        OOF metrics and training RMSE pooled over all fold-training predictions.
        Residuals are label minus prediction. Input row order and index persist.

    Raises:
        ValueError: If identities, text, targets, or frozen assignments are invalid.
        RuntimeError: If OOF coverage or predictions are invalid.
    """
    _validate_filenames(train)
    _validate_filenames(frozen_folds)
    if not {"label", "text"}.issubset(train.columns):
        raise ValueError("Training table requires label and text columns")
    if "fold" not in frozen_folds:
        raise ValueError("Frozen assignments require a fold column")
    if "split" in train and not train["split"].eq("train").all():
        raise ValueError("E002 accepts only train transcripts")
    if any(not isinstance(text, str) for text in train["text"]):
        raise ValueError("Transcript text must contain strings")
    if set(train["filename"]) != set(frozen_folds["filename"]):
        raise ValueError("Frozen fold filenames must exactly match training filenames")
    aligned = frozen_folds.set_index("filename").reindex(train["filename"])
    if "label" in aligned and not np.array_equal(
        train["label"].to_numpy(), aligned["label"].to_numpy()
    ):
        raise ValueError("Frozen fold labels differ from training labels")
    folds = aligned["fold"].to_numpy()
    validate_cv_folds(train["label"].to_numpy(), folds, n_splits)
    targets = train["label"].to_numpy(dtype=np.float64)
    texts = train["text"].to_numpy()
    predictions = np.full(len(train), np.nan)
    coverage = np.zeros(len(train), dtype=np.int64)
    diagnostics = []
    training_targets = []
    training_predictions = []
    for fold in range(n_splits):
        valid = folds == fold
        model = build_tfidf_ridge(include_char=include_char)
        model.fit(texts[~valid], targets[~valid])
        predicted = model.predict(texts[valid])
        predictions[valid] = predicted
        coverage[valid] += 1
        training_targets.append(targets[~valid])
        training_predictions.append(model.predict(texts[~valid]))
        diagnostics.append(
            {
                "fold": fold,
                "n_train": int((~valid).sum()),
                "n_valid": int(valid.sum()),
                **regression_metrics(targets[valid], predicted),
            }
        )
    if not np.all(coverage == 1) or not np.isfinite(predictions).all():
        raise RuntimeError("Every row must receive exactly one finite OOF prediction")
    oof = train.loc[:, ["filename", "label"]].copy()
    oof["fold"] = folds
    oof["prediction"] = predictions
    oof["residual"] = targets - predictions
    oof["abs_error"] = np.abs(oof["residual"])
    return TfidfBaselineResult(
        oof=oof,
        per_fold=pd.DataFrame(diagnostics),
        oof_metrics=regression_metrics(targets, predictions),
        training_rmse=rmse(
            np.concatenate(training_targets), np.concatenate(training_predictions)
        ),
    )


def run_tfidf_experiments() -> dict[str, tuple[TfidfBaselineResult, Path]]:
    """Run both fixed E002 variants against canonical training artifacts.

    Returns:
        Results and saved OOF paths keyed by experiment name.

    Raises:
        ValueError: If canonical data or frozen assignments are invalid.
        OSError: If inputs cannot be read or outputs cannot be written.
        RuntimeError: If OOF coverage is incomplete.
    """
    train = load_competition_transcripts("train")
    if len(train) != 769:
        raise ValueError("Canonical E002 training data must have exactly 769 rows")
    folds = pd.read_csv(FEATURES_DIR / "train_folds.csv")
    results = {}
    for name, include_char in (
        ("E002a_tfidf_word_ridge", False),
        ("E002b_tfidf_word_char_ridge", True),
    ):
        result = evaluate_tfidf_baseline(train, folds, include_char=include_char)
        output = OOF_DIR / f"{name}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.oof.to_csv(output, index=False)
        results[name] = (result, output)
    return results
