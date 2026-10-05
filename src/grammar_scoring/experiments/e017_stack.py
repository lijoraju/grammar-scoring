"""E017: stack explicit grammar features on top of the E014 ensemble.

A small Ridge model takes the E014 ensemble prediction plus label-free
grammar features (GEC edit rate, LLM rubric judgement) as inputs. Each
validation fold's stacked predictions come from a scaler and Ridge fitted on
the other four folds only. Like E012a/E013 this is not nested cross-validation:
the ensemble OOF predictions of training rows come from fold models that saw
the validation fold, a small dependency that is standard for stacking.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def stack_oof(
    features: pd.DataFrame,
    labels: np.ndarray,
    folds: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Return cross-fitted stacked predictions over the frozen folds.

    Args:
        features: Stacking inputs, one row per training example.
        labels: Targets aligned with ``features``.
        folds: Fold identifier of each row.
        alpha: Ridge regularization strength.

    Returns:
        OOF predictions; fold ``k`` never uses fold-``k`` labels.
    """
    values = features.to_numpy(dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    folds = np.asarray(folds)
    predictions = np.empty(len(values))
    for fold in np.unique(folds):
        held_out = folds == fold
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(values[~held_out], labels[~held_out])
        predictions[held_out] = model.predict(values[held_out])
    return predictions


def stack_test(
    train_features: pd.DataFrame,
    labels: np.ndarray,
    test_features: pd.DataFrame,
    alpha: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit the stacker on all training rows and predict the test rows.

    Args:
        train_features: Stacking inputs of training rows.
        labels: Training targets.
        test_features: Stacking inputs of test rows (same columns).
        alpha: Ridge regularization strength.

    Returns:
        ``(test predictions, standardized coefficients by column)``.
    """
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_features.to_numpy(dtype=np.float64), labels)
    coefficients = dict(
        zip(train_features.columns, model[-1].coef_.tolist(), strict=True)
    )
    return model.predict(test_features.to_numpy(dtype=np.float64)), coefficients


def feature_columns(frame: pd.DataFrame, prefixes: Sequence[str]) -> list[str]:
    """Select columns starting with any of the given prefixes, in frame order."""
    return [c for c in frame.columns if any(c.startswith(p) for p in prefixes)]
