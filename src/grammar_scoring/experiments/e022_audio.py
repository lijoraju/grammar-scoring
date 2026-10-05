"""E022: frozen WavLM audio regressor blended with the E020 text ensemble.

Transcripts lose pronunciation, rhythm, pauses and self-repairs. E022 regresses
the grammar score from frozen ``microsoft/wavlm-base-plus`` utterance embeddings
(``features/wavlm.py``) with a standardized RBF support-vector regressor, then
blends it with the text ensemble.

Leakage controls:

* The scaler and SVR are fitted inside each training fold only.
* SVR hyperparameters are chosen by an inner cross-validation within each outer
  training fold (nested CV), so OOF predictions never see their own labels.
* The blend weight is chosen on the other four folds for each held-out fold
  (nested), and the final weight on all OOF predictions.

E009 used the same embeddings with ``Ridge(alpha=1.0)``, which is badly
under-regularized for 768 features and ~585 training rows (OOF RMSE 0.90).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from grammar_scoring.evaluation.metrics import regression_metrics
from grammar_scoring.experiments.e014_ensemble import load_groups

PARAM_GRID: dict[str, list[float]] = {
    "svr__C": [1.0, 3.0, 10.0, 30.0],
    "svr__epsilon": [0.05, 0.2],
}
BLEND_WEIGHTS: NDArray[np.float64] = np.round(np.arange(0.0, 0.71, 0.05), 2)
SEED = 42


def svr_search(n_splits: int) -> GridSearchCV:
    """Return the standardized RBF-SVR with a seeded hyperparameter search.

    Args:
        n_splits: Number of shuffled K-fold splits used to score the grid.

    Returns:
        Unfitted ``GridSearchCV`` minimizing RMSE.
    """
    pipeline: Pipeline = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
    return GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=KFold(n_splits, shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
    )


def nested_audio_oof(
    embeddings: NDArray[np.float64], labels: NDArray[np.float64], folds: NDArray
) -> tuple[NDArray[np.float64], list[dict[str, float]]]:
    """Predict each outer fold with an SVR tuned and fitted on the other folds.

    Args:
        embeddings: Utterance embeddings, one row per training example.
        labels: Targets aligned with ``embeddings``.
        folds: Outer fold identifier of each row.

    Returns:
        ``(oof predictions, chosen hyperparameters per outer fold)``.
    """
    predictions = np.empty(len(labels), dtype=np.float64)
    chosen = []
    for fold in np.unique(folds):
        held_out = folds == fold
        search = svr_search(4).fit(embeddings[~held_out], labels[~held_out])
        predictions[held_out] = search.predict(embeddings[held_out])
        chosen.append({k: float(v) for k, v in search.best_params_.items()})
    return predictions, chosen


def best_blend_weight(
    text: NDArray[np.float64],
    audio: NDArray[np.float64],
    labels: NDArray[np.float64],
    weights: NDArray[np.float64] = BLEND_WEIGHTS,
) -> float:
    """Return the audio weight minimizing RMSE of ``(1 - w) * text + w * audio``."""
    errors = [
        np.sqrt(np.mean((labels - ((1 - w) * text + w * audio)) ** 2)) for w in weights
    ]
    return float(weights[int(np.argmin(errors))])


def nested_blend(
    text: NDArray[np.float64],
    audio: NDArray[np.float64],
    labels: NDArray[np.float64],
    folds: NDArray,
) -> tuple[NDArray[np.float64], list[float]]:
    """Blend each fold with a weight chosen on the other folds only.

    Args:
        text: Text-ensemble OOF predictions.
        audio: Audio OOF predictions.
        labels: Targets.
        folds: Fold identifier of each row.

    Returns:
        ``(nested blended OOF predictions, audio weight used per fold)``.
    """
    blended = np.empty(len(labels), dtype=np.float64)
    used = []
    for fold in np.unique(folds):
        held_out = folds == fold
        weight = best_blend_weight(text[~held_out], audio[~held_out], labels[~held_out])
        blended[held_out] = (1 - weight) * text[held_out] + weight * audio[held_out]
        used.append(weight)
    return blended, used


def load_audio(
    embeddings_path: Path, metadata_path: Path, filenames: Sequence[str]
) -> NDArray[np.floating]:
    """Load embeddings and return them in the requested filename order.

    Args:
        embeddings_path: ``.npy`` matrix, one row per metadata row.
        metadata_path: CSV with a ``filename`` column aligned with the matrix.
        filenames: Required output order.

    Returns:
        Finite embedding matrix aligned with ``filenames``, at stored precision.

    Raises:
        ValueError: If shapes disagree, values are not finite or files are missing.
    """
    matrix = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    if matrix.ndim != 2 or len(matrix) != len(metadata):
        raise ValueError("Embedding rows must match metadata rows")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be finite")
    index = {name: row for row, name in enumerate(metadata["filename"])}
    missing = [name for name in filenames if name not in index]
    if missing:
        raise ValueError(f"{len(missing)} filenames lack audio embeddings")
    # Keep the stored precision (E009 cached float32): the submitted file was
    # produced this way, and the SVR computes in float64 internally either way.
    return matrix[[index[name] for name in filenames]]


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate E022 with nested CV and write the blended submission."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    groups = {g: sorted((args.runs_dir / g).iterdir()) for g in args.groups}
    oof, test = load_groups(groups)
    labels = oof["label"].to_numpy(dtype=np.float64)
    folds = oof["fold"].to_numpy()
    text_oof = oof[args.groups].mean(axis=1).to_numpy()
    text_test = test[args.groups].mean(axis=1).to_numpy()
    train_x = load_audio(
        args.audio_dir / "train_embeddings.npy",
        args.audio_dir / "train_metadata.csv",
        oof["filename"].tolist(),
    )
    test_x = load_audio(
        args.audio_dir / "test_embeddings.npy",
        args.audio_dir / "test_metadata.csv",
        test["filename"].tolist(),
    )

    audio_oof, fold_params = nested_audio_oof(train_x, labels, folds)
    blended_oof, fold_weights = nested_blend(text_oof, audio_oof, labels, folds)
    weight = best_blend_weight(text_oof, audio_oof, labels)
    final = svr_search(5).fit(train_x, labels)
    prediction = (1 - weight) * text_test + weight * final.predict(test_x)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "filename": oof["filename"],
            "label": labels,
            "fold": folds,
            "text": text_oof,
            "audio": audio_oof,
            "prediction": blended_oof,
        }
    ).to_csv(args.output_dir / "oof.csv", index=False)
    pd.DataFrame({"filename": test["filename"], "label": prediction}).to_csv(
        args.output_dir / "submission.csv", index=False
    )
    report = {
        "experiment": "E022",
        "text_groups": args.groups,
        "text_oof": regression_metrics(labels, text_oof),
        "audio_oof": regression_metrics(labels, audio_oof),
        "blend_oof_nested": regression_metrics(labels, blended_oof),
        "fold_svr_params": fold_params,
        "fold_blend_weights": fold_weights,
        "final_blend_weight": weight,
        "final_svr_params": {k: float(v) for k, v in final.best_params_.items()},
        "param_grid": PARAM_GRID,
        "seed": SEED,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
