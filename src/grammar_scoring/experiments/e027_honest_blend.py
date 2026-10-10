"""E027: speaker-honest validation and a three-channel blend.

Two properties of the data make random-fold validation optimistic:

* Speakers repeat. Most training clips have another clip by the same speaker
  with almost the same score, while few test speakers occur in training. Models
  validated on random folds can score a clip by recognizing its speaker.
* The test mix differs. Short recordings (under 50 s) are a minority of the
  training set but the majority of the test set.

E027 therefore (1) groups clips into speakers by voice similarity, (2) validates
frozen-feature channels with speaker-grouped folds, (3) evaluates fine-tuned text
predictions only on rows whose speaker was absent from their training folds, and
(4) fits blend weights and the calibration line on those unseen-speaker rows,
weighted to the test set's share of short clips.

Channels: the E020 text ensemble, a Ridge audio channel (WavLM-large layer 20
and Whisper-encoder layers 22-32) and a Ridge on Voxtral audio-LLM states.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from grammar_scoring.evaluation.metrics import regression_metrics
from grammar_scoring.experiments.e014_ensemble import load_groups

SPEAKER_THRESHOLD = 0.93
SHORT_SECONDS = 50.0
RUBRIC_RANGE = (1.0, 5.0)
WEIGHT_STEP = 0.05
AUDIO_FEATURES = ("wavlm_large_l20", "whisper_enc_l22_32")
AUDIO_ALPHA = 1000.0
VOXTRAL_FEATURE = "voxtral_l12_14"
VOXTRAL_ALPHA = 3000.0
CHANNELS = ("text", "audio", "voxtral")


def speaker_groups(
    embeddings: NDArray[np.floating],
    reference: NDArray[np.floating],
    threshold: float = SPEAKER_THRESHOLD,
) -> NDArray[np.int64]:
    """Group recordings into speakers by voice similarity.

    Embeddings are centred on the mean of ``reference`` and length-normalized;
    recordings whose cosine similarity exceeds ``threshold`` are linked, and
    connected components are the speaker groups.

    Args:
        embeddings: Speaker embeddings ``[n, dim]`` to group.
        reference: Embeddings whose mean is removed (e.g. train and test).
        threshold: Cosine similarity above which two clips share a speaker.

    Returns:
        Integer group label per recording.
    """
    centred = np.asarray(embeddings, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    ).mean(axis=0)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)
    similarity = centred @ centred.T
    np.fill_diagonal(similarity, 0.0)
    _, labels = connected_components(csr_matrix(similarity > threshold), directed=False)
    return labels.astype(np.int64)


def unseen_speaker_mask(groups: NDArray, folds: NDArray) -> NDArray[np.bool_]:
    """Mark rows whose speaker has no clip in any other fold.

    A row's out-of-fold prediction comes from a model trained on the other
    folds; it is an unseen-speaker prediction only if none of the speaker's
    other clips were in those folds.

    Args:
        groups: Speaker group per row.
        folds: Fold identifier per row.

    Returns:
        Boolean mask, True for unseen-speaker rows.
    """
    frame = pd.DataFrame({"group": groups, "fold": folds})
    folds_per_group = frame.groupby("group")["fold"].nunique()
    return (frame["group"].map(folds_per_group) == 1).to_numpy()


def importance_weights(
    short: NDArray[np.bool_], mask: NDArray[np.bool_], test_short_share: float
) -> NDArray[np.float64]:
    """Weight rows so the masked rows match the test share of short clips.

    Args:
        short: True for short recordings.
        mask: Rows the weights will be used on.
        test_short_share: Fraction of short recordings in the test set.

    Returns:
        Per-row importance weights (meaningful on masked rows).

    Raises:
        ValueError: If the masked rows lack short or long recordings.
    """
    share = float(short[mask].mean())
    if not 0.0 < share < 1.0:
        raise ValueError("Masked rows must contain short and long recordings")
    return np.where(
        short, test_short_share / share, (1 - test_short_share) / (1 - share)
    )


def weighted_rmse(
    labels: NDArray[np.floating],
    predictions: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> float:
    """Return the importance-weighted root mean squared error."""
    return float(np.sqrt(np.average((labels - predictions) ** 2, weights=weights)))


def grouped_ridge_oof(
    features: NDArray[np.floating],
    labels: NDArray[np.floating],
    groups: NDArray,
    alpha: float,
    n_splits: int = 5,
) -> NDArray[np.float64]:
    """Predict every row with a Ridge fitted on other speakers only.

    The scaler and Ridge are fitted inside each speaker-grouped training fold.

    Args:
        features: Feature matrix ``[n, dim]``.
        labels: Targets.
        groups: Speaker group per row.
        alpha: Ridge regularization strength.
        n_splits: Number of speaker-grouped folds.

    Returns:
        Out-of-fold predictions on unseen speakers.
    """
    predictions = np.empty(len(labels), dtype=np.float64)
    for train, valid in GroupKFold(n_splits).split(features, labels, groups):
        predictions[valid] = ridge_fit_predict(
            features[train], labels[train], features[valid], alpha
        )
    return predictions


def ridge_fit_predict(
    train_x: NDArray[np.floating],
    train_y: NDArray[np.floating],
    test_x: NDArray[np.floating],
    alpha: float,
) -> NDArray[np.float64]:
    """Fit a standardized Ridge on the training rows and predict the test rows."""
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=alpha).fit(scaler.transform(train_x), train_y)
    return model.predict(scaler.transform(test_x))


def ridge_channel(
    data: dict[str, NDArray],
    names: Sequence[str],
    rows: NDArray[np.int64],
    test_rows: NDArray[np.int64],
    labels: NDArray[np.floating],
    groups: NDArray,
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build one channel as the mean of Ridge models over several feature sets.

    Args:
        data: Packed features with ``<name>_train`` and ``<name>_test`` arrays.
        names: Feature sets averaged into the channel.
        rows: Training rows of ``data`` to use, in label order.
        test_rows: Test rows of ``data`` to use, in submission order.
        labels: Training targets.
        groups: Speaker group per training row.
        alpha: Ridge regularization strength.

    Returns:
        ``(speaker-grouped OOF predictions, test predictions)``; the test
        models are fitted on all training rows.
    """
    grouped, final = [], []
    for name in names:
        train_x, test_x = data[f"{name}_train"][rows], data[f"{name}_test"][test_rows]
        grouped.append(grouped_ridge_oof(train_x, labels, groups, alpha))
        final.append(ridge_fit_predict(train_x, labels, test_x, alpha))
    return np.mean(grouped, axis=0), np.mean(final, axis=0)


def simplex_grid(n_channels: int, step: float = WEIGHT_STEP) -> NDArray[np.float64]:
    """Return all non-negative weight vectors on a grid that sum to one."""
    ticks = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    rows = [
        g for g in itertools.product(ticks, repeat=n_channels) if abs(sum(g) - 1) < 1e-9
    ]
    return np.array(rows, dtype=np.float64)


def best_weights(
    channels: NDArray[np.floating],
    labels: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Choose non-negative blend weights minimizing weighted RMSE.

    Args:
        channels: Channel predictions ``[n, n_channels]``.
        labels: Targets.
        weights: Row importance weights.

    Returns:
        Weight vector summing to one.
    """
    grid = simplex_grid(channels.shape[1])
    errors = [weighted_rmse(labels, channels @ g, weights) for g in grid]
    return grid[int(np.argmin(errors))]


def fit_calibration(
    blended: NDArray[np.floating],
    labels: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> tuple[float, float]:
    """Fit the weighted line ``label ~ intercept + slope * blended``."""
    slope, intercept = np.polyfit(blended, labels, 1, w=np.sqrt(weights))
    return float(intercept), float(slope)


def apply_calibration(
    blended: NDArray[np.floating], line: tuple[float, float]
) -> NDArray[np.float64]:
    """Apply a calibration line and clip to the rubric range."""
    return np.clip(line[0] + line[1] * blended, *RUBRIC_RANGE)


def nested_honest_blend(
    channels: NDArray[np.floating],
    labels: NDArray[np.floating],
    folds: NDArray,
    mask: NDArray[np.bool_],
    weights: NDArray[np.floating],
) -> tuple[NDArray[np.float64], list[list[float]]]:
    """Blend and calibrate each fold using the other folds' honest rows only.

    Args:
        channels: Channel predictions ``[n, n_channels]``.
        labels: Targets.
        folds: Fold identifier per row.
        mask: Unseen-speaker rows (the only rows fitted and predicted).
        weights: Row importance weights.

    Returns:
        ``(predictions, blend weights per fold)``; predictions are NaN outside
        the mask.
    """
    predictions = np.full(len(labels), np.nan)
    chosen = []
    for fold in np.unique(folds):
        fit, apply = mask & (folds != fold), mask & (folds == fold)
        blend = best_weights(channels[fit], labels[fit], weights[fit])
        line = fit_calibration(channels[fit] @ blend, labels[fit], weights[fit])
        predictions[apply] = apply_calibration(channels[apply] @ blend, line)
        chosen.append(blend.round(2).tolist())
    return predictions, chosen


def load_features(path: Path) -> dict[str, NDArray]:
    """Load the packed E027 feature file (see ``artifacts/final_e027``)."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate E027 honestly and write the blended, calibrated submission."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    oof, test = load_groups(
        {g: sorted((args.runs_dir / g).iterdir()) for g in args.groups}
    )
    labels = oof["label"].to_numpy(dtype=np.float64)
    folds = oof["fold"].to_numpy()
    data = load_features(args.features)
    train_index = {name: i for i, name in enumerate(data["train_filenames"])}
    test_index = {name: i for i, name in enumerate(data["test_filenames"])}
    rows = np.array([train_index[name] for name in oof["filename"]])
    test_rows = np.array([test_index[name] for name in test["filename"]])

    all_speakers = np.vstack([data["speaker_train"], data["speaker_test"]])
    # Group in sorted-filename order so group ids, and therefore the
    # speaker-grouped fold assignment, do not depend on the file's row order.
    order = np.argsort(data["train_filenames"])
    ordered_groups = speaker_groups(data["speaker_train"][order], all_speakers)
    all_groups = np.empty(len(order), dtype=np.int64)
    all_groups[order] = ordered_groups
    groups = all_groups[rows]
    short = data["duration_train"][rows] < SHORT_SECONDS
    test_short = data["duration_test"][test_rows] < SHORT_SECONDS
    mask = unseen_speaker_mask(groups, folds)
    weights = importance_weights(short, mask, float(test_short.mean()))

    audio_oof, audio_test = ridge_channel(
        data, AUDIO_FEATURES, rows, test_rows, labels, groups, AUDIO_ALPHA
    )
    voxtral_oof, voxtral_test = ridge_channel(
        data, (VOXTRAL_FEATURE,), rows, test_rows, labels, groups, VOXTRAL_ALPHA
    )
    channels = np.column_stack(
        [oof[args.groups].mean(axis=1).to_numpy(), audio_oof, voxtral_oof]
    )
    test_channels = np.column_stack(
        [test[args.groups].mean(axis=1).to_numpy(), audio_test, voxtral_test]
    )

    nested, fold_weights = nested_honest_blend(channels, labels, folds, mask, weights)
    blend = best_weights(channels[mask], labels[mask], weights[mask])
    line = fit_calibration(channels[mask] @ blend, labels[mask], weights[mask])
    prediction = apply_calibration(test_channels @ blend, line)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "filename": oof["filename"],
            "label": labels,
            "fold": folds,
            "speaker": groups,
            "short": short,
            "unseen_speaker": mask,
            "weight": weights,
            **{name: channels[:, i] for i, name in enumerate(CHANNELS)},
            "prediction": nested,
        }
    )
    frame.to_csv(args.output_dir / "oof.csv", index=False)
    pd.DataFrame({"filename": test["filename"], "label": prediction}).to_csv(
        args.output_dir / "submission.csv", index=False
    )
    honest = frame[mask]
    report = {
        "experiment": "E027",
        "n_speakers": int(len(np.unique(groups))),
        "unseen_speaker_rows": int(mask.sum()),
        "short_share_train": float(short.mean()),
        "short_share_test": float(test_short.mean()),
        "honest_test_mix_rmse": {
            name: weighted_rmse(labels[mask], channels[mask, i], weights[mask])
            for i, name in enumerate(CHANNELS)
        }
        | {"blend_nested": weighted_rmse(labels[mask], nested[mask], weights[mask])},
        "honest_unweighted": regression_metrics(honest["label"], honest["prediction"]),
        "speaker_grouped_all_rows": {
            "audio": regression_metrics(labels, audio_oof),
            "voxtral": regression_metrics(labels, voxtral_oof),
        },
        "fold_blend_weights": fold_weights,
        "final_blend_weights": dict(
            zip(CHANNELS, blend.round(2).tolist(), strict=True)
        ),
        "calibration": {"intercept": line[0], "slope": line[1]},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
