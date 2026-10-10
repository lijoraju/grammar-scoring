"""E028: every channel validated on unseen speakers, blend fitted on all rows.

E027 fitted its blend on the 255 rows whose speaker happened to be absent from
the random training folds. E028 retrains the text models on speaker-grouped
folds (``e014_finetune --folds-csv``), so every training row is predicted by
models that never heard its speaker, and uses the same folds for the frozen-
feature Ridge channels. Blend weights and the calibration line are then fitted
on all rows, weighted to the test set's mix of two properties:

* short recordings (under 50 s), and
* whether the clip has another clip by the same speaker in its own split.

Most test clips are short and from single-clip speakers; few training clips are.

Channels: text (speaker-grouped DeBERTa ensemble), audio (Ridge on WavLM-large
layer 20 and Whisper-encoder layers 22-32) and Voxtral (Ridge on audio-token
states, averaged over three views: audio before the question, and two questions
placed before the audio). Voxtral reads left to right, so its audio states only
depend on the question when the question comes first.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from grammar_scoring.evaluation.metrics import regression_metrics
from grammar_scoring.experiments.e014_ensemble import load_groups
from grammar_scoring.experiments.e027_honest_blend import (
    AUDIO_ALPHA,
    AUDIO_FEATURES,
    SHORT_SECONDS,
    SPEAKER_THRESHOLD,
    VOXTRAL_ALPHA,
    apply_calibration,
    best_weights,
    fit_calibration,
    load_features,
    ridge_fit_predict,
    speaker_groups,
    weighted_rmse,
)

VOXTRAL_VIEWS = (
    "voxtral_l12_14",
    "voxtral_qfirst_grammar_l12_14",
    "voxtral_qfirst_errors_l12_14",
)
CHANNELS = ("text", "audio", "voxtral")


def has_partner(
    embeddings: NDArray[np.floating],
    reference: NDArray[np.floating],
    threshold: float = SPEAKER_THRESHOLD,
) -> NDArray[np.bool_]:
    """Mark clips that have another clip by the same speaker in the same set.

    Args:
        embeddings: Speaker embeddings ``[n, dim]`` of one split.
        reference: Embeddings whose mean is removed before comparing.
        threshold: Cosine similarity above which two clips share a speaker.

    Returns:
        Boolean mask, True where some other clip exceeds the threshold.
    """
    centred = np.asarray(embeddings, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    ).mean(axis=0)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)
    similarity = centred @ centred.T
    np.fill_diagonal(similarity, 0.0)
    return similarity.max(axis=1) > threshold


def cell_weights(cells: NDArray, target_cells: NDArray) -> NDArray[np.float64]:
    """Weight rows so each cell has the share it has in the target set.

    Args:
        cells: Cell identifier per training row.
        target_cells: Cell identifier per target (test) row.

    Returns:
        Importance weight per training row.

    Raises:
        ValueError: If a target cell has no training rows.
    """
    missing = set(np.unique(target_cells)) - set(np.unique(cells))
    if missing:
        raise ValueError(f"Target cells without training rows: {sorted(missing)}")
    weights = np.empty(len(cells), dtype=np.float64)
    for cell in np.unique(cells):
        weights[cells == cell] = np.mean(target_cells == cell) / np.mean(cells == cell)
    return weights


def fold_ridge_oof(
    features: NDArray[np.floating],
    labels: NDArray[np.floating],
    folds: NDArray,
    alpha: float,
) -> NDArray[np.float64]:
    """Ridge out-of-fold predictions over predefined (speaker-grouped) folds."""
    predictions = np.empty(len(labels), dtype=np.float64)
    for fold in np.unique(folds):
        held_out = folds == fold
        predictions[held_out] = ridge_fit_predict(
            features[~held_out], labels[~held_out], features[held_out], alpha
        )
    return predictions


def ridge_channel(
    data: dict[str, NDArray],
    names: Sequence[str],
    rows: NDArray[np.int64],
    test_rows: NDArray[np.int64],
    labels: NDArray[np.floating],
    folds: NDArray,
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average Ridge models over feature sets, validated on the given folds.

    Returns:
        ``(out-of-fold predictions, test predictions from all training rows)``.
    """
    oof, final = [], []
    for name in names:
        train_x, test_x = data[f"{name}_train"][rows], data[f"{name}_test"][test_rows]
        oof.append(fold_ridge_oof(train_x, labels, folds, alpha))
        final.append(ridge_fit_predict(train_x, labels, test_x, alpha))
    return np.mean(oof, axis=0), np.mean(final, axis=0)


def nested_blend(
    channels: NDArray[np.floating],
    labels: NDArray[np.floating],
    folds: NDArray,
    weights: NDArray[np.floating],
) -> tuple[NDArray[np.float64], list[list[float]]]:
    """Blend and calibrate each fold with parameters fitted on the other folds.

    Args:
        channels: Honest channel predictions ``[n, n_channels]``.
        labels: Targets.
        folds: Speaker-grouped fold per row.
        weights: Row importance weights.

    Returns:
        ``(nested predictions, blend weights per fold)``.
    """
    predictions = np.empty(len(labels), dtype=np.float64)
    chosen = []
    for fold in np.unique(folds):
        fit, apply = folds != fold, folds == fold
        blend = best_weights(channels[fit], labels[fit], weights[fit])
        line = fit_calibration(channels[fit] @ blend, labels[fit], weights[fit])
        predictions[apply] = apply_calibration(channels[apply] @ blend, line)
        chosen.append(blend.round(2).tolist())
    return predictions, chosen


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate E028 on unseen speakers and write the blended submission."""
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
    order = np.argsort(data["train_filenames"])
    all_groups = np.empty(len(order), dtype=np.int64)
    all_groups[order] = speaker_groups(data["speaker_train"][order], all_speakers)
    groups = all_groups[rows]
    if pd.DataFrame({"g": groups, "f": folds}).groupby("g")["f"].nunique().max() > 1:
        raise ValueError("Text runs must use speaker-grouped folds")

    short = data["duration_train"][rows] < SHORT_SECONDS
    test_short = data["duration_test"][test_rows] < SHORT_SECONDS
    partner = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy() > 1
    test_partner = has_partner(data["speaker_test"][test_rows], all_speakers)
    weights = cell_weights(2 * short + partner, 2 * test_short + test_partner)

    audio_oof, audio_test = ridge_channel(
        data, AUDIO_FEATURES, rows, test_rows, labels, folds, AUDIO_ALPHA
    )
    voxtral_oof, voxtral_test = ridge_channel(
        data, VOXTRAL_VIEWS, rows, test_rows, labels, folds, VOXTRAL_ALPHA
    )
    channels = np.column_stack(
        [oof[args.groups].mean(axis=1).to_numpy(), audio_oof, voxtral_oof]
    )
    test_channels = np.column_stack(
        [test[args.groups].mean(axis=1).to_numpy(), audio_test, voxtral_test]
    )

    nested, fold_weights = nested_blend(channels, labels, folds, weights)
    blend = best_weights(channels, labels, weights)
    line = fit_calibration(channels @ blend, labels, weights)
    prediction = apply_calibration(test_channels @ blend, line)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "filename": oof["filename"],
            "label": labels,
            "fold": folds,
            "speaker": groups,
            "short": short,
            "partner": partner,
            "weight": weights,
            **{name: channels[:, i] for i, name in enumerate(CHANNELS)},
            "prediction": nested,
        }
    ).to_csv(args.output_dir / "oof.csv", index=False)
    pd.DataFrame({"filename": test["filename"], "label": prediction}).to_csv(
        args.output_dir / "submission.csv", index=False
    )
    report = {
        "experiment": "E028",
        "n_rows": int(len(labels)),
        "n_speakers": int(len(np.unique(groups))),
        "shares": {
            "short_train": float(short.mean()),
            "short_test": float(test_short.mean()),
            "partner_train": float(partner.mean()),
            "partner_test": float(test_partner.mean()),
        },
        "test_like_rmse": {
            name: weighted_rmse(labels, channels[:, i], weights)
            for i, name in enumerate(CHANNELS)
        }
        | {"blend_nested": weighted_rmse(labels, nested, weights)},
        "unweighted": {
            name: regression_metrics(labels, channels[:, i])
            for i, name in enumerate(CHANNELS)
        }
        | {"blend_nested": regression_metrics(labels, nested)},
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
