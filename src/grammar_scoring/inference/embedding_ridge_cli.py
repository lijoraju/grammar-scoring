"""CLI for cached E003b inference and the fixed E005 blend."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grammar_scoring.config.paths import (
    ARTIFACT_DIR,
    EMBEDDINGS_DIR,
    FEATURES_DIR,
    OOF_DIR,
)
from grammar_scoring.data.dataset import TEST_CSV, TRAIN_CSV
from grammar_scoring.features.embeddings import load_embeddings
from grammar_scoring.inference.embedding_ridge import (
    E003B_WEIGHT,
    E005_WEIGHT,
    align_prediction_frame,
    blend_predictions,
    infer_embedding_ridge,
    prediction_summary,
    write_predictions,
)


def main(argv: list[str] | None = None) -> int:
    """Generate a parity-checked fixed blend using existing cached artifacts.

    Args:
        argv: Explicit arguments or None for process arguments.

    Returns:
        Zero on successful artifact generation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "train-csv": TRAIN_CSV,
        "test-csv": TEST_CSV,
        "train-embeddings": EMBEDDINGS_DIR / "deberta_v3_base_train.npz",
        "test-embeddings": EMBEDDINGS_DIR / "deberta_v3_base_test.npz",
        "folds": FEATURES_DIR / "train_folds.csv",
        "e003b-oof": OOF_DIR / "E003b_deberta_ridge.csv",
        "e005-predictions": ARTIFACT_DIR / "submissions/E005_test_predictions.csv",
        "output": ARTIFACT_DIR / "submissions/E005_E003b_w065_test_predictions.csv",
        "e003b-output": ARTIFACT_DIR / "submissions/E003b_test_predictions.csv",
        "diagnostics": ARTIFACT_DIR / "experiments/E005_E003b_w065/diagnostics.json",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    options = parser.parse_args(argv)
    # These are the exact competition labels used by load_competition_transcripts;
    # inference needs no transcript text and never reads test placeholder labels.
    train = pd.read_csv(options.train_csv, dtype={"filename": "string"})
    test = pd.read_csv(
        options.test_csv, usecols=["filename"], dtype={"filename": "string"}
    )
    result = infer_embedding_ridge(
        train,
        pd.read_csv(options.folds),
        load_embeddings(options.train_embeddings, 768),
        test,
        load_embeddings(options.test_embeddings, 768),
        pd.read_csv(options.e003b_oof),
    )
    print(
        f"E003b OOF parity passed: {result.oof_metrics}; "
        f"max absolute error={result.parity_max_absolute_error}"
    )
    e005 = align_prediction_frame(pd.read_csv(options.e005_predictions), test)
    blended = blend_predictions(e005, result.predictions)
    outputs = [options.output, options.e003b_output, options.diagnostics]
    inputs = [
        getattr(options, name.replace("-", "_"))
        for name in defaults
        if name not in ("output", "e003b-output", "diagnostics")
    ]
    if len({p.resolve() for p in outputs}) != len(outputs) or any(
        p.resolve() in {q.resolve() for q in inputs} for p in outputs
    ):
        raise ValueError("Output paths must be distinct and must not overwrite inputs")
    diagnostics = {
        "weights": {"E005": E005_WEIGHT, "E003b": E003B_WEIGHT},
        "fold_ids": list(result.fold_ids),
        "fold_count": 5,
        "ridge": {"alpha": 1.0, "solver": "lsqr"},
        "random_seed": None,
        "determinism": "lsqr uses no randomness; frozen folds",
        "inputs": {
            name: str(getattr(options, name.replace("-", "_")))
            for name in defaults
            if name not in ("output", "e003b-output", "diagnostics")
        },
        "oof_metrics": result.oof_metrics,
        "oof_parity_max_absolute_error": result.parity_max_absolute_error,
        "fold_test": [prediction_summary(row) for row in result.fold_test_predictions],
        "E003b": prediction_summary(result.predictions.label.to_numpy()),
        "E005": prediction_summary(e005.label.to_numpy()),
        "blend": prediction_summary(blended.label.to_numpy()),
        "blend_below_0": int(np.sum(blended.label < 0)),
        "blend_above_5": int(np.sum(blended.label > 5)),
    }
    write_predictions(result.predictions, test, options.e003b_output)
    write_predictions(blended, test, options.output)
    options.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    options.diagnostics.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))
    print(f"Outputs: {options.e003b_output}, {options.output}, {options.diagnostics}")
    return 0
