"""Command-line orchestration without E005 hyperparameter overrides."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from grammar_scoring.config.paths import ARTIFACT_DIR, FEATURES_DIR, TRANSCRIPTS_DIR
from grammar_scoring.experiments.deberta_finetuning import (
    CONFIG,
    load_inputs,
    run_experiment,
)


def main() -> None:
    """Validate canonical inputs, display the protocol and execute requested folds."""
    parser = argparse.ArgumentParser(description="Canonical E005 DeBERTa fine-tuning")
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument(
        "--fold-path", type=Path, default=FEATURES_DIR / "train_folds.csv"
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    frame = load_inputs(args.transcript_dir, args.fold_path)
    print(json.dumps(asdict(CONFIG), indent=2))
    result = run_experiment(
        frame, args.artifact_dir, device=args.device, resume=args.resume, fold=args.fold
    )
    print(json.dumps(result, indent=2))
