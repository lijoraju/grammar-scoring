"""Run E006a and save offline E005 complementarity/blend diagnostics."""

import argparse
import json

from grammar_scoring.experiments.acoustic_baseline import run_acoustic_experiment


def main() -> None:
    """Parse overwrite permission and delegate to the canonical E006a runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_acoustic_experiment(overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
