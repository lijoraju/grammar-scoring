"""Run diagnostics on canonical E005 and existing secondary OOF artifacts."""

import sys

from grammar_scoring.config.paths import ARTIFACT_DIR, OOF_DIR
from grammar_scoring.experiments.oof_analysis import print_summary, run_analysis


def main() -> int:
    """Write analysis artifacts and print metrics; return one on invalid inputs."""
    output = ARTIFACT_DIR / "experiments" / "E005_oof_analysis"
    try:
        run_analysis(OOF_DIR, output)
        print_summary(output)
    except (OSError, ValueError) as exc:
        print(f"OOF analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
