"""Inspect configured competition WAV files without processing audio samples."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from grammar_scoring.data.inspection import (
    format_audio_report,
    inspect_dataset_audio,
)


def main(argv: list[str] | None = None) -> int:
    """Print audio metadata summaries and optionally export readable files.

    Args:
        argv: CLI arguments, or None to use process arguments.

    Returns:
        Zero on completion, or one on fatal inspection or export failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save per-file metadata as CSV")
    args = parser.parse_args(argv)
    try:
        inspections = inspect_dataset_audio()
        for inspection in inspections:
            print(format_audio_report(inspection))
        if any(inspection.metadata.empty for inspection in inspections):
            print(
                "Inspection failed: a split has no readable WAV files.", file=sys.stderr
            )
            return 1
        if args.output is not None:
            pd.concat(
                [inspection.metadata for inspection in inspections], ignore_index=True
            ).to_csv(args.output, index=False)
    except (OSError, ValueError) as exc:
        print(f"Inspection failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
