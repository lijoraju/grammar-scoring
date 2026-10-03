"""Generate raw ASR artifacts, resuming completed split/filename identities."""

import argparse
import sys
from pathlib import Path

from grammar_scoring.transcription.transcriber import (
    WhisperTranscriber,
    WhisperTranscriptionConfig,
    transcribe_dataset,
)


def main(argv: list[str] | None = None) -> int:
    """Parse ASR options and delegate artifact generation to the package."""
    defaults = WhisperTranscriptionConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "all"), required=True)
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--compute-type", default=defaults.compute_type)
    parser.add_argument("--beam-size", type=int, default=defaults.beam_size)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL file for one split; directory for --split all",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = WhisperTranscriptionConfig(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
        )
        transcribe_dataset(
            args.split,
            WhisperTranscriber(config),
            output=args.output,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
