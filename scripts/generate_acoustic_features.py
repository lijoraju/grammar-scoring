"""Generate label-free E006a acoustic feature artifacts."""

import argparse
import json

from grammar_scoring.features.acoustic_artifacts import generate_acoustic_features


def main() -> None:
    """Parse overwrite permission and delegate to the acoustic artifact generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = generate_acoustic_features(overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "extraction_seconds": metadata["extraction_seconds"],
                "model_features": len(metadata["feature_columns"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
