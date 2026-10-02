"""Generate and save the fixed training cross-validation assignments."""

import argparse
import sys
from pathlib import Path

from grammar_scoring.config.paths import FEATURES_DIR
from grammar_scoring.data.dataset import TRAIN_CSV, load_train_dataframe
from grammar_scoring.evaluation import make_cv_folds, summarize_cv_folds


def main(argv: list[str] | None = None) -> int:
    """Create training folds using the configured dataset and artifact paths.

    Args:
        argv: CLI arguments, or None to use process arguments.

    Returns:
        Zero on success, or one on invalid training data or filesystem failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=FEATURES_DIR / "train_folds.csv")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == TRAIN_CSV.resolve():
            raise ValueError("Output must not overwrite train.csv")
        train = load_train_dataframe()
        if not {"filename", "label"}.issubset(train.columns):
            raise ValueError("Training table must contain filename and label columns")
        filenames = train["filename"]
        if (
            filenames.isna().any()
            or filenames.astype(str).str.strip().eq("").any()
            or filenames.duplicated().any()
        ):
            raise ValueError("Training filenames must be non-empty and unique")
        folds = make_cv_folds(
            train["label"], args.n_splits, args.n_bins, args.random_state
        )
        print(
            f"CV: n_splits={args.n_splits}, n_bins={args.n_bins}, "
            f"random_state={args.random_state}"
        )
        print(summarize_cv_folds(train["label"], folds).to_string(index=False))
        output = train.loc[:, ["filename", "label"]].copy()
        output["fold"] = folds
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Saved folds to {args.output}")
    except (OSError, ValueError) as exc:
        print(f"Fold creation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
