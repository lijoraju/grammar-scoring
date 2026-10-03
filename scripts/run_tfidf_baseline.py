"""Run E002a and E002b using canonical raw train transcripts and frozen folds."""

import sys

from grammar_scoring.experiments.tfidf_baseline import run_tfidf_experiments


def main() -> int:
    """Print E002 diagnostics and save OOF artifacts.

    Returns:
        Zero on success or one on invalid inputs or filesystem failure.
    """
    try:
        print("Frozen CV: n_splits=5, n_bins=5, random_state=42; Ridge alpha=1.0")
        for name, (result, output) in run_tfidf_experiments().items():
            print(f"\n{name}\nPer-fold:")
            print(result.per_fold.to_string(index=False))
            print(f"OOF RMSE: {result.oof_metrics['rmse']:.6f}")
            print(f"OOF Pearson: {result.oof_metrics['pearson_correlation']:.6f}")
            print(f"Diagnostic fold-training RMSE: {result.training_rmse:.6f}")
            print(f"Saved: {output}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"E002 failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
