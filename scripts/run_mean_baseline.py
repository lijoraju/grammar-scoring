"""Run E001 and save its training OOF predictions."""

import sys

from grammar_scoring.config.paths import OOF_DIR
from grammar_scoring.data.dataset import load_train_dataframe
from grammar_scoring.experiments.mean_baseline import evaluate_mean_baseline


def main() -> int:
    """Run the fixed E001 configuration against the configured training CSV.

    Returns:
        Zero on success, or one on invalid data or filesystem failure.
    """
    try:
        result = evaluate_mean_baseline(
            load_train_dataframe(), n_splits=5, n_bins=5, random_state=42
        )
        output = OOF_DIR / "E001_mean_baseline.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.oof.to_csv(output, index=False)
        print("E001 - Mean Baseline\n====================")
        print("\nCV configuration:\n  folds: 5\n  bins: 5\n  random_state: 42")
        print("\nPer-fold:")
        print(result.per_fold.to_string(index=False, na_rep="NaN"))
        print("\nOverall:")
        print(f"  OOF RMSE: {result.oof_metrics['rmse']:.6f}")
        print(f"  OOF Pearson: {result.oof_metrics['pearson_correlation']:.6f}")
        print(f"  Training RMSE: {result.training_rmse:.6f}")
        print(f"\nSaved OOF predictions to {output}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"E001 failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
