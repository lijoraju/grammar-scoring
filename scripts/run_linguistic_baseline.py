"""Run E004 Ridge variants on cached linguistic features and frozen folds."""

import sys

from grammar_scoring.experiments.linguistic_baseline import (
    experiment_features,
    run_linguistic_experiments,
)


def main() -> int:
    """Save E004 OOF artifacts and print validation and diagnostic metrics.

    Returns:
        Zero on success, one on invalid inputs or filesystem failure.
    """
    try:
        print(
            "Frozen CV: seed=42, n_splits=5; StandardScaler; "
            "Ridge alpha=1.0, solver=lsqr"
        )
        results, correlations = run_linguistic_experiments()
        for name, (result, output) in results.items():
            print(f"\n{name}: {len(experiment_features(name))} features")
            print("Per-fold OOF validation metrics:")
            print(result.per_fold.to_string(index=False))
            print(f"OOF RMSE: {result.oof_metrics['rmse']:.6f}")
            print(f"OOF Pearson: {result.oof_metrics['pearson_correlation']:.6f}")
            print(f"Diagnostic pooled fold-training RMSE: {result.training_rmse:.6f}")
            print(f"Saved: {output}")
        print("\nRepresentation correlations (diagnostic only):")
        print(correlations.to_string(index=False))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"E004 failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
