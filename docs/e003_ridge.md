# E003 frozen-embedding Ridge evaluation

Run with `uv run python scripts/run_embedding_baseline.py`.

Both variants use fixed `Ridge(alpha=1.0, solver="lsqr")`, with scikit-learn
solver defaults and no preprocessing or tuning. LSQR is deterministic and uses
no random seed. Existing five-fold assignments in
`artifacts/features/train_folds.csv` are reused (original fold seed: 42).
Cached train embeddings are validated by the existing artifact loader and
aligned by filename to the 769 canonical training rows. Test embeddings are
never loaded.

OOF residuals are label minus prediction. Diagnostic training RMSE pools all
fold-training targets and predictions, matching E002's definition. Pairwise
comparisons validate filenames, labels, folds, finite values, and residuals;
they align by filename before calculating correlations.

## Local results

| Fold | Train | Valid | MiniLM RMSE | MiniLM Pearson | DeBERTa RMSE | DeBERTa Pearson |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 615 | 154 | 1.053761 | 0.522508 | 0.939603 | 0.696233 |
| 1 | 615 | 154 | 0.998060 | 0.515458 | 0.840440 | 0.712366 |
| 2 | 615 | 154 | 1.119560 | 0.441830 | 0.911654 | 0.721722 |
| 3 | 615 | 154 | 1.266011 | 0.382689 | 0.957249 | 0.716348 |
| 4 | 616 | 153 | 1.067502 | 0.426907 | 0.894833 | 0.689035 |

| Variant | OOF RMSE | OOF Pearson | Diagnostic pooled training RMSE |
| --- | --- | --- | --- |
| E003a MiniLM | 1.104792 | 0.453247 | 0.878460 |
| E003b DeBERTa-v3-base | 0.909674 | 0.705234 | 0.439768 |

| Pair | Prediction Pearson | Residual Pearson |
| --- | --- | --- |
| E002b / E003a | 0.640311 | 0.893311 |
| E002b / E003b | 0.579165 | 0.602726 |
| E003a / E003b | 0.506660 | 0.560835 |

MiniLM does not improve on E002b's OOF RMSE (1.094199). DeBERTa improves OOF
RMSE and Pearson over E002b (Pearson 0.468539). Correlations are diagnostics;
no ensemble is fitted.

Generated, gitignored OOF artifacts:

- `artifacts/oof/E003a_minilm_ridge.csv`
- `artifacts/oof/E003b_deberta_ridge.csv`

Both use `filename,label,fold,prediction,residual,abs_error`, contain 769 unique
training filenames with finite predictions, and preserve canonical row order.

Validation: `uv run pytest` (283 passed), `uv run ruff check .` (passed), and
`uv run ruff format --check .` (passed). Tests use offline synthetic data.
