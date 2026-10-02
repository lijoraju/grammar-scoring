# Engineering rules

- Reusable production code belongs under `src/grammar_scoring/`.
- Notebooks are for experimentation, visualization, and reporting.
- Scripts are thin CLI entry points that delegate to reusable package code.
- Tests belong under `tests/`. Write or update tests for production functionality.
- Never use test labels for training, tuning, preprocessing, or model selection.
- Prevent cross-validation leakage: keep validation folds isolated from fitting
  and tuning, and respect relevant groups when defining splits.
- Fit supervised preprocessing within each training fold, then apply the fitted
  preprocessing to its validation fold.
- Every model experiment should eventually report out-of-fold (OOF) RMSE and
  Pearson correlation. Do not fabricate results.
- Use deterministic random seeds and record them with experiment configuration.
- Cache expensive generated artifacts with enough metadata to identify their
  inputs and configuration.
- Never commit competition data, credentials, generated embeddings, model weights,
  or other large artifacts.
- Follow PEP 8 and Ruff with a maximum line length of 88 characters.
- Require type hints for production code and Google-style docstrings for public
  functions and classes.
- Do not introduce dependencies without justification.