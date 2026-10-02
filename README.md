# Grammar Scoring Engine

Predict a continuous grammar score from 0 to 5 for spoken English WAV recordings
of approximately 45–60 seconds. The competition has approximately 769 training
samples and 216 test samples, and evaluates RMSE and Pearson correlation.

## High-level architecture

The planned workflow is audio data → transcription → features and embeddings →
grammar scoring models → evaluation. Reusable components will live in the Python
package, with thin scripts for execution and notebooks for exploration and reports.

## Repository structure

```text
grammar-scoring/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── transcripts/
│   ├── embeddings/
│   ├── features/
│   ├── oof/
│   └── models/
├── notebooks/
├── scripts/
├── src/
│   └── grammar_scoring/
│       ├── __init__.py
│       ├── data/__init__.py
│       ├── transcription/__init__.py
│       ├── features/__init__.py
│       ├── models/__init__.py
│       ├── evaluation/__init__.py
│       └── utils/__init__.py
├── tests/
│   └── __init__.py
├── .gitignore
├── AGENTS.md
├── README.md
└── pyproject.toml
```

## Development setup

Placeholder: detailed environment setup will be documented when implementation
begins. The bootstrap targets Python 3.11+ and declares a `dev` extra for Ruff and
pytest in `pyproject.toml`.

## Current status

**Project Bootstrap** — repository structure and development configuration only.

## Fixed validation protocol

Create the shared training folds once with:

```bash
uv run python scripts/create_folds.py
```

This loads only the configured training table and writes
`artifacts/features/train_folds.csv` (ignored by Git), containing `filename`,
`label`, and `fold`. `GRAMMAR_DATA_DIR` and `GRAMMAR_ARTIFACT_DIR` override the
project paths. Optional flags are `--n-splits`, `--n-bins`, `--random-state`, and
`--output`; defaults are 5, 5, and 42.

All future experiments should reuse these assignments by filename. Regeneration
requires the same training row order, configuration, and dependency versions.
Stratification uses target quantiles with duplicate boundaries dropped, reducing
bin counts until each occupied stratum has at least five samples. A single
stratum is the fallback for constant or heavily quantized targets. Duration and
test data are never used. Targets remain unchanged.

`grammar_scoring.evaluation` exports `rmse`, `pearson_correlation`,
`regression_metrics`, `make_regression_stratification_bins`, `make_cv_folds`,
`validate_cv_folds`, and `summarize_cv_folds`. Diagnostics report population
standard deviations (`ddof=0`). Pearson is `NaN` when either input is constant or
there are fewer than two samples. Every future model experiment must report OOF
RMSE and Pearson correlation using this protocol.
