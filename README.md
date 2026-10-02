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
