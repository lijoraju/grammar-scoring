# Grammar Scoring Engine

An end-to-end speech grammar scoring system that predicts continuous grammar
scores from spoken English audio. **E008**, a fine-tuned DeBERTa text regressor,
is the frozen final model.

## Problem

Given a spoken English WAV recording, predict a continuous grammar score.
The competition evaluates RMSE and Pearson correlation, so both absolute
prediction accuracy and agreement with score ordering matter. No combined
metric formula is assumed here. The dataset contains 769 training recordings
and 216 test recordings; test labels are never used for development.

## Approach

```text
audio → canonical ASR transcript → DeBERTa-v3-base tokenizer
      → fine-tuned DeBERTa regression model → five-fold prediction average
      → grammar score
```

Canonical ASR uses **faster-whisper large-v3**, English, beam size 5,
`condition_on_previous_text=False`, and VAD disabled. Raw transcripts are retained
without grammar correction or rewriting. Final inference loads these cached
transcripts rather than rerunning ASR.

E008 fully fine-tunes `microsoft/deberta-v3-base` with attention-mask-aware mean
pooling of the final hidden states, dropout 0.1, and a linear regression head.
Training uses MSE, maximum token length 256, AdamW with learning rate `2e-5` and
weight decay `0.01`, five frozen folds, and seed 42. Inference takes the raw,
equal-weight average of the five selected checkpoints, with no clipping,
calibration, or rounding. The full [training protocol](docs/E008.md) documents
checkpoint selection and remaining parameters.

## Data Investigation

Analysis identified 37 zero-label examples forming an anomalous, source-specific
block that behaved differently from the remaining training population. E005,
trained on all 769 rows, achieved OOF RMSE **0.8352476523** and Pearson
**0.7419439817**. Its diagnostic results were:

| Population | Rows | OOF RMSE | OOF Pearson |
| --- | ---: | ---: | ---: |
| Non-zero labels | 732 | 0.645118 | 0.792181 |
| Zero-label block | 37 | ≈2.503 | Undefined (constant labels) |

Manual inspection suggested source and audio-quality differences; it did not
establish that the labels were erroneous. E008 tested exclusion of this block
as an empirical distribution-mismatch experiment, retaining `label > 0` rows
and their original fold assignments. Filename IDs were diagnostic evidence
only, never predictive features or test-time rules.

## Experiments

Metrics below are pooled out-of-fold (OOF) development results. Population size
matters: results on 769 rows and the retained 732 rows are not directly
comparable. The matched E005 diagnostic above provides the relevant E008 baseline.

| Experiment | Approach | Rows | OOF RMSE | OOF Pearson | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| E001 | Mean baseline | 769 | 1.238497 | -0.029490 | Baseline |
| E002 | TF-IDF word + character Ridge | 769 | 1.094199 | 0.468539 | Baseline |
| E003b | Frozen DeBERTa embeddings + Ridge | 769 | 0.909674 | 0.705234 | Baseline |
| E005 | Fine-tuned DeBERTa, all rows | 769 | 0.8352476523 | 0.7419439817 | Reference |
| E006a | Acoustic features + Ridge | 769 | 0.868546 | 0.714764 | Exploratory |
| E007 | CORAL ordinal DeBERTa | 769 | 0.837852 | 0.743331 | Not selected |
| **E008** | **Fine-tuned DeBERTa, anomalous block excluded** | **732** | **0.6197891638194158** | **0.7942308488890498** | **Selected** |
| E009 | Frozen WavLM + Ridge | 732 | 0.8997869512 | 0.6428639790 | Not selected |
| E012a | Acoustic stack with E008 | 732 | 0.618117 | 0.795419 | Rejected |
| E013 | ASR-aligned fluency stack | 732 | 0.6146677237 | 0.7954336933 | Rejected |

E008 was retained despite later OOF improvements. Earlier multimodal/fusion
gains did not transfer reliably to the public leaderboard. E012a's improvement
was too small to justify that risk. E013 improved RMSE by about 0.0051, but one
fold regressed and the predefined acceptance gate was not met. Neither stack
is part of final inference. Their downstream OOF protocols are not nested CV;
base-model predictions introduce cross-fold dependencies, as detailed in
[E012a](docs/E012a.md) and [E013](docs/E013.md).

## Results

| Evaluation | Final E008 result |
| --- | ---: |
| Development OOF RMSE (732 rows) | 0.6197891638194158 |
| Development OOF Pearson (732 rows) | 0.7942308488890498 |
| Public Kaggle leaderboard score | 0.3925 |

The leaderboard score is a separate external evaluation signal, not reported
here as RMSE or Pearson. A clean Kaggle **Restart Session → Run All** execution
reproduced the frozen OOF metrics and generated a validated **216-row** submission.
The leaderboard result and clean execution are reported from the completed
Kaggle validation; local unit tests do not reproduce those external results.

## Repository Structure

```text
notebooks/                  Experiments, reports, and final submission notebook
scripts/                    Thin CLI entry points
src/grammar_scoring/
    config/                 Paths and configuration
    data/                   Dataset, audio, and transcript validation
    evaluation/             Metrics and shared cross-validation folds
    experiments/            Reusable experiment implementations
    features/               Text, acoustic, and embedding features
    inference/              Checkpoint inference and submission validation
    models/                 Regression and ordinal models
    transcription/          Canonical ASR generation and caching
    utils/                  Package utilities
tests/                      Automated tests
docs/                       Detailed experiment protocols
```

Competition data, transcripts, generated features/embeddings, OOF artifacts,
predictions, and model checkpoints are intentionally excluded from Git.
Credentials must remain external.

## Reproducing the Final Submission

The canonical entry point is
[notebooks/02_final_e008_submission.ipynb](notebooks/02_final_e008_submission.ipynb).
It performs frozen inference, with no retraining or model selection.

1. Create a Kaggle notebook with GPU enabled. Enable Internet for the initial
   repository clone, missing runtime packages, and Hugging Face downloads;
   a complete existing checkout and model/tokenizer cache support offline reuse.
2. Attach competition `shl-hiring-assessment-2026`, private model dataset
   `lijoraju94/grammar-scoring-e008-e009-artifacts`, and canonical transcript
   dataset `lijoraju94/grammar-scoring-canonical-artifacts`. Access to the private
   artifacts is required to reproduce inference.
3. Use the final notebook and review its configuration cell for repository and
   mounted input paths. Run it top-to-bottom in Python 3.11+.
4. The notebook validates input alignment, checkpoint structure and available
   checksums, and frozen OOF population/metrics. Provenance depends on the
   externally mounted frozen artifact dataset.
5. It averages the five E008 predictions and writes
   `/kaggle/working/submission.csv`, checking 216 rows, schema, filename order,
   uniqueness, finite scores, and serialized prediction equality.

The artifact dataset also contains E009 resources; only E008 checkpoints are
used by the final notebook.

## Engineering / Validation

Reusable code follows a `src` layout with type hints and Google-style docstrings
for public APIs. Scripts delegate to the package; notebooks handle exploration
and reporting. Ruff checks lint and formatting, and pytest covers production
behavior, including deterministic folds, artifact compatibility, transcript/test
alignment, and submission schema, order, and non-finite checks. Supervised
preprocessing fits within each training fold; validation folds remain isolated
from that fitting.

For local development and validation:

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

GPU modeling/inference additionally requires the ML runtime described in the
final notebook; the core project environment alone does not provide it.
Fresh local validation: **722 tests passed** (three warnings). Ruff lint and format checks passed.

## Key Findings

- Fine-tuning contextual language representations substantially outperformed
  sparse text and frozen embeddings on the full training population.
- Investigating dataset/source differences materially influenced model selection.
- Audio features offered complementary OOF information, but fusion gains did not
  transfer reliably to the public leaderboard.
- E008 was preferred over marginally better stacks with weaker robustness evidence.
- Reproducibility and artifact provenance validation were first-class requirements.

## Limitations

ASR errors propagate into the text model. The small dataset and source effects
limit confidence under distribution shift. Excluding the zero-label block is an
empirical population decision, not proof of labeling error; E008's OOF metrics
cover only the retained population. The public leaderboard is one external
signal and does not establish general performance across speakers or sources.

## Reproducibility

[uv.lock](uv.lock) records the local dependency resolution. Training uses seed 42
and five frozen fold assignments; E008 retains those assignments after filtering.
The final notebook sets runtime seed 42, prints repository provenance, validates
frozen artifacts, and recomputes the reference OOF metrics before inference.
GPU runtime dependencies and externally stored artifacts must also be available.
Bitwise deterministic GPU inference across environments is not guaranteed.
