# Grammar Scoring Engine

An end-to-end speech grammar scoring system that predicts continuous grammar
scores from spoken English audio. The final model, **E022**, blends two channels:
an equal-weight ensemble of fine-tuned DeBERTa-v3 regressors that read three kinds
of transcript (default Whisper, disfluency-preserving "verbatim" Whisper, and a
literal wav2vec2 CTC transcript), and a frozen WavLM audio regressor that hears
what transcripts lose. Final score = 0.6 × text + 0.4 × audio.

## Problem

Given a spoken English WAV recording, predict a continuous grammar score.
The competition evaluates RMSE and Pearson correlation, so both absolute
prediction accuracy and agreement with score ordering matter. No combined
metric formula is assumed here. The dataset contains 769 training recordings
and 216 test recordings; test labels are never used for development.

## Approach

```text
               ┌─ Whisper large-v3, default            → canonical transcript ─┐
audio (.wav) ──┼─ Whisper large-v3, disfluent prompt   → verbatim v1 ──────────┴─ DeBERTa (dual) ─┐
               ├─ Whisper large-v3, per-window cue     → verbatim v2 ─────────── DeBERTa ─────────┤
               └─ wav2vec2-large CTC, no LM             → literal CTC ─────────── DeBERTa ─────────┤
                                                                                                  ▼
      text = equal-weight mean of 6 groups: {large, base} × {dual, v2, CTC}, 2–3 seeds each

audio (.wav) ── frozen WavLM-base-plus (20 s chunks, mean-pooled) ── RBF-SVR ── audio

                     grammar score = 0.6 × text + 0.4 × audio
```

**Why verbatim transcripts.** Default Whisper is built to produce clean text: its
language model drops fillers and false starts and silently repairs grammatical
slips ("I liked *to* playground" becomes "I liked *the* playground"), which removes
exactly the evidence the rubric scores. Prompting Whisper with disfluent text makes
it transcribe what was actually said (fillers rise from 0.3 to about 4 per 100
words). Verbatim v1 conditions on previous text; v2 re-inserts a short style cue
into every 30 s window instead, which removes v1's rare repetition loops. Code:
[`transcription/verbatim.py`](src/grammar_scoring/transcription/verbatim.py).

**Why a CTC transcript.** `wav2vec2-large-960h-lv60-self` emits characters frame by
frame and greedy decoding uses no language model, so nothing smooths the words
("i like to playground", "a lot kids"). It is weaker alone (no punctuation, more
spelling errors on accented speech) but its errors differ from Whisper's, which is
what an ensemble needs. Code:
[`transcription/ctc.py`](src/grammar_scoring/transcription/ctc.py).

**Why audio.** Transcripts drop pronunciation, rhythm, pauses and self-repairs.
Frozen `microsoft/wavlm-base-plus` embeddings
([`features/wavlm.py`](src/grammar_scoring/features/wavlm.py)) feed a standardized
RBF support-vector regressor
([`experiments/e022_audio.py`](src/grammar_scoring/experiments/e022_audio.py)).
The SVR's `C`/`epsilon` are tuned by an inner CV inside each training fold, and the
40% audio weight is chosen by nested CV (every fold independently picked 0.40–0.45).
The earlier E009 used the same embeddings with `Ridge(alpha=1.0)`, which is badly
under-regularized for 768 features and scored only 0.90.

**Models** ([`experiments/e014_finetune.py`](src/grammar_scoring/experiments/e014_finetune.py)).
Fully fine-tuned `microsoft/deberta-v3-large` (lr 1e-5, layer-wise LR decay 0.9)
and `deberta-v3-base` (lr 2e-5); masked mean pooling, linear head initialised at the
training-fold mean, encoder dropout off, MSE, AdamW, 10% warm-up, 5 epochs,
effective batch 16, max 320 tokens, fp16. *Dual* models train on both the canonical
and the verbatim-v1 transcript of every clip and average their predictions over the
two texts. Per fold, the best validation epoch is kept (the same rule as E005/E008);
test predictions are the five-fold mean. Seeds 42, 7 and 2024 are averaged inside a
group and the four groups with equal weight
([`experiments/e014_ensemble.py`](src/grammar_scoring/experiments/e014_ensemble.py));
no blend weights are tuned and predictions are not clipped or calibrated.

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
| E014 | DeBERTa base ×3 + large ×1 seeds, canonical | 732 | 0.5732 | 0.8250 | Superseded |
| E015 | + base on verbatim v1 (3 groups) | 732 | 0.5670 | 0.8290 | Superseded |
| E015 | 4 groups: base/large × canonical/verbatim v1 | 732 | 0.5626 | 0.8319 | Superseded |
| E016 | Qwen2.5-7B QLoRA regressor | 732 | 0.7095 | 0.7370 | Rejected |
| E017 | + GEC edit rate / LLM rubric judge features | 732 | 0.5725 | 0.8253 | Rejected |
| E019 | + disfluency / fluency / length features | 732 | 0.5642 | 0.8308 | Rejected |
| E015 | Dual-transcript large ×3 + base ×3 | 732 | 0.5603 | 0.8336 | Superseded |
| E015 | Dual large/base + verbatim-v2 large/base | 732 | 0.5545 | 0.8372 | Superseded |
| E020 | + wav2vec2 CTC large/base (6 text groups) | 732 | 0.5466 | 0.8426 | Superseded |
| E021 | + RoBERTa-large dual as a 7th text group | 732 | 0.5443 | 0.8442 | Rejected (3/5 folds) |
| **E022** | **E020 text (60%) + WavLM RBF-SVR audio (40%), nested (final)** | **732** | **0.5073** | **0.8720** | **Selected** |

E014–E019 ran on the same 732 rows and frozen folds as E008, so their OOF numbers
are directly comparable with it. A candidate was accepted only if it improved pooled
OOF RMSE *and* at least four of five folds. Rows E017 and E019 are stacks on the
then-current ensemble (0.5745 / 0.5680 references); their gains were within noise or
regressed folds. Before E014, E008 was retained despite later OOF improvements. Earlier multimodal/fusion
gains did not transfer reliably to the public leaderboard. E012a's improvement
was too small to justify that risk. E013 improved RMSE by about 0.0051, but one
fold regressed and the predefined acceptance gate was not met. Neither stack
is part of final inference. Their downstream OOF protocols are not nested CV;
base-model predictions introduce cross-fold dependencies, as detailed in
[E012a](docs/E012a.md) and [E013](docs/E013.md).

## Results

| Evaluation | Final E022 result |
| --- | ---: |
| Training-data RMSE (5-fold OOF, nested, 732 rows) | **0.5073** |
| Training-data Pearson (5-fold OOF, nested, 732 rows) | **0.8720** |
| Per-fold RMSE (mean ± std) | 0.5053 ± 0.0425 |
| Public Kaggle leaderboard score | **0.3605** |

| Submission | OOF RMSE | OOF Pearson | Public LB |
| --- | ---: | ---: | ---: |
| E008 | 0.6198 | 0.7942 | 0.3925 |
| E014 | 0.5732 | 0.8250 | 0.3894 |
| E015, 3 groups | 0.5670 | 0.8290 | 0.3816 |
| E015, 4 groups | 0.5626 | 0.8319 | 0.3755 |
| E015, dual large + base | 0.5603 | 0.8336 | 0.3800 |
| E015, dual + verbatim v2 | 0.5545 | 0.8372 | 0.3778 |
| E020, + CTC | 0.5466 | 0.8426 | 0.3712 |
| **E022, + WavLM audio (final)** | **0.5073** | **0.8720** | **0.3605** |

The leaderboard score is a separate external signal computed on part of the 216
test rows and moves by a few thousandths from noise alone, so model selection used
OOF metrics only (no leaderboard probing or selection by public score). E022 was the
one exception in spirit: its audio channel is exposed to recording-batch effects that
random CV cannot rule out (see below), so a single submission at the CV-chosen weight
served as the external check. OOF RMSE fell 18.2% relative to E008, and E022 is also
the best public score.

**Audio caveat.** Labels of neighbouring file IDs correlate (0.51), so recordings come
in batches. Grouping folds by file-ID blocks raises the audio-only RMSE from 0.598 to
0.664–0.719, while a text stand-in loses only about 0.03, and a classifier separates
train from test audio (AUC ≈ 0.76–0.82). Part of the audio CV gain is therefore batch
recognition, and 0.507 is optimistic for new recording sources; the public score still
improved from 0.3712 to 0.3605.

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

The final report and submission notebook is
[notebooks/03_final_submission.ipynb](notebooks/03_final_submission.ipynb).
It recomputes every metric and figure from the stored per-run OOF and test
predictions, refits the small audio SVR, and writes the 216-row `submission.csv`
(CPU, about a minute). Locally it reads
`artifacts/final_e015/`; on Kaggle, attach the competition, the private dataset
`lijoraju94/grammar-scoring-e015-final-artifacts` (per-run predictions, the
verbatim and CTC transcripts, and the WavLM train/test embeddings) and `lijoraju94/grammar-scoring-canonical-artifacts`.

Training each run (GPU; about 6–40 min per seed on a T4):

```sh
python scripts/transcribe_verbatim.py --data-dir <Dataset_Final> --output-dir <out> --variant v1
python scripts/transcribe_ctc.py --data-dir <Dataset_Final> --output-dir <ctc_out>
python scripts/run_e014_finetune.py --data-dir <Dataset_Final> \
    --transcript-dir <canonical> --alt-transcript-dir <verbatim_v1> \
    --model-name microsoft/deberta-v3-large --learning-rate 1e-5 --layer-decay 0.9 \
    --gradient-checkpointing --seeds 42 7 2024 --output-dir <runs>
python scripts/run_e014_ensemble.py --group <run dirs> --group <run dirs> ... --output submission.csv
python scripts/run_e022_audio.py --runs-dir <runs> --groups large_dual base_dual ... \
    --audio-dir <WavLM train/test embeddings> --output-dir <out>
```

WavLM embeddings come from `features/wavlm.py` (E009 extraction for train; the test
set was extracted with the same code on Colab and verified to reproduce training
embeddings at cosine 1.0000).

### Previous final model (E008)

[notebooks/02_final_e008_submission.ipynb](notebooks/02_final_e008_submission.ipynb)
performs frozen E008 inference, with no retraining or model selection.

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
Fresh local validation: **753 tests passed** (three warnings). Ruff lint and format checks passed.

## Key Findings

- **The transcript is the bottleneck.** Verbatim Whisper decoding, which keeps
  disfluencies and uncorrected errors, gave the largest consistent gains; training
  each model on both transcripts of a clip helped base (≈0.605 → 0.583 seed-mean
  OOF RMSE) and large (≈0.594 → 0.580) alike.
- A literal CTC transcript (no language model) is weaker alone but adds diversity:
  two CTC groups improved OOF RMSE from 0.5545 to 0.5466 across all five folds.
- **Audio was the largest single gain once modelled properly.** A frozen WavLM +
  RBF-SVR blend moved OOF RMSE from 0.5466 to 0.5073 (all five folds) and the public
  score from 0.3712 to 0.3605. The earlier conclusion that audio adds little came from
  an under-regularized Ridge, not from the data.
- Larger encoders and seed averaging helped; a 7B LLM regressor, explicit grammar
  features (GEC edit rate, LLM rubric judge) and fluency features added nothing
  beyond the fine-tuned ensemble.
- Fine-tuning contextual language representations substantially outperformed
  sparse text and frozen embeddings on the full training population.
- Investigating dataset/source differences materially influenced model selection.
- Earlier handcrafted acoustic fusion did not transfer to the leaderboard, largely
  because it was trained with the zero-label block and learned to spot that batch.
- E008 was preferred over marginally better stacks with weaker robustness evidence.
- Reproducibility and artifact provenance validation were first-class requirements.

## Limitations

ASR errors propagate into the text model, and verbatim decoding is prompt-driven:
a few files still contain repetition loops. The ensemble shrinks extreme scores
towards the mean (weak speakers are over-scored by about 0.6, the strongest
under-scored by about 0.4). The audio model partly recognizes recording batches,
and test audio is partly from different batches; speaker or source IDs would allow
properly grouped folds. The small dataset and source effects
limit confidence under distribution shift. Excluding the zero-label block is an
empirical population decision, not proof of labeling error; E008's OOF metrics
cover only the retained population. The public leaderboard is one external
signal and does not establish general performance across speakers or sources.

## Reproducibility

[uv.lock](uv.lock) records the local dependency resolution. All experiments use the
same five frozen fold assignments. E014/E015 runs record their seed and full
configuration in each run's `result.json`; E008 used seed 42.
The final notebook sets runtime seed 42, prints repository provenance, validates
frozen artifacts, and recomputes the reference OOF metrics before inference.
GPU runtime dependencies and externally stored artifacts must also be available.
Bitwise deterministic GPU inference across environments is not guaranteed.
