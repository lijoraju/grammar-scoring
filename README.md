# Grammar Scoring Engine

An end-to-end speech grammar scoring system that predicts continuous grammar
scores from spoken English audio. The final model, **E027**, is a calibrated blend of
three channels: an ensemble of fine-tuned DeBERTa-v3 regressors that read several
kinds of transcript (40%), Ridge regressions on frozen WavLM-large and Whisper-encoder
audio states (30%), and a Ridge on the hidden states of the Voxtral audio language
model (30%). Blend weights and calibration are fitted only on predictions for
**speakers the models never heard**, weighted to the test set's mix of recordings.

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

audio (.wav) ─┬─ WavLM-large layer 20 ──────────── Ridge ─┐
              ├─ Whisper encoder layers 22–32 ──── Ridge ─┴─ mean ─ audio
              └─ Voxtral-Mini-3B (audio + question), layers 12–14 ── Ridge ─ voxtral

   grammar score = clip(−0.441 + 1.109 × (0.40 × text + 0.30 × audio + 0.30 × voxtral), 1, 5)
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

**Why audio, and why frozen models.** Transcripts drop pronunciation, rhythm, pauses
and self-repairs, and raters hear them. With 732 clips a speech model cannot be
trained, so large pretrained models stay frozen and only small Ridge regressors learn
from their time-averaged states: WavLM-large layer 20 and Whisper-encoder layers 22–32
([`features/audio_layers.py`](src/grammar_scoring/features/audio_layers.py)), and
Voxtral-Mini-3B LLM layers 12–14 at the audio-token positions, where the model is
given the clip together with the question "How accurate and complex is the speaker's
grammar?" ([`features/speaker_voxtral.py`](src/grammar_scoring/features/speaker_voxtral.py)).

**Why speaker-honest validation.** Clips are grouped into speakers by voice similarity
(x-vectors, cosine > 0.93). 69% of training clips have another clip by the same speaker
with almost the same score (spread 0.19 within a speaker, 1.01 overall), but only 13%
of test clips match a training speaker. Random folds therefore let models score a clip
by recognizing its speaker: the text ensemble's RMSE is 0.521 when the speaker is also
in the training folds and 0.591 when not, and an earlier audio model (RBF-SVR, last
WavLM layer) went from 0.50 to 0.73. In addition, recordings shorter than 50 s are 24%
of the training set but 69% of the test set. E027
([`experiments/e027_honest_blend.py`](src/grammar_scoring/experiments/e027_honest_blend.py))
validates the audio channels with speaker-grouped folds and fits blend weights and the
calibration line on unseen-speaker rows weighted to the test share of short clips.

**Why calibration.** Averaging channels pulls predictions towards the mean; a weighted
line `label ≈ a + b × blend` (slope 1.11) stretches them back and predictions are
clipped to the rubric range [1, 5].

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
| E022 | E020 text (60%) + WavLM RBF-SVR audio (40%), nested | 732 | 0.5073 | 0.8720 | Superseded |
| E023 | E020 text + WavLM-large & Whisper-encoder audio (55%) | 732 | 0.4809 | 0.8862 | Rejected (LB 0.3649) |
| E024 | E022 + cross-fitted calibration, clip [1, 5] | 732 | 0.4942 | 0.8734 | Superseded |
| E025 | E023 audio, blend and calibration per duration batch | 732 | 0.4628 | 0.8900 | Rejected (LB tie; leaky validation) |
| E026 | Text + Ridge audio on upper layers, speaker-honest fit | 255 unseen | 0.549 (honest) | — | Superseded |
| **E027** | **E026 + Voxtral audio-LLM channel (final)** | **255 unseen** | **0.544 (honest)** | **0.836** | **Selected** |

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

| Evaluation on the training data | Final E027 result |
| --- | ---: |
| **RMSE, unseen speakers at the test mix (honest estimate)** | **0.544** |
| RMSE / Pearson, unseen speakers, unweighted (255 rows) | 0.545 / 0.836 |
| RMSE / Pearson, all 732 rows, cross-validated (optimistic) | 0.508 / 0.868 |
| Public Kaggle leaderboard score | **0.3362** |

"Honest" means every clip is scored by models that never heard its speaker, with short
and long clips weighted as in the test set. The all-rows figure is optimistic because
the fine-tuned text models saw other clips by the same speakers.

| Submission | Random-fold OOF RMSE | Honest RMSE | Public LB |
| --- | ---: | ---: | ---: |
| E008 DeBERTa-base, canonical transcript | 0.6198 | — | 0.3925 |
| E014 multi-seed base and large | 0.5732 | — | 0.3894 |
| E015 + verbatim transcripts | 0.5545 | — | 0.3778 |
| E020 + CTC transcripts | 0.5466 | 0.590 | 0.3712 |
| E022 + WavLM audio (RBF-SVR, random folds) | 0.5073 | 0.588 | 0.3605 |
| E023 stronger audio, same validation (rejected) | 0.4809 | — | 0.3649 |
| E024 + calibration | 0.4942 | 0.579 | 0.3501 |
| E025 per-duration blend (rejected) | 0.4628 | — | 0.3499 |
| E026 honest audio | — | 0.549 | 0.3448 |
| **E027 + Voxtral (final)** | — | **0.544** | **0.3362** |

Up to E024, models were selected on random-fold out-of-fold metrics. Those numbers kept
improving while the leaderboard reacted less and less (E023 and E025 improved the
random-fold RMSE a lot and the public score not at all), which led to the speaker and
test-mix analysis. From E026 on, the honest estimate and the leaderboard move together.
No blend weight or calibration parameter was ever tuned on the leaderboard; each
candidate got a single submission.

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
It recomputes every metric and figure from the stored per-run text predictions and
the packed frozen features, refits the small Ridge models, and writes the 216-row
`submission.csv` (CPU, about a minute). Locally it reads `artifacts/final_e015/`; on
Kaggle, attach the competition, the private dataset
`lijoraju94/grammar-scoring-e015-final-artifacts` (per-run predictions, transcripts and
`features_e027.npz`) and `lijoraju94/grammar-scoring-canonical-artifacts`.

```sh
# transcripts and text models (GPU; about 6–40 min per seed on a T4)
python scripts/transcribe_verbatim.py --data-dir <Dataset_Final> --output-dir <out> --variant v1
python scripts/transcribe_ctc.py --data-dir <Dataset_Final> --output-dir <ctc_out>
python scripts/run_e014_finetune.py --data-dir <Dataset_Final> \
    --transcript-dir <canonical> --alt-transcript-dir <verbatim_v1> \
    --model-name microsoft/deberta-v3-large --learning-rate 1e-5 --layer-decay 0.9 \
    --gradient-checkpointing --seeds 42 7 2024 --output-dir <runs>
# frozen audio features (GPU, once)
python scripts/extract_audio_layers.py --data-dir <Dataset_Final> --output-dir <layers>
python -m grammar_scoring.features.speaker_voxtral --data-dir <Dataset_Final> \
    --output-dir <e026> --what speaker voxtral
# speaker-honest evaluation, blend, calibration and submission (CPU)
python scripts/run_e027_honest_blend.py --runs-dir <runs> \
    --groups large_dual base_dual large_v2 base_v2 large_ctc base_ctc \
    --features <features.npz> --output-dir <out>
```

`features.npz` packs, per recording, the selected layer averages (WavLM-large layer 20,
Whisper-encoder layers 22–32, Voxtral layers 12–14), the speaker x-vector and the
duration.

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
Fresh local validation: **772 tests passed** (three warnings). Ruff lint and format checks passed.

## Key Findings

- **Validate the way the test set is built.** Speakers repeat in training and are
  mostly new in the test set, and the test set has far more short recordings. Random
  folds overstated every model, most of all audio models that can recognize voices.
  Switching to unseen-speaker, test-mix validation changed which audio model was best
  and moved the public score from 0.3501 to 0.3362.
- **The transcript matters.** Verbatim Whisper decoding, which keeps disfluencies and
  uncorrected errors, and a literal CTC transcript each improved the text ensemble.
- **Frozen models plus small regressors generalize to new speakers.** Upper layers
  with a strongly regularized Ridge beat a flexible RBF-SVR, which memorized voices.
  An audio language model (Voxtral) was the strongest single channel on new speakers
  (0.553 speaker-grouped RMSE) and is equally good on short and long clips.
- **Calibrate blends.** Averaging channels shrinks predictions; a fitted line plus
  clipping to the rubric range corrects it.
- **Negative results.** A 7B LLM regressor, GEC edit rates, an LLM rubric judge,
  fluency features, RoBERTa, tree stackers, rounding and test-time audio
  normalization added nothing; see [docs/E014_E027.md](docs/E014_E027.md).
- A speaker's known score would improve the 13% of test clips that match a training
  speaker, but that is speaker recognition, not grammar scoring, and is left out.

## Limitations

The honest validation set is small (255 unseen-speaker rows), so the blend and the
honest RMSE carry about ±0.01 of fold-assignment noise. The fine-tuned text models
were trained on random folds and saw other clips by the same speakers; retraining them
on speaker-grouped folds would make every row's prediction honest. Speaker grouping
relies on a voice-similarity threshold and is approximate. ASR errors propagate into
the text models, and verbatim decoding is prompt-driven. The blend still shrinks
extreme scores towards the middle; only four training clips score below 2. Excluding
the 37 zero-label clips is an empirical decision about a noisy batch (a CTC model hears
no speech in about half of them), not proof of labelling error. The public leaderboard
is one external signal on part of 216 clips.

## Reproducibility

[uv.lock](uv.lock) records the local dependency resolution. All experiments use the
same five frozen fold assignments. E014/E015 runs record their seed and full
configuration in each run's `result.json`; E008 used seed 42.
The final notebook sets runtime seed 42, prints repository provenance, validates
frozen artifacts, and recomputes the reference OOF metrics before inference.
GPU runtime dependencies and externally stored artifacts must also be available.
Bitwise deterministic GPU inference across environments is not guaranteed.
