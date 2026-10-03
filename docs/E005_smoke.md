# E005 CUDA integration smoke test

**SMOKE TEST ONLY — NOT E005 RESULT**

Run this on Kaggle GPU before the canonical experiment. Do not interpret its
loss, RMSE or Pearson as model-quality evidence. The runner does not call E005
fold orchestration or write canonical model, OOF or experiment artifacts.

Use the runtime and input setup described in [E005.md](E005.md): CUDA PyTorch,
Transformers, SentencePiece/protobuf, the project runtime, canonical train CSV,
raw train Whisper JSONL and frozen fold CSV. Set `GRAMMAR_DATA_DIR` before
imports so the existing train-label loader locates `Dataset_Final/train.csv`.
Internet or a complete Hugging Face cache must provide the real original
`microsoft/deberta-v3-base` tokenizer and model. CUDA is mandatory.

```sh
python scripts/smoke_test_deberta_finetuning.py \
  --transcript-dir /kaggle/input/your-canonical-artifacts/transcripts \
  --fold-path /kaggle/input/your-canonical-artifacts/features/train_folds.csv \
  --smoke-dir /kaggle/working/artifacts/smoke/E005
```

The deterministic subset is the first 32 canonical rows where frozen fold is
not 0 and the first 16 where fold is 0. All selected filenames are printed.
Raw text, tokenizer settings, regression architecture and optimization settings
come from production E005. The smoke trains exactly one epoch: four mini-batches
of eight examples, two accumulation groups, and two expected successful
optimizer/scheduler updates. The linear scheduler uses the smoke workload's two
steps and `ceil(2 * 0.10) = 1` warmup step. This is not E005's five-epoch schedule.
If AMP skips an update, the expected-count assertion fails explicitly.

The runner checks CUDA FP16 training predictions, finite nonzero unscaled head
gradients, finite loss and 16 scalar validation predictions. It inspects actual
input, mask, backbone, pooled and prediction shapes without printing tensors.
It checks the production head receives the pooled representation and verifies
that replacing padded token states does not alter masked pooling. Inspection
prefers a padded batch across all 48 selected examples. If no selected batch
contains padding, an informational message states that limitation.

The trained state is saved, the original GPU model and optimizer are released,
and a fresh production model reloads the checkpoint. Validation predictions
must agree at `rtol=1e-3, atol=1e-3`. CUDA allocated/reserved/peak allocated memory
and device capacity are reported. OOM or any failed check exits with status 1;
batch size and canonical settings are never adjusted.

Each invocation writes only to a newly created directory:

```text
artifacts/smoke/E005/run-<unique-id>/
    smoke.pt
    smoke_summary.json
```

The output root must resolve to `.../smoke/E005`, outside any canonical
`models/E005` or `experiments/E005` tree. Symlink aliases into canonical trees
are rejected. Artifacts remain gitignored. The summary contains smoke counts,
selected filenames, runtime/AMP information, shapes, token-length diagnostics,
smoke metrics, memory and checkpoint round-trip error. A failed run may leave an
incomplete smoke directory, but it never creates an E005 completion marker.

A successful run prints a compact checklist and final PASS with the smoke-only
notice. No GPU smoke run is performed by the offline unit tests; they exercise
subset selection, path guards, counters, validation/round-trip checks and failure
handling with synthetic inputs. No package dependencies or frozen E005 files
are changed.
