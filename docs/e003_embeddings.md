# E003 frozen embedding infrastructure

This stage audits tokenizer lengths and generates embeddings only. It does not
read competition CSVs, labels, or CV folds, and does not evaluate regressors.
The canonical transcript JSONL loader preserves artifact order and raw text.

## Public API

- `features.token_lengths.audit_token_lengths(frame, tokenizer, limit)` returns a
  `TokenLengthAudit` with ordered `samples`, `summary`, `configured_limit`, and
  `tokenizer_limit`. Counts include special tokens and disable truncation and
  padding. Empty audits have zero samples/exceedances and `None` for distribution
  statistics. Percentages range from 0 to 100; p95 uses NumPy's linear percentile.
- `features.token_lengths.load_audit_tokenizer(variant)` lazily loads the canonical
  Hugging Face tokenizer. The experiment limits remain 256/512 even when
  tokenizer metadata differs; the CLI prints the discrepancy.
- `features.embeddings.FrozenEmbedder(variant, device="cpu", batch_size=32)` accepts
  `minilm` or `deberta`. `encode(frame)` returns `EmbeddingResult`, associating
  each float32 row with `(split, filename)` in input order. Empty inputs return
  `(0, dimension)` matrices. Optional model/tokenizer/runtime injection supports
  offline tests.
- `features.embeddings.masked_mean_pool(hidden, attention_mask)` excludes padded
  tokens and clamps the denominator for all-masked inputs.
- `features.embeddings.save_embeddings(result, path, overwrite=False)` and
  `load_embeddings(path, expected_dimension)` validate shapes, identities,
  float32 dtype, and finiteness. NPZ files contain only `split`, `filenames`, and
  `embeddings`; loading explicitly disables pickle.
- `FrozenEmbedder.metadata()` reports model configuration and installed package
  versions. The CLI adds per-split input/output SHA-256 hashes and sample counts.
  Shared metadata rejects incompatible configurations unless every existing
  split is regenerated with `--overwrite`.

MiniLM uses native SentenceTransformers pooling, explicitly sets
`max_seq_length=256`, and disables normalization. DeBERTa uses final hidden states,
masked mean pooling, batch padding, and truncation at 512 tokens. Both models run
in eval mode with frozen parameters; DeBERTa additionally uses inference mode.
Real model initialization seeds Python, NumPy, and Torch with 42. GPU numerical
results may still vary across hardware and library versions.

## Kaggle commands

Run from the project checkout, with canonical `artifacts/transcripts/train.jsonl`
and `test.jsonl` available there. Use Kaggle's managed GPU Python rather than the
local uv environment. Install these inference-only packages in Kaggle if absent;
Torch is normally supplied by the managed GPU image. `sentencepiece` supports
the DeBERTa tokenizer. No project dependency or lockfile changes are needed.

```bash
python -m pip install sentence-transformers transformers sentencepiece
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/audit_token_lengths.py \
  --model all --split all --transcript-dir artifacts/transcripts

python scripts/generate_embeddings.py \
  --model minilm --split all --device cuda --batch-size 32 \
  --transcript-dir artifacts/transcripts \
  --output-dir /kaggle/working/embeddings

python scripts/generate_embeddings.py \
  --model deberta --split all --device cuda --batch-size 16 \
  --transcript-dir artifacts/transcripts \
  --output-dir /kaggle/working/embeddings
```

Model downloads require Kaggle internet access or an already populated Hugging
Face cache. Adjust batch size to GPU memory. Outputs use the canonical names
`minilm_{train,test}.npz`, `minilm.meta.json`,
`deberta_v3_base_{train,test}.npz`, and `deberta_v3_base.meta.json`.
Use `--overwrite` only when regeneration is intended. Both split artifacts reuse
the same model instance. For persisted project artifacts, copy the generated
files into `artifacts/embeddings/`, which is already gitignored.

## Offline verification

Tests inject fake tokenizers/models and a NumPy tensor adapter; no model download,
Torch installation, competition artifacts, labels, or network access is needed.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
