"""Label-free command workflows for transcript audits and embedding generation."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from grammar_scoring.config.paths import EMBEDDINGS_DIR, TRANSCRIPTS_DIR
from grammar_scoring.data.transcripts import load_transcript_jsonl
from grammar_scoring.features.embeddings import MODELS, FrozenEmbedder, save_embeddings
from grammar_scoring.features.token_lengths import (
    audit_token_lengths,
    load_audit_tokenizer,
)


def _parser(description: str, models: tuple[str, ...]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=models, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPTS_DIR)
    return parser


def _splits(split: str) -> tuple[str, ...]:
    return ("train", "test") if split == "all" else (split,)


def _load(directory: Path, split: str) -> pd.DataFrame:
    frame = load_transcript_jsonl(directory / f"{split}.jsonl")
    if not frame["split"].eq(split).all():
        raise ValueError(f"Transcript artifact contains unexpected split: {split}")
    return frame


def audit_main(argv: list[str] | None = None) -> int:
    """Print canonical tokenizer audits without reading labels or folds."""
    args = _parser("Audit untruncated transcript tokens", tuple(MODELS) + ("all",))
    options = args.parse_args(argv)
    variants = tuple(MODELS) if options.model == "all" else (options.model,)
    for variant in variants:
        tokenizer = load_audit_tokenizer(variant)
        for split in _splits(options.split):
            audit = audit_token_lengths(
                _load(options.transcript_dir, split), tokenizer, MODELS[variant][2]
            )
            print(
                f"\n{variant} / {split}: configured limit={audit.configured_limit}; "
                f"tokenizer metadata limit={audit.tokenizer_limit}"
            )
            if audit.tokenizer_limit not in (None, audit.configured_limit):
                print(
                    "Limit discrepancy: configured experiment limit is authoritative."
                )
            for key, value in audit.summary.items():
                print(f"  {key}: {value}")
    return 0


def generate_main(argv: list[str] | None = None) -> int:
    """Generate split artifacts using one frozen model and reproducible metadata."""
    parser = _parser("Generate frozen embeddings", tuple(MODELS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    options = parser.parse_args(argv)
    stem = MODELS[options.model][3]
    splits = _splits(options.split)
    paths = {split: options.output_dir / f"{stem}_{split}.npz" for split in splits}
    metadata_path = options.output_dir / f"{stem}.meta.json"
    for path in paths.values():
        if path.exists() and not options.overwrite:
            raise FileExistsError(f"Artifact exists: {path}; use --overwrite")
    existing = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    frames = {split: _load(options.transcript_dir, split) for split in splits}
    embedder = FrozenEmbedder(options.model, options.device, options.batch_size)
    metadata = embedder.metadata()
    if existing:
        old_configuration = {
            key: value for key, value in existing.items() if key != "artifacts"
        }
        if old_configuration != metadata:
            # A shared metadata file cannot describe two differing configurations.
            other_splits = set(existing.get("artifacts", {})) - set(splits)
            if not options.overwrite or other_splits:
                raise ValueError(
                    "Shared metadata configuration differs; regenerate all existing "
                    "splits with --split all --overwrite or use a new output directory"
                )
    artifacts = dict(existing.get("artifacts", {}))
    for split, frame in frames.items():
        result = embedder.encode(frame)
        save_embeddings(result, paths[split], overwrite=options.overwrite)
        source = options.transcript_dir / f"{split}.jsonl"
        artifacts[split] = {
            "filename": paths[split].name,
            "n_samples": len(frame),
            "transcript_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(paths[split].read_bytes()).hexdigest(),
        }
        print(f"Saved {paths[split]}: {result.embeddings.shape}")
    metadata["artifacts"] = artifacts
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0
