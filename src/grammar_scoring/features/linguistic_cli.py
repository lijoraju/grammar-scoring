"""Label-free linguistic extraction workflow with reproducible shared metadata."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from grammar_scoring.config.paths import TRANSCRIPTS_DIR
from grammar_scoring.data.transcripts import load_transcript_jsonl
from grammar_scoring.features.linguistic import (
    FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    FILLERS,
    SCHEMA_VERSION,
    VERY_SHORT_THRESHOLD,
    extract_frame,
)
from grammar_scoring.features.linguistic_artifacts import save_features


def generate_main(argv: list[str] | None = None) -> int:
    """Load en_core_web_sm once and generate requested split CSVs and metadata."""
    parser = argparse.ArgumentParser(description="Extract raw linguistic features")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=TRANSCRIPTS_DIR.parent / "features"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    options = parser.parse_args(argv)
    if options.batch_size < 1:
        parser.error("--batch-size must be positive")
    splits = ("train", "test") if options.split == "all" else (options.split,)
    paths = {s: options.output_dir / f"linguistic_{s}.csv" for s in splits}
    meta_path = options.output_dir / "linguistic_features.meta.json"
    for path in (*paths.values(), meta_path):
        if path.exists() and not options.overwrite:
            raise FileExistsError(f"Artifact exists: {path}; use --overwrite")
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    frames = {
        s: load_transcript_jsonl(options.transcript_dir / f"{s}.jsonl") for s in splits
    }
    for split, frame in frames.items():
        if not frame["split"].eq(split).all():
            raise ValueError(f"Unexpected transcript split: {split}")
    import spacy

    nlp = spacy.load("en_core_web_sm")
    metadata = {
        "feature_schema_version": SCHEMA_VERSION,
        "spacy_version": spacy.__version__,
        "spacy_model_name": "en_core_web_sm",
        "spacy_model_version": nlp.meta.get("version"),
        "python_version": platform.python_version(),
        "feature_families": {
            key: list(value) for key, value in FEATURE_FAMILIES.items()
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "very_short_sentence_threshold": VERY_SHORT_THRESHOLD,
        "filler_lexicon": [list(filler) for filler in FILLERS],
        "batch_size": options.batch_size,
    }
    if existing and {k: v for k, v in existing.items() if k != "artifacts"} != metadata:
        if set(existing.get("artifacts", {})) - set(splits):
            raise ValueError(
                "Configuration differs: regenerate all splits or use new directory"
            )
    artifacts = dict(existing.get("artifacts", {}))
    results = {
        s: extract_frame(frame, nlp, options.batch_size) for s, frame in frames.items()
    }
    for split, result in results.items():
        save_features(
            result, paths[split], frames[split], split, overwrite=options.overwrite
        )
        source = options.transcript_dir / f"{split}.jsonl"
        artifacts[split] = {
            "filename": paths[split].name,
            "n_samples": len(result),
            "transcript_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(paths[split].read_bytes()).hexdigest(),
        }
        print(
            f"Saved {paths[split]}: {len(result)} rows, {len(FEATURE_COLUMNS)} features"
        )
    metadata["artifacts"] = artifacts
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0
