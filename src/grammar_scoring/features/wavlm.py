"""Frozen WavLM extraction and identity-bound train-only caches for E009."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from grammar_scoring.data.dataset import _audio_path
from grammar_scoring.features.acoustic import load_waveform
from grammar_scoring.features.embeddings import masked_mean_pool

if TYPE_CHECKING:
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

MODEL = "microsoft/wavlm-base-plus"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 20 * SAMPLE_RATE
DIMENSION = 768
SEED = 42


def minimum_input_samples(
    kernels: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2),
    strides: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2),
) -> int:
    """Derive the receptive field for one frame from WavLM's valid convolutions."""
    required = 1
    for kernel, stride in reversed(list(zip(kernels, strides, strict=True))):
        required = (required - 1) * stride + kernel
    return required


def chunk_sample_counts(
    samples: int, minimum_samples: int = minimum_input_samples()
) -> list[int]:
    """Plan disjoint chunks, merging only a sub-receptive-field final tail."""
    if samples <= 0 or minimum_samples <= 0:
        raise ValueError("Sample counts and minimum size must be positive")
    full, tail = divmod(samples, CHUNK_SAMPLES)
    counts = [CHUNK_SAMPLES] * full
    if tail:
        if counts and tail < minimum_samples:
            counts[-1] += tail
        else:
            counts.append(tail)
    return counts


def chunk_audio(
    waveform: NDArray, minimum_samples: int = minimum_input_samples()
) -> list[NDArray]:
    """Split finite mono audio into sequential 20-second chunks without loss."""
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("Audio must be nonempty finite mono samples")
    edges = np.cumsum([0, *chunk_sample_counts(len(waveform), minimum_samples)])
    return [
        waveform[start:end] for start, end in zip(edges[:-1], edges[1:], strict=True)
    ]


def weighted_embedding(embeddings: NDArray, samples: NDArray) -> NDArray:
    """Combine chunk means using original valid sample counts as duration weights."""
    if (
        embeddings.ndim != 2
        or samples.shape != (len(embeddings),)
        or not np.isfinite(embeddings).all()
        or not np.isfinite(samples).all()
        or not (samples > 0).all()
    ):
        raise ValueError("Invalid chunk embeddings or weights")
    return np.average(embeddings, axis=0, weights=samples).astype(np.float32)


def digest(path: Path) -> str:
    """Hash a file incrementally without loading a recording collection into RAM."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def population_identity(frame: pd.DataFrame) -> str:
    """Bind ordered filenames, labels and frozen folds to an extraction cache."""
    records = frame[["filename", "label", "fold"]].to_dict("records")
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def validate_embeddings(
    embeddings: NDArray, metadata: pd.DataFrame, expected: pd.DataFrame
) -> None:
    """Require exact ordered canonical identities and finite 732-by-768 features."""
    from grammar_scoring.experiments.deberta_no_zero import validate_oof

    validate_oof(metadata.assign(prediction=0.0), expected)
    if not metadata.filename.tolist() == expected.filename.tolist():
        raise ValueError("Embedding metadata row order differs from canonical inputs")
    if embeddings.shape != (732, DIMENSION) or embeddings.dtype.kind != "f":
        raise ValueError("Expected floating 732-by-768 embeddings")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite")
    for column in ("duration_seconds", "num_chunks"):
        if column not in metadata or not np.isfinite(metadata[column]).all():
            raise ValueError(f"Invalid extraction metadata: {column}")
        if not metadata[column].gt(0).all():
            raise ValueError(f"Nonpositive extraction metadata: {column}")
    samples = np.rint(metadata.duration_seconds.to_numpy() * SAMPLE_RATE).astype(
        np.int64
    )
    chunks = [len(chunk_sample_counts(int(n))) for n in samples]
    if not np.array_equal(chunks, metadata.num_chunks):
        raise ValueError("Chunk counts differ from valid duration")


class FrozenWavLM:
    """Extract final hidden states with injectable model/processor for offline tests."""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        *,
        model: WavLMModel | None = None,
        processor: Wav2Vec2FeatureExtractor | None = None,
        revision: str | None = None,
    ) -> None:
        """Load a frozen float32 encoder and canonical feature extractor lazily."""
        import torch

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("Chunk batch size must be a positive integer")
        if (model is None) != (processor is None):
            raise ValueError("Inject both model and processor")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.use_deterministic_algorithms(True)
        if model is None:
            from transformers import AutoFeatureExtractor, WavLMModel

            model = WavLMModel.from_pretrained(MODEL, revision=revision)
            resolved = getattr(model.config, "_commit_hash", None) or revision
            processor = AutoFeatureExtractor.from_pretrained(MODEL, revision=resolved)
        if model.config.hidden_size != DIMENSION:
            raise ValueError("WavLM hidden size must be 768")
        self.model = model.to(device).float().eval()
        self.model.requires_grad_(False)
        self.processor = processor
        self.device, self.batch_size = device, batch_size
        self.revision = getattr(model.config, "_commit_hash", None) or revision
        self.minimum_samples = minimum_input_samples(
            tuple(model.config.conv_kernel), tuple(model.config.conv_stride)
        )

    def encode(self, waveform: NDArray) -> tuple[NDArray, int]:
        """Pool every chunk without padding contamination, then duration-weight means.

        Equal-length chunks batch together; the final shorter chunk runs alone.
        Sub-receptive-field final tails merge into the preceding chunk.
        An entire recording shorter than the receptive field still raises explicitly.
        """
        import torch

        chunks = chunk_audio(waveform, self.minimum_samples)
        pooled = []
        start = 0
        with torch.no_grad():
            while start < len(chunks):
                end = start + 1
                while (
                    end < len(chunks)
                    and end - start < self.batch_size
                    and len(chunks[end]) == len(chunks[start])
                ):
                    end += 1
                batch = chunks[start:end]
                lengths = torch.tensor([len(chunk) for chunk in batch])
                frames = self.model._get_feat_extract_output_lengths(lengths)
                if (frames <= 0).any():
                    raise ValueError(
                        "Chunk too short for a valid WavLM frame; no audio discarded"
                    )
                inputs = self.processor(
                    [chunk.astype(np.float32) for chunk in batch],
                    sampling_rate=SAMPLE_RATE,
                    padding=False,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                hidden = self.model(**inputs).last_hidden_state
                mask = (
                    torch.arange(hidden.shape[1], device=self.device)[None, :]
                    < frames.to(self.device)[:, None]
                )
                pooled.extend(
                    np.asarray(masked_mean_pool(hidden, mask).float().cpu().tolist())
                )
                start = end
        return weighted_embedding(
            np.stack(pooled), np.array([len(c) for c in chunks])
        ), len(chunks)


def extract_embeddings(
    retained: pd.DataFrame,
    audio_dir: Path,
    directory: Path,
    encoder: FrozenWavLM,
) -> tuple[NDArray, pd.DataFrame]:
    """Reuse compatible verified caches or extract all retained original recordings."""
    import importlib.metadata

    matrix_path = directory / "train_embeddings.npy"
    metadata_path = directory / "train_metadata.csv"
    manifest_path = directory / "extraction.json"
    present = [p.exists() for p in (matrix_path, metadata_path, manifest_path)]
    if any(present) and not all(present):
        raise ValueError("Incomplete E009 extraction cache")
    paths = [_audio_path(audio_dir, name) for name in retained.filename]
    configuration = {
        "model": MODEL,
        "revision": encoder.revision,
        "sample_rate": SAMPLE_RATE,
        "chunk_seconds": 20,
        "minimum_chunk_samples": encoder.minimum_samples,
        "tail_policy": "merge sub-receptive-field final tail into preceding chunk",
        "pooling": "final valid-frame mean; valid-sample-weighted utterance mean",
        "dimension": DIMENSION,
        "population_identity": population_identity(retained),
        "audio_sha256": {p.name: digest(p) for p in paths},
        "seed": SEED,
        "dtype": "float32",
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "transformers")
        },
    }
    if all(present):
        manifest = json.loads(manifest_path.read_text())
        if manifest["configuration"] != configuration:
            raise ValueError("Incompatible existing E009 extraction cache")
        if any(
            digest(directory / name) != value
            for name, value in manifest["checksums"].items()
        ):
            raise ValueError("E009 cache checksum mismatch")
        matrix = np.load(matrix_path, allow_pickle=False)
        metadata = pd.read_csv(metadata_path)
    else:
        rows, vectors = [], []
        for row, path in zip(retained.to_dict("records"), paths, strict=True):
            waveform = load_waveform(path)
            vector, count = encoder.encode(waveform)
            vectors.append(vector)
            rows.append(
                {
                    **row,
                    "duration_seconds": len(waveform) / SAMPLE_RATE,
                    "num_chunks": count,
                }
            )
            print(
                f"E009 extracted {len(rows)}/{len(retained)}: {path.name}", flush=True
            )
        matrix, metadata = np.stack(vectors), pd.DataFrame(rows)
        validate_embeddings(matrix, metadata, retained)
        directory.mkdir(parents=True, exist_ok=True)
        with matrix_path.open("xb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        metadata.to_csv(metadata_path, index=False, mode="x")
        manifest = {
            "configuration": configuration,
            "checksums": {p.name: digest(p) for p in (matrix_path, metadata_path)},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate_embeddings(matrix, metadata, retained)
    return matrix, metadata
