"""E020: literal transcripts from a CTC acoustic model without a language model.

Whisper's decoder is a language model that prefers fluent, grammatical text. A
CTC model such as wav2vec2 emits characters frame by frame and greedy decoding
uses no language model, so it writes what was said, including errors Whisper
repairs. Its output has no punctuation or casing; the text is lower-cased and
whitespace-normalized. Long recordings are processed in overlapping chunks.
Output JSONL records are compatible with the canonical transcript artifacts.

This module only depends on third-party packages so it can run in Colab or a
Kaggle script; ``torch``, ``transformers`` and ``librosa`` are imported lazily.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class CTCConfig:
    """Settings recorded with the transcripts.

    Attributes:
        model_name: Hugging Face CTC checkpoint.
        sampling_rate: Audio sampling rate expected by the model.
        chunk_length_s: Chunk length for long-form inference, in seconds.
        stride_length_s: Overlap on each side of a chunk, in seconds.
        batch_size: Chunks per forward pass.
    """

    model_name: str = "facebook/wav2vec2-large-960h-lv60-self"
    sampling_rate: int = 16_000
    chunk_length_s: float = 20.0
    stride_length_s: float = 4.0
    batch_size: int = 8


def normalize_ctc_text(text: str) -> str:
    """Lower-case CTC output and collapse whitespace.

    Args:
        text: Raw decoded string (upper-case letters, apostrophes, spaces).

    Returns:
        Normalized transcript text.
    """
    return _SPACES.sub(" ", text).strip().lower()


def transcript_record(
    split: str, filename: str, text: str, duration: float
) -> dict[str, object]:
    """Build a transcript JSONL record compatible with canonical artifacts.

    Args:
        split: ``train`` or ``test``.
        filename: Audio basename.
        text: Raw decoded text.
        duration: Audio duration in seconds.

    Returns:
        Record with normalized text, English language and no segments (CTC
        greedy decoding here does not produce timestamps).

    Raises:
        ValueError: If the split is unknown.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split!r}")
    return {
        "split": split,
        "filename": filename,
        "text": normalize_ctc_text(text),
        "language": "en",
        "duration_seconds": float(duration),
        "segments": [],
    }


def transcribe_split(
    files: Sequence[Path], split: str, output: Path, config: CTCConfig
) -> None:
    """Transcribe WAV files into a JSONL artifact, resuming completed records.

    Args:
        files: WAV paths to transcribe.
        split: ``train`` or ``test``, stored in every record.
        output: Destination JSONL path; existing records are kept.
        config: Model and chunking settings.
    """
    import librosa
    import torch
    from transformers import pipeline

    done: set[str] = set()
    if output.exists():
        done = {json.loads(line)["filename"] for line in output.open() if line.strip()}
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "model": config.model_name,
        "device": 0 if torch.cuda.is_available() else -1,
    }
    try:  # Transformers >= 4.56 names the argument ``dtype``.
        asr = pipeline("automatic-speech-recognition", dtype=dtype, **kwargs)
    except TypeError:
        asr = pipeline("automatic-speech-recognition", torch_dtype=dtype, **kwargs)
    started = time.time()
    with output.open("a", encoding="utf-8") as stream:
        for index, path in enumerate(files, start=1):
            if path.name in done:
                continue
            audio, _ = librosa.load(path, sr=config.sampling_rate, mono=True)
            result = asr(
                audio,
                chunk_length_s=config.chunk_length_s,
                stride_length_s=config.stride_length_s,
                batch_size=config.batch_size,
            )
            record = transcript_record(
                split, path.name, result["text"], len(audio) / config.sampling_rate
            )
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if index % 100 == 0:
                rate = (time.time() - started) / index
                print(f"{split}: {index}/{len(files)} ({rate:.2f}s/file)", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Transcribe the train and test splits with the CTC model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = CTCConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        files = sorted((args.data_dir / split).glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No WAV files in {args.data_dir / split}")
        transcribe_split(files, split, args.output_dir / f"{split}.jsonl", config)
        (args.output_dir / f"{split}.meta.json").write_text(
            json.dumps({"backend": "transformers-ctc", "configuration": asdict(config)})
        )


if __name__ == "__main__":
    main()
