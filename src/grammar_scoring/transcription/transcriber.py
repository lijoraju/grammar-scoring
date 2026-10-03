"""Raw Whisper transcription and resumable, label-free transcript artifacts."""

import json
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Protocol

from grammar_scoring.config.paths import TRANSCRIPTS_DIR
from grammar_scoring.data import dataset


def _identity(split: str, filename: str) -> None:
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split!r}")
    # Reuse the dataset layer's basename validation.
    dataset.get_train_audio_path(filename)


@dataclass(frozen=True)
class WhisperTranscriptionConfig:
    """Immutable model and decoding settings for reproducible raw ASR."""

    model_name: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 5
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    word_timestamps: bool = False

    def __post_init__(self) -> None:
        """Validate decoding settings."""
        if self.beam_size < 1:
            raise ValueError("beam_size must be positive")


@dataclass(frozen=True)
class TranscriptionSegment:
    """One verbatim ASR segment and its timestamps in seconds."""

    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        """Validate segment timestamps and raw text."""
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("Segment timestamps must be finite")
        if self.start < 0 or self.end < self.start:
            raise ValueError("Segment requires 0 <= start <= end")
        if not isinstance(self.text, str):
            raise ValueError("Segment text must be a string")


@dataclass(frozen=True)
class TranscriptionResult:
    """Raw transcript identified by split and basename, without labels."""

    split: str
    filename: str
    text: str
    language: str
    duration_seconds: float
    segments: tuple[TranscriptionSegment, ...]

    def __post_init__(self) -> None:
        """Validate artifact identity and duration."""
        _identity(self.split, self.filename)
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("Duration must be finite and nonnegative")
        if not isinstance(self.text, str) or not isinstance(self.language, str):
            raise ValueError("Text and language must be strings")

    @property
    def key(self) -> tuple[str, str]:
        """Return the exact cache identity."""
        return self.split, self.filename


class _RawSegment(Protocol):
    start: float
    end: float
    text: str


class _TranscriptionInfo(Protocol):
    language: str
    duration: float


class WhisperModelLike(Protocol):
    """Minimal injectable faster-whisper model interface."""

    def transcribe(
        self, audio: str, **kwargs: object
    ) -> tuple[Iterable[_RawSegment], _TranscriptionInfo]:
        """Return a lazy segment iterable and transcription information."""
        ...


class WhisperTranscriber:
    """Reuse one lazily constructed model across multiple recordings."""

    def __init__(
        self,
        config: WhisperTranscriptionConfig | None = None,
        *,
        model: WhisperModelLike | None = None,
    ) -> None:
        """Configure decoding and optionally inject an already constructed model."""
        self.config = config or WhisperTranscriptionConfig()
        self._model = model

    def transcribe(self, path: Path, *, split: str) -> TranscriptionResult:
        """Transcribe a file, concatenating segment text without any cleaning.

        Args:
            path: Existing audio file, whose basename supplies its identity.
            split: Competition split, either train or test.

        Returns:
            Verbatim segments and their exact concatenated text.
        """
        _identity(split, path.name)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "ASR requires faster-whisper installed in the execution environment"
                ) from exc
            self._model = WhisperModel(
                self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        decoding = asdict(self.config)
        for key in ("model_name", "device", "compute_type"):
            del decoding[key]
        raw_segments, info = self._model.transcribe(str(path), **decoding)
        segments = tuple(
            TranscriptionSegment(float(s.start), float(s.end), s.text)
            for s in raw_segments
        )
        return TranscriptionResult(
            split,
            path.name,
            "".join(s.text for s in segments),
            info.language,
            float(info.duration),
            segments,
        )


def serialize_result(result: TranscriptionResult) -> str:
    """Encode one deterministic UTF-8 compatible JSON record without labels."""
    return json.dumps(
        asdict(result), ensure_ascii=False, sort_keys=True, allow_nan=False
    )


def read_jsonl(path: Path) -> list[TranscriptionResult]:
    """Read validated records; reject corrupt lines and duplicate cache keys."""
    results: list[TranscriptionResult] = []
    keys: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                record["segments"] = tuple(
                    TranscriptionSegment(**segment) for segment in record["segments"]
                )
                result = TranscriptionResult(**record)
                if result.key in keys:
                    raise ValueError(f"Duplicate transcript identity: {result.key}")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid transcript {path}:{number}: {exc}") from exc
            keys.add(result.key)
            results.append(result)
    return results


def write_jsonl(path: Path, results: Iterable[TranscriptionResult]) -> None:
    """Write deterministic JSONL, rejecting duplicate identities before writing."""
    records = list(results)
    if len({r.key for r in records}) != len(records):
        raise ValueError("Duplicate transcript identities")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(serialize_result(r) + "\n" for r in records), encoding="utf-8"
    )


def transcription_metadata(config: WhisperTranscriptionConfig) -> dict[str, object]:
    """Describe configuration and installed backend versions without timestamps."""
    versions: dict[str, str | None] = {}
    for package in ("faster-whisper", "ctranslate2", "av"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "backend": "faster-whisper",
        "configuration": asdict(config),
        "versions": versions,
    }


def transcribe_files(
    files: Iterable[tuple[str, Path]],
    output: Path,
    transcriber: WhisperTranscriber,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Resume an artifact by exact identity and persist every successful result.

    Args:
        files: Split/path pairs in the desired processing order.
        output: JSONL artifact; metadata uses the sibling .meta.json suffix.
        transcriber: Reusable configured transcriber.
        overwrite: Explicitly replace the existing artifact and its metadata.
        progress: Receive per-record progress messages.

    Returns:
        Counts of newly processed and cached recordings.
    """
    metadata = transcription_metadata(transcriber.config)
    meta_path = output.with_suffix(".meta.json")
    cached = read_jsonl(output) if output.exists() and not overwrite else []
    if cached:
        if (
            not meta_path.is_file()
            or json.loads(meta_path.read_text(encoding="utf-8")) != metadata
        ):
            raise ValueError("Artifact metadata is missing or differs; use --overwrite")
    keys = {result.key for result in cached}
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    processed = skipped = 0
    with output.open("w" if overwrite else "a", encoding="utf-8") as stream:
        # A valid final record without a newline must not merge with the next one.
        if cached and not output.read_bytes().endswith(b"\n"):
            stream.write("\n")
        for split, path in files:
            _identity(split, path.name)
            key = (split, path.name)
            if key in keys:
                skipped += 1
                progress(
                    f"split={split} processed={processed} cached={skipped} "
                    f"filename={path.name} skipped"
                )
                continue
            start = perf_counter()
            result = transcriber.transcribe(path, split=split)
            if result.key != key:
                raise ValueError(f"Unexpected transcript identity: {result.key}")
            stream.write(serialize_result(result) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            keys.add(key)
            processed += 1
            progress(
                f"split={split} processed={processed} cached={skipped} "
                f"filename={path.name} duration={result.duration_seconds:.3f}s "
                f"runtime={perf_counter() - start:.3f}s"
            )
    return processed, skipped


def transcribe_dataset(
    split: str,
    transcriber: WhisperTranscriber,
    *,
    output: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Load filenames through the dataset layer and transcribe selected splits.

    Args:
        split: Train, test, or all.
        transcriber: One model shared across the selected splits.
        output: JSONL path for one split, or output directory for all.
        overwrite: Replace existing artifacts instead of resuming.
    """
    if split not in {"train", "test", "all"}:
        raise ValueError(f"Invalid split: {split!r}")
    for name in ("train", "test") if split == "all" else (split,):
        loader = (
            dataset.load_train_dataframe
            if name == "train"
            else dataset.load_test_dataframe
        )
        resolver = (
            dataset.get_train_audio_path
            if name == "train"
            else dataset.get_test_audio_path
        )
        target = (
            (output or TRANSCRIPTS_DIR) / f"{name}.jsonl"
            if split == "all"
            else output or TRANSCRIPTS_DIR / f"{name}.jsonl"
        )
        transcribe_files(
            ((name, resolver(filename)) for filename in loader()["filename"]),
            target,
            transcriber,
            overwrite=overwrite,
        )
