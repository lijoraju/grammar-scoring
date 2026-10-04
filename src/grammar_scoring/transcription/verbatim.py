"""E015: disfluency-preserving ("verbatim") Whisper transcription.

Whisper's decoder language model tends to drop fillers, repetitions and false
starts and to repair small grammatical slips, which removes evidence relevant
to grammar scoring. Whisper follows the style of its prompt, so E015 decodes
with a disfluent ``initial_prompt`` and conditions each window on the previous
text, keeping that style for the whole recording. Everything else matches the
canonical large-v3 transcription. Output JSONL records are compatible with the
canonical transcript artifacts (``filename``, ``split``, ``text``, ...).

This module only depends on third-party packages so it can run as a Kaggle
script; ``faster_whisper`` is imported lazily.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

VERBATIM_PROMPT = (
    "Umm, so, uh, I- I went to the, the park and, like, we was playing... "
    "Hmm, let me think. And then, uh, he don't- he didn't come."
)


@dataclass(frozen=True)
class VerbatimConfig:
    """Decoding settings recorded with the transcripts.

    Attributes:
        model_name: faster-whisper model identifier.
        compute_type: CTranslate2 compute type.
        language: Forced decoding language.
        beam_size: Beam search width.
        initial_prompt: Disfluent style prompt for the first window.
        condition_on_previous_text: Carry decoded text (and style) forward.
        vad_filter: Whether to drop non-speech with Silero VAD.
    """

    model_name: str = "large-v3"
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 5
    initial_prompt: str = VERBATIM_PROMPT
    condition_on_previous_text: bool = True
    vad_filter: bool = False


def audio_files(audio_dir: Path) -> list[Path]:
    """Return the WAV files of a split directory in filename order.

    Args:
        audio_dir: Directory containing ``*.wav`` files.

    Returns:
        Sorted list of WAV paths.

    Raises:
        FileNotFoundError: If the directory has no WAV files.
    """
    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files in {audio_dir}")
    return files


def transcribe_split(
    files: Sequence[Path], split: str, output: Path, config: VerbatimConfig
) -> None:
    """Transcribe files into a JSONL artifact, resuming completed records.

    Args:
        files: WAV paths to transcribe.
        split: ``train`` or ``test``, stored in every record.
        output: Destination JSONL path; existing records are kept.
        config: Decoding configuration.
    """
    from faster_whisper import WhisperModel

    done: set[str] = set()
    if output.exists():
        done = {json.loads(line)["filename"] for line in output.open() if line.strip()}
    model = WhisperModel(
        config.model_name, device="cuda", compute_type=config.compute_type
    )
    started = time.time()
    with output.open("a", encoding="utf-8") as stream:
        for index, path in enumerate(files, start=1):
            if path.name in done:
                continue
            segments, info = model.transcribe(
                str(path),
                language=config.language,
                beam_size=config.beam_size,
                initial_prompt=config.initial_prompt,
                condition_on_previous_text=config.condition_on_previous_text,
                vad_filter=config.vad_filter,
            )
            segments = list(segments)
            record = {
                "split": split,
                "filename": path.name,
                "text": "".join(segment.text for segment in segments),
                "language": info.language,
                "duration_seconds": info.duration,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in segments
                ],
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if index % 50 == 0:
                rate = (time.time() - started) / index
                print(f"{split}: {index}/{len(files)} ({rate:.1f}s/file)", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Transcribe the train and test splits with the verbatim configuration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = VerbatimConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        transcribe_split(
            audio_files(args.data_dir / split),
            split,
            args.output_dir / f"{split}.jsonl",
            config,
        )
        (args.output_dir / f"{split}.meta.json").write_text(
            json.dumps({"backend": "faster-whisper", "configuration": asdict(config)})
        )


if __name__ == "__main__":
    main()
