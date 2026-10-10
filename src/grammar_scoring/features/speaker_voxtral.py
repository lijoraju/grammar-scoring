"""E026: speaker-identity embeddings and Voxtral audio-LLM states.

Two label-free extractions used to validate on unseen speakers and to add an
audio language model channel:

* Speaker embeddings from ``microsoft/wavlm-base-plus-sv`` (x-vectors), averaged
  over 10 s windows and L2-normalized. Clips of the same speaker are later
  grouped by cosine similarity so cross-validation folds can keep speakers
  together.
* Voxtral-Mini-3B hidden states: the model receives the audio and a question
  about the speaker's grammar; the states at the audio-token positions are
  averaged per layer. A small regressor is trained on these averages later.

This module depends only on third-party packages so it can run as a Kaggle
script; ``torch`` and ``transformers`` are imported lazily.
"""

from __future__ import annotations

import argparse
import time
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"
SPEAKER_WINDOW_SECONDS = 10
VOXTRAL_MODEL = "mistralai/Voxtral-Mini-3B-2507"
VOXTRAL_QUESTION = "How accurate and complex is the speaker's grammar?"
# Alternative questions give further "views" of the same audio (E028).
VOXTRAL_QUESTIONS = {
    "grammar": VOXTRAL_QUESTION,
    "errors": "What grammatical errors does the speaker make?",
    "structure": "Does the speaker use correct sentence structure and verb forms?",
}


def read_wav(path: Path) -> NDArray[np.float32]:
    """Read a 16 kHz mono PCM16 WAV file as float32 in [-1, 1].

    Raises:
        ValueError: If the file is not 16 kHz mono 16-bit PCM.
    """
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getframerate(),
            stream.getnchannels(),
            stream.getsampwidth(),
        ) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"Expected 16 kHz mono PCM16 WAV: {path}")
        raw = stream.readframes(stream.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def windows(n_samples: int, window: int) -> list[tuple[int, int]]:
    """Return consecutive full windows, or one short window for a short clip.

    Args:
        n_samples: Recording length in samples.
        window: Window length in samples.

    Returns:
        ``(start, end)`` ranges; a trailing partial window is dropped unless
        the recording is shorter than one window.

    Raises:
        ValueError: If lengths are not positive.
    """
    if n_samples <= 0 or window <= 0:
        raise ValueError("Lengths must be positive")
    if n_samples < window:
        return [(0, n_samples)]
    return [(s, s + window) for s in range(0, n_samples - window + 1, window)]


def unit_mean(vectors: NDArray[np.floating]) -> NDArray[np.float64]:
    """Average row vectors after L2-normalizing each, and normalize the result.

    Args:
        vectors: Array ``[n, dim]`` with at least one non-zero row.

    Returns:
        Unit-length mean direction.

    Raises:
        ValueError: If the input is empty or the mean has zero length.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("Expected a non-empty [n, dim] array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    mean = (vectors / np.maximum(norms, 1e-12)).mean(axis=0)
    length = np.linalg.norm(mean)
    if length == 0:
        raise ValueError("Mean embedding has zero length")
    return mean / length


def extract_speaker(files: Sequence[Path], output: Path) -> None:
    """Save one unit-length speaker embedding per recording."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    extractor = AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL)
    model = WavLMForXVector.from_pretrained(SPEAKER_MODEL).to("cuda").eval()
    window = SPEAKER_WINDOW_SECONDS * SAMPLE_RATE
    rows = []
    for index, path in enumerate(files, start=1):
        audio = read_wav(path)
        chunks = [audio[a:b] for a, b in windows(len(audio), window)]
        inputs = extractor(
            chunks, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        ).to("cuda")
        with torch.no_grad():
            embeddings = model(**inputs).embeddings.float().cpu().numpy()
        rows.append(unit_mean(embeddings))
        if index % 200 == 0:
            print(f"speaker: {index}/{len(files)}", flush=True)
    np.savez_compressed(
        output,
        filenames=np.array([p.name for p in files]),
        embeddings=np.stack(rows).astype(np.float32),
    )


def extract_voxtral(
    files: Sequence[Path],
    output: Path,
    question: str = VOXTRAL_QUESTION,
    layers: Sequence[int] | None = None,
    question_first: bool = False,
) -> None:
    """Save per-layer means of Voxtral states over the audio-token positions.

    Args:
        files: WAV paths.
        output: Destination ``.npz``.
        question: Text question given to the model together with the audio.
        layers: Hidden-state indices to keep (all layers when ``None``).
        question_first: Put the question before the audio. Voxtral reads left to
            right, so the audio-token states depend on the question only when it
            comes first; with the default order they are identical for every
            question. The state of the final token, which has seen both, is
            always saved as ``last``.
    """
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    processor = AutoProcessor.from_pretrained(VOXTRAL_MODEL)
    model = VoxtralForConditionalGeneration.from_pretrained(
        VOXTRAL_MODEL, torch_dtype=torch.float16, device_map={"": 0}
    ).eval()
    audio_token_id = model.config.audio_token_id
    rows, lasts, counts = [], [], []
    started = time.time()
    for index, path in enumerate(files, start=1):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "audio", "path": str(path)},
                ]
                if question_first
                else [
                    {"type": "audio", "path": str(path)},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to("cuda", dtype=torch.float16)
        with torch.no_grad():
            states = model(**inputs, output_hidden_states=True).hidden_states
        positions = inputs["input_ids"][0] == audio_token_id
        if not bool(positions.any()):
            raise RuntimeError(f"No audio tokens found for {path.name}")
        stacked = torch.stack(states)[:, 0].float()  # [layers, tokens, dim]
        means = stacked[:, positions].mean(dim=1)
        last = stacked[:, -1]
        if not bool(torch.isfinite(means).all() and torch.isfinite(last).all()):
            raise FloatingPointError(f"Non-finite Voxtral states for {path.name}")
        kept = means if layers is None else means[list(layers)]
        rows.append(kept.cpu().numpy().astype(np.float16))
        kept_last = last if layers is None else last[list(layers)]
        lasts.append(kept_last.cpu().numpy().astype(np.float16))
        counts.append(int(positions.sum()))
        if index % 50 == 0:
            rate = (time.time() - started) / index
            print(f"voxtral: {index}/{len(files)} ({rate:.2f}s/file)", flush=True)
    np.savez_compressed(
        output,
        filenames=np.array([p.name for p in files]),
        mean=np.stack(rows),
        last=np.stack(lasts),
        question_first=np.array(question_first),
        audio_tokens=np.array(counts),
        question=np.array(question),
        layers=np.array(list(layers) if layers is not None else [-1]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Extract speaker embeddings and/or Voxtral states for train and test."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--what", nargs="+", choices=["speaker", "voxtral"], required=True
    )
    parser.add_argument(
        "--question", choices=sorted(VOXTRAL_QUESTIONS), default="grammar"
    )
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--question-first", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def voxtral(files: Sequence[Path], output: Path) -> None:
        extract_voxtral(
            files,
            output,
            VOXTRAL_QUESTIONS[args.question],
            args.layers,
            args.question_first,
        )

    extractors = {"speaker": extract_speaker, "voxtral": voxtral}
    for what in args.what:
        tag = what
        if what == "voxtral" and (args.question != "grammar" or args.question_first):
            tag = f"voxtral_{args.question}" + (
                "_qfirst" if args.question_first else ""
            )
        for split in ("train", "test"):
            output = args.output_dir / f"{tag}_{split}.npz"
            if output.exists():
                print("skip", output)
                continue
            files = sorted((args.data_dir / split).glob("*.wav"))
            if not files:
                raise FileNotFoundError(f"No WAV files in {args.data_dir / split}")
            extractors[what](files, output)
            print("wrote", output, flush=True)


if __name__ == "__main__":
    main()
