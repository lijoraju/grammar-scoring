"""E023: per-layer utterance statistics from frozen speech encoders.

E022 used only the final WavLM layer averaged over time. Middle layers of
self-supervised speech models usually carry more pronunciation and fluency
information, and the spread of a layer over time describes how steady the
speech is. This module stores, for every encoder layer, the mean and standard
deviation over all valid frames of a recording, so layer choice can be made
later on CPU by cross-validation.

Supported encoders:

* self-supervised models loaded with ``AutoModel`` (WavLM, HuBERT, wav2vec2),
  run on 20 s chunks without padding;
* the Whisper encoder, run on 30 s log-mel windows; only frames covering real
  audio (50 per second) are pooled, never the zero padding.

Statistics are accumulated exactly across chunks from per-chunk frame sums and
sums of squares. This module depends only on third-party packages so it can run
in Colab; ``torch`` and ``transformers`` are imported lazily.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
SSL_CHUNK_SECONDS = 20
WHISPER_WINDOW_SECONDS = 30
WHISPER_FRAMES_PER_SECOND = 50
MIN_TAIL_SECONDS = 1.0

ENCODERS = {
    "wavlm_base_plus": "microsoft/wavlm-base-plus",
    "wavlm_large": "microsoft/wavlm-large",
    "whisper_large_v3": "openai/whisper-large-v3",
}


def plan_chunks(
    n_samples: int, chunk_samples: int, min_tail_samples: int
) -> list[tuple[int, int]]:
    """Split a recording into consecutive chunks, merging a too-short tail.

    Args:
        n_samples: Recording length in samples.
        chunk_samples: Target chunk length in samples.
        min_tail_samples: A final chunk shorter than this merges into the
            previous chunk (unless it is the only chunk).

    Returns:
        ``(start, end)`` sample ranges covering every sample exactly once.

    Raises:
        ValueError: If the recording is empty or lengths are not positive.
    """
    if n_samples <= 0 or chunk_samples <= 0 or min_tail_samples < 0:
        raise ValueError("Lengths must be positive")
    edges = list(range(0, n_samples, chunk_samples)) + [n_samples]
    chunks = list(zip(edges[:-1], edges[1:], strict=True))
    if len(chunks) > 1 and chunks[-1][1] - chunks[-1][0] < min_tail_samples:
        last = chunks.pop()
        chunks[-1] = (chunks[-1][0], last[1])
    return chunks


def whisper_valid_frames(n_samples: int) -> int:
    """Return how many Whisper encoder frames cover real audio in one window.

    Args:
        n_samples: Samples in the window before padding to 30 s.

    Returns:
        ``ceil(seconds * 50)``, capped at the 1500 frames of a full window.
    """
    frames = math.ceil(n_samples / SAMPLE_RATE * WHISPER_FRAMES_PER_SECOND)
    return min(frames, WHISPER_WINDOW_SECONDS * WHISPER_FRAMES_PER_SECOND)


@dataclass
class LayerStats:
    """Running per-layer frame sums for an exact mean and standard deviation."""

    total: NDArray[np.float64] | None = None
    squares: NDArray[np.float64] | None = None
    frames: int = 0
    history: list[int] = field(default_factory=list)

    def add(
        self,
        frame_sum: NDArray[np.floating],
        frame_square_sum: NDArray[np.floating],
        n_frames: int,
    ) -> None:
        """Add one chunk's per-layer sums over its valid frames.

        Args:
            frame_sum: Array ``[layers, dim]`` of summed hidden states.
            frame_square_sum: Array ``[layers, dim]`` of summed squares.
            n_frames: Number of valid frames summed.

        Raises:
            ValueError: If shapes disagree or no frames are added.
        """
        if n_frames <= 0:
            raise ValueError("A chunk must contribute at least one frame")
        frame_sum = np.asarray(frame_sum, dtype=np.float64)
        frame_square_sum = np.asarray(frame_square_sum, dtype=np.float64)
        if frame_sum.shape != frame_square_sum.shape or frame_sum.ndim != 2:
            raise ValueError("Sums must share one [layers, dim] shape")
        if self.total is None or self.squares is None:
            self.total, self.squares = frame_sum.copy(), frame_square_sum.copy()
        elif self.total.shape != frame_sum.shape:
            raise ValueError("Chunks disagree on [layers, dim]")
        else:
            self.total += frame_sum
            self.squares += frame_square_sum
        self.frames += n_frames
        self.history.append(n_frames)

    def finalize(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return per-layer ``(mean, std)`` over all added frames.

        Raises:
            ValueError: If nothing was added.
        """
        if self.total is None or self.squares is None or self.frames == 0:
            raise ValueError("No frames were added")
        mean = self.total / self.frames
        variance = np.maximum(self.squares / self.frames - mean**2, 0.0)
        return mean, np.sqrt(variance)


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


class Encoder:
    """Frozen speech encoder returning per-layer sums for one audio chunk."""

    def __init__(self, key: str, device: str = "cuda") -> None:
        """Load the encoder named by ``key`` (see ``ENCODERS``) in eval mode."""
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.key, self.name, self.device = key, ENCODERS[key], device
        self.is_whisper = key.startswith("whisper")
        # Whisper runs in fp16 as usual; self-supervised encoders stay fp32 because
        # their convolutional front end can overflow in fp16.
        use_fp16 = self.is_whisper and device == "cuda"
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.processor = AutoFeatureExtractor.from_pretrained(self.name)
        try:  # Transformers >= 4.56 names the argument ``dtype``.
            model = AutoModel.from_pretrained(self.name, dtype=self.dtype)
        except TypeError:
            model = AutoModel.from_pretrained(self.name, torch_dtype=self.dtype)
        self.model = (model.get_encoder() if self.is_whisper else model).to(device)
        self.model.eval().requires_grad_(False)
        self.chunk_samples = SAMPLE_RATE * (
            WHISPER_WINDOW_SECONDS if self.is_whisper else SSL_CHUNK_SECONDS
        )

    def chunk_sums(
        self, chunk: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], int]:
        """Return per-layer frame sums, sums of squares and the valid frame count."""
        import torch

        inputs = self.processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        key = "input_features" if self.is_whisper else "input_values"
        values = inputs[key].to(self.device, self.dtype)
        with torch.no_grad():
            states = self.model(values, output_hidden_states=True).hidden_states
        hidden = torch.stack(states)[:, 0].float()  # [layers, frames, dim]
        if not torch.isfinite(hidden).all():
            raise FloatingPointError(f"{self.key}: non-finite hidden states")
        if self.is_whisper:
            hidden = hidden[:, : whisper_valid_frames(len(chunk))]
        return (
            hidden.sum(1).cpu().numpy(),
            hidden.pow(2).sum(1).cpu().numpy(),
            int(hidden.shape[1]),
        )

    def encode(
        self, waveform: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return per-layer ``(mean, std)`` over all valid frames of a recording."""
        stats = LayerStats()
        min_tail = int(MIN_TAIL_SECONDS * SAMPLE_RATE)
        for start, end in plan_chunks(len(waveform), self.chunk_samples, min_tail):
            stats.add(*self.chunk_sums(waveform[start:end]))
        return stats.finalize()


def extract(
    key: str, files: Sequence[Path], output: Path, device: str = "cuda"
) -> None:
    """Encode recordings and save float16 per-layer statistics to ``output``.

    The ``.npz`` holds ``filenames``, ``mean`` and ``std`` arrays of shape
    ``[recordings, layers, dim]`` and the encoder name.
    """
    encoder = Encoder(key, device)
    means, stds = [], []
    started = time.time()
    for index, path in enumerate(files, start=1):
        mean, std = encoder.encode(read_wav(path))
        means.append(mean.astype(np.float16))
        stds.append(std.astype(np.float16))
        if index % 100 == 0:
            rate = (time.time() - started) / index
            print(f"{key}: {index}/{len(files)} ({rate:.2f}s/file)", flush=True)
    np.savez_compressed(
        output,
        filenames=np.array([p.name for p in files]),
        mean=np.stack(means),
        std=np.stack(stds),
        encoder=np.array(encoder.name),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Extract per-layer statistics for train and test with each encoder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in args.encoders:
        for split in ("train", "test"):
            output = args.output_dir / f"{key}_{split}.npz"
            if output.exists():
                print("skip", output)
                continue
            files = sorted((args.data_dir / split).glob("*.wav"))
            extract(key, files, output)
    (args.output_dir / "extraction.json").write_text(
        json.dumps({"encoders": {k: ENCODERS[k] for k in args.encoders}})
    )


if __name__ == "__main__":
    main()
