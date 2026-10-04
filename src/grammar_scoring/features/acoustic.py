"""Deterministic waveform descriptors for E006a; no labels or learned DSP.

Activity means energy activity, not phonetic voicing. Complete 25 ms frames
advance by 10 ms, without padding or altering samples. A file shorter than one
frame uses its available samples. Trailing samples beyond the last complete
frame are ignored in frame summaries but retained in duration/peak metadata.
"""

import wave
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct, rfft

from grammar_scoring.data.audio import read_audio_metadata


@dataclass(frozen=True)
class AcousticConfig:
    """Fixed label-free extraction settings; amplitudes use PCM / 32768."""

    sample_rate: int = 16000
    frame_samples: int = 400
    hop_samples: int = 160
    activity_relative_threshold: float = 0.1
    activity_absolute_floor: float = 1e-5
    minimum_pause_seconds: float = 0.2
    fft_samples: int = 512
    mel_filters: int = 26
    mfcc_coefficients: int = 13
    rolloff_fraction: float = 0.85
    log_power_floor: float = 1e-12


CONFIG = AcousticConfig()
ACTIVITY_COLUMNS = (
    "active_ratio",
    "silence_ratio",
    "pause_count",
    "pause_rate_per_active_second",
    "mean_pause_duration",
    "median_pause_duration",
    "std_pause_duration",
    "max_pause_duration",
    "mean_active_segment_duration",
    "std_active_segment_duration",
)
ENERGY_COLUMNS = (
    "rms_mean",
    "rms_std",
    "rms_cv",
    "rms_q10",
    "rms_q50",
    "rms_q90",
    "peak_to_rms",
)
SPECTRAL_COLUMNS = tuple(
    f"{name}_{stat}"
    for name in ("zcr", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff")
    for stat in ("mean", "std")
)
FEATURE_COLUMNS = (
    ACTIVITY_COLUMNS
    + ENERGY_COLUMNS
    + SPECTRAL_COLUMNS
    + tuple(
        f"mfcc_{coefficient:02d}_{stat}"
        for coefficient in range(1, CONFIG.mfcc_coefficients + 1)
        for stat in ("mean", "std")
    )
)
ARTIFACT_COLUMNS = ("filename", "split", "duration_seconds") + FEATURE_COLUMNS


def extraction_config() -> dict[str, object]:
    """Return serializable settings and interpretation for the fixed extractor."""
    return {
        **asdict(CONFIG),
        "activity_rule": "RMS > max(1e-5, 0.1 * file RMS 90th percentile)",
        "pause_rule": "Internal inactive runs >= 0.2 s; exclude edge silence",
        "segment_time": "run frame count * hop / sample rate",
        "spectral_rule": "Hann-windowed FFT power; all frames including silence",
        "mfcc_rule": "26 triangular HTK mel filters 0..8kHz; log power; DCT-II ortho",
        "mfcc_numbering": "01..13 represent DCT indices 0..12 (includes energy)",
        "pitch": "Omitted: no reliable existing pitch dependency",
        "duration": "Audit only; sample count is never a model feature",
    }


def validate_waveform(waveform: NDArray[np.float64], sample_rate: int = 16000) -> None:
    """Reject empty, nonfinite, nonmono, or unexpected-rate waveform arrays."""
    if sample_rate != CONFIG.sample_rate:
        raise ValueError(f"Expected 16000 Hz audio, got {sample_rate}")
    if waveform.ndim != 1 or not waveform.size:
        raise ValueError("Expected nonempty mono waveform")
    if not np.isfinite(waveform).all():
        raise ValueError("Waveform must contain only finite values")


def load_waveform(path: Path) -> NDArray[np.float64]:
    """Read canonical mono 16 kHz PCM16 WAV, rejecting truncated sample data."""
    metadata = read_audio_metadata(path)
    if (metadata.sample_rate, metadata.num_channels, metadata.sample_width_bytes) != (
        CONFIG.sample_rate,
        1,
        2,
    ):
        raise ValueError(f"Expected 16 kHz mono PCM16 WAV: {path}; got {metadata}")
    with wave.open(str(path), "rb") as stream:
        raw = stream.readframes(metadata.num_frames)
    if len(raw) != 2 * metadata.num_frames:
        raise ValueError(f"Truncated WAV sample data: {path}")
    waveform = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    validate_waveform(waveform, metadata.sample_rate)
    return waveform


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(
        zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    )


def activity_features(rms: NDArray[np.float64]) -> dict[str, float]:
    """Summarize RMS activity and internal pauses using fixed label-free rules.

    No activity yields zero pause/segment statistics; silence ratio remains one.
    Durations describe the hop grid, not precise phonetic speech boundaries.
    """
    if rms.ndim != 1 or not rms.size or not np.isfinite(rms).all() or (rms < 0).any():
        raise ValueError("RMS must be a nonempty finite nonnegative vector")
    threshold = max(
        CONFIG.activity_absolute_floor,
        CONFIG.activity_relative_threshold * float(np.quantile(rms, 0.9)),
    )
    active = rms > threshold
    step = CONFIG.hop_samples / CONFIG.sample_rate
    segments = np.array([(end - start) * step for start, end in _runs(active)])
    pauses = np.array(
        [
            (end - start) * step
            for start, end in _runs(~active)
            if start > 0
            and end < len(active)
            and (end - start) * step >= CONFIG.minimum_pause_seconds
        ]
    )
    active_seconds = float(active.sum() * step)
    return dict(
        zip(
            ACTIVITY_COLUMNS,
            (
                float(active.mean()),
                float((~active).mean()),
                float(pauses.size),
                float(pauses.size / active_seconds) if active_seconds else 0.0,
                float(pauses.mean()) if pauses.size else 0.0,
                float(np.median(pauses)) if pauses.size else 0.0,
                float(pauses.std()) if pauses.size else 0.0,
                float(pauses.max()) if pauses.size else 0.0,
                float(segments.mean()) if segments.size else 0.0,
                float(segments.std()) if segments.size else 0.0,
            ),
            strict=True,
        )
    )


@lru_cache(maxsize=1)
def _mel_bank() -> NDArray[np.float64]:
    hz = np.linspace(0, CONFIG.sample_rate / 2, CONFIG.fft_samples // 2 + 1)
    mel_max = 2595 * np.log10(1 + (CONFIG.sample_rate / 2) / 700)
    edges = 700 * (10 ** (np.linspace(0, mel_max, CONFIG.mel_filters + 2) / 2595) - 1)
    bank = np.array(
        [
            np.maximum(
                0,
                np.minimum(
                    (hz - left) / (middle - left), (right - hz) / (right - middle)
                ),
            )
            for left, middle, right in zip(
                edges[:-2], edges[1:-1], edges[2:], strict=True
            )
        ]
    )
    bank.setflags(write=False)
    return bank


def extract_features(waveform: NDArray[np.float64]) -> dict[str, float]:
    """Extract 51 finite, ordered descriptors from an unmodified mono waveform.

    Spectral moments use FFT power weights, with zero moments for silent frames.
    Near-zero RMS denominators yield zero ratios; all stds are population stds.
    MFCCs use log-floored power, so silence yields finite constant coefficients.
    """
    validate_waveform(waveform)
    if waveform.size < CONFIG.frame_samples:
        frames = waveform[None, :]
    else:
        frames = np.lib.stride_tricks.sliding_window_view(
            waveform, CONFIG.frame_samples
        )[:: CONFIG.hop_samples]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    values = activity_features(rms)
    mean_rms = float(rms.mean())
    global_rms = float(np.sqrt(np.mean(waveform**2)))
    values.update(
        dict(
            zip(
                ENERGY_COLUMNS,
                (
                    mean_rms,
                    float(rms.std()),
                    float(rms.std() / mean_rms) if mean_rms > 1e-12 else 0.0,
                    *map(float, np.quantile(rms, [0.1, 0.5, 0.9])),
                    float(np.abs(waveform).max() / global_rms)
                    if global_rms > 1e-12
                    else 0.0,
                ),
                strict=True,
            )
        )
    )
    zcr = (
        np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
        if frames.shape[1] > 1
        else np.zeros(len(frames))
    )
    power = (
        np.abs(rfft(frames * np.hanning(frames.shape[1]), n=CONFIG.fft_samples)) ** 2
    )
    frequencies = np.linspace(0, CONFIG.sample_rate / 2, power.shape[1])
    total = power.sum(axis=1)
    weights = np.divide(
        power, total[:, None], out=np.zeros_like(power), where=total[:, None] > 0
    )
    centroid = weights @ frequencies
    bandwidth = np.sqrt(
        np.sum(weights * (frequencies - centroid[:, None]) ** 2, axis=1)
    )
    rolloff = frequencies[
        np.argmax(
            np.cumsum(power, axis=1) >= CONFIG.rolloff_fraction * total[:, None], axis=1
        )
    ]
    for name, series in zip(
        ("zcr", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"),
        (zcr, centroid, bandwidth, rolloff),
        strict=True,
    ):
        values[f"{name}_mean"] = float(series.mean())
        values[f"{name}_std"] = float(series.std())
    mfcc = dct(
        np.log(np.maximum(power @ _mel_bank().T, CONFIG.log_power_floor)),
        type=2,
        norm="ortho",
        axis=1,
    )[:, : CONFIG.mfcc_coefficients]
    for index in range(CONFIG.mfcc_coefficients):
        values[f"mfcc_{index + 1:02d}_mean"] = float(mfcc[:, index].mean())
        values[f"mfcc_{index + 1:02d}_std"] = float(mfcc[:, index].std())
    if tuple(values) != FEATURE_COLUMNS or not np.isfinite(list(values.values())).all():
        raise ValueError("Invalid extracted acoustic features")
    return values
