"""Dataset-level audio inspection and plain-text reporting."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from grammar_scoring.data import dataset
from grammar_scoring.data.audio import collect_audio_metadata


@dataclass
class SplitInspection:
    """CSV row count, readable WAV metadata, and per-file inspection failures."""

    split: str
    csv_rows: int
    metadata: pd.DataFrame
    errors: list[tuple[Path, str]]


def inspect_dataset_audio() -> list[SplitInspection]:
    """Inspect both configured splits, including unreferenced WAV files.

    Returns:
        Train and test inspections in that order. Invalid headers are recorded
        as per-file errors so that other files can still be inspected.

    Raises:
        FileNotFoundError: If required dataset files or directories are missing.
        ValueError: If dataset integrity validation fails.
        OSError: If dataset tables or directories cannot be accessed.
    """
    extras = dataset.validate_dataset(identify_unreferenced=True)
    results = []
    for split, loader, resolver in (
        ("train", dataset.load_train_dataframe, dataset.get_train_audio_path),
        ("test", dataset.load_test_dataframe, dataset.get_test_audio_path),
    ):
        table = loader()
        paths = sorted(
            {resolver(name) for name in table["filename"]} | set(extras[split])
        )
        frames = []
        errors = []
        for path in paths:
            try:
                frames.append(collect_audio_metadata([path]))
            except (ValueError, OSError) as exc:
                errors.append((path, str(exc)))
        metadata = (
            pd.concat(frames, ignore_index=True)
            if frames
            else collect_audio_metadata([])
        )
        metadata.insert(0, "split", split)
        results.append(SplitInspection(split, len(table), metadata, errors))
    return results


def format_audio_report(inspection: SplitInspection) -> str:
    """Format deterministic statistics and anomalies for one split.

    Args:
        inspection: Dataset inspection result to summarize.

    Returns:
        Human-readable report using seconds and population standard deviation.
    """
    frame = inspection.metadata
    lines = [
        f"{inspection.split}:",
        f"  CSV rows: {inspection.csv_rows}",
        f"  WAV files inspected: {len(frame) + len(inspection.errors)} "
        f"({len(frame)} readable, {len(inspection.errors)} failed)",
    ]
    if frame.empty:
        lines.append("  Duration: unavailable (no readable WAV files)")
    else:
        durations = frame["duration_seconds"]
        lines.append(
            "  Duration (seconds): "
            f"min={durations.min():.3f}, max={durations.max():.3f}, "
            f"mean={durations.mean():.3f}, median={durations.median():.3f}, "
            f"std={durations.std(ddof=0):.3f}, total={durations.sum():.3f}"
        )
    for column, label in (
        ("sample_rate", "Sample rates (Hz)"),
        ("num_channels", "Channel counts"),
        ("sample_width_bytes", "Sample widths (bytes)"),
    ):
        counts = frame[column].value_counts().sort_index()
        distribution = ", ".join(f"{value}: {count}" for value, count in counts.items())
        lines.append(f"  {label}: {distribution or 'none'}")
    lines.append(f"  Unreadable/invalid WAV files: {len(inspection.errors)}")
    for path, error in inspection.errors:
        lines.append(f"    {path.name}: {error}")
    lines.append(
        "  Zero/invalid duration: rejected by the WAV reader and listed above "
        "when frame count or sample rate is invalid"
    )
    for label, mask in (
        ("Short recordings (<30 seconds)", frame["duration_seconds"] < 30),
        ("Long recordings (>90 seconds)", frame["duration_seconds"] > 90),
    ):
        anomalies = frame.loc[mask].sort_values(["duration_seconds", "filename"])
        lines.append(f"  {label}: {len(anomalies)}")
        for row in anomalies.itertuples(index=False):
            lines.append(f"    {row.filename}: {row.duration_seconds:.3f} seconds")
    return "\n".join(lines)
