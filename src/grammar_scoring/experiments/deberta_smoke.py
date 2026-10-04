"""Real CUDA integration checks, isolated from canonical E005 experiments."""

from __future__ import annotations

import argparse
import gc
import json
import math
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from grammar_scoring.config.paths import ARTIFACT_DIR, FEATURES_DIR, TRANSCRIPTS_DIR
from grammar_scoring.evaluation.metrics import regression_metrics
from grammar_scoring.experiments.deberta_finetuning import (
    CONFIG,
    _loader,
    create_scaler,
    load_inputs,
    predict,
    runtime_info,
    seed_everything,
    split_fold,
    train_epoch,
)

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch import device as TorchDevice
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

NOTICE = "SMOKE TEST ONLY — NOT E005 RESULT"
TRAIN_ROWS = 32
VALID_ROWS = 16
REQUIRED_SUCCESSFUL_UPDATES = 2
MAX_UPDATE_ATTEMPTS = 16


def smoke_subset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the first 32/16 canonical rows from frozen fold 0 partitions."""
    training, validation = split_fold(frame, 0)
    training = training.iloc[:TRAIN_ROWS].copy()
    validation = validation.iloc[:VALID_ROWS].copy()
    if len(training) != TRAIN_ROWS or len(validation) != VALID_ROWS:
        raise ValueError("Smoke requires 32 training and 16 validation examples")
    if set(training.filename) & set(validation.filename):
        raise ValueError("Smoke partitions overlap")
    return training, validation


def smoke_root(path: Path, experiment: str = "E005") -> Path:
    """Require a resolved smoke/E005 namespace outside canonical output trees."""
    resolved = path.resolve()
    if resolved.name != experiment or resolved.parent.name != "smoke":
        raise ValueError(
            f"Smoke output must be a separate .../smoke/{experiment} directory"
        )
    pairs = zip(resolved.parts, resolved.parts[1:], strict=False)
    if any(
        left in {"models", "experiments"} and right == experiment
        for left, right in pairs
    ):
        raise ValueError(f"Existing {experiment} configuration mismatch")
    return resolved


@dataclass
class SmokeCounts:
    """Observed workload counts; scheduler construction is not an update."""

    minibatches: int = 0
    optimizer_update_attempts: int = 0
    optimizer_steps: int = 0
    scheduler_steps: int = 0
    maximum_padded_length: int = 0
    padded_batches: int = 0

    def validate(self) -> None:
        """Require two genuine updates within the bounded smoke workload."""
        if not 2 <= self.optimizer_update_attempts <= MAX_UPDATE_ATTEMPTS:
            raise RuntimeError("Smoke update attempts must be between 2 and 16")
        if self.minibatches != self.optimizer_update_attempts * 2:
            raise RuntimeError("Every smoke attempt requires 2 mini-batches")
        if self.optimizer_steps != REQUIRED_SUCCESSFUL_UPDATES:
            raise RuntimeError(
                f"Expected 2 successful optimizer steps after at most 16 attempts, "
                f"observed {self.optimizer_steps}"
            )
        if self.scheduler_steps != REQUIRED_SUCCESSFUL_UPDATES:
            raise RuntimeError("Scheduler steps must equal actual optimizer steps (2)")


class BoundedSmokeLoader:
    """Repeat a fixed subset ordering until two updates or sixteen attempts.

    Production requests each new group after scheduler stepping and gradient
    clearing, allowing this loader to stop without changing the training loop.
    """

    def __init__(self, loader: DataLoader, counts: SmokeCounts) -> None:
        """Freeze one seeded traversal of the same 32 training rows."""
        self.batches = list(loader)
        self.counts = counts
        if len(self.batches) != 4 or any(
            len(batch["labels"]) != CONFIG.train_batch_size for batch in self.batches
        ):
            raise RuntimeError("Smoke requires four batches of eight rows")

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yield two unchanged microbatches per bounded update attempt."""
        for attempt in range(MAX_UPDATE_ATTEMPTS):
            if self.counts.optimizer_steps >= REQUIRED_SUCCESSFUL_UPDATES:
                return
            self.counts.optimizer_update_attempts += 1
            start = (attempt * CONFIG.gradient_accumulation_steps) % len(self.batches)
            yield from self.batches[start : start + CONFIG.gradient_accumulation_steps]


class ObservedLoader:
    """Count the batches actually consumed by the unchanged production loop."""

    def __init__(self, loader: DataLoader, counts: SmokeCounts) -> None:
        """Wrap a loader without changing its batches, ordering or padding."""
        self.loader = loader
        self.counts = counts

    def __len__(self) -> int:
        """Return the underlying loader's mini-batch count."""
        return len(self.loader)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Observe dynamic batch lengths and yield unchanged production batches."""
        for batch in self.loader:
            ids, mask = batch["input_ids"], batch["attention_mask"]
            if ids.ndim != 2 or ids.shape != mask.shape:
                raise RuntimeError("Invalid input_ids/attention_mask shapes")
            if ids.shape[1] != int(mask.sum(1).max().item()):
                raise RuntimeError("Batch is not padded dynamically to its longest row")
            self.counts.minibatches += 1
            self.counts.maximum_padded_length = max(
                self.counts.maximum_padded_length, ids.shape[1]
            )
            self.counts.padded_batches += int(bool((mask == 0).any().item()))
            yield batch


class ObservedScheduler:
    """Count only scheduler steps invoked by the production training loop."""

    def __init__(self, scheduler: LRScheduler, counts: SmokeCounts) -> None:
        """Wrap the constructed scheduler without altering its schedule."""
        self.scheduler = scheduler
        self.counts = counts

    def step(self) -> None:
        """Advance the smoke schedule and count its actual update."""
        self.scheduler.step()
        self.counts.scheduler_steps += 1


def validation_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Require 16 finite scalar predictions and return smoke-only loss/metrics."""
    if predictions.shape != (VALID_ROWS,) or labels.shape != (VALID_ROWS,):
        raise RuntimeError("Smoke validation requires exactly 16 scalar predictions")
    if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
        raise RuntimeError("Nonfinite smoke validation predictions or labels")
    loss = float(np.mean(np.square(labels - predictions)))
    metrics = regression_metrics(labels, predictions)
    if not math.isfinite(loss) or not math.isfinite(metrics["rmse"]):
        raise RuntimeError("Nonfinite smoke validation loss/RMSE")
    return {"validation_loss": loss, **metrics}


def check_round_trip(before: np.ndarray, after: np.ndarray) -> None:
    """Require matching finite scalar predictions within FP16 inference tolerance."""
    if before.shape != (VALID_ROWS,) or after.shape != before.shape:
        raise RuntimeError("Checkpoint prediction round-trip shape mismatch")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise RuntimeError("Nonfinite checkpoint round-trip predictions")
    if not np.allclose(before, after, rtol=1e-3, atol=1e-3):
        difference = float(np.max(np.abs(before - after)))
        raise RuntimeError(
            f"Checkpoint prediction round trip failed: max error {difference}"
        )


def inspect_forward(
    model: nn.Module, batch: dict[str, Tensor], device: TorchDevice
) -> dict[str, Any]:
    """Inspect actual backbone, production pooling and regression output shapes.

    Eval-mode hooks also verify that the production dropout/head receives the
    masked mean. Replacing padded hidden states must not change that mean.
    """
    import torch

    from grammar_scoring.models.deberta_regressor import masked_mean_pool

    inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
    captured = {}

    def backbone_hook(module: nn.Module, args: tuple, output: object) -> None:
        captured["hidden"] = output.last_hidden_state

    def head_hook(module: nn.Module, args: tuple) -> None:
        captured["head_input"] = args[0]

    handles = [
        model.backbone.register_forward_hook(backbone_hook),
        model.head.register_forward_pre_hook(head_hook),
    ]
    model.eval()
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            prediction = model(**inputs)
            hidden = captured["hidden"]
            mask = inputs["attention_mask"]
            pooled = masked_mean_pool(hidden, mask)
            if hidden.shape[:2] != inputs["input_ids"].shape:
                raise RuntimeError("Real backbone hidden-state shape mismatch")
            if pooled.shape != (len(mask), model.backbone.config.hidden_size):
                raise RuntimeError("Real pooled representation shape mismatch")
            if prediction.shape != (len(mask),):
                raise RuntimeError("Real regression prediction must have shape [batch]")
            if prediction.dtype != torch.float16 or prediction.device.type != "cuda":
                raise RuntimeError("Real prediction must use CUDA FP16 autocast")
            if not all(
                torch.isfinite(value).all().item()
                for value in (hidden, pooled, prediction)
            ):
                raise RuntimeError("Nonfinite real forward-pass tensors")
            torch.testing.assert_close(captured["head_input"], pooled)
            mask_expanded = mask.unsqueeze(-1).bool()
            replaced = torch.where(mask_expanded, hidden, torch.full_like(hidden, 1000))
            torch.testing.assert_close(masked_mean_pool(replaced, mask), pooled)
            # Independently check the formula against only the unmasked states.
            for row in range(len(mask)):
                expected = hidden[row][mask[row].bool()].float().mean(0)
                torch.testing.assert_close(
                    pooled[row].float(), expected, rtol=2e-3, atol=2e-3
                )
            return {
                "input_ids": list(inputs["input_ids"].shape),
                "attention_mask": list(mask.shape),
                "last_hidden_state": list(hidden.shape),
                "pooled_representation": list(pooled.shape),
                "prediction": list(prediction.shape),
                "prediction_dtype": str(prediction.dtype),
                "padding_present": bool((mask == 0).any().item()),
                "padded_positions_excluded": True,
            }
    finally:
        for handle in handles:
            handle.remove()


def gradient_diagnostics(model: nn.Module) -> dict[str, Any]:
    """Summarize gradients and name tensors containing nonfinite values.

    Args:
        model: Model whose current gradients are inspected without mutation.

    Returns:
        Finite status, maximum magnitude, and NaN/Inf tensor names. Names
        distinguish backbone and head; no gradient tensors are printed.
    """
    import torch

    names: dict[str, list[str]] = {"nan": [], "positive_inf": [], "negative_inf": []}
    maxima = []
    norms = []
    tensor_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        tensor_count += 1
        for key, predicate in (
            ("nan", torch.isnan),
            ("positive_inf", torch.isposinf),
            ("negative_inf", torch.isneginf),
        ):
            if predicate(gradient).any().item():
                names[key].append(name)
        maxima.append(float(gradient.abs().max()))
        norms.append(torch.linalg.vector_norm(gradient))
    return {
        "total_gradient_norm": (
            float(torch.linalg.vector_norm(torch.stack(norms))) if norms else None
        ),
        "gradient_tensor_count": tensor_count,
        "all_finite": not any(names.values()) if tensor_count else None,
        "maximum_absolute_gradient": (
            float("nan")
            if any(math.isnan(value) for value in maxima)
            else max(maxima, default=None)
        ),
        **names,
    }


class SmokeUpdateDiagnostics:
    """Observe update attempts, copying only one small sentinel parameter."""

    def __init__(self) -> None:
        """Initialize smoke-only reports and sentinel storage."""
        self.reports: list[dict[str, Any]] = []
        self._sentinel: Tensor | None = None
        self._before: Tensor | None = None

    def __call__(self, stage: str, model: nn.Module, report: dict[str, Any]) -> None:
        """Collect stage diagnostics and emit each completed attempt immediately.

        Args:
            stage: Update stage provided by the production training loop.
            model: Training model inspected without modifying its state.
            report: Current accumulation group's mutable diagnostic report.
        """
        import torch

        if stage == "after_backward":
            report["microbatches"][-1]["scaled_gradients_after_backward"] = (
                gradient_diagnostics(model)
            )
        elif stage in {"before_unscale", "after_unscale", "after_clip"}:
            report[stage] = gradient_diagnostics(model)
        elif stage == "before_step":
            candidates = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            # Prefer the small regression head over a backbone tensor.
            name, parameter = min(candidates, key=lambda item: item[1].numel())
            report["sentinel_parameter"] = name
            self._sentinel = parameter
            self._before = parameter.detach().clone()
        elif stage == "after_step":
            assert self._sentinel is not None and self._before is not None
            report["sentinel_parameter_changed"] = not torch.equal(
                self._before, self._sentinel.detach()
            )
            self._before = None
            self._sentinel = None
            self.reports.append(report)
            print("AMP update diagnostics:", json.dumps(report), flush=True)


def dtype_diagnostics(stage: str, model: nn.Module, loss: Tensor) -> dict[str, Any]:
    """Report storage/loss or accumulated gradient dtypes without tensor values.

    Args:
        stage: Either before_backward or before_unscale.
        model: Production regressor with a backbone and linear head.
        loss: Unscaled training loss.

    Returns:
        JSON-compatible dtype names for the requested stage.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    backbone = next(p for p in model.backbone.parameters() if p.requires_grad)
    if stage == "before_backward":
        return {
            "stage": stage,
            "trainable_parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "head_weight_dtype": str(model.head.weight.dtype),
            "head_bias_dtype": str(model.head.bias.dtype),
            "backbone_parameter_dtype": str(backbone.dtype),
            "loss_dtype": str(loss.dtype),
        }
    if stage != "before_unscale":
        raise ValueError(f"Unknown dtype diagnostic stage: {stage}")
    return {
        "stage": stage,
        "gradient_dtypes": sorted(
            {str(p.grad.dtype) for p in parameters if p.grad is not None}
        ),
        "head_gradient_dtype": (
            str(model.head.weight.grad.dtype)
            if model.head.weight.grad is not None
            else None
        ),
        "backbone_gradient_dtype": (
            str(backbone.grad.dtype) if backbone.grad is not None else None
        ),
    }


def gpu_memory(device: TorchDevice) -> dict[str, int]:
    """Read allocated, reserved, peak allocated and total CUDA capacity in bytes."""
    import torch

    torch.cuda.synchronize(device)
    values = {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "total_capacity": torch.cuda.get_device_properties(device).total_memory,
    }
    if (
        max(values["allocated"], values["reserved"], values["peak_allocated"])
        > values["total_capacity"]
    ):
        raise RuntimeError("GPU memory exceeded available device capacity")
    return values


def run_smoke(
    transcript_dir: Path,
    fold_path: Path,
    output_root: Path,
    *,
    frame: pd.DataFrame | None = None,
    experiment: str = "E005",
) -> Path:
    """Run bounded GPU smoke updates without canonical fold orchestration."""
    import torch

    notice = f"SMOKE TEST ONLY — NOT {experiment} RESULT"
    print(notice, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; the smoke test cannot run on CPU")
    root = smoke_root(output_root, experiment)
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from grammar_scoring.models.deberta_regressor import DebertaRegressor

    device = torch.device("cuda")
    seed_everything()
    runtime = {**runtime_info(device), "amp_enabled": True}
    print(json.dumps(runtime, indent=2), flush=True)
    training, validation = smoke_subset(
        load_inputs(transcript_dir, fold_path) if frame is None else frame
    )
    selected = {
        "training": training.filename.tolist(),
        "validation": validation.filename.tolist(),
    }
    print("Selected canonical filenames:", json.dumps(selected, indent=2), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)
    untruncated = tokenizer(
        training.text.tolist() + validation.text.tolist(),
        truncation=False,
        padding=False,
    )
    lengths = [len(tokens) for tokens in untruncated["input_ids"]]
    token_diagnostics = {
        "maximum_untruncated_token_length": max(lengths),
        "number_exceeding_256": sum(length > CONFIG.max_length for length in lengths),
    }
    train_loader, _ = _loader(training, tokenizer, shuffle=True)
    valid_loader, _ = _loader(validation, tokenizer, shuffle=False)
    counts = SmokeCounts()
    observed_train = ObservedLoader(BoundedSmokeLoader(train_loader, counts), counts)
    # Prefer an actually padded batch across the selected train/validation rows.
    valid_batches = list(valid_loader)
    inspection_loader, _ = _loader(
        pd.concat([training, validation]), tokenizer, shuffle=False
    )
    inspection_batch = max(
        inspection_loader, key=lambda batch: int((batch["attention_mask"] == 0).sum())
    )
    model = DebertaRegressor.from_pretrained().to(device)
    shapes = inspect_forward(model, inspection_batch, device)
    print("Real runtime tensor shapes:", json.dumps(shapes), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
    )
    expected_steps = REQUIRED_SUCCESSFUL_UPDATES
    schedule = {
        "required_successful_updates": REQUIRED_SUCCESSFUL_UPDATES,
        "max_update_attempts": MAX_UPDATE_ATTEMPTS,
        "total_optimizer_steps": expected_steps,
        "warmup_steps": math.ceil(expected_steps * CONFIG.warmup_ratio),
    }
    scheduler = ObservedScheduler(
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=schedule["warmup_steps"],
            num_training_steps=expected_steps,
        ),
        counts,
    )
    scaler = create_scaler(device)
    if not scaler.is_enabled():
        raise RuntimeError("CUDA FP16 AMP scaler is not enabled")

    initial_scale = scaler.get_scale()
    head_weight = model.head.weight

    def stepped(actual_optimizer: Optimizer, args: tuple, kwargs: dict) -> None:
        gradient = head_weight.grad
        if gradient is None or gradient.device.type != "cuda":
            raise RuntimeError("CUDA backward pass did not produce head gradients")
        if not torch.isfinite(gradient).all().item() or not gradient.any().item():
            raise RuntimeError("Head gradients must be finite and nonzero")
        counts.optimizer_steps += 1

    def fp16_forward(module: nn.Module, args: tuple, output: Tensor) -> None:
        if (
            output.shape != (CONFIG.train_batch_size,)
            or output.dtype != torch.float16
            or output.device.type != "cuda"
            or not torch.is_autocast_enabled()
        ):
            raise RuntimeError("Smoke training forward pass must use CUDA FP16 AMP")
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Nonfinite smoke training forward predictions")

    handle = optimizer.register_step_post_hook(stepped)
    forward_handle = model.register_forward_hook(fp16_forward)
    dtype_reports = []
    update_diagnostics = SmokeUpdateDiagnostics()

    def observe_update(
        stage: str, observed_model: nn.Module, report: dict[str, Any]
    ) -> None:
        if stage == "before_step":
            report["optimizer_hook_count_before"] = counts.optimizer_steps
        elif stage == "after_step":
            report["optimizer_hook_count_after"] = counts.optimizer_steps
            report["optimizer_step_hook_fired"] = (
                counts.optimizer_steps > report["optimizer_hook_count_before"]
            )
        report["update_index"] = counts.optimizer_update_attempts
        if stage == "after_step":
            if (
                report["optimizer_step_hook_fired"]
                != report["update_not_skipped_by_scale"]
            ):
                raise RuntimeError("Optimizer hook and AMP scale disagree on success")
        update_diagnostics(stage, observed_model, report)

    def observe_dtypes(stage: str, observed_model: nn.Module, loss: Tensor) -> None:
        report = dtype_diagnostics(stage, observed_model, loss)
        dtype_reports.append(report)
        print("Training dtype diagnostics:", json.dumps(report), flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    try:
        loss = train_epoch(
            model,
            observed_train,
            optimizer,
            scheduler,
            scaler,
            device,
            dtype_observer=observe_dtypes,
            update_observer=observe_update,
        )
    finally:
        handle.remove()
        forward_handle.remove()
    successes = [
        r for r in update_diagnostics.reports if r["optimizer_step_hook_fired"]
    ]
    amp_summary = {
        "total_update_attempts": counts.optimizer_update_attempts,
        "skipped_amp_updates": counts.optimizer_update_attempts
        - counts.optimizer_steps,
        "successful_optimizer_updates": counts.optimizer_steps,
        "scheduler_steps": counts.scheduler_steps,
        "initial_grad_scaler_scale": initial_scale,
        "final_grad_scaler_scale": scaler.get_scale(),
        "scale_sequence": [initial_scale]
        + [r["scale_after_update"] for r in update_diagnostics.reports],
        "first_successful_update_attempt": (
            successes[0]["update_index"] if successes else None
        ),
        "successful_update_finite_status": [
            {
                "attempt": r["update_index"],
                "after_unscale": r["after_unscale"]["all_finite"],
                "after_clip": r["after_clip"]["all_finite"],
                "gradient_norm": r["gradient_norm_is_finite"],
            }
            for r in successes
        ],
    }
    print("AMP startup summary:", json.dumps(amp_summary), flush=True)
    counts.validate()
    if any(
        not all(
            status[key] is True
            for key in ("after_unscale", "after_clip", "gradient_norm")
        )
        for status in amp_summary["successful_update_finite_status"]
    ):
        raise RuntimeError("Successful updates require finite gradients and norms")
    if not math.isfinite(loss):
        raise RuntimeError("Nonfinite smoke training loss")
    before = predict(model, valid_loader, device)
    metrics = validation_metrics(validation.label.to_numpy(), before)
    memory_training = gpu_memory(device)
    token_diagnostics["maximum_padded_batch_sequence_length"] = max(
        counts.maximum_padded_length,
        max(batch["input_ids"].shape[1] for batch in valid_batches),
    )
    padding_present = shapes["padding_present"] or counts.padded_batches > 0
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    checkpoint = directory / "smoke.pt"
    torch.save(model.state_dict(), checkpoint)
    # Release original GPU weights and optimizer state before the fresh instance.
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    restored = DebertaRegressor.from_pretrained().to(device)
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    after = predict(restored, valid_loader, device)
    check_round_trip(before, after)
    memory_reload = gpu_memory(device)
    finite_metrics = {
        key: value for key, value in metrics.items() if math.isfinite(value)
    }
    report = {
        "notice": notice,
        "status": "PASS",
        "runtime": runtime,
        "canonical_reference_config": asdict(CONFIG),
        "smoke_schedule": schedule,
        "selected_filenames": selected,
        "counts": asdict(counts),
        "tokenization": token_diagnostics,
        "real_tensor_shapes": shapes,
        "training_dtype_diagnostics": dtype_reports,
        "amp_update_diagnostics": update_diagnostics.reports,
        "amp_startup_summary": amp_summary,
        "train_loss": loss,
        "smoke_only_metrics": finite_metrics,
        "memory_after_training": memory_training,
        "memory_after_reload": memory_reload,
        "checkpoint": str(checkpoint),
        "round_trip_rtol": 1e-3,
        "round_trip_atol": 1e-3,
        "round_trip_max_abs_error": float(np.max(np.abs(before - after))),
        "model_revision": getattr(restored.backbone.config, "_commit_hash", None),
    }
    (directory / "smoke_summary.json").write_text(
        json.dumps(_finite_json(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(notice, json.dumps(report, indent=2), flush=True)
    checks = [
        "CUDA available",
        "real DeBERTa model loaded",
        "tokenizer loaded",
        "dynamic padding",
        "forward pass",
        "FP16 backward pass",
        "finite loss",
        "gradient accumulation",
        "optimizer steps == expected (2)",
        "scheduler steps == optimizer steps (2)",
        "validation predictions finite",
        "checkpoint save/reload",
        "prediction round trip",
        "GPU memory within available capacity",
    ]
    for check in checks:
        print(f"[PASS] {check}", flush=True)
    if not padding_present:
        print(
            "[INFO] No padding in these batches; padded-position exclusion is vacuous."
        )
    print(notice, "PASS", flush=True)
    return directory


def _finite_json(value: object) -> object:
    """Encode legitimate overflow diagnostics as valid JSON string values."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def main() -> None:
    """Run CUDA integration checks with a compact failure summary and nonzero exit."""
    parser = argparse.ArgumentParser(description=NOTICE)
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument(
        "--fold-path", type=Path, default=FEATURES_DIR / "train_folds.csv"
    )
    parser.add_argument("--smoke-dir", type=Path, default=ARTIFACT_DIR / "smoke/E005")
    args = parser.parse_args()
    try:
        directory = run_smoke(args.transcript_dir, args.fold_path, args.smoke_dir)
        print("Smoke artifacts:", directory, flush=True)
    except Exception as exc:
        print(NOTICE, flush=True)
        print(f"[FAIL] {type(exc).__name__}: {exc}", flush=True)
        if "out of memory" in str(exc).lower():
            print(
                "CUDA OOM: stopping; canonical batch size and settings are unchanged."
            )
        raise SystemExit(1) from exc
