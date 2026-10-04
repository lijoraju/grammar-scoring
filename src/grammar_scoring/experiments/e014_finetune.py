"""E014: parameterized transformer fine-tuning for backbone and seed ensembles.

E014 keeps the E008 population (``label > 0``) and the frozen five-fold
assignments, but makes the backbone, seed and optimizer settings explicit so
that larger encoders and multiple seeds can be trained and averaged. Each run
trains one model per fold, selects its epoch on that fold's validation RMSE
(the same rule as E005/E008), and predicts the test set with the selected
epoch. Only predictions are written, so runs need no checkpoint storage.

This module deliberately imports only third-party packages so that it can run
as a self-contained Kaggle script. PyTorch and Transformers are imported lazily
inside the training functions; data assembly is importable without them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch

N_FOLDS = 5


@dataclass(frozen=True)
class E014Config:
    """Fine-tuning hyperparameters recorded with every run.

    Attributes:
        model_name: Hugging Face backbone identifier.
        seed: Seed for Python, NumPy, Torch and the data loader order.
        learning_rate: Peak learning rate of the top encoder layer.
        head_learning_rate: Learning rate of the regression head.
        layer_decay: Multiplicative learning-rate decay per layer below the top
            (1.0 disables layer-wise decay).
        weight_decay: AdamW weight decay for non-bias, non-norm weights.
        epochs: Training epochs; the best validation epoch is selected.
        batch_size: Per-step batch size.
        grad_accum: Gradient accumulation steps per optimizer update.
        max_length: Maximum token length (dynamic padding below it).
        warmup_ratio: Fraction of optimizer updates used for linear warmup.
        max_grad_norm: Gradient clipping norm.
    """

    model_name: str = "microsoft/deberta-v3-base"
    seed: int = 42
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-3
    layer_decay: float = 1.0
    weight_decay: float = 0.01
    epochs: int = 5
    batch_size: int = 8
    grad_accum: int = 2
    max_length: int = 320
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    @property
    def run_name(self) -> str:
        """Return a filesystem-safe run identifier."""
        backbone = self.model_name.split("/")[-1]
        return f"{backbone}_lr{self.learning_rate:g}_s{self.seed}"


def load_transcripts(path: Path) -> dict[str, str]:
    """Map filename to raw ASR text from a transcript JSONL artifact.

    Args:
        path: JSONL file with ``filename`` and ``text`` fields per line.

    Returns:
        Dictionary from filename to unmodified transcript text.

    Raises:
        ValueError: If a filename appears more than once.
    """
    texts: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["filename"] in texts:
                raise ValueError(f"Duplicate transcript: {record['filename']}")
            texts[record["filename"]] = str(record["text"]).strip()
    return texts


def build_frames(
    train_csv: Path,
    test_csv: Path,
    folds_csv: Path,
    train_transcripts: Path,
    test_transcripts: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the E008 training population and the ordered test frame.

    Training rows keep ``label > 0`` only, with their original frozen folds.
    Labels come from ``train.csv`` and must agree with the fold artifact.

    Args:
        train_csv: Competition training labels.
        test_csv: Competition test rows (placeholder labels are ignored).
        folds_csv: Frozen fold assignments with filename, label and fold.
        train_transcripts: Training transcript JSONL.
        test_transcripts: Test transcript JSONL.

    Returns:
        ``(train, test)``; train has filename, label, fold and text columns,
        test has filename and text in ``test.csv`` order.

    Raises:
        ValueError: If identities, labels, folds or transcripts do not align.
    """
    labels = pd.read_csv(train_csv)
    folds = pd.read_csv(folds_csv)
    test = pd.read_csv(test_csv)[["filename"]]
    if not labels["filename"].is_unique or not test["filename"].is_unique:
        raise ValueError("Duplicate filenames in competition CSVs")
    if set(labels["filename"]) != set(folds["filename"]) or len(labels) != len(folds):
        raise ValueError("train.csv and fold artifact contain different filenames")
    # A left merge keeps train.csv row order.
    merged = labels.merge(folds, on="filename", how="left", suffixes=("", "_fold"))
    if not np.allclose(merged["label"], merged["label_fold"]):
        raise ValueError("Fold artifact labels disagree with train.csv")
    train = merged.loc[merged["label"] > 0, ["filename", "label", "fold"]]
    train = train.reset_index(drop=True)
    if sorted(train["fold"].unique().tolist()) != list(range(N_FOLDS)):
        raise ValueError(f"Expected folds 0..{N_FOLDS - 1}")
    for frame, path in ((train, train_transcripts), (test, test_transcripts)):
        texts = load_transcripts(path)
        missing = set(frame["filename"]) - set(texts)
        if missing:
            raise ValueError(f"{len(missing)} filenames lack transcripts in {path}")
        frame["text"] = frame["filename"].map(texts)
    return train, test


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch (CPU and CUDA)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_model(model_name: str) -> torch.nn.Module:
    import torch
    from transformers import AutoConfig, AutoModel

    class Regressor(torch.nn.Module):
        """Backbone, masked mean pooling and a linear regression head."""

        def __init__(self) -> None:
            super().__init__()
            config = AutoConfig.from_pretrained(model_name)
            # Dropout hurts regression targets; disable it inside the encoder.
            for key in ("hidden_dropout_prob", "attention_probs_dropout_prob"):
                if hasattr(config, key):
                    setattr(config, key, 0.0)
            self.backbone = AutoModel.from_pretrained(
                model_name, config=config, torch_dtype=torch.float32
            )
            self.head = torch.nn.Linear(config.hidden_size, 1)

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            states = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
            return self.head(pooled).squeeze(-1)

    return Regressor()


def layer_depth(name: str, n_layers: int) -> int:
    """Return the encoder depth of a parameter for layer-wise LR decay.

    Embeddings have depth 0, encoder layer ``i`` has depth ``i + 1`` and any
    other backbone parameter (e.g. shared relative embeddings) the top depth.

    Args:
        name: Parameter name relative to the regressor module.
        n_layers: Number of encoder layers.

    Returns:
        Integer depth in ``[0, n_layers]``.
    """
    if ".embeddings." in name:
        return 0
    marker = ".layer."
    if marker in name:
        index = name.split(marker, 1)[1].split(".", 1)[0]
        if index.isdigit():
            return int(index) + 1
    return n_layers


def parameter_groups(
    model: torch.nn.Module, config: E014Config
) -> list[dict[str, Any]]:
    """Build AdamW groups with layer-wise LR decay and no decay on norms/biases.

    Args:
        model: Regressor with ``backbone`` and ``head`` submodules.
        config: Run configuration.

    Returns:
        Parameter group dictionaries for ``torch.optim.AdamW``.
    """
    n_layers = int(model.backbone.config.num_hidden_layers)
    groups: dict[tuple[float, float], list[Any]] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("head."):
            lr = config.head_learning_rate
        else:
            depth = layer_depth(name, n_layers)
            lr = config.learning_rate * config.layer_decay ** (n_layers - depth)
        no_decay = name.endswith("bias") or "LayerNorm" in name or "norm" in name
        decay = 0.0 if no_decay else config.weight_decay
        groups.setdefault((lr, decay), []).append(parameter)
    return [
        {"params": params, "lr": lr, "weight_decay": decay}
        for (lr, decay), params in groups.items()
    ]


def _batches(
    encodings: list[list[int]],
    targets: np.ndarray | None,
    batch_size: int,
    pad_id: int,
    order: np.ndarray,
) -> list[dict[str, torch.Tensor]]:
    import torch

    batches = []
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        width = max(len(encodings[i]) for i in index)
        ids = torch.full((len(index), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(index), width), dtype=torch.long)
        for row, i in enumerate(index):
            ids[row, : len(encodings[i])] = torch.tensor(encodings[i])
            mask[row, : len(encodings[i])] = 1
        batch = {"input_ids": ids, "attention_mask": mask}
        if targets is not None:
            batch["labels"] = torch.tensor(targets[index], dtype=torch.float32)
        batches.append(batch)
    return batches


def _predict(
    model: torch.nn.Module,
    encodings: list[list[int]],
    pad_id: int,
    device: torch.device,
) -> np.ndarray:
    import torch

    model.eval()
    # Length-sorted inference reduces padding; results are restored in order.
    order = np.argsort([len(e) for e in encodings], kind="stable")
    outputs = np.empty(len(encodings), dtype=np.float64)
    with torch.no_grad():
        for start, batch in zip(
            range(0, len(order), 32),
            _batches(encodings, None, 32, pad_id, order),
            strict=True,
        ):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            outputs[order[start : start + 32]] = pred.float().cpu().numpy()
    return outputs


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _pearson(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else float("nan")


def train_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold: int,
    config: E014Config,
    device: torch.device,
) -> dict[str, Any]:
    """Train one fold and return selected validation and test predictions.

    Args:
        train: E008 population with filename, label, fold and text.
        test: Test frame with filename and text.
        fold: Validation fold index.
        config: Run configuration.
        device: Torch device.

    Returns:
        Dictionary with per-epoch history, the selected epoch, validation
        predictions (``valid``) and test predictions (``test``) for it, plus
        last-epoch predictions for selection-free diagnostics.
    """
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    seed_everything(config.seed + fold)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    pad_id = tokenizer.pad_token_id or 0

    def encode(texts: pd.Series) -> list[list[int]]:
        return tokenizer(texts.tolist(), max_length=config.max_length, truncation=True)[
            "input_ids"
        ]

    is_valid = train["fold"].to_numpy() == fold
    fit, valid = train[~is_valid], train[is_valid]
    fit_enc, valid_enc, test_enc = (
        encode(fit.text),
        encode(valid.text),
        encode(test.text),
    )
    fit_y = fit["label"].to_numpy(dtype=np.float64)
    valid_y = valid["label"].to_numpy(dtype=np.float64)

    model = _build_model(config.model_name).to(device)
    # Start the head at the training mean so early updates are not wasted.
    with torch.no_grad():
        model.head.bias.fill_(float(fit_y.mean()))
    optimizer = torch.optim.AdamW(parameter_groups(model, config))
    updates_per_epoch = math.ceil(
        math.ceil(len(fit_enc) / config.batch_size) / config.grad_accum
    )
    total = updates_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(total * config.warmup_ratio), total
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    rng = np.random.default_rng(config.seed + fold)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        batches = _batches(
            fit_enc, fit_y, config.batch_size, pad_id, rng.permutation(len(fit_enc))
        )
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batches, start=1):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            loss = torch.nn.functional.mse_loss(
                pred.float(), batch["labels"].to(device)
            )
            scaler.scale(loss / config.grad_accum).backward()
            losses.append(float(loss.detach()))
            if step % config.grad_accum == 0 or step == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        valid_pred = _predict(model, valid_enc, pad_id, device)
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "valid_rmse": _rmse(valid_y, valid_pred),
            "valid_pearson": _pearson(valid_y, valid_pred),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        last = {"valid": valid_pred, "test": _predict(model, test_enc, pad_id, device)}
        if best is None or row["valid_rmse"] < best["valid_rmse"]:
            best = {"epoch": epoch, "valid_rmse": row["valid_rmse"], **last}
    assert best is not None
    del model, optimizer
    torch.cuda.empty_cache()
    return {
        "fold": fold,
        "history": history,
        "selected_epoch": best["epoch"],
        "valid_index": np.flatnonzero(is_valid),
        "valid": best["valid"],
        "test": best["test"],
        "valid_last": last["valid"],
        "test_last": last["test"],
    }


def run(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: E014Config,
    output_dir: Path,
) -> dict[str, Any]:
    """Train all folds for one configuration and write OOF/test predictions.

    Writes ``<run>/oof.csv`` (filename, label, fold, prediction,
    prediction_last), ``<run>/test.csv`` (filename, prediction,
    prediction_last; fold-averaged) and ``<run>/result.json``.

    Args:
        train: E008 population with filename, label, fold and text.
        test: Test frame with filename and text.
        config: Run configuration.
        output_dir: Root directory for run outputs.

    Returns:
        The result summary written to ``result.json``.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = output_dir / config.run_name
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    oof = np.full(len(train), np.nan)
    oof_last = np.full(len(train), np.nan)
    test_preds, test_last, folds = [], [], []
    for fold in range(N_FOLDS):
        result = train_fold(train, test, fold, config, device)
        oof[result["valid_index"]] = result["valid"]
        oof_last[result["valid_index"]] = result["valid_last"]
        test_preds.append(result["test"])
        test_last.append(result["test_last"])
        folds.append({k: result[k] for k in ("fold", "selected_epoch", "history")})
    y = train["label"].to_numpy(dtype=np.float64)
    pd.DataFrame(
        {
            "filename": train["filename"],
            "label": y,
            "fold": train["fold"],
            "prediction": oof,
            "prediction_last": oof_last,
        }
    ).to_csv(directory / "oof.csv", index=False)
    pd.DataFrame(
        {
            "filename": test["filename"],
            "prediction": np.mean(test_preds, axis=0),
            "prediction_last": np.mean(test_last, axis=0),
        }
    ).to_csv(directory / "test.csv", index=False)
    summary = {
        "experiment": "E014",
        "config": asdict(config),
        "run_name": config.run_name,
        "n_train": len(train),
        "oof_rmse": _rmse(y, oof),
        "oof_pearson": _pearson(y, oof),
        "oof_rmse_last_epoch": _rmse(y, oof_last),
        "oof_pearson_last_epoch": _pearson(y, oof_last),
        "folds": folds,
        "minutes": (time.time() - started) / 60,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    (directory / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse E014 command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=E014Config.model_name)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--layer-decay", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=320)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Train one run per seed with the requested configuration."""
    args = parse_args(argv)
    train, test = build_frames(
        args.data_dir / "train.csv",
        args.data_dir / "test.csv",
        args.transcript_dir / "train_folds.csv",
        args.transcript_dir / "train.jsonl",
        args.transcript_dir / "test.jsonl",
    )
    print(f"train rows={len(train)} test rows={len(test)}", flush=True)
    for seed in args.seeds:
        config = E014Config(
            model_name=args.model_name,
            seed=seed,
            learning_rate=args.learning_rate,
            head_learning_rate=args.head_learning_rate,
            layer_decay=args.layer_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_length=args.max_length,
        )
        run(train, test, config, args.output_dir)


if __name__ == "__main__":
    main()
