import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from grammar_scoring.data import dataset
from grammar_scoring.evaluation import make_cv_folds

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/create_folds.py"


def test_create_folds_cli(tmp_path, monkeypatch, capsys):
    train_path = tmp_path / "train.csv"
    train = pd.DataFrame(
        {"filename": [f"{i}.wav" for i in range(30)], "label": np.arange(30) / 6}
    )
    train.to_csv(train_path, index=False)
    original = train_path.read_bytes()
    monkeypatch.setattr(dataset, "TRAIN_CSV", train_path)
    main = runpy.run_path(str(SCRIPT))["main"]
    output = tmp_path / "features" / "folds.csv"
    assert main(["--output", str(output), "--random-state", "17"]) == 0
    saved = pd.read_csv(output)
    assert list(saved.columns) == ["filename", "label", "fold"]
    np.testing.assert_array_equal(
        saved["fold"], make_cv_folds(train.label, random_state=17)
    )
    assert train_path.read_bytes() == original
    assert "target_mean" in capsys.readouterr().out


def test_create_folds_cli_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dataset, "TRAIN_CSV", tmp_path / "missing.csv")
    main = runpy.run_path(str(SCRIPT))["main"]
    output = tmp_path / "folds.csv"
    assert main(["--output", str(output)]) == 1
    assert not output.exists()
    assert "Fold creation failed" in capsys.readouterr().err


def test_create_folds_rejects_overwriting_train(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    train_path.write_text("filename,label\na.wav,1\n")
    original = train_path.read_bytes()
    monkeypatch.setattr(dataset, "TRAIN_CSV", train_path)
    main = runpy.run_path(str(SCRIPT))["main"]
    assert main(["--output", str(train_path)]) == 1
    assert train_path.read_bytes() == original
