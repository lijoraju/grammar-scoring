import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from grammar_scoring.config import paths


@pytest.fixture
def load_paths(monkeypatch):
    monkeypatch.delenv("GRAMMAR_DATA_DIR", raising=False)
    monkeypatch.delenv("GRAMMAR_ARTIFACT_DIR", raising=False)

    def load(source: Path = Path(paths.__file__)) -> ModuleType:
        spec = importlib.util.spec_from_file_location("test_project_paths", source)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


def test_default_local_paths(load_paths, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    configured = load_paths()
    root = Path(__file__).resolve().parents[2]

    assert configured.PROJECT_ROOT == root
    assert configured.DATA_DIR == root / "data" / "raw"
    assert configured.ARTIFACT_DIR == root / "artifacts"


def test_data_dir_override(load_paths, monkeypatch, tmp_path):
    data_dir = tmp_path / "custom_data"
    monkeypatch.setenv("GRAMMAR_DATA_DIR", str(data_dir))
    configured = load_paths()

    assert configured.DATA_DIR == data_dir
    assert configured.ARTIFACT_DIR == configured.PROJECT_ROOT / "artifacts"


@pytest.mark.parametrize("override", [False, True])
def test_artifact_subdirectories(load_paths, monkeypatch, tmp_path, override):
    if override:
        monkeypatch.setenv("GRAMMAR_ARTIFACT_DIR", str(tmp_path / "outputs"))
    configured = load_paths()
    artifact_dir = (
        tmp_path / "outputs" if override else configured.PROJECT_ROOT / "artifacts"
    )

    assert configured.ARTIFACT_DIR == artifact_dir
    assert configured.DATA_DIR == configured.PROJECT_ROOT / "data" / "raw"
    for name in ("transcripts", "embeddings", "features", "oof", "models"):
        assert getattr(configured, f"{name.upper()}_DIR") == artifact_dir / name


def test_all_constants_use_pathlib(load_paths):
    configured = load_paths()

    for name in (
        "PROJECT_ROOT",
        "DATA_DIR",
        "ARTIFACT_DIR",
        "TRANSCRIPTS_DIR",
        "EMBEDDINGS_DIR",
        "FEATURES_DIR",
        "OOF_DIR",
        "MODELS_DIR",
    ):
        assert isinstance(getattr(configured, name), Path)


@pytest.mark.parametrize("override", [False, True])
def test_import_does_not_create_directories(
    load_paths, monkeypatch, tmp_path, override
):
    source = tmp_path / "src" / "grammar_scoring" / "config" / "paths.py"
    source.parent.mkdir(parents=True)
    source.write_text(Path(paths.__file__).read_text())
    if override:
        monkeypatch.setenv("GRAMMAR_DATA_DIR", str(tmp_path / "custom_data"))
        monkeypatch.setenv("GRAMMAR_ARTIFACT_DIR", str(tmp_path / "outputs"))

    configured = load_paths(source)

    assert configured.PROJECT_ROOT == tmp_path
    for directory in (
        configured.DATA_DIR,
        configured.ARTIFACT_DIR,
        configured.TRANSCRIPTS_DIR,
        configured.EMBEDDINGS_DIR,
        configured.FEATURES_DIR,
        configured.OOF_DIR,
        configured.MODELS_DIR,
    ):
        assert not directory.exists()
