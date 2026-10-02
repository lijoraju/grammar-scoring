"""Filesystem paths configured at import time without creating directories.

Set GRAMMAR_DATA_DIR and GRAMMAR_ARTIFACT_DIR before importing this module to
override the local defaults. Relative overrides retain their relative form.
"""

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = Path(os.environ.get("GRAMMAR_DATA_DIR", PROJECT_ROOT / "data/raw"))
ARTIFACT_DIR: Path = Path(
    os.environ.get("GRAMMAR_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")
)

TRANSCRIPTS_DIR: Path = ARTIFACT_DIR / "transcripts"
EMBEDDINGS_DIR: Path = ARTIFACT_DIR / "embeddings"
FEATURES_DIR: Path = ARTIFACT_DIR / "features"
OOF_DIR: Path = ARTIFACT_DIR / "oof"
MODELS_DIR: Path = ARTIFACT_DIR / "models"
