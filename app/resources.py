from __future__ import annotations

import sys
from pathlib import Path


def app_base_dir() -> Path:
    """Return the runtime base directory for files bundled by PyInstaller.

    - Frozen app: use sys._MEIPASS (temporary extraction dir).
    - Source run: use project root (parent of app/).
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resolve_default_config_path() -> Path | None:
    base = app_base_dir()
    candidates = [
        base / "assets" / "config.yaml",
        base / "config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def list_embedded_checkpoints() -> list[Path]:
    base = app_base_dir()
    patterns = [
        base / "assets" / "checkpoints",
        base / "checkpoints",
    ]
    files: list[Path] = []
    for folder in patterns:
        if not folder.exists() or not folder.is_dir():
            continue
        files.extend(sorted(folder.glob("*.pt")))
        files.extend(sorted(folder.glob("*.pth")))

    # de-duplicate while preserving order
    dedup: list[Path] = []
    seen = set()
    for f in files:
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(f)
    return dedup


def resolve_default_checkpoint_path() -> Path | None:
    checkpoints = list_embedded_checkpoints()
    return checkpoints[0] if checkpoints else None
