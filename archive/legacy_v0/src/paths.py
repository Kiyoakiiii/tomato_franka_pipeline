from __future__ import annotations

import ctypes
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
LAB_CODE_DIR = PROJECT_DIR.parent


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def add_lab_code_import_paths() -> None:
    """Make copied lab-code packages importable from this standalone folder."""
    vendor_fmt8 = (
        PROJECT_DIR
        / "vendor"
        / "franky_021"
        / "libfmt8"
        / "usr"
        / "lib"
        / "x86_64-linux-gnu"
        / "libfmt.so.8"
    )
    if vendor_fmt8.exists():
        ctypes.CDLL(str(vendor_fmt8), mode=ctypes.RTLD_GLOBAL)

    candidates = [
        PROJECT_DIR / "vendor" / "franky_021",
        LAB_CODE_DIR / "franky" / "franky",
        LAB_CODE_DIR / "Pin Brake" / "Pin Brake",
    ]
    for candidate in reversed(candidates):
        if candidate.exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)

