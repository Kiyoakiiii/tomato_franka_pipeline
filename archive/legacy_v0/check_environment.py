from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from src.paths import add_lab_code_import_paths


PROJECT_DIR = Path(__file__).resolve().parent
LAB_CODE_DIR = PROJECT_DIR.parent


def add_local_paths() -> None:
    add_lab_code_import_paths()


def check_import(name: str, pip_hint: str | None = None, required: bool = True) -> bool:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "")
        suffix = f" {version}" if version else ""
        print(f"[OK]   import {name}{suffix}")
        return True
    except Exception as exc:
        level = "MISS" if required else "WARN"
        hint = f" | install/check: {pip_hint}" if pip_hint else ""
        print(f"[{level}] import {name} failed: {exc}{hint}")
        return False


def main() -> int:
    add_local_paths()
    print("Tomato Franka pipeline environment check")
    print(f"Project: {PROJECT_DIR}")
    print(f"Python:  {sys.executable}")
    print(f"Version: {platform.python_version()} ({platform.system()} {platform.machine()})")
    print()

    ok = True
    ok &= check_import("numpy", "python3 -m pip install numpy")
    ok &= check_import("cv2", "python3 -m pip install opencv-python")
    ok &= check_import("ultralytics", "python3 -m pip install ultralytics")
    check_import("torch", "installed automatically with ultralytics", required=False)
    check_import("serial", "python3 -m pip install pyserial", required=False)
    check_import("pyzed.sl", "Install Stereolabs ZED SDK; not a pip package", required=False)

    franky_path = LAB_CODE_DIR / "franky" / "franky"
    print()
    if franky_path.exists():
        print(f"[OK]   local franky path exists: {franky_path}")
    else:
        print(f"[MISS] local franky path missing: {franky_path}")
        ok = False

    # Franky is required only for robot scripts. It may fail on non-lab Windows/Python versions.
    franky_ok = check_import("franky", "Use lab computer or install franky-panda", required=False)
    if not franky_ok:
        print("       Robot scripts will not run until franky imports successfully.")

    model_path = PROJECT_DIR / "models" / "stem_yolo11n_v3_clean_one_stem.pt"
    config_path = PROJECT_DIR / "config.json"
    print()
    if model_path.exists():
        print(f"[OK]   model exists: {model_path}")
    else:
        print(f"[MISS] model missing: {model_path}")
        ok = False
    if config_path.exists():
        print(f"[OK]   config exists: {config_path}")
    else:
        print(f"[MISS] config missing: {config_path}")
        ok = False

    print()
    if ok:
        print("Basic files and core Python packages look ready.")
        print("If pyzed is missing, vision can still be tested with --source webcam or --image, but not ZED.")
        print("If franky is missing, vision scripts still run, but robot scripts will not.")
        return 0

    print("Some required items are missing. Install requirements or fix copied paths before running robot tests.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

