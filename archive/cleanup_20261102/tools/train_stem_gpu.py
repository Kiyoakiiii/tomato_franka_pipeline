#!/usr/bin/env python3
"""Explicit GPU entry: run only during the user-coordinated shared GPU session."""
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_stem_cpu import main
from vision_runtime import local

if __name__ == "__main__":
    if "--name" in sys.argv and sys.argv.index("--name") + 1 < len(sys.argv):
        name = sys.argv[sys.argv.index("--name") + 1]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise SystemExit("Invalid run name")
        log = local(f".tmp/{name}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"GPU run console log: {log}", flush=True)
        with local(".tmp/vision_gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with log.open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                main(device="0")
        print(f"Run completed: experiments/vision/{name}/summary.json", flush=True)
    else:
        main(device="0")
