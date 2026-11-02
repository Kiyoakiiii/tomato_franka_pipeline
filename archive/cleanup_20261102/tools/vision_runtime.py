"""Project-local runtime shared by the new offline vision tools; CPU is the default."""
import hashlib
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]


def local(value):
    path = PROJECT / value
    path.resolve().relative_to(PROJECT)
    for parent in (path, *path.parents):
        if parent == PROJECT:
            break
        if parent.is_symlink():
            raise ValueError(f"Refusing a symlink path: {parent}")
    return path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure(device="cpu", cpu_count=2, threads=2):
    """Constrain only this new process, before importing any compute libraries."""
    if device not in ("cpu", "0"):
        raise ValueError("Only CPU or explicitly selected GPU 0 is supported")
    available = sorted(os.sched_getaffinity(0))
    if not (1 <= threads <= cpu_count <= max(2, int(len(available) * 0.8))):
        raise ValueError("CPU allocation must leave at least 20% of available CPUs unused")
    environment = ".venvs/vision-cpu" if device == "cpu" else ".venvs/vision-gpu"
    if Path(sys.prefix).resolve() != local(environment).resolve():
        raise RuntimeError(f"Run with {environment}/bin/python -B")
    os.chdir(PROJECT)
    for key, value in {
        "TMPDIR": ".tmp", "XDG_CACHE_HOME": ".cache", "XDG_CONFIG_HOME": ".config",
        "PIP_CACHE_DIR": ".cache/pip", "TORCH_HOME": ".cache/torch",
        "MPLCONFIGDIR": ".cache/matplotlib", "HF_HOME": ".cache/huggingface",
        "YOLO_CONFIG_DIR": ".config/ultralytics",
        "CUDA_CACHE_PATH": ".cache/cuda", "TRITON_CACHE_DIR": ".cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor", "TORCH_EXTENSIONS_DIR": ".cache/torch_extensions",
    }.items():
        path = local(value)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(CUDA_VISIBLE_DEVICES="" if device == "cpu" else "0",
                      PYTORCH_NVML_BASED_CUDA_CHECK="0", CUDA_MODULE_LOADING="LAZY",
                      OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                      OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_MAX_THREADS=str(threads), POLARS_MAX_THREADS=str(threads),
                      PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", WANDB_MODE="disabled",
                      YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", MPLBACKEND="Agg")
    sys.dont_write_bytecode = True
    os.sched_setaffinity(0, available[:cpu_count])
    return available[:cpu_count]


def load_libraries():
    import torch
    import cv2
    from ultralytics import YOLO, settings
    import ultralytics.utils
    import ultralytics.utils.torch_utils
    # select_device resets torch threads from this module constant.
    threads = int(os.environ["OMP_NUM_THREADS"])
    ultralytics.utils.NUM_THREADS = threads
    ultralytics.utils.torch_utils.NUM_THREADS = threads
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    disabled = {key: False for key in ("sync", "clearml", "comet", "dvc", "hub", "mlflow",
                                      "neptune", "raytune", "tensorboard", "wandb", "vscode_msg", "openvino_msg")}
    settings.update({**disabled, "datasets_dir": str(local("data/curated")),
                     "weights_dir": str(local("models/pretrained")),
                     "runs_dir": str(local("experiments/vision"))})
    return YOLO
