#!/usr/bin/env python3
"""Bounded CPU development training. Does not import the robot/camera project code."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import re
import resource
import shutil
import sys
import time
import urllib.request
import zipfile

# Only this project's tools directory is added; no old src package is imported.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from vision_runtime import PROJECT, configure, digest, load_libraries, local


def pretrained(name):
    path = local("models/pretrained") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}"
        partial = path.with_suffix(".pt.partial")
        if partial.exists():
            partial = path.with_suffix(f".pt.partial.{time.time_ns()}")
        print(f"Downloading official weights: {url}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as output:
            expected_length = response.headers.get("Content-Length")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if partial.stat().st_size < 100_000:
            raise ValueError("Unexpectedly small pretrained weight file")
        if expected_length and partial.stat().st_size != int(expected_length):
            raise ValueError("Incomplete weight download; partial file preserved")
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip() is not None:
                raise ValueError("Weight archive checksum failure")
        partial.rename(path)
        path.with_suffix(".source.json").write_text(json.dumps({
            "url": url, "sha256": digest(path), "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "hash_note": "Locally recorded digest, not an independently published signature"
        }, indent=2) + "\n")
    provenance = json.loads(path.with_suffix(".source.json").read_text())
    if digest(path) != provenance["sha256"]:
        raise ValueError("Pretrained weight checksum changed")
    return path


def main(device="cpu"):
    parser = argparse.ArgumentParser(description=f"Bounded {'CPU' if device == 'cpu' else 'GPU 0'} development training; no camera or robot access.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--stage", choices=("smoke", "pilot", "train"), default="smoke")
    parser.add_argument("--model", choices=("yolo11n-seg.pt", "yolo11s-seg.pt", "yolo26n-seg.pt", "yolo26s-seg.pt"), default="yolo26s-seg.pt")
    parser.add_argument("--resource-profile", choices=("conservative", "shared80"), default="conservative",
                        help="shared80: user-authorized <=80%% CPU affinity, approximate GPU duty; GPU only")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minutes", type=float, default=15)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--nbs", type=int, choices=(2, 16, 32, 64), default=2)
    parser.add_argument("--optimizer", choices=("AdamW", "auto"), default="AdamW")
    parser.add_argument("--warmup-epochs", type=float, default=1)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", help="Preserved last.pt of an inactive local run; fork to this new run")
    args = parser.parse_args()
    expanded = args.resource_profile == "shared80"
    if args.resume_from and (not expanded or args.stage != "train"):
        parser.error("Checkpoint continuation requires shared80 and stage=train")
    if expanded and device == "cpu":
        parser.error("shared80 is an explicit GPU training profile")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("Name must contain only letters, digits, hyphens or underscores")
    if not (320 <= args.imgsz <= 960 and args.imgsz % 32 == 0
            and 1 <= args.epochs <= (200 if expanded else 100)
            and 0 < args.minutes <= (120 if expanded else 30)):
        parser.error("Resource guard: size 320..960; conservative <=100 epochs/30 min, shared80 <=200 epochs/120 min")
    if not (1 <= args.patience <= 80 and 0 < args.lr0 <= 0.01 and 0 < args.lrf <= 1):
        parser.error("Invalid optimizer/patience settings")
    if not expanded and args.batch > 4:
        parser.error("Batch >4 requires shared80")
    if (device == "cpu" and args.batch > 2) or args.nbs < args.batch:
        parser.error("CPU batch <=2; nbs must be >= physical batch")
    if not (0 <= args.warmup_bias_lr <= 0.1 and 1 <= args.warmup_epochs <= 5):
        parser.error("Warmup must be 1..5 epochs and bias LR 0..0.1")
    cpu_count = min(16, max(2, int(len(os.sched_getaffinity(0)) * 0.8))) if expanded else 2
    cores = configure(device, cpu_count=cpu_count, threads=min(4, cpu_count) if expanded else 2)
    allocator_fraction = 0.70 if expanded else 0.30
    active_fraction = 0.80 if expanded else 0.25
    run = local(f"experiments/vision/{args.name}")
    run.mkdir(parents=True, exist_ok=False)
    dataset = local("data/curated/seedlings_dev_v3")
    data = dataset / "data.yaml"
    with (dataset / "manifest.csv").open() as stream:
        samples = [r for r in csv.DictReader(stream) if r["status"] == "included"]
    # Detect accidental edits/missing labels before a long training run.
    for r in samples:
        if digest(local(r["derived_image"])) != r["image_file_sha256"]:
            raise ValueError(f"Derived image checksum changed: {r['sample_id']}")
        if digest(local(r["derived_label"])) != r["derived_label_sha256"]:
            raise ValueError(f"Derived label checksum changed: {r['sample_id']}")
    pixel_cache = dataset / "pixel_cache_manifest.json"
    cache_record = json.loads(pixel_cache.read_text()) if pixel_cache.exists() else None
    recorded_caches = {r["sample_id"]: r for r in cache_record["files"]} if cache_record else {}
    for r in samples:
        cache_path = local(r["derived_image"]).with_suffix(".npy")
        if cache_path.exists():
            recorded = recorded_caches.get(r["sample_id"])
            if (not recorded or recorded["png_sha256"] != r["image_file_sha256"]
                    or digest(local(cache_path)) != recorded["cache_sha256"]):
                raise ValueError(f"Unverified decoded-pixel cache: {cache_path}")
    if args.stage == "smoke":
        for split, count in (("train", 8), ("val", max(2, args.batch * 2))):
            subset = [r for r in samples if r["split"] == split][:count]
            (run / f"{split}.txt").write_text("".join(str(local(r["derived_image"]))+"\n" for r in subset))
        data = run / "data.yaml"
        data.write_text(f"path: {json.dumps(str(run))}\ntrain: train.txt\nval: val.txt\nnames:\n  0: stem\n")
        args.epochs = 1
    weight = pretrained(args.model)
    YOLO = load_libraries()
    import torch
    gpu_info = None
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime unavailable; no automatic fallback")
        torch.cuda.set_per_process_memory_fraction(allocator_fraction, 0)
        free, total = torch.cuda.mem_get_info(0)
        if free < 8 * 1024**3:
            raise RuntimeError("Less than 8 GiB GPU memory free; declining this new training run")
        # A real kernel is required: successful driver initialization alone is not enough.
        x = torch.arange(256, device="cuda", dtype=torch.float32)
        assert float((x*x).sum().cpu()) == float(sum(i*i for i in range(256)))
        del x
        gpu_info = {"name": torch.cuda.get_device_name(0), "free_bytes_at_start": free,
                    "total_bytes": total, "allocator_fraction_limit": allocator_fraction,
                    "training_target_active_fraction": active_fraction,
                    "note": "Allocator cap excludes driver/context memory. Batch pauses approximate duty; not a hardware utilization quota."}
        # Make the official AMP-check weight provenance explicit and keep it in models/pretrained.
        pretrained("yolo26n.pt")
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (run / "environment.freeze.txt").write_text("\n".join(packages)+"\n")
    config = dict(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                  device=device, workers=0, cache=False, project=str(run), name="train",
                  exist_ok=False, seed=args.seed, deterministic=True, amp=device != "cpu", plots=True,
                  save=True, val=True, patience=args.patience, mask_ratio=2, overlap_mask=True,
                  optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf, weight_decay=0.0005,
                  warmup_epochs=args.warmup_epochs, warmup_bias_lr=args.warmup_bias_lr,
                  nbs=args.nbs, cos_lr=True,
                  mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                  degrees=5.0, translate=0.05, scale=0.15, shear=0.0, perspective=0.0,
                  fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.25, hsv_v=0.2,
                  close_mosaic=0)
    resume_info = None
    model_weight = weight
    if args.resume_from:
        from vision_status import task_processes
        checkpoint_path = local(args.resume_from)
        source_run = checkpoint_path.parents[2]
        if source_run.name in task_processes():
            raise RuntimeError("Refusing to copy checkpoints from an active training run")
        source_metadata = json.loads((source_run / "run_metadata.json").read_text())
        if (source_metadata["dataset_manifest_sha256"] != digest(dataset / "manifest.csv")
                or source_metadata["model_sha256"] != digest(weight)):
            raise ValueError("Resume provenance does not match current dataset and pretrained model")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved = checkpoint["train_args"]
        for key, value in config.items():
            if key not in ("project", "name", "exist_ok") and saved.get(key) != value:
                raise ValueError(f"Resume would change training setting {key}: {saved.get(key)} -> {value}")
        if checkpoint.get("optimizer") is None or checkpoint.get("ema") is None:
            raise ValueError("Checkpoint lacks optimizer/EMA state; not a resumable last.pt")
        completed = checkpoint["epoch"] + 1
        if not 0 < completed < args.epochs:
            raise ValueError("Checkpoint has no remaining epochs")
        source_csv = source_run / "train/results.csv"
        prior_rows = list(csv.DictReader(source_csv.open()))
        if int(prior_rows[-1]["epoch"]) != completed:
            raise ValueError("Checkpoint and logged complete epochs disagree")
        destination = run / "train"
        (destination / "weights").mkdir(parents=True)
        shutil.copyfile(source_csv, destination / "results.csv")
        shutil.copyfile(source_run / "train/weights/best.pt", destination / "weights/best.pt")
        previous_best = torch.load(source_run / "train/weights/best.pt", map_location="cpu", weights_only=False)
        previous_best_epoch = previous_best["epoch"] + 1
        previous_best_fitness = float(checkpoint["best_fitness"])
        del previous_best
        saved.update(project=str(run), name="train", save_dir=str(destination))
        model_weight = run / "resume_source.pt"
        torch.save(checkpoint, model_weight)
        del checkpoint
        config["resume"] = str(model_weight)
        resume_info = {"source_checkpoint": str(checkpoint_path), "source_sha256": digest(checkpoint_path),
                       "forked_checkpoint_sha256": digest(model_weight), "completed_epochs_before_resume": completed,
                       "prior_logged_elapsed_s": float(prior_rows[-1]["time"]),
                       "previous_best_epoch": previous_best_epoch,
                       "previous_best_fitness": previous_best_fitness,
                       "optimizer_and_ema_restored": True,
                       "note": "Original interrupted run preserved. RNG state is not fully checkpointed by Ultralytics; not bitwise continuation. CSV time restarts for the resumed phase."}
    metadata = {"purpose": f"{device} pipeline smoke test" if args.stage == "smoke" else f"bounded {device} development training",
                "not_independent_test": True, "stage": args.stage, "model": str(weight),
                "model_sha256": digest(weight), "dataset_manifest_sha256": digest(dataset / "manifest.csv"),
                "cpu_affinity": cores, "compute_device": device, "gpu_info": gpu_info, "training_minutes_limit": args.minutes,
                "resource_profile": args.resource_profile,
                "pixel_cache_manifest_sha256": digest(pixel_cache) if cache_record else None,
                "resume_info": resume_info,
                "resource_authorization": "User explicitly authorized up to 80% CPU/GPU on 2026-10-01" if expanded else "Prior modest-use authorization",
                "budget_note": "Training stops at a batch boundary; validation/checkpoint saving adds time.",
                "started_utc": datetime.now(timezone.utc).isoformat(), "config": config}
    snapshot = run / "tool_snapshot"
    snapshot.mkdir()
    for tool in ("train_stem_cpu.py", "train_stem_gpu.py", "vision_runtime.py"):
        (snapshot / tool).write_bytes(local(f"tools/{tool}").read_bytes())
    metadata["tool_sha256"] = {p.name: digest(p) for p in snapshot.iterdir()}
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    model = YOLO(str(model_weight))
    started = time.monotonic()
    train_started = [None]
    budget_reached = [False]
    low_gpu_memory = [False]
    low_host_memory = [False]
    batch_started = [None]
    val_batch_started = [None]
    resource_log = (run / "resources.csv").open("x", newline="")
    writer = csv.writer(resource_log)
    writer.writerow(("elapsed_s", "epoch", "peak_rss_mib", "torch_threads", "cpu_affinity",
                     "gpu_peak_reserved_mib", "gpu_free_mib"))

    def on_train_start(trainer):
        train_started[0] = time.monotonic()
        if resume_info:
            trainer.stopper.best_epoch = resume_info["previous_best_epoch"]
            trainer.stopper.best_fitness = resume_info["previous_best_fitness"]
        resolved = {"optimizer": type(trainer.optimizer).__name__,
                    "physical_batch": trainer.batch_size, "target_accumulation_steps": trainer.accumulate,
                    "warmup_epochs": trainer.args.warmup_epochs,
                    "warmup_bias_lr": trainer.args.warmup_bias_lr,
                    "parameter_groups": [{key: group.get(key) for key in
                                          ("param_group", "initial_lr", "lr", "weight_decay", "betas")}
                                         for group in trainer.optimizer.param_groups]}
        (run / "resolved_training_settings.json").write_text(json.dumps(resolved, indent=2)+"\n")

    def on_batch_end(trainer):
        if device != "cpu":
            torch.cuda.synchronize()
            # Voluntarily leave idle time between this process's batches; no priority changes.
            elapsed = time.monotonic() - batch_started[0]
            pause = elapsed * (1 / active_fraction - 1)
            time.sleep(pause if expanded else min(pause, 3.0))
            free, total = torch.cuda.mem_get_info(0)
            if free < (total * 0.20 if expanded else 4 * 1024**3):
                trainer.stop = True
                low_gpu_memory[0] = True
        if expanded:
            mem = dict((line.split(':')[0], int(line.split()[1]))
                       for line in __import__('pathlib').Path('/proc/meminfo').read_text().splitlines())
            if mem['MemAvailable'] < mem['MemTotal'] * 0.20:
                trainer.stop = True
                low_host_memory[0] = True
        if time.monotonic() - train_started[0] >= args.minutes * 60:
            trainer.stop = True
            budget_reached[0] = True

    def on_epoch_end(trainer):
        writer.writerow((round(time.monotonic()-started, 2), trainer.epoch + 1,
                         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                         torch.get_num_threads(), sorted(os.sched_getaffinity(0)),
                         torch.cuda.max_memory_reserved(0)/1024**2 if device != "cpu" else 0,
                         torch.cuda.mem_get_info(0)[0]/1024**2 if device != "cpu" else 0))
        resource_log.flush()

    def on_batch_start(trainer):
        batch_started[0] = time.monotonic()

    def on_val_batch_end(validator):
        if device != "cpu":
            torch.cuda.synchronize()
            elapsed = time.monotonic() - val_batch_started[0]
            time.sleep(elapsed * (1 / active_fraction - 1) if expanded else 0.15)

    def on_val_batch_start(validator):
        val_batch_started[0] = time.monotonic()

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_start", on_batch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_fit_epoch_end", on_epoch_end)
    model.add_callback("on_val_batch_start", on_val_batch_start)
    model.add_callback("on_val_batch_end", on_val_batch_end)
    try:
        metrics = model.train(**config)
        result = {"elapsed_s": time.monotonic()-started, "budget_reached": budget_reached[0],
                  "low_gpu_memory_stop": low_gpu_memory[0],
                  "low_host_memory_stop": low_host_memory[0],
                  "epochs_logged": model.trainer.epoch+1, "last_epoch_may_be_partial": budget_reached[0] or low_gpu_memory[0] or low_host_memory[0],
                  "resume_info": resume_info,
                  "epochs_completed_this_invocation": model.trainer.epoch+1 - (resume_info["completed_epochs_before_resume"] if resume_info else 0),
                  "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(0)/1024**2 if device != "cpu" else 0,
                  "best_weights": str(model.trainer.best), "last_weights": str(model.trainer.last),
                  "development_metrics": {k: float(v) for k, v in metrics.results_dict.items()},
                  "not_independent_test": True}
        (run / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        resource_log.close()


if __name__ == "__main__":
    main()
