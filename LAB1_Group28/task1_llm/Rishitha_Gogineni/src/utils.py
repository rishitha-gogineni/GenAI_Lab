from pathlib import Path
import csv
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root():
    return Path(__file__).resolve().parents[1]


def make_dirs(root):
    names = [
        "data_processed", "checkpoints", "outputs/plots",
        "outputs/generations", "outputs/failure_cases",
        "logs/raw", "manifest"
    ]
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)


def write_environment(root):
    versions = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda": getattr(torch.version, "cuda", None),
    }
    try:
        import datasets
        versions["datasets"] = datasets.__version__
    except ImportError:
        versions["datasets"] = "not installed"
    try:
        import matplotlib
        versions["matplotlib"] = matplotlib.__version__
    except ImportError:
        versions["matplotlib"] = "not installed"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        versions["gpu"] = torch.cuda.get_device_name(0)
        versions["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)

    lines = [f"{k}={v}" for k, v in versions.items()]
    (root / "manifest" / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True
        )
        (root / "manifest" / "pip_freeze.txt").write_text(result.stdout, encoding="utf-8")
    except Exception:
        pass


def new_run_id():
    return time.strftime("%Y%m%d_%H%M%S")


def new_log_path(root, run_id):
    return root / "logs" / "raw" / f"train_{run_id}.log"


def append_log(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + os.linesep)


def append_checkpoint_manifest(root, rows):
    path = root / "manifest" / "checkpoint_manifest.csv"
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["run_id", "checkpoint", "result", "validation_loss", "source_log"])
        writer.writerows(rows)
