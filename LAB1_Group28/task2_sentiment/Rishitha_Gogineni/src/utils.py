import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def project_root():
    return Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dirs(root):
    folders = [
        "checkpoints",
        "logs/raw",
        "manifest",
        "outputs/plots",
        "outputs/predictions",
        "outputs/error_analysis",
    ]

    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)


def new_run_id():
    return time.strftime("%Y%m%d_%H%M%S")


def append_log(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + os.linesep)


def write_environment(root):
    lines = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"cpu={platform.processor()}",
        f"pytorch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"pytorch_cuda={torch.version.cuda}",
    ]

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines.extend([
            f"gpu={torch.cuda.get_device_name(0)}",
            f"gpu_memory_gb={props.total_memory / 1024**3:.2f}",
        ])

    path = root / "manifest" / "environment.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
