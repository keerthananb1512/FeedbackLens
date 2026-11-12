from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def environment():
    import torch

    packages = {}
    for name in (
        "torch",
        "transformers",
        "unsloth",
        "unsloth_zoo",
        "peft",
        "datasets",
        "accelerate",
        "triton",
        "flash-linear-attention",
        "causal-conv1d",
        "streamlit",
        "numpy",
        "scikit-learn",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        result["gpu"] = {
            "name": p.name,
            "total_memory_bytes": p.total_memory,
            "visible_device": str(torch.cuda.current_device()),
        }
        result["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu",
                "--format=csv",
            ],
            text=True,
        )
        result["gpu_processes"] = gpu_processes()
    return result


def gpu_processes():
    return (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
