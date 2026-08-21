from __future__ import annotations

import os
from pathlib import Path


def run_mode() -> str:
    return os.getenv("RUN_MODE", "LOCAL").upper()


def model_path(config: dict) -> Path:
    configured = os.getenv("MODEL_PATH") or os.getenv("KAGGLE_MODEL_PATH")
    if configured:
        return Path(configured)
    if run_mode() == "KAGGLE_GPU":
        candidates = sorted(Path("/kaggle/input").rglob("*-00001-of-00002.gguf"))
        if candidates:
            return candidates[0]
    return Path(config["model"]["path"])


def require_model(config: dict) -> Path:
    path = model_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Local model is missing: {path}. Download it once into the configured models directory.")
    if path.stat().st_size > 7 * 1024**3:
        raise ValueError(f"Model shard exceeds the 7 GB project limit: {path.stat().st_size / 1024**3:.2f} GB")
    shard_count = config["model"].get("shard_count", 1)
    for shard_number in range(1, shard_count + 1):
        shard = path.with_name(path.name.replace("00001-of-00002", f"{shard_number:05d}-of-{shard_count:05d}"))
        if not shard.exists():
            raise FileNotFoundError(f"Model shard is missing: {shard}")
    return path


def require_cuda_for_heavy_workload() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution is required. Please start/enable the Kaggle GPU session and confirm when it is ready.")
    device = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"CUDA device: {device}; total memory: {memory_gb:.2f} GB")
