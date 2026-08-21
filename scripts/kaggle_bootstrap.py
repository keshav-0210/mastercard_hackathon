from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_kaggle_runtime() -> Path:
    repo_root = Path.cwd()
    source_root = repo_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    model_files = sorted(Path("/kaggle/input").rglob("*.gguf"))
    if len(model_files) != 2:
        raise FileNotFoundError(f"Expected two GGUF shards under /kaggle/input, found: {model_files}")
    first_shard = next(path for path in model_files if "00001-of-00002" in path.name)
    os.environ["RUN_MODE"] = "KAGGLE_GPU"
    os.environ["MODEL_PATH"] = str(first_shard)
    print(f"Repository: {repo_root}")
    print(f"Model shards: {model_files}")
    print(f"Model path: {first_shard}")
    return first_shard


if __name__ == "__main__":
    configure_kaggle_runtime()