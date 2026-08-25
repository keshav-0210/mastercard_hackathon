from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mastercard_defence.loop import ClosedLoop, load_config


def main() -> None:
    config = load_config(str(ROOT / "config" / "default.yaml"))
    config["paths"]["memory_db"] = str(ROOT / "artifacts" / "adaptive_v2_memory.sqlite")
    config["generator_backend"] = "ctgan"
    config["detector_mode"] = "static"
    loop = ClosedLoop(config)
    try:
        loop.run_robustness_suite(seeds=1, rounds=max(50, config["pipeline"]["rounds"]))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
