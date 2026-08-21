from __future__ import annotations

from mastercard_defence.runtime import require_cuda_for_heavy_workload

print("GPU workload: neural generator training or large synthetic generation")
print("Expected requirement: CUDA-enabled Kaggle GPU and sufficient VRAM for the selected workload")
require_cuda_for_heavy_workload()
print("GPU check passed; expensive execution may proceed only after this point.")
