# Adaptive v2

This folder is the versioned Kaggle execution entry point for the adaptive intelligence framework.

## Execution

Open `adaptive/adaptive.ipynb` in Kaggle with GPU enabled and run the cells in order. The notebook pulls the repository, verifies CUDA, runs QwenAgents with conditional CTGAN for one seed and 50 rounds, and writes these outputs under `/kaggle/working/mastercard_hackathon/artifacts/`:

- `adaptive_v2_run_<timestamp>.log`
- `adaptive_v2_results_<timestamp>.json`
- `adaptive_v2_memory_<timestamp>.sqlite`

Charts are written under `adaptive/charts/<family>/<metric>.png`.

The implementation lives in `src/mastercard_defence/`. Agent 3 weakness families and confidence influence Agent 1 sampling, while all-family inference probes are kept separate from detector training.
