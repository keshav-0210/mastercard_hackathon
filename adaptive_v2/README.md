# Adaptive v2

This folder is the versioned Kaggle execution entry point for the adaptive intelligence framework.

## Execution

Open `adaptive/adaptive.ipynb` in Kaggle with GPU enabled and run the cells in order. The notebook pulls the repository, verifies CUDA, runs QwenAgents with conditional CTGAN for one seed and 50 rounds, and writes these outputs under `/kaggle/working/mastercard_hackathon/artifacts/`:

- `adaptive_v2_run_<timestamp>.log`
- `adaptive_v2_results_<timestamp>.json`
- `adaptive_v2_memory_<timestamp>.sqlite`

The implementation lives in `src/mastercard_defence/`. Each round selects and evaluates one of the seven approved families. Agent 3 weakness families and confidence influence Agent 1 sampling in the next round.
