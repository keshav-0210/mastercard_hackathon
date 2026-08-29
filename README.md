# Mastercard AI Defence Lab

Final Mastercard Innovation Challenge 2026 submission for an adaptive, synthetic-only red-team/blue-team fraud-defence loop.

## Submission Entry Point

Run [FINAL_SUBMISSION.ipynb](FINAL_SUBMISSION.ipynb) from top to bottom in a Kaggle GPU notebook. It performs the complete workflow:

1. Clones or updates this repository.
2. Installs the CTGAN and CUDA-enabled llama.cpp dependencies.
3. Verifies CUDA, llama.cpp GPU offload, and both Qwen2.5-7B GGUF shards.
4. Runs a five-round Qwen+CTGAN qualification gate.
5. Runs one independent 50-round experiment only when the gate passes.
6. Writes raw results, metrics, trend evidence, configuration, package versions, and logs.
7. Builds and verifies the HTML dashboard.
8. Creates a checksummed submission zip and downloadable dashboard copy.

The repository revision pulled by Kaggle must contain this notebook, `src/mastercard_defence/submission.py`, and the final dashboard generator before the run starts.

## Kaggle Prerequisites

- Enable a Kaggle GPU accelerator.
- Attach the private dataset containing both matching Qwen2.5-7B-Instruct Q4_K_M GGUF shards.
- Enable internet access so the notebook can clone the repository and install packages.
- No API key or runtime secret is required.

The notebook refuses to fall back to CPU or `HeuristicAgents` for Qwen. CTGAN uses Torch CUDA only when the installed Torch wheel supports the attached GPU architecture; otherwise CTGAN trains on CPU and records that choice in the run configuration. Paths and the repository URL can be overridden with `MASTERCARD_PROJECT_DIR`, `KAGGLE_INPUT_DIR`, `SUBMISSION_DIR`, and `MASTERCARD_REPOSITORY_URL`.

## Experiment Protocol

The closed loop is:

`Reviewed RAG + Attack Memory -> Agent1 -> Agent2 -> CTGAN -> Detector -> Agent3 -> Attack Memory -> Agent1`

All transactions are generated synthetically. Each round uses disjoint detector-training and current-family evaluation attacks. Longitudinal blue-team metrics use one fixed unseen benchmark containing all seven approved fraud families, reused unchanged across rounds. The detector operating threshold is calibrated on a separate legitimate-only holdout, not replay data or benchmark labels.

The detector retrains continually with representative replay and prior misses. The target synthetic fraud rate is 2%, within the required 1-3% range.

## Trend Gates

The five-round gate compares the first two and last two fixed-benchmark windows and requires:

- Precision, Recall, F1, and ROC-AUC to increase with positive linear slopes.
- False-positive rate to decrease with a negative linear slope.

The 50-round assessment compares the first and last ten rounds and applies the same blue-team directions. It also requires increasing attack fidelity, complete non-decreasing Family Coverage Diversity, and sustained final-window attack novelty. Failed runs retain raw diagnostic artifacts but cannot generate a submission-ready dashboard. Metrics are never rewritten to force a pass.

## Metrics

- **Precision, Recall, F1:** fixed seven-family unseen benchmark.
- **AUC-ROC / ROC-AUC:** calculated from fraud probability scores, not class predictions.
- **False-positive rate:** fixed benchmark; lower is better.
- **Attack Novelty:** structured distance from prior Attack Memory context.
- **Attack Fidelity:** behavioural plausibility against the synthetic reference distribution.
- **Family Coverage Diversity:** cumulative proportion of the seven approved fraud families explored.

Attack-channel diversity is not exported or displayed. Dashboard solid lines are stored per-round values; dashed lines are trailing three-round visualization averages.

## Outputs

Successful runs write timestamped artifacts under `/kaggle/working/mastercard_hackathon/artifacts/` and a downloadable bundle under `/kaggle/working/`:

- `adaptive_v2_results_<timestamp>.json`
- `adaptive_v2_metrics_dump_<timestamp>.json`
- `adaptive_v2_metrics_dump_<timestamp>.csv`
- `adaptive_v2_summary_<timestamp>.json`
- `adaptive_v2_config_<timestamp>.json`
- `adaptive_v2_package_versions_<timestamp>.json`
- `adaptive_v2_run_<timestamp>.log`
- `submission_dashboard_<timestamp>.html`
- `manifest_<timestamp>.json`
- `mastercard_submission_<timestamp>.zip`

The local dashboard source is [src/ui/generate_dashboard.py](src/ui/generate_dashboard.py). The generated local copy is [artifacts/submission_dashboard.html](artifacts/submission_dashboard.html).

## Local Validation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[tests]"
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe src\ui\generate_dashboard.py
```

Local tests do not claim to reproduce the Kaggle Qwen+CTGAN GPU run.

## Cleanup

The final notebook prints a cleanup dry run. Set `CONFIRM_SUBMISSION_CLEANUP=1` only after reviewing it. Cleanup is allowlisted to generated caches, temporary gate databases, and obsolete adaptive-v2 outputs; source, data, rules, the final notebook, README, current evidence, and dashboards are protected.

## Safety

Do not add real cardholder data, PII, production payment data, confidential material, or unauthorized sources. Red-team generation is offline and must not target live systems, payment infrastructure, or third parties.
