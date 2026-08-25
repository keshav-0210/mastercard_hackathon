# Mastercard AI Defence Lab

First-cut closed-loop red-team/blue-team payment-security prototype.

## Safety and data policy

The first iteration uses synthetic transactions only. Do not add real cardholder data, PII, production payment data, confidential material, or data from unauthorized sources. Red-team generation is offline experimentation and must not target live systems, payment infrastructure, or third parties.

## Local quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
streamlit run app.py
```

The first local demo uses deterministic agent fallbacks so the pipeline can be tested before the local model is installed. The shared Qwen GGUF model is configured in `config/default.yaml`; the current Q4_K_M distribution is two matching GGUF shards, and both must remain in `models/` before the model loader is used.

## Kaggle GPU

The local VS Code checkout is the source of truth, but code that uses the Kaggle model or Tesla T4 must execute in the Kaggle kernel. In VS Code, connect the notebook to the active Kaggle Jupyter server, clone or sync this repository into `/kaggle/working`, and run:

```python
%run scripts/kaggle_bootstrap.py
from mastercard_defence.runtime import require_cuda_for_heavy_workload
require_cuda_for_heavy_workload()
```

The bootstrap discovers both GGUF shards under `/kaggle/input`, sets `RUN_MODE=KAGGLE_GPU`, and sets `MODEL_PATH` without downloading anything. `llama-cpp-python` then uses `n_gpu_layers=-1` automatically in Kaggle mode. GPU-dependent commands must check CUDA availability and stop if the Kaggle GPU session is not enabled. Heavy neural generator work belongs in Kaggle; local execution remains for orchestration and lightweight tests.

### VS Code-only execution workflow

1. Keep editing `.py` files in this workspace.
2. Push the repository to GitHub or sync the changed files into `/kaggle/working` from the connected Kaggle kernel.
3. Attach the private model Dataset and enable the Kaggle GPU accelerator.
4. In the VS Code notebook connected to the Kaggle kernel, run the bootstrap cell above.
5. Execute project files with `%run scripts/...py` or import modules from `src`.

Running a local Python process cannot directly access `/kaggle/input` or the Tesla T4. A remote Kaggle kernel connection is therefore required while keeping VS Code as the editor and execution interface.

## Closed-loop path

`Reviewed public RAG + Attack Memory -> Agent 1 -> Agent 2 -> Generator -> Fidelity + Diversity + Detector -> Agent 3 -> Attack Memory -> Agent 1`

The configured protocol selects and evaluates one of seven approved families per round and evaluates unseen attack rows against a legitimate holdout. Detector-training attacks are separate from evaluation attacks, and the fidelity report includes amount moments, behavioural signal deltas, and channel coverage. Agent 1 receives a round-specific query built from the previous Attack Memory weakness/recommendation, and the allowlisted public summaries are retrieved again for that query on every round. Each round also records an internal novelty score based on structured similarity to prior hypotheses. The detector split targets a two-percent synthetic fraud rate, within the requested one-to-three-percent range.
