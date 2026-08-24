# Adaptive Experiment Archive

This folder contains the versioned adaptive red-team/blue-team experiment design and its reproducibility assets.

## Experiment

```text
QwenAgents + conditional CTGAN
3 seeds x 5 rounds
Kaggle GPU
```

Round 1 starts from the seeded approved taxonomy. After each evaluation, Agent 3 writes the detector weakness to Attack Memory. Agent 1 uses that weakness to recommend the next approved family. The controller validates the recommendation before Agent 2 creates the next structured specification.

## Taxonomy

See `taxonomy_v1.json`. The taxonomy is a controlled benchmark vocabulary, not a claim that it covers every possible payment attack. New combinations are recorded as candidates before promotion to a later taxonomy version.

## Expected Kaggle outputs

The full adaptive run creates these files under the Kaggle `artifacts/` directory:

```text
adaptive_qwen_ctgan_results_20260824T105805Z.json
adaptive_qwen_ctgan_graphs_20260824T105805Z.png
Mastercard_AI_Defence_Lab_Adaptive_Experiment_v1_20260824.pdf
```

Download those files from Kaggle and place them in this folder. The JSON contains round metrics and family decisions. The PNG contains red-team quality, blue-team response, and challenge-frontier graphs. The PDF contains the decision trace and aggregate metrics.

## Reproduction

Run the adaptive cells in `my.ipynb` after connecting the Kaggle GPU and mounting the two Qwen GGUF shards. The notebook installs CTGAN and CUDA-enabled llama.cpp, runs the adaptive smoke test, runs the full experiment, saves the JSON and graphs, and builds the PDF bundle.

All data is synthetic. No real cardholder, production payment, PII, confidential, or live-system data is used. Metrics are internal experiment indicators, not official Mastercard scores.
