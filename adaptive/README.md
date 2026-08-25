# Adaptive Experiment Archive

This folder contains the versioned adaptive red-team/blue-team experiment design and its reproducibility assets.

## Experiment

```text
QwenAgents + conditional CTGAN
3 seeds x 50 rounds minimum
Kaggle GPU
```

Round 1 starts from the seeded seven-family taxonomy. After each evaluation, Agent 3 writes the weak family and confidence to Attack Memory. Agent 1 uses those values as sampling weights when selecting one of the same seven approved families for the next round. The controller validates the recommendation before Agent 2 creates the next structured specification.

## Taxonomy

See `taxonomy_v1.json`. The taxonomy is a controlled benchmark vocabulary, not a claim that it covers every possible payment attack.

## Expected Kaggle outputs

The full adaptive run creates a JSON result file and a SQLite Attack Memory database under the Kaggle `artifacts/` directory. The JSON contains round metrics and family decisions.

## Reproduction

Run the adaptive cells in `adaptive.ipynb` after connecting the Kaggle GPU and mounting the two Qwen GGUF shards. The notebook installs CTGAN and CUDA-enabled llama.cpp, runs the adaptive smoke test, and saves the JSON result.

All data is synthetic. No real cardholder, production payment, PII, confidential, or live-system data is used. Metrics are internal experiment indicators, not official Mastercard scores.
