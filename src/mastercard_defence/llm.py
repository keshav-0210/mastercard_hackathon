from __future__ import annotations

import json
import os
from typing import Any

from .runtime import require_model


class SharedLocalLLM:
    """One lazily loaded GGUF model shared by all logical agent roles."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._engine = None

    def _load(self):
        if self._engine is None:
            path = require_model(self.config)
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError("llama-cpp-python is required to use the local GGUF model.") from exc
            configured_layers = os.getenv("GPU_LAYERS")
            gpu_layers = int(configured_layers) if configured_layers is not None else (-1 if os.getenv("RUN_MODE", "LOCAL").upper() == "KAGGLE_GPU" else self.config["model"].get("gpu_layers", 0))
            self._engine = Llama(model_path=str(path), n_ctx=self.config["model"]["context_size"], n_gpu_layers=gpu_layers, verbose=False)
        return self._engine

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self._load().create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1200,
        )
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)
