from __future__ import annotations

import json
import os
import re
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
        messages = [
            {"role": "system", "content": system_prompt + " Output one compact JSON object only. Do not use markdown."},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(3):
            response = self._load().create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=int(self.config.get("model", {}).get("max_output_tokens", 256)),
            )
            content = response["choices"][0]["message"]["content"]
            try:
                return self._parse_json(content)
            except ValueError:
                if attempt == 2:
                    raise
                messages.append({"role": "user", "content": "Your previous response was invalid JSON. Return a shorter valid JSON object with no commentary."})
        raise RuntimeError("JSON generation failed unexpectedly.")

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("The local model response did not contain a JSON object.")

        candidate = cleaned[start:]
        in_string = False
        escape = False
        depth = 0
        end_index = None
        for index, char in enumerate(candidate):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end_index = index + 1
                        break
        if end_index is None:
            raise ValueError("The local model response did not contain a complete JSON object.")

        candidate = candidate[:end_index]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"The local model returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("The local model response must be a JSON object.")
        return parsed
