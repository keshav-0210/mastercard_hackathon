import importlib
import sys
import types

import pandas as pd
import pytest


class FakeCTGAN:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.device = None

    def set_device(self, device) -> None:
        self.device = device

    def set_random_state(self, seed: int) -> None:
        self.seed = seed

    def fit(self, data: pd.DataFrame, discrete_columns: list[str]) -> None:
        self.fit_rows = len(data)
        self.discrete_columns = discrete_columns


def load_generator_module(monkeypatch: pytest.MonkeyPatch, cuda_available: bool):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        set_device=lambda device: None,
    )
    fake_torch.device = lambda value: value
    fake_ctgan = types.ModuleType("ctgan")
    fake_ctgan.CTGAN = FakeCTGAN
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ctgan", fake_ctgan)
    monkeypatch.delitem(sys.modules, "mastercard_defence.learned_generator", raising=False)
    return importlib.import_module("mastercard_defence.learned_generator")


def test_explicit_cpu_overrides_visible_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    learned_generator = load_generator_module(monkeypatch, cuda_available=True)
    monkeypatch.setattr(learned_generator, "CTGAN", FakeCTGAN)
    generator = learned_generator.ConditionalCTGANGenerator(seed=7, epochs=1, cuda=False)
    training_data = learned_generator.build_training_corpus(seed=7, attack_size=10, reference_size=10)

    generator.fit(training_data)

    assert generator.model is not None
    assert generator.model.kwargs["cuda"] is False
    assert generator.model.device is None


def test_explicit_cuda_requires_torch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    learned_generator = load_generator_module(monkeypatch, cuda_available=False)

    with pytest.raises(RuntimeError, match="Torch CUDA is unavailable"):
        learned_generator.ConditionalCTGANGenerator(cuda=True)