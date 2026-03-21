"""Tests for Hugging Face token propagation in model loaders."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_detector.compat as compat  # noqa: E402


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    """Ensure env vars and caches don't leak across tests."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    # Reset cached models between tests
    compat._PPL_MODEL = None
    compat._PPL_TOKENIZER = None
    compat._PPL_MODEL_ID = None
    compat._BINO_MODEL = None
    compat._BINO_TOKENIZER = None
    yield
    compat._PPL_MODEL = None
    compat._PPL_TOKENIZER = None
    compat._PPL_MODEL_ID = None
    compat._BINO_MODEL = None
    compat._BINO_TOKENIZER = None


def test_get_hf_token_prefers_HF_TOKEN(monkeypatch):
    print("\n-- HF TOKEN LOOKUP --")
    monkeypatch.setenv("HF_TOKEN", "primary_token")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "backup_token")
    token = compat._get_hf_token()
    assert token == "primary_token", token


def test_get_hf_token_falls_back(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "backup_token")
    token = compat._get_hf_token()
    assert token == "backup_token", token


def test_perplexity_loader_passes_token(monkeypatch):
    print("\n-- PPL LOADER TOKEN PROPAGATION --")
    monkeypatch.setenv("HF_TOKEN", "abc123")
    monkeypatch.setattr(compat, "HAS_PERPLEXITY", True)

    class DummyAutoModel:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.last_kwargs = kwargs
            return cls()

        def eval(self):
            return None

    class DummyAutoTokenizer:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.last_kwargs = kwargs
            return cls()

    dummy_mod = types.ModuleType("transformers")
    dummy_mod.AutoModelForCausalLM = DummyAutoModel
    dummy_mod.AutoTokenizer = DummyAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", dummy_mod)

    # Ensure torch module is available (may not be installed in test env)
    if "torch" not in sys.modules:
        dummy_torch = types.ModuleType("torch")
        dummy_torch.float32 = "float32"
        monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    compat.get_perplexity_model("dummy-model")

    assert DummyAutoModel.last_kwargs.get("token") == "abc123"
    assert DummyAutoTokenizer.last_kwargs.get("token") == "abc123"


def test_perplexity_loader_passes_float32_dtype(monkeypatch):
    """get_perplexity_model passes torch_dtype=torch.float32 to prevent BFloat16 mismatch."""
    print("\n-- PPL LOADER DTYPE --")
    monkeypatch.setattr(compat, "HAS_PERPLEXITY", True)

    class DummyAutoModel:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.last_kwargs = kwargs
            return cls()

        def eval(self):
            return None

    class DummyAutoTokenizer:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.last_kwargs = kwargs
            return cls()

    dummy_mod = types.ModuleType("transformers")
    dummy_mod.AutoModelForCausalLM = DummyAutoModel
    dummy_mod.AutoTokenizer = DummyAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", dummy_mod)

    # Ensure torch module is available with float32
    if "torch" not in sys.modules:
        dummy_torch = types.ModuleType("torch")
        dummy_torch.float32 = "float32_sentinel"
        monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    torch_mod = sys.modules["torch"]
    compat.get_perplexity_model("dummy-model")
    assert "torch_dtype" in DummyAutoModel.last_kwargs, "torch_dtype not passed to from_pretrained"
    assert DummyAutoModel.last_kwargs["torch_dtype"] == torch_mod.float32


def test_binoculars_loader_passes_float32_dtype(monkeypatch):
    """get_binoculars_model passes torch_dtype=torch.float32 to prevent BFloat16 mismatch."""
    print("\n-- BINOCULARS LOADER DTYPE --")
    monkeypatch.setattr(compat, "HAS_BINOCULARS", True)

    class DummyAutoModel:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.last_kwargs = kwargs
            return cls()

        def eval(self):
            return None

    class DummyAutoTokenizer:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.last_kwargs = kwargs
            return cls()

    dummy_mod = types.ModuleType("transformers")
    dummy_mod.AutoModelForCausalLM = DummyAutoModel
    dummy_mod.AutoTokenizer = DummyAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", dummy_mod)

    # Ensure torch module is available with float32
    if "torch" not in sys.modules:
        dummy_torch = types.ModuleType("torch")
        dummy_torch.float32 = "float32_sentinel"
        monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    torch_mod = sys.modules["torch"]
    compat.get_binoculars_model()
    assert "torch_dtype" in DummyAutoModel.last_kwargs, "torch_dtype not passed to from_pretrained"
    assert DummyAutoModel.last_kwargs["torch_dtype"] == torch_mod.float32


def test_binoculars_loader_passes_token(monkeypatch):
    print("\n-- BINOCULARS TOKEN PROPAGATION --")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "xyz789")
    monkeypatch.setattr(compat, "HAS_BINOCULARS", True)

    class DummyAutoModel:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.last_kwargs = kwargs
            return cls()

        def eval(self):
            return None

    class DummyAutoTokenizer:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.last_kwargs = kwargs
            return cls()

    dummy_mod = types.ModuleType("transformers")
    dummy_mod.AutoModelForCausalLM = DummyAutoModel
    dummy_mod.AutoTokenizer = DummyAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", dummy_mod)

    # Ensure torch module is available (may not be installed in test env)
    if "torch" not in sys.modules:
        dummy_torch = types.ModuleType("torch")
        dummy_torch.float32 = "float32"
        monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    compat.get_binoculars_model()

    assert DummyAutoModel.last_kwargs.get("token") == "xyz789"
    assert DummyAutoTokenizer.last_kwargs.get("token") == "xyz789"
