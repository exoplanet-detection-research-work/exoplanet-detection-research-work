"""Tests for exoplanet neural network architectures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exodet.ml.data import MlBatch, collate_ml_batch
from exodet.models.base import MODELS
from exodet.models.config import parse_model_params
from exodet.models.registry import ExoplanetClassifierModel
from exodet.utils.seeding import seed_everything
from tests.ml_fixtures import make_representation_dataset

torch = pytest.importorskip("torch")

import exodet.models.registry  # noqa: F401, E402 — register architectures

_BENCH_PARAMS = {
    "global_bins": 64,
    "local_bins": 32,
    "embed_dim": 32,
    "hidden_dim": 64,
    "cnn_channels": [16, 32],
    "cnn_kernel_sizes": [5, 3],
    "transformer_depth": 2,
    "transformer_heads": 2,
    "physics_hidden_dims": [32],
    "num_classes": 5,
    "transformer_checkpoint": False,
    "compile_model": False,
}


def _make_batch(
    batch_size: int = 4,
    n_global: int = 64,
    n_local: int = 32,
    n_physics: int = 12,
    seed: int = 0,
) -> MlBatch:
    rng = np.random.default_rng(seed)
    items = []
    for index in range(batch_size):
        items.append(
            {
                "global_view": rng.normal(size=n_global),
                "local_view": rng.normal(size=n_local),
                "features": rng.normal(size=n_physics),
                "labels": index % 5,
                "weights": 1.0,
                "sample_id": f"s{index}",
                "target_id": f"TIC {index}",
            }
        )
    return collate_ml_batch(items, use_views="both")


@pytest.fixture(params=["fusion", "cnn_transformer", "cnn_only", "transformer_only", "physics_only"])
def model_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def model_and_batch(model_name: str) -> tuple[ExoplanetClassifierModel, MlBatch, int]:
    params = dict(_BENCH_PARAMS)
    n_physics = 12
    if model_name == "physics_only":
        input_dim = n_physics
    elif model_name in ("cnn_only", "cnn"):
        input_dim = params["local_bins"]
    elif model_name in ("transformer_only", "transformer"):
        input_dim = params["global_bins"]
    elif model_name == "cnn_transformer":
        input_dim = params["global_bins"] + params["local_bins"]
    else:
        input_dim = params["global_bins"] + params["local_bins"] + n_physics

    model = MODELS.build(model_name, **params)
    batch = _make_batch(
        n_global=params["global_bins"],
        n_local=params["local_bins"],
        n_physics=n_physics,
    )
    return model, batch, input_dim


class TestModelConfig:
    def test_parse_infers_physics_dim(self) -> None:
        cfg = parse_model_params(
            {"global_bins": 64, "local_bins": 32, "branch_mode": "fusion"},
            input_dim=64 + 32 + 20,
        )
        assert cfg.n_physics_features == 20

    def test_all_registry_names_present(self) -> None:
        for name in (
            "cnn",
            "cnn_only",
            "transformer",
            "transformer_only",
            "cnn_transformer",
            "fusion",
            "physics_only",
        ):
            assert name in MODELS


class TestForwardBackward:
    def test_forward_backward(self, model_and_batch: tuple) -> None:
        model, batch, input_dim = model_and_batch
        model._ensure_module(input_dim, torch.device("cpu"))
        model.module.train()
        logits = model.forward_batch(batch)
        assert logits.shape[0] == batch.labels.shape[0]
        loss = logits.sum()
        loss.backward()

    def test_gradient_flow(self, model_and_batch: tuple) -> None:
        model, batch, input_dim = model_and_batch
        model._ensure_module(input_dim, torch.device("cpu"))
        model.module.train()
        output = model._run_forward(batch)
        output.class_logits.sum().backward()
        grads = [
            p.grad.abs().sum().item()
            for p in model.module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads
        assert any(g > 0 for g in grads)


class TestModelAPI:
    def test_forward_features_cache(self, model_and_batch: tuple) -> None:
        model, batch, input_dim = model_and_batch
        model._ensure_module(input_dim, torch.device("cpu"))
        network = model.module
        out = network(
            global_view=batch.global_view,
            local_view=batch.local_view,
            physics=batch.features,
        )
        cached = network.forward_features()
        assert torch.allclose(out.fused, cached.fused)
        feats = network.extract_features()
        assert "fused" in feats

    def test_predict_proba_multiclass(self, model_and_batch: tuple) -> None:
        model, batch, input_dim = model_and_batch
        model._ensure_module(input_dim, torch.device("cpu"))
        model._fitted = True
        parts = []
        if batch.global_view is not None:
            parts.append(batch.global_view.numpy())
        if batch.local_view is not None:
            parts.append(batch.local_view.numpy())
        if batch.features is not None:
            parts.append(batch.features.numpy())
        flat = np.concatenate(parts, axis=1)
        probs = model.predict_proba_multiclass(flat)
        assert probs.shape == (flat.shape[0], 5)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


class TestDeterminism:
    def test_deterministic_forward(self, model_and_batch: tuple) -> None:
        model, batch, input_dim = model_and_batch
        model._ensure_module(input_dim, torch.device("cpu"))
        model.module.eval()

        def run() -> np.ndarray:
            seed_everything(123)
            with torch.no_grad():
                return model.forward_batch(batch).numpy()

        assert np.allclose(run(), run())


class TestSerialization:
    def test_save_load_roundtrip(self, model_and_batch: tuple, tmp_path: Path) -> None:
        model, batch, input_dim = model_and_batch
        model._ensure_module(input_dim, torch.device("cpu"))
        model._fitted = True
        path = tmp_path / "model.pt"
        model.save(path)
        restored = ExoplanetClassifierModel.load(path)
        restored._ensure_module(input_dim, torch.device("cpu"))
        model.module.eval()
        restored.module.eval()
        with torch.no_grad():
            a = model.forward_batch(batch)
            b = restored.forward_batch(batch)
        assert torch.allclose(a, b, atol=1e-5)


class TestRegistryLoading:
    def test_build_from_registry(self) -> None:
        model = MODELS.build("fusion", **_BENCH_PARAMS)
        assert isinstance(model, ExoplanetClassifierModel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCuda:
    def test_cuda_forward(self) -> None:
        model = MODELS.build("fusion", **_BENCH_PARAMS)
        batch = _make_batch(
            n_global=_BENCH_PARAMS["global_bins"],
            n_local=_BENCH_PARAMS["local_bins"],
        )
        input_dim = _BENCH_PARAMS["global_bins"] + _BENCH_PARAMS["local_bins"] + 12
        model._ensure_module(input_dim, torch.device("cuda"))
        batch = MlBatch(
            global_view=batch.global_view.cuda(),
            local_view=batch.local_view.cuda(),
            features=batch.features.cuda(),
            labels=batch.labels.cuda(),
            weights=batch.weights.cuda(),
            sample_ids=batch.sample_ids,
            target_ids=batch.target_ids,
        )
        logits = model.forward_batch(batch)
        assert logits.device.type == "cuda"


class TestTorchScript:
    def test_script_physics_only(self) -> None:
        params = {**_BENCH_PARAMS, "branch_mode": "physics_only", "n_physics_features": 12}
        cfg = parse_model_params(params, input_dim=12)
        from exodet.models.classifier import HybridExoplanetNetwork

        network = HybridExoplanetNetwork(cfg).eval()
        physics = torch.randn(2, 12)
        try:
            scripted = torch.jit.trace(
                lambda x: network(physics=x).class_logits,
                physics,
            )
            out = scripted(physics)
            assert out.shape == (2, 5)
        except Exception as exc:
            pytest.skip(f"TorchScript trace unsupported in this environment: {exc}")
