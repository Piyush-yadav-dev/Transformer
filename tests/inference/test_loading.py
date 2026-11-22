"""Trust boundaries and strict artifact compatibility without network or actual run dependencies."""

import json
import weakref

import pytest
import torch
from inference_support import read_registry, rewrite_checkpoint

from transformer.data.preparation import file_sha256
from transformer.inference import GenerationService, ModelRegistry
from transformer.tokenization.character_tokenizer import CharacterTokenizer


def test_registered_load_freezes_model_and_preserves_rng(inference_fixture):
    state = torch.get_rng_state().clone()
    original_dtype = torch.get_default_dtype()
    original_threads = torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    try:
        with GenerationService(inference_fixture, device="cpu") as service:
            assert len(service.list_models()) == 2
            metadata = service.load("fixture-baseline-v1")
            assert metadata["model_id"] == "fixture-baseline-v1"
            model = service._loaded.model
            assert not model.training
            assert all(
                not p.requires_grad and p.grad is None and p.dtype == torch.float32
                for p in model.parameters()
            )
            assert service.load("fixture-baseline-v1")["cache_hit"]
            assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(original_dtype)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.get_num_threads() == original_threads


@pytest.mark.parametrize("artifact", ["checkpoint", "configuration", "tokenizer", "metrics"])
def test_missing_registered_artifacts_fail(inference_fixture, artifact):
    _, data = read_registry(inference_fixture)
    (inference_fixture / data["models"][0][artifact]["path"]).unlink()
    with GenerationService(inference_fixture, device="cpu") as service:
        with pytest.raises(ValueError, match="Missing regular local project artifact"):
            service.load("fixture-baseline-v1")


def test_checksum_failure_prevents_deserialization(inference_fixture, monkeypatch):
    _, data = read_registry(inference_fixture)
    (inference_fixture / data["models"][0]["checkpoint"]["path"]).write_bytes(b"bad pickle")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Unverified file deserialized"))
    with GenerationService(inference_fixture, device="cpu") as service:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            service.load("fixture-baseline-v1")


def test_only_verified_file_handle_uses_restricted_deserialization(inference_fixture, monkeypatch):
    original = torch.load
    calls = []

    def inspect_load(source, **kwargs):
        calls.append(kwargs)
        assert hasattr(source, "read") and source.tell() == 0
        return original(source, **kwargs)

    monkeypatch.setattr(torch, "load", inspect_load)
    with GenerationService(inference_fixture, device="cpu") as service:
        service.load("fixture-baseline-v1")
    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.update(format_version=2), "format"),
        (lambda p: p["position"].update(updates=0), "position"),
        (lambda p: p["identity"]["configuration"]["model"].update(num_layers=2), "configuration"),
        (lambda p: p["model"].pop("final_norm.weight"), "Missing key"),
        (lambda p: p["model"]["final_norm.weight"].fill_(float("nan")), "finite float32"),
        (
            lambda p: p["model"].update(
                {"final_norm.weight": p["model"]["final_norm.weight"].double()}
            ),
            "float32",
        ),
        (
            lambda p: p["model"].update(
                {"vocabulary_projection.weight": p["model"]["vocabulary_projection.weight"] + 1}
            ),
            "Tied",
        ),
    ],
)
def test_incompatible_checkpoint_rejected(inference_fixture, mutate, match):
    rewrite_checkpoint(inference_fixture, mutate)
    with GenerationService(inference_fixture, device="cpu") as service:
        with pytest.raises(ValueError, match=match):
            service.load("fixture-baseline-v1")


def test_tokenizer_identity_and_vocabulary_validation(inference_fixture):
    path, registry = read_registry(inference_fixture)
    entry = registry["models"][0]
    entry["tokenizer_fingerprint"] = "0" * 64
    path.write_text(json.dumps(registry))
    with GenerationService(inference_fixture, device="cpu") as service:
        with pytest.raises(ValueError, match="fingerprint"):
            service.load("fixture-baseline-v1")
    tokenizer_path = inference_fixture / entry["tokenizer"]["path"]
    tokenizer_path.unlink()
    other = CharacterTokenizer(tuple("abcde"))
    other.save(tokenizer_path)
    entry["tokenizer_fingerprint"] = other.fingerprint
    entry["tokenizer"]["sha256"] = file_sha256(tokenizer_path)
    configuration = inference_fixture / entry["configuration"]["path"]
    config = json.loads(configuration.read_text())
    config["tokenizer_fingerprint"] = other.fingerprint
    configuration.write_text(json.dumps(config))
    entry["configuration"]["sha256"] = file_sha256(configuration)
    path.write_text(json.dumps(registry))
    with GenerationService(inference_fixture, device="cpu") as service:
        with pytest.raises(ValueError, match="vocabulary size"):
            service.load("fixture-baseline-v1")


@pytest.mark.parametrize(
    "invalid", ["../weights.pt", "/tmp/weights.pt", "https://example.com/a.pt", "missing", None]
)
def test_request_cannot_supply_file_paths(inference_fixture, invalid):
    registry = ModelRegistry(inference_fixture)
    with pytest.raises(ValueError, match="Unknown model ID"):
        registry.resolve(invalid)


@pytest.mark.parametrize(
    "value", ["../outside.pt", "/tmp/outside.pt", "runs/synthetic-model_a/../bad.pt"]
)
def test_catalog_path_traversal_is_rejected(inference_fixture, value):
    path, registry = read_registry(inference_fixture)
    registry["models"][0]["checkpoint"]["path"] = value
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="registry"):
        ModelRegistry(inference_fixture)


def test_symlinked_artifact_or_parent_is_rejected(inference_fixture):
    _, registry = read_registry(inference_fixture)
    checkpoint = inference_fixture / registry["models"][0]["checkpoint"]["path"]
    original = checkpoint.with_suffix(".saved")
    checkpoint.rename(original)
    checkpoint.symlink_to(original)
    with GenerationService(inference_fixture, device="cpu") as service:
        with pytest.raises(ValueError, match="Symbolic links"):
            service.load("fixture-baseline-v1")
    checkpoint.unlink()
    original.rename(checkpoint)
    parent = checkpoint.parent
    saved_parent = parent.with_name("moved-checkpoints")
    parent.rename(saved_parent)
    parent.symlink_to(saved_parent, target_is_directory=True)
    with GenerationService(inference_fixture, device="cpu") as service:
        with pytest.raises(ValueError, match="Symbolic links"):
            service.load("fixture-baseline-v1")


def test_switch_and_close_evict_previous_model(inference_fixture):
    with GenerationService(inference_fixture, device="cpu") as service:
        service.load("fixture-baseline-v1")
        previous = weakref.ref(service._loaded.model)
        service.load("fixture-combined-v1")
        assert previous() is None
        assert service.cache_info() == {
            "capacity": 1,
            "cached_models": 1,
            "model_id": "fixture-combined-v1",
            "device": "cpu",
        }
    assert service.cache_info()["cached_models"] == 0


def test_cached_artifact_change_does_not_adopt_new_weights(inference_fixture):
    with GenerationService(inference_fixture, device="cpu") as service:
        service.load("fixture-baseline-v1")
        path = inference_fixture / service._loaded.version.checkpoint.path
        path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="SHA-256"):
            service.load("fixture-baseline-v1")
        assert service.cache_info()["cached_models"] == 0
