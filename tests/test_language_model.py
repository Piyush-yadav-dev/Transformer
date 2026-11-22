"""Phase 4 interface and tiny execution checks; mathematical validation belongs to Phase 5."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from transformer.configuration.settings import TransformerConfig
from transformer.model.transformer import TransformerLanguageModel, TransformerOutput
from transformer.model.transformer import LayerNormalization
from transformer.runtime.device import check_model


@pytest.fixture
def config():
    return TransformerConfig(
        vocab_size=11,
        num_layers=2,
        num_heads=2,
        embedding_dim=16,
        context_length=8,
        feed_forward_expansion=2,
        dropout=0.1,
    )


def test_default_architecture_and_configuration_round_trip(tmp_path):
    config = TransformerConfig(vocab_size=78)
    assert (config.num_layers, config.num_heads, config.embedding_dim, config.context_length) == (
        4,
        4,
        256,
        256,
    )
    destination = tmp_path / "nested/model.json"
    config.save(destination)
    assert TransformerConfig.load(destination) == config
    assert TransformerConfig.from_dict(json.loads(config.to_json())) == config
    with pytest.raises(FileExistsError):
        config.save(destination)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vocab_size", 1),
        ("vocab_size", True),
        ("vocab_size", 3.5),
        ("num_layers", 0),
        ("num_heads", -1),
        ("embedding_dim", 0),
        ("context_length", 0),
        ("feed_forward_expansion", 0),
        ("feed_forward_expansion", 2.5),
        ("embedding_dim", 15),
        ("dropout", -0.1),
        ("dropout", 1),
        ("dropout", float("nan")),
        ("dropout", True),
        ("dropout", "0.1"),
        ("layer_norm_epsilon", 0),
        ("layer_norm_epsilon", float("inf")),
        ("initialization_std", 0),
        ("initialization_std", float("nan")),
        ("tie_embeddings", 1),
        ("schema_version", 2),
        ("schema_version", True),
    ],
)
def test_invalid_configuration_is_explained(config, field, value):
    with pytest.raises(ValueError, match=field if field != "schema_version" else "schema_version"):
        replace(config, **{field: value})


@pytest.mark.parametrize("payload", [[], {}, {"vocab_size": 11, "extra": 2}])
def test_configuration_structure_is_validated(payload):
    with pytest.raises(ValueError):
        TransformerConfig.from_dict(payload)


def test_malformed_configuration_json(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text('{"vocab_size":', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid Transformer JSON"):
        TransformerConfig.load(source)


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("length", [1, 8])
def test_tiny_forward_backward_shapes_loss_and_all_gradients(config, tied, length):
    torch.manual_seed(42)
    model = TransformerLanguageModel(replace(config, tie_embeddings=tied))
    stream = torch.randint(config.vocab_size, (2, length + 1))
    output = model(stream[:, :-1], stream[:, 1:])
    assert isinstance(output, TransformerOutput)
    assert output.logits.shape == (2, length, config.vocab_size)
    assert torch.isfinite(output.logits).all()
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    output.loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_optional_loss_and_already_shifted_targets(config):
    model = TransformerLanguageModel(config).eval()
    stream = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    inputs, targets = stream[:, :-1], stream[:, 1:]
    with torch.no_grad():
        output = model(inputs, targets)
        prediction_only = model(inputs)
    assert prediction_only.loss is None
    assert torch.equal(prediction_only.logits, output.logits)
    expected = F.cross_entropy(output.logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    torch.testing.assert_close(output.loss, expected)


def test_final_initialization_ties_once_and_is_seeded(config, monkeypatch):
    config = replace(config, initialization_std=0.037)
    calls = []
    original = nn.init.normal_

    def record(tensor, mean=0.0, std=1.0, **kwargs):
        if std == config.initialization_std:
            calls.append(id(tensor))
        return original(tensor, mean=mean, std=std, **kwargs)

    monkeypatch.setattr(nn.init, "normal_", record)
    torch.manual_seed(17)
    first = TransformerLanguageModel(config)
    assert first.embeddings.token.weight is first.vocabulary_projection.weight
    assert calls.count(id(first.embeddings.token.weight)) == 1
    assert len(calls) == len(set(calls))
    torch.manual_seed(17)
    second = TransformerLanguageModel(config)
    for key, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[key]), key
    for module in first.modules():
        if isinstance(module, LayerNormalization):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0


def test_parameter_count_deduplicates_shared_weights(config):
    tied = TransformerLanguageModel(config)
    untied = TransformerLanguageModel(replace(config, tie_embeddings=False))
    assert untied.embeddings.token.weight is not untied.vocabulary_projection.weight
    assert untied.trainable_parameter_count() - tied.trainable_parameter_count() == (
        config.vocab_size * config.embedding_dim
    )
    unique = {id(parameter): parameter for parameter in tied.parameters()}
    assert tied.trainable_parameter_count() == sum(
        parameter.numel() for parameter in unique.values()
    )
    count = tied.trainable_parameter_count()
    tied.embeddings.token.weight.requires_grad_(False)
    assert tied.trainable_parameter_count() == count - config.vocab_size * config.embedding_dim


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ([[1, 2]], "shaped"),
        (torch.ones(2, dtype=torch.int64), "shaped"),
        (torch.zeros(2, 3, 1, dtype=torch.int64), "shaped"),
        (torch.zeros(2, 3), "int64"),
        (torch.zeros(2, 3, dtype=torch.bool), "int64"),
        (torch.zeros(0, 3, dtype=torch.int64), "non-empty"),
        (torch.zeros(2, 0, dtype=torch.int64), "non-empty"),
        (torch.zeros(2, 9, dtype=torch.int64), "exceeds context_length"),
        (torch.tensor([[-1, 0]]), "IDs in"),
        (torch.tensor([[0, 11]]), "IDs in"),
    ],
)
def test_invalid_model_inputs_fail_clearly(config, inputs, message):
    with pytest.raises(ValueError, match=message):
        TransformerLanguageModel(config)(inputs)


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (torch.zeros(2, 2, dtype=torch.int64), "same.*shape"),
        (torch.zeros(2, 3), "int64"),
        (torch.full((2, 3), -100), "IDs in"),
        (torch.zeros(2, 3, dtype=torch.int64, device="meta"), "same device"),
    ],
)
def test_invalid_targets_fail_clearly(config, targets, message):
    with pytest.raises(ValueError, match=message):
        TransformerLanguageModel(config)(torch.zeros(2, 3, dtype=torch.int64), targets)


def test_zero_dropout_and_single_embedding_dimension_are_supported():
    config = TransformerConfig(vocab_size=2, embedding_dim=1, num_heads=1, num_layers=1, dropout=0)
    output = TransformerLanguageModel(config)(torch.tensor([[0]]), torch.tensor([[1]]))
    assert torch.isfinite(output.loss)


def test_synthetic_check_reports_no_optimizer_steps(config):
    result = check_model(config, "cpu", seed=42)
    assert result["forward_backward_check"] == "passed"
    assert result["selected_device"] == "cpu"
    assert result["weights_tied"] is True
    assert result["optimizer_steps"] == 0


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Real model MPS smoke check pending."
)
def test_real_tiny_model_mps_forward_backward(config):
    result = check_model(config, "mps")
    assert result["selected_device"] == "mps"
    assert result["forward_backward_check"] == "passed"
    assert result["weights_tied"] is True


def test_installed_model_check_cli(config, tmp_path):
    config_path = tmp_path / "model.json"
    config.save(config_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "transformer",
            "check-model",
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["logits_shape"] == [2, 8, 11]
    assert report["optimizer_steps"] == 0
