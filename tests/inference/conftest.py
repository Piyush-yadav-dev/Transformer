"""Synthetic local checkpoint bundles for offline loader tests, not trained experiments."""

import pytest
import torch

from transformer.configuration.settings import TrainingConfig
from transformer.configuration.settings import TransformerConfig
from transformer.data.preparation import file_sha256, write_json
from transformer.model import TransformerLanguageModel
from transformer.tokenization.character_tokenizer import CharacterTokenizer


@pytest.fixture
def inference_fixture(tmp_path):
    root = tmp_path
    tokenizer = CharacterTokenizer(tuple("abcd"))
    tokenizer_path = root / "data/tokens/fixture/tokenizer.json"
    tokenizer.save(tokenizer_path)
    architecture = TransformerConfig(
        vocab_size=6,
        num_layers=1,
        num_heads=2,
        embedding_dim=8,
        context_length=4,
        dropout=0.4,
    )
    versions, records, scores = [], {}, {}

    def reference(path):
        return {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}

    metrics_path = root / "artifacts/fixture/results.json"
    metrics_path.parent.mkdir(parents=True)
    for i, (member, kind) in enumerate((("model_a", "baseline"), ("model_b", "combined"))):
        run_id = f"synthetic-{member}"
        run = root / "runs" / run_id
        (run / "checkpoints").mkdir(parents=True)
        config = TrainingConfig(
            run_id=run_id,
            token_data_dir="data/tokens/fixture",
            tokenizer_path="data/tokens/fixture/tokenizer.json",
            tokenizer_fingerprint=tokenizer.fingerprint,
            dataset_fingerprint="a" * 64,
            token_data_fingerprint="b" * 64,
            model=architecture,
            train_stream="train_tinystories" if member == "model_a" else "train",
            max_updates=1,
            cpu_threads=1,
        )
        config.save(run / "configuration.json")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(i)
            model = TransformerLanguageModel(architecture)
        checkpoint = run / "checkpoints/fixture.pt"
        torch.save(
            {
                "format_version": 1,
                "identity": {"configuration": config.to_dict()},
                "model": model.state_dict(),
                "position": {"updates": 1, "tokens": 32},
            },
            checkpoint,
        )
        entry = {
            "model_id": f"fixture-{kind}-v1",
            "run_id": run_id,
            "kind": kind,
            "comparison_member": member,
            "checkpoint_update": 1,
            "checkpoint": reference(checkpoint),
            "configuration": reference(run / "configuration.json"),
            "tokenizer": reference(tokenizer_path),
            "tokenizer_fingerprint": tokenizer.fingerprint,
        }
        versions.append(entry)
        records[member] = {
            "run_id": run_id,
            "configuration": config.to_dict(),
            "checkpoint": {"sha256": entry["checkpoint"]["sha256"]},
            "selected_update": 1,
            "selected_training_tokens": 32,
            "parameter_count": model.trainable_parameter_count(),
            "selection_rule": "synthetic fixture, not measured model quality",
        }
        scores[member] = {"packed": {"combined": {"fixture": True}, "sources": {}}}
    write_json(
        metrics_path,
        {
            "identity": {"tokenizer_fingerprint": tokenizer.fingerprint},
            "runs": records,
            "scores": scores,
        },
    )
    for entry in versions:
        entry["metrics"] = reference(metrics_path)
    registry = root / "configs/inference/model_registry.json"
    registry.parent.mkdir(parents=True)
    write_json(registry, {"schema_version": 1, "models": versions})
    return root
