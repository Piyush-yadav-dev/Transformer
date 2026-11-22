"""Synthetic paired runs: target ownership, isolated validation, and unchanged training math."""

import hashlib
import json
import signal
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import transformer
from transformer.configuration.settings import ComparisonProtocol
from transformer.configuration.settings import TokenPreparationConfig
from transformer.configuration.settings import TrainingConfig
from transformer.configuration.settings import TransformerConfig
from transformer.data.preparation import file_sha256, fingerprint, write_json
from transformer.data.loading import (
    SourceProvenance,
    audit_partitions,
    record_source,
    source_exposure,
)
from transformer.data.preparation import CLEANING_POLICY
from transformer.evaluation.metrics import SourceValidationDataset
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.tokenization.token_preparation import prepare_tokens
from transformer.training.experiments import run_baseline, run_experiment
from transformer.training.engine import CheckpointManager
from transformer.training.experiments import run_combined
from transformer.data.loading import WindowSampler
from transformer.training.engine import run_training


@pytest.fixture
def paired_fixture(tmp_path):
    prepared = tmp_path / "data/prepared/paired-fixture"
    prepared.mkdir(parents=True)

    def row(text, source, split):
        return {
            "text": text,
            "id": hashlib.sha256(text.encode()).hexdigest(),
            "source": source,
            "origins": [{"source": source, "official_split": split}],
        }

    tiny = [row("abcd" * 3, "tinystories", "train")]
    simple = [row("dcba" * 5, "simplestories", "train")]
    streams = {
        "train": tiny + simple,
        "train_tinystories": tiny,
        "train_simplestories": simple,
        "validation": [
            row("badc" * 5, "tinystories", "validation"),
            row("adcb" * 6, "simplestories", "test"),
        ],
        "test": [
            row("acbd" * 3, "tinystories", "validation"),
            row("bcad" * 3, "simplestories", "test"),
        ],
    }
    files = {}
    for name, records in streams.items():
        path = prepared / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        files[path.name] = {"sha256": file_sha256(path), "documents": len(records)}
    corpus = fingerprint({name: r["sha256"] for name, r in files.items()})
    write_json(
        prepared / "manifest.json",
        {
            "identity": {"configuration": {"sources": [{"source_id": "synthetic-fixture"}]}},
            "files": files,
            "corpus_fingerprint": corpus,
            "cleaning": {**CLEANING_POLICY, "max_story_characters": 100000},
        },
    )
    token = prepare_tokens(
        TokenPreparationConfig("paired-fixture", "paired-fixture", corpus), tmp_path
    )
    model = TransformerConfig(
        vocab_size=6, num_layers=1, num_heads=2, embedding_dim=8, context_length=4, dropout=0.25
    )
    a = TrainingConfig(
        run_id="paired-a",
        token_data_dir="data/tokens/paired-fixture",
        tokenizer_path="data/tokens/paired-fixture/tokenizer.json",
        tokenizer_fingerprint=token.manifest["tokenizer_fingerprint"],
        dataset_fingerprint=corpus,
        token_data_fingerprint=token.manifest["token_data_fingerprint"],
        model=model,
        microbatch_size=2,
        gradient_accumulation_steps=2,
        max_training_tokens=64,
        evaluation_interval=2,
        checkpoint_interval=2,
        max_validation_batches=20,
        cpu_threads=1,
        console_interval=100,
    )
    protocol = ComparisonProtocol(
        a,
        replace(a, run_id="paired-b", train_stream="train"),
        "protocol.md",
        (2, 4),
        ("abcd",),
        max_new_tokens=2,
    )
    (tmp_path / "protocol.md").write_text("Synthetic paired test protocol.")
    (tmp_path / "supplement.md").write_text("Synthetic source diagnostics, not selection.")
    return tmp_path, protocol


def baseline_with_hash_evidence(root, protocol):
    run_baseline(protocol, root)
    package = Path(transformer.__file__).parent
    write_json(
        root / "runs" / protocol.model_a.run_id / "prerequisite_verification.json",
        {
            "source_hashes": {
                str(p.relative_to(package)): file_sha256(p) for p in package.rglob("*.py")
            }
        },
    )


def test_target_attribution_handles_shift_eos_and_mixed_windows(paired_fixture):
    root, protocol = paired_fixture
    p = SourceProvenance(
        root,
        "train",
        np.array([0, 3, 7, 10]),
        np.array([0, 1, 0]),
        ("a", "b", "c"),
        protocol.model_a.tokenizer_fingerprint,
        protocol.model_a.dataset_fingerprint,
        4,
    )
    assert p.target_counts().tolist() == [[2, 2], [2, 2]]
    description = p.description()
    assert description["source_documents"] == {"tinystories": 2, "simplestories": 1}
    assert description["source_eligible_target_fraction"] == {
        "tinystories": 0.5,
        "simplestories": 0.5,
    }
    sampler = WindowSampler(2, protocol.model_b.seed, True)
    sampler.take(6)
    exposure = source_exposure(p, protocol.model_b, 24, sampler.state_dict())
    assert exposure["source_target_tokens"] == {"tinystories": 12, "simplestories": 12}
    assert exposure["repeated_window_draws"] == 4
    bad = sampler.state_dict()
    bad["cursor"] = 0
    with pytest.raises(ValueError, match="sampler"):
        source_exposure(p, protocol.model_b, 24, bad)


def test_isolated_source_windows_match_independent_concatenation(paired_fixture):
    root, protocol = paired_fixture
    views, audit = audit_partitions(root, protocol.model_b)
    assert audit["exact_story_overlap_across_splits"] == 0
    tokenizer = CharacterTokenizer.load(root / protocol.model_b.tokenizer_path)
    for source, text in [("tinystories", "badc" * 5), ("simplestories", "adcb" * 6)]:
        ids = tokenizer.encode(text, add_eos=True)
        with SourceValidationDataset(views["validation"], source) as data:
            for i in range(len(data)):
                x, y = data[i]
                assert x.tolist() == ids[i * 4 : i * 4 + 4]
                assert y.tolist() == ids[i * 4 + 1 : i * 4 + 5]
    with pytest.raises(ValueError, match="validation partition"):
        SourceValidationDataset(views["test"], "tinystories")
    planned = source_exposure(views["train"], protocol.model_b, 64)
    assert planned["source_target_tokens"] == {"tinystories": 24, "simplestories": 40}
    assert planned["unique_windows"] == 8 and planned["repeated_window_draws"] == 8


@pytest.mark.parametrize(
    "origins",
    [
        [],
        ["bad"],
        [{"source": "unknown"}],
        [{"source": "tinystories"}, {"source": "simplestories"}],
    ],
)
def test_ambiguous_provenance_is_not_silently_assigned(origins):
    with pytest.raises(ValueError):
        record_source({"origins": origins})


def test_combined_run_preserves_baseline_and_matches_uninstrumented_weights(paired_fixture):
    root, protocol = paired_fixture
    baseline_with_hash_evidence(root, protocol)
    a_dir = root / "runs" / protocol.model_a.run_id
    before = {str(p.relative_to(a_dir)): file_sha256(p) for p in a_dir.rglob("*") if p.is_file()}
    output = root / "artifacts/preparation"
    prepared = run_combined(protocol, root, output, root / "supplement.md", prepare_only=True)
    assert (
        prepared["status"] == "prepared" and not (root / "runs" / protocol.model_b.run_id).exists()
    )
    record = run_combined(protocol, root, output, root / "supplement.md")
    assert record["status"] == "completed" and record["training_tokens"] == 64
    assert record["source_exposure"]["source_target_tokens"] == {
        "tinystories": 24,
        "simplestories": 40,
    }
    assert record["source_exposure"]["checkpoint_sampler_state_verified"]
    assert set(record["source_only_best_validation"]["sources"]) == {"tinystories", "simplestories"}
    assert before == {
        str(p.relative_to(a_dir)): file_sha256(p) for p in a_dir.rglob("*") if p.is_file()
    }
    b_dir = root / "runs" / protocol.model_b.run_id
    assert (
        json.loads((a_dir / "initialization.json").read_text())["weights_sha256"]
        == json.loads((b_dir / "initialization.json").read_text())["weights_sha256"]
    )
    reference = replace(protocol.model_b, run_id="uninstrumented-b")
    run_training(reference, root)
    observed = CheckpointManager(b_dir).load()
    uninterrupted = CheckpointManager(root / "runs" / reference.run_id).load()
    for name, value in observed["model"].items():
        assert torch.equal(value, uninterrupted["model"][name])
    assert torch.equal(observed["rng"]["torch_cpu"], uninterrupted["rng"]["torch_cpu"])
    with pytest.raises(FileExistsError):
        run_combined(protocol, root, output, root / "supplement.md")
    resumed = run_combined(protocol, root, output, root / "supplement.md", resume=True)
    assert resumed["training_tokens"] == 64
    (root / "supplement.md").write_text("Changed after preparation.")
    with pytest.raises(ValueError, match="identity differs"):
        run_combined(protocol, root, output, root / "supplement.md", resume=True)


def test_interrupt_at_final_observer_does_not_start_next_session(paired_fixture):
    root, protocol = paired_fixture

    def observer(event, directory):
        if event["event"] == "update" and event["optimizer_update"] == 2:
            signal.raise_signal(signal.SIGINT)

    result = run_experiment(protocol, root, member="model_b", observer_extension=observer)
    assert result["status"] == "interrupted" and result["updates"] == 2
    resumed = run_experiment(protocol, root, member="model_b", resume=True)
    assert resumed["optimizer_updates"] == 4


def test_stale_baseline_configuration_is_rejected_before_training(paired_fixture):
    root, protocol = paired_fixture
    baseline_with_hash_evidence(root, protocol)
    changed = replace(
        protocol,
        model_a=replace(protocol.model_a, seed=1),
        model_b=replace(protocol.model_b, seed=1),
    )
    with pytest.raises(ValueError, match="Actual baseline settings"):
        run_combined(changed, root, root / "artifact", root / "supplement.md")
    assert not (root / "runs" / protocol.model_b.run_id).exists()


def test_verified_manifests_do_not_hide_cross_split_overlap(paired_fixture):
    root, protocol = paired_fixture
    prepared = root / "data/prepared/paired-fixture"
    validation = prepared / "validation.jsonl"
    validation.write_text(
        validation.read_text() + (prepared / "train_tinystories.jsonl").read_text()
    )
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["files"]["validation.jsonl"] = {"sha256": file_sha256(validation), "documents": 3}
    manifest["corpus_fingerprint"] = fingerprint(
        {name: info["sha256"] for name, info in manifest["files"].items()}
    )
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    result = prepare_tokens(
        TokenPreparationConfig("overlap-fixture", "paired-fixture", manifest["corpus_fingerprint"]),
        root,
    )
    config = replace(
        protocol.model_b,
        token_data_dir="data/tokens/overlap-fixture",
        tokenizer_path="data/tokens/overlap-fixture/tokenizer.json",
        dataset_fingerprint=manifest["corpus_fingerprint"],
        tokenizer_fingerprint=result.manifest["tokenizer_fingerprint"],
        token_data_fingerprint=result.manifest["token_data_fingerprint"],
    )
    with pytest.raises(ValueError, match="split overlap"):
        audit_partitions(root, config)
