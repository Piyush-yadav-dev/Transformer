"""Small synthetic story artifacts; these are not downloaded data or production runs."""

import hashlib
import json

import pytest
import torch

from transformer.configuration.settings import TokenPreparationConfig
from transformer.configuration.settings import TrainingConfig
from transformer.configuration.settings import TransformerConfig
from transformer.data.preparation import file_sha256, fingerprint, write_json
from transformer.tokenization.token_preparation import prepare_tokens


@pytest.fixture
def training_fixture(tmp_path):
    prepared = tmp_path / "data/prepared/training-fixtures"
    prepared.mkdir(parents=True)

    def record(text, source, split):
        return {
            "text": text,
            "id": hashlib.sha256(text.encode()).hexdigest(),
            "origins": [{"source": source, "official_split": split}],
        }

    first = [record("abcdabcdabcd", "tinystories", "train")]
    second = [record("dcbaabcd", "simplestories", "train")]
    streams = {
        "train": first + second,
        "train_tinystories": first,
        "train_simplestories": second,
        "validation": [record("badc" * 5, "tinystories", "validation")],
        "test": [record("cdab", "simplestories", "test")],
    }
    files = {}
    for name, records in streams.items():
        path = prepared / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        files[path.name] = {"sha256": file_sha256(path), "documents": len(records)}
    manifest = {
        "identity": {"configuration": {"sources": [{"source_id": "synthetic-test-fixture"}]}},
        "files": files,
        "corpus_fingerprint": fingerprint({name: value["sha256"] for name, value in files.items()}),
    }
    write_json(prepared / "manifest.json", manifest)
    output = prepare_tokens(
        TokenPreparationConfig(
            "training-fixtures", "training-fixtures", manifest["corpus_fingerprint"]
        ),
        tmp_path,
    )
    model = TransformerConfig(
        vocab_size=6,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        context_length=4,
        feed_forward_expansion=2,
        dropout=0.25,
    )
    config = TrainingConfig(
        run_id="synthetic-run",
        token_data_dir="data/tokens/training-fixtures",
        tokenizer_path="data/tokens/training-fixtures/tokenizer.json",
        tokenizer_fingerprint=output.manifest["tokenizer_fingerprint"],
        dataset_fingerprint=manifest["corpus_fingerprint"],
        token_data_fingerprint=output.manifest["token_data_fingerprint"],
        model=model,
        train_stream="train",
        max_updates=4,
        microbatch_size=2,
        gradient_accumulation_steps=2,
        evaluation_interval=2,
        checkpoint_interval=2,
        max_validation_batches=10,
        console_interval=100,
        warmup_updates=1,
        cpu_threads=1,
    )
    return tmp_path, config


@pytest.fixture
def synthetic_windows():
    # Tiny deterministic token windows, with no dataset/network dependency.
    stream = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [5, 3, 1, 4, 2, 0],
            [2, 4, 1, 5, 0, 3],
            [3, 0, 5, 2, 1, 4],
        ]
    )
    return [(row[:4], row[1:5]) for row in stream]
