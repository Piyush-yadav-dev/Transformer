"""Offline token preparation/loading against tiny synthetic prepared story artifacts."""

import hashlib
import json
import pickle
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from transformer.configuration.settings import DataLoaderConfig, TokenPreparationConfig
from transformer.data.preparation import file_sha256, fingerprint, iter_jsonl, write_json
from transformer.data.loading import TokenWindowDataset, create_data_loader
from transformer.tokenization import token_preparation
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.tokenization.token_preparation import (
    STREAM_SPLITS,
    prepare_tokens,
    select_storage_dtype,
)


def synthetic_prepared(root, *, first=None, second=None, validation=None, test=None):
    """Explicit test fixtures; names represent experiment membership, not real dataset content."""
    first = ["abC!\n" * 20] if first is None else first
    second = ["Déf?" * 20] if second is None else second
    validation = ["baΩ"] if validation is None else validation
    test = ["Ω🚀"] if test is None else test
    directory = root / "data/prepared/synthetic-fixtures"
    directory.mkdir(parents=True)

    def story(text, source, split):
        return {
            "id": hashlib.sha256(text.encode()).hexdigest(),
            "text": text,
            "source": source,
            "origins": [{"source": source, "official_split": split, "source_index": 0}],
        }

    a = [story(text, "tinystories", "train") for text in first]
    b = [story(text, "simplestories", "train") for text in second]
    streams = {
        "train": a + b,
        "train_tinystories": a,
        "train_simplestories": b,
        "validation": [story(text, "tinystories", "validation") for text in validation],
        "test": [story(text, "simplestories", "test") for text in test],
    }
    files = {}
    for name, records in streams.items():
        path = directory / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        files[path.name] = {"sha256": file_sha256(path), "documents": len(records)}
    manifest = {
        "identity": {
            "synthetic_test_fixture": True,
            "configuration": {
                "sources": [
                    {
                        "name": name,
                        "source_id": f"fixtures/{name}",
                        "license": "synthetic-test-fixture",
                    }
                    for name in ("tinystories", "simplestories")
                ]
            },
        },
        "files": files,
        "corpus_fingerprint": fingerprint({name: info["sha256"] for name, info in files.items()}),
    }
    write_json(directory / "manifest.json", manifest)
    return TokenPreparationConfig(
        "synthetic-tokens", "synthetic-fixtures", manifest["corpus_fingerprint"]
    )


@pytest.fixture
def tokens(tmp_path):
    config = synthetic_prepared(tmp_path)
    result = prepare_tokens(config, tmp_path)
    tokenizer = CharacterTokenizer.load(result.directory / "tokenizer.json")
    return config, result, tokenizer


def test_vocabulary_reads_only_allowed_training_views(tmp_path, monkeypatch):
    config = synthetic_prepared(tmp_path)
    directory = tmp_path / "data/prepared" / config.prepared_name
    manifest = json.loads((directory / "manifest.json").read_text())
    reads = []

    def training_only(path):
        reads.append(path.name)
        assert path.name in ("train_tinystories.jsonl", "train_simplestories.jsonl")
        yield from iter_jsonl(path)

    monkeypatch.setattr(token_preparation, "iter_jsonl", training_only)
    tokenizer, provenance = token_preparation.fit_shared_tokenizer(directory, manifest)
    assert reads == ["train_tinystories.jsonl", "train_simplestories.jsonl"]
    assert set(tokenizer.characters) == set("abC!\nDéf?")
    assert "Ω" not in tokenizer.char_to_id and "🚀" not in tokenizer.char_to_id
    assert provenance["validation_or_test_used_for_vocabulary"] is False


def test_all_streams_share_frozen_vocabulary_and_keep_boundaries(tokens):
    _, result, tokenizer = tokens
    assert result.manifest["streams"]["validation"]["unknown_tokens"] == 1
    assert result.manifest["streams"]["test"]["unknown_tokens"] == 2
    assert result.manifest["streams"]["train"]["unknown_tokens"] == 0
    for stream, info in result.manifest["streams"].items():
        assert info["tokenizer_fingerprint"] == tokenizer.fingerprint
        assert info["dtype"] == "|u1"
        ids = np.fromfile(result.directory / info["token_file"], dtype=info["dtype"])
        offsets = np.fromfile(result.directory / info["offset_file"], dtype="<u8")
        assert offsets[0] == 0 and offsets[-1] == len(ids)
        assert len(offsets) == info["documents"] + 1
        assert all(ids[int(end) - 1] == tokenizer.eos_id for end in offsets[1:])
        assert info["split"] == STREAM_SPLITS[stream]


def test_window_shift_types_memory_map_and_story_crossing(tmp_path):
    config = synthetic_prepared(tmp_path, first=["ab"], second=["cd"])
    result = prepare_tokens(config, tmp_path)
    tokenizer = CharacterTokenizer.load(result.directory / "tokenizer.json")
    with TokenWindowDataset(
        result.directory, "train", tokenizer.fingerprint, context_length=3
    ) as dataset:
        assert dataset._mapping is None
        x, y = dataset[0]
        assert dataset._mapping is not None
        assert not dataset._tokens.flags.writeable
        assert x.tolist() == tokenizer.encode("ab", add_eos=True)
        assert y.tolist() == [
            tokenizer.char_to_id["b"],
            tokenizer.eos_id,
            tokenizer.char_to_id["c"],
        ]
        assert x.dtype == y.dtype == torch.int64
        assert x.device.type == y.device.type == "cpu"
        assert torch.equal(x[1:], y[:-1])
        assert len(dataset) == 1 and dataset.dropped_tail_pairs == 2
        x[0] = 999
        assert dataset[0][0][0].item() == tokenizer.char_to_id["a"]
        restored = pickle.loads(pickle.dumps(dataset))
        assert restored._mapping is None
        assert torch.equal(restored[0][1], y)
        restored.close()
        with pytest.raises(IndexError):
            dataset[-1]
        with pytest.raises(IndexError):
            dataset[len(dataset)]
    assert dataset._mapping is None


def test_windows_never_reach_another_split(tokens):
    _, result, tokenizer = tokens
    for stream in STREAM_SPLITS:
        info = result.manifest["streams"][stream]
        raw = np.fromfile(result.directory / info["token_file"], dtype=info["dtype"])
        with TokenWindowDataset(
            result.directory, stream, tokenizer.fingerprint, context_length=1
        ) as dataset:
            assert len(dataset) == len(raw) - 1
            for index in range(len(dataset)):
                x, y = dataset[index]
                assert (x.item(), y.item()) == (int(raw[index]), int(raw[index + 1]))
            with pytest.raises(IndexError):
                dataset[len(dataset)]


def test_incompatible_fingerprints_fail(tokens):
    _, result, tokenizer = tokens
    with pytest.raises(ValueError, match="Incompatible tokenizer fingerprint"):
        TokenWindowDataset(result.directory, "train", "0" * 64, context_length=2)
    with pytest.raises(ValueError, match="Incompatible prepared dataset fingerprint"):
        TokenWindowDataset(
            result.directory,
            "train",
            tokenizer.fingerprint,
            context_length=2,
            dataset_fingerprint="0" * 64,
        )


def test_empty_and_short_streams_have_clear_policy(tmp_path):
    config = synthetic_prepared(tmp_path, first=["a"], second=["b"], validation=[], test=[""])
    result = prepare_tokens(config, tmp_path)
    tokenizer = CharacterTokenizer.load(result.directory / "tokenizer.json")
    assert (result.directory / "validation.bin").stat().st_size == 0
    assert (result.directory / "test.bin").read_bytes() == bytes([tokenizer.eos_id])
    for stream, expected in [("validation", 0), ("test", 1), ("train", 4)]:
        with pytest.raises(ValueError, match=f"has {expected} tokens.*requires at least 5"):
            TokenWindowDataset(result.directory, stream, tokenizer.fingerprint, context_length=4)


def test_empty_training_refuses_fitting_and_publishes_nothing(tmp_path):
    config = synthetic_prepared(tmp_path, first=[], second=[])
    with pytest.raises(ValueError, match="training text has no characters"):
        prepare_tokens(config, tmp_path)
    assert not (tmp_path / "data/tokens" / config.name).exists()


def test_reproducible_preparation_reuse_and_conflict_protection(tokens, tmp_path):
    config, result, _ = tokens
    assert prepare_tokens(config, tmp_path).reused
    rebuilt = prepare_tokens(replace(config, name="second-token-build"), tmp_path)
    assert rebuilt.manifest["token_data_fingerprint"] == result.manifest["token_data_fingerprint"]
    with pytest.raises(ValueError, match="identity differs"):
        prepare_tokens(replace(config, storage_dtype="uint16"), tmp_path)
    assert result.manifest == json.loads((result.directory / "manifest.json").read_text())


def test_corrupted_binary_is_rejected_without_overwriting(tokens):
    _, result, tokenizer = tokens
    path = result.directory / "train.bin"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum mismatch"):
        TokenWindowDataset(result.directory, "train", tokenizer.fingerprint, context_length=2)
    assert path.read_bytes() == b"bad"


def test_out_of_range_ids_fail_even_if_checksums_were_updated(tokens):
    _, result, tokenizer = tokens
    path = result.directory / "train.bin"
    contents = bytearray(path.read_bytes())
    contents[0] = 255
    path.write_bytes(contents)
    manifest = result.manifest
    manifest["files"]["train.bin"]["sha256"] = file_sha256(path)
    manifest["token_data_fingerprint"] = fingerprint(
        {name: info["sha256"] for name, info in manifest["files"].items()}
    )
    (result.directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Out-of-vocabulary token ID"):
        TokenWindowDataset(result.directory, "train", tokenizer.fingerprint, context_length=2)


@pytest.mark.parametrize(
    ("size", "dtype"), [(256, "|u1"), (257, "<u2"), (65536, "<u2"), (65537, "<u4")]
)
def test_storage_capacity_boundaries(size, dtype):
    assert select_storage_dtype(size).str == dtype


def test_storage_dtype_overflow_rejected():
    with pytest.raises(ValueError, match="cannot represent"):
        select_storage_dtype(257, "uint8")
    with pytest.raises(ValueError, match="cannot represent"):
        select_storage_dtype(2**32 + 1)


def test_seeded_batches_are_reproducible_and_sequential_order_is_supported(tokens):
    _, result, tokenizer = tokens
    with TokenWindowDataset(
        result.directory, "train", tokenizer.fingerprint, context_length=4
    ) as dataset:
        settings = DataLoaderConfig(context_length=4, batch_size=3, seed=47)
        first = list(create_data_loader(dataset, settings))
        second = list(create_data_loader(dataset, settings))
        assert len(first) == len(second)
        assert all(
            torch.equal(a, b) and torch.equal(c, d)
            for (a, c), (b, d) in zip(first, second, strict=True)
        )
        sequential = create_data_loader(dataset, replace(settings, shuffle=False))
        assert torch.equal(next(iter(sequential))[0][0], dataset[0][0])
        with pytest.raises(ValueError, match="produce no batches"):
            create_data_loader(dataset, replace(settings, drop_last=True, batch_size=1000))


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_workers", 3), ("batch_size", 0), ("context_length", True), ("seed", -1), ("shuffle", 1)],
)
def test_loader_configuration_validation(field, value):
    with pytest.raises(ValueError):
        DataLoaderConfig(**{field: value})


def test_spawn_worker_and_cli_execute_without_training(tokens, tmp_path):
    _, result, _ = tokens
    settings = DataLoaderConfig(context_length=4, batch_size=2, num_workers=1)
    path = tmp_path / "loader.json"
    path.write_text(json.dumps(settings.to_dict()))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "transformer",
            "check-token-data",
            "--tokens",
            str(result.directory),
            "--loader-config",
            str(path),
            "--stream",
            "train_tinystories",
            "--batches",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert len(report["batches"]) == 2
    assert all(
        batch["input"] == [2, 4] and batch["dtype"] == "torch.int64" and batch["device"] == "cpu"
        for batch in report["batches"]
    )
