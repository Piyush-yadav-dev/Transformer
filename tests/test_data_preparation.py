"""Offline preparation tests using explicitly synthetic document fixtures."""

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformer.cli import main
from transformer.configuration.settings import DataPreparationConfig, DatasetSourceConfig
from transformer.data.preparation import artifact_lock, iter_jsonl
from transformer.data.preparation import HubStorySource, iter_parquet_records
from transformer.data.preparation import document_split, prepare_data
from transformer.data.preparation import clean_record

FIXTURES = Path(__file__).parent / "fixtures/stories"


class FixtureProvider:
    """No Hub access: these are local synthetic records, never training samples."""

    def __init__(self):
        self.read = Counter()

    def verify(self, source):
        return {
            "synthetic_test_fixture": True,
            "source_id": source.source_id,
            "resolved_revision": source.revision,
        }

    def records(self, source, split):
        for record in iter_jsonl(FIXTURES / f"{source.name}_{split}.jsonl"):
            self.read[source.name, split] += 1
            yield record


@pytest.fixture
def config():
    return DataPreparationConfig(
        name="synthetic-test-fixtures",
        sources=(
            DatasetSourceConfig(
                "first",
                "fixtures/first",
                "a" * 40,
                "default",
                "text",
                "test-fixture",
                {"train": 8, "validation": 2},
                validation_test_fraction=0.5,
            ),
            DatasetSourceConfig(
                "second",
                "fixtures/second",
                "b" * 40,
                "default",
                "story",
                "test-fixture",
                {"train": 3, "test": 2},
                train_validation_fraction=0.2,
            ),
        ),
    )


def test_cleaning_preserves_capitalization_unicode_punctuation_and_paragraphs():
    cleaned = clean_record({"text": '  "Café!"\r\n\rA CAT.\t  End.  '}, "text", 100)
    assert cleaned.text == '"Café!"\n\nA CAT.\t  End.'
    assert cleaned.rejection is None
    equivalent = clean_record({"text": '"Cafe\u0301!"\n\nA CAT.\t  End.'}, "text", 100)
    assert equivalent.content_hash == cleaned.content_hash
    assert (
        clean_record({"text": cleaned.text.lower()}, "text", 100).content_hash
        != cleaned.content_hash
    )


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        ({"text": " \r\n\t "}, "empty_text"),
        ({}, "missing_text_column"),
        ([], "record_not_object"),
        ({"text": None}, "text_not_string"),
        ({"text": "broken\ud800"}, "invalid_unicode_surrogate"),
        ({"text": "nul\x00"}, "unsupported_control_character"),
        ({"text": "a" * 101}, "story_too_long"),
    ],
)
def test_rejection_reasons(record, reason):
    assert clean_record(record, "text", 100).rejection == reason


def test_limits_global_duplicates_and_heldout_precedence(config, tmp_path):
    provider = FixtureProvider()
    result = prepare_data(config, tmp_path, provider=provider)
    assert provider.read == {
        ("first", "train"): 8,
        ("first", "validation"): 2,
        ("second", "train"): 3,
        ("second", "test"): 2,
    }
    counts = result.manifest["counts"]
    assert counts["scanned"] == 15
    assert counts["accepted_before_deduplication"] == 11
    assert counts["duplicate_occurrences"] == 3
    assert counts["duplicates_across_sources"] == 2
    assert sum(counts["rejected"].values()) == 4
    splits = {
        name: list(iter_jsonl(result.directory / f"{name}.jsonl"))
        for name in ("train", "validation", "test")
    }
    ids = {name: {record["id"] for record in rows} for name, rows in splits.items()}
    assert not (
        ids["train"] & ids["validation"]
        or ids["train"] & ids["test"]
        or ids["validation"] & ids["test"]
    )
    assert sum(map(len, ids.values())) == 8
    assert "Held by second." in {record["text"] for record in splits["test"]}
    assert "Repeat me." not in {record["text"] for record in splits["train"]}
    assert all(
        origin["official_split"] == "train" for row in splits["train"] for origin in row["origins"]
    )
    assert all(
        "must not be acquired" not in record["text"] for rows in splits.values() for record in rows
    )
    assert {r["id"] for r in iter_jsonl(result.directory / "train_first.jsonl")} <= ids["train"]


def test_preparation_is_reproducible_and_offline_reuse_does_not_call_provider(config, tmp_path):
    first = prepare_data(config, tmp_path / "first", provider=FixtureProvider())
    second = prepare_data(config, tmp_path / "second", provider=FixtureProvider())
    assert first.manifest["corpus_fingerprint"] == second.manifest["corpus_fingerprint"]
    assert first.manifest == second.manifest
    for name in first.manifest["files"]:
        assert (first.directory / name).read_bytes() == (second.directory / name).read_bytes()
    reused = prepare_data(config, tmp_path / "first", offline=True, provider=object())
    assert reused.reused
    assert reused.manifest == first.manifest
    from_cache = prepare_data(
        replace(config, name="new-output-same-subsets"),
        tmp_path / "first",
        offline=True,
        provider=object(),
    )
    assert from_cache.manifest["corpus_fingerprint"] == first.manifest["corpus_fingerprint"]


def test_changed_configuration_cannot_overwrite_existing_artifact(config, tmp_path):
    result = prepare_data(config, tmp_path, provider=FixtureProvider())
    before = (result.directory / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="identity differs"):
        prepare_data(replace(config, max_story_characters=50), tmp_path, provider=FixtureProvider())
    assert (result.directory / "manifest.json").read_bytes() == before


def test_corrupt_output_is_detected_and_preserved(config, tmp_path):
    result = prepare_data(config, tmp_path, provider=FixtureProvider())
    (result.directory / "train.jsonl").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_data(config, tmp_path, offline=True)
    assert (result.directory / "train.jsonl").read_text() == "corrupt"


def test_corrupt_cache_is_detected_without_publishing_output(config, tmp_path):
    result = prepare_data(config, tmp_path, provider=FixtureProvider())
    cached = (
        tmp_path
        / "data/cache/subsets"
        / result.manifest["sources"][0]["cache_key"]
        / "records.jsonl"
    )
    cached.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_data(replace(config, name="second-output"), tmp_path, offline=True)
    assert not (tmp_path / "data/prepared/second-output").exists()
    assert not list((tmp_path / "data/prepared").glob(".second-output*"))


def test_failure_does_not_publish_partial_cache_or_output(config, tmp_path):
    class InterruptedProvider(FixtureProvider):
        def records(self, source, split):
            yield {source.text_column: "First partial record."}
            raise OSError("simulated interrupted acquisition")

    with pytest.raises(RuntimeError, match="simulated interrupted acquisition"):
        prepare_data(config, tmp_path, provider=InterruptedProvider())
    assert not (tmp_path / "data/prepared" / config.name).exists()
    assert not list((tmp_path / "data/cache/subsets").glob("*/manifest.json"))
    result = prepare_data(config, tmp_path, provider=FixtureProvider())
    assert result.manifest["counts"]["scanned"] == 15


def test_existing_lock_fails_clearly(config, tmp_path):
    target = tmp_path / "data/prepared" / config.name
    with artifact_lock(target), pytest.raises(RuntimeError, match="Another process may be working"):
        prepare_data(config, tmp_path, provider=FixtureProvider())


def test_split_assignment_is_order_independent_and_seeded(config):
    source = config.sources[0]
    hashes = [f"{value:064x}" for value in range(100)]
    expected = {value: document_split(source, "validation", value) for value in hashes}
    assert expected == {
        value: document_split(source, "validation", value) for value in reversed(hashes)
    }
    assert set(expected.values()) == {"validation", "test"}
    assert expected != {
        value: document_split(replace(source, seed=43), "validation", value) for value in hashes
    }
    assert all(document_split(source, "test", value) == "test" for value in hashes)


def test_malformed_json_line_has_path_and_line_number(tmp_path):
    path = tmp_path / "malformed.jsonl"
    path.write_text('{"text":"valid"}\n{broken}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"malformed.jsonl, line 2"):
        list(iter_jsonl(path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revision", "main", "commit SHA"),
        ("splits", {"train": 0, "test": 2}, "limit"),
        ("splits", {"train": True, "test": 2}, "limit"),
        ("splits", {"train": 2}, "held-out"),
        ("seed", -1, "seed"),
        ("validation_test_fraction", 1.1, "fraction"),
    ],
)
def test_source_configuration_validation(config, field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(config.sources[0], **{field: value})


def test_data_configuration_round_trip_and_unknown_fields(config):
    assert DataPreparationConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError, match="Unknown data configuration"):
        DataPreparationConfig.from_dict({"name": "x", "sources": [], "typo": 1})


def test_offline_cli_failure_is_clear_and_has_no_network_dependency(config, tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["prepare-data", "--config", str(path), "--project-root", str(tmp_path), "--offline"])
    assert error.value.code == 2
    assert "Offline subset cache missing" in capsys.readouterr().err


@pytest.fixture
def mocked_hub(config, tmp_path, monkeypatch):
    import datasets
    import huggingface_hub

    source = config.sources[0]
    metadata = SimpleNamespace(
        sha=source.revision,
        card_data=SimpleNamespace(to_dict=lambda: {"license": source.license}),
        siblings=[],
    )
    info = SimpleNamespace(
        features=datasets.Features({"text": datasets.Value("string")}),
        splits={"train": {"name": "train"}, "validation": SimpleNamespace(num_examples=10)},
    )
    card = tmp_path / "fixture-card.md"
    card.write_text("Synthetic metadata test fixture; no network data.", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", lambda *a, **kw: metadata)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(card))
    monkeypatch.setattr(datasets, "get_dataset_config_names", lambda *a, **kw: ["default"])
    monkeypatch.setattr(datasets, "get_dataset_config_info", lambda *a, **kw: info)
    monkeypatch.setattr(
        datasets,
        "load_dataset_builder",
        lambda *a, **kw: SimpleNamespace(
            config=SimpleNamespace(
                data_files={
                    split: [f"hf://datasets/{source.source_id}@{source.revision}/{split}.parquet"]
                    for split in source.splits
                }
            )
        ),
    )
    return source, HubStorySource(tmp_path / "cache")


def test_source_verification_accepts_split_objects_and_metadata_without_row_counts(mocked_hub):
    source, provider = mocked_hub
    verified = provider.verify(source)
    assert verified["available_splits"] == {"train": None, "validation": 10}
    assert verified["resolved_revision"] == source.revision
    assert "Synthetic metadata test fixture" in verified["dataset_card"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revision", "c" * 40, "Revision mismatch"),
        ("license", "different-license", "License mismatch"),
        ("configuration", "nonexistent", "Unknown dataset configuration"),
        ("text_column", "wrong_column", "not a string feature"),
    ],
)
def test_source_verification_rejects_mismatches_before_download(mocked_hub, field, value, message):
    source, provider = mocked_hub
    with pytest.raises(ValueError, match=message):
        provider.verify(replace(source, **{field: value}))


def test_synchronous_parquet_reader_closes_on_early_stop(tmp_path):
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    path = tmp_path / "synthetic.parquet"
    parquet.write_table(
        arrow.table({"text": [f"Synthetic fixture {i}." for i in range(300)]}), path
    )
    opened = []

    def opener(path):
        handle = open(path, "rb")
        opened.append(handle)
        return handle

    records = iter_parquet_records([path], "text", opener)
    assert next(records) == {"text": "Synthetic fixture 0."}
    records.close()
    assert opened[0].closed
