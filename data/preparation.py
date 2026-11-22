# Artifact store
"""Integrity-checked artifacts with exclusive locks and atomic directory publication."""

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")


def verify_artifact(directory: Path, identity: dict) -> dict:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != json.loads(canonical_json(identity)):
            raise ValueError("configuration or pipeline identity differs")
        if not isinstance(manifest["files"], dict) or not manifest["files"]:
            raise ValueError("empty artifact file list")
        for name, expected in manifest["files"].items():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("unsafe artifact filename")
            path = directory / name
            if path.is_symlink() or file_sha256(path) != expected["sha256"]:
                raise ValueError(f"checksum mismatch for {name}")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Artifact {directory} is incomplete, incompatible or corrupt: {exc}. "
            "Existing files were preserved; inspect them or choose a new preparation name."
        ) from exc


@contextmanager
def artifact_lock(target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(f".{target.name}.lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"Artifact lock exists: {lock}. Another process may be working. "
            "After confirming it has stopped, inspect and remove only that stale lock."
        ) from exc
    try:
        yield
    finally:
        lock.rmdir()


@contextmanager
def staged_directory(target: Path):
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        yield stage
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {target}")
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON record in {path}, line {number}: {exc.msg}."
                ) from exc


# Text cleaning
"""Conservative Unicode cleaning and exact normalized-content identities."""

import hashlib
import unicodedata
from dataclasses import dataclass

CLEANING_POLICY = {
    "version": 1,
    "unicode_normalization": "NFC",
    "line_endings": "CRLF and CR converted to LF",
    "outer_whitespace": "strip",
    "internal_whitespace": "preserve, including paragraphs and tabs",
    "capitalization_and_punctuation": "preserve",
    "invalid_unicode": "reject surrogate code points; never replace or ASCII-fold",
    "control_characters": "reject Cc except LF and TAB",
    "exact_hash": "SHA-256 of cleaned UTF-8 text",
}


@dataclass(frozen=True, slots=True)
class CleanedRecord:
    text: str | None
    content_hash: str | None
    rejection: str | None


def clean_record(record: object, text_column: str, max_characters: int) -> CleanedRecord:
    if not isinstance(record, dict):
        return CleanedRecord(None, None, "record_not_object")
    if text_column not in record:
        return CleanedRecord(None, None, "missing_text_column")
    text = record[text_column]
    if not isinstance(text, str):
        return CleanedRecord(None, None, "text_not_string")
    if len(text) > max_characters:
        return CleanedRecord(None, None, "story_too_long")
    if any(unicodedata.category(char) == "Cs" for char in text):
        return CleanedRecord(None, None, "invalid_unicode_surrogate")
    if any(unicodedata.category(char) == "Cc" and char not in "\r\n\t" for char in text):
        return CleanedRecord(None, None, "unsupported_control_character")
    # Case aur punctuation bachao; sirf equivalent Unicode aur line endings normalize karo.
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not text:
        return CleanedRecord(None, None, "empty_text")
    return CleanedRecord(text, hashlib.sha256(text.encode("utf-8")).hexdigest(), None)


# Dataset acquisition
"""Pinned Hub metadata verification and reusable bounded streaming subsets."""

from contextlib import closing
from importlib.metadata import version
from itertools import islice
from pathlib import Path

from transformer.configuration.settings import DatasetSourceConfig


class HubStorySource:
    """Verify the exact dataset before streaming only its configured text column."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.verified: dict[str, dict] = {}

    def verify(self, source: DatasetSourceConfig) -> dict:
        from datasets import get_dataset_config_info, get_dataset_config_names, load_dataset_builder
        from huggingface_hub import HfApi, hf_hub_download

        key = fingerprint(source.to_dict())
        if key in self.verified:
            return self.verified[key]
        metadata = HfApi().dataset_info(source.source_id, revision=source.revision)
        if metadata.sha != source.revision:
            raise ValueError(f"Revision mismatch for {source.source_id}.")
        declared_license = (
            metadata.card_data.to_dict().get("license") if metadata.card_data else None
        )
        if declared_license != source.license:
            raise ValueError(
                f"License mismatch for {source.source_id}: configured {source.license}, "
                f"repository declares {declared_license}. Review source configuration."
            )
        kwargs = {"revision": source.revision, "cache_dir": str(self.cache_dir / "datasets")}
        configurations = get_dataset_config_names(source.source_id, **kwargs)
        if source.configuration not in configurations:
            raise ValueError(
                f"Unknown dataset configuration {source.configuration}: {configurations}."
            )
        info = get_dataset_config_info(source.source_id, source.configuration, **kwargs)
        if (
            not info.features
            or getattr(info.features.get(source.text_column), "dtype", None) != "string"
        ):
            raise ValueError(
                f"Text column {source.text_column!r} is not a string feature in {source.source_id}."
            )
        if not info.splits:
            raise ValueError(f"No official split metadata available for {source.source_id}.")
        available_splits = {
            name: split.get("num_examples") if isinstance(split, dict) else split.num_examples
            for name, split in info.splits.items()
        }
        if missing := source.splits.keys() - available_splits.keys():
            raise ValueError(f"Missing official splits in {source.source_id}: {sorted(missing)}.")
        builder = load_dataset_builder(source.source_id, name=source.configuration, **kwargs)
        data_files = {split: list(builder.config.data_files[split]) for split in source.splits}
        pinned_prefix = f"hf://datasets/{source.source_id}@{source.revision}/"
        if any(
            not files
            or any(
                not path.startswith(pinned_prefix) or not path.endswith(".parquet")
                for path in files
            )
            for files in data_files.values()
        ):
            raise ValueError(
                "This reader requires revision-pinned Parquet files in the source repository."
            )
        card_path = hf_hub_download(
            source.source_id,
            "README.md",
            repo_type="dataset",
            revision=source.revision,
            cache_dir=str(self.cache_dir / "hub"),
        )
        result = {
            "source_id": source.source_id,
            "source_url": f"https://huggingface.co/datasets/{source.source_id}",
            "resolved_revision": metadata.sha,
            "configuration": source.configuration,
            "available_configurations": configurations,
            "text_column": source.text_column,
            "features": info.features.to_dict(),
            "available_splits": available_splits,
            "data_files": data_files,
            "license": declared_license,
            "dataset_card": Path(card_path).read_text(encoding="utf-8"),
            "license_files": [
                file.rfilename for file in metadata.siblings if "license" in file.rfilename.lower()
            ],
        }
        self.verified[key] = result
        return result

    def records(self, source: DatasetSourceConfig, split: str):
        from huggingface_hub import HfFileSystem

        metadata = self.verify(source)
        filesystem = HfFileSystem()
        yield from iter_parquet_records(
            metadata["data_files"][split],
            source.text_column,
            lambda path: filesystem.open(path, "rb", block_size=1 << 20),
        )


def iter_parquet_records(data_files, text_column, opener):
    """Read small synchronous batches; explicitly release partially consumed readers."""
    import pyarrow.parquet as parquet

    for path in data_files:
        with opener(path) as handle:
            # Early stop par background Arrow workers hang na hon, isliye synchronous reading.
            with closing(parquet.ParquetFile(handle, pre_buffer=False)) as reader:
                for batch in reader.iter_batches(
                    batch_size=128, columns=[text_column], use_threads=False
                ):
                    yield from batch.to_pylist()


def acquire_subset(
    source: DatasetSourceConfig, split: str, cache_root: Path, provider, *, offline=False
):
    identity = {
        "format_version": 1,
        "source_id": source.source_id,
        "revision": source.revision,
        "configuration": source.configuration,
        "text_column": source.text_column,
        "license": source.license,
        "split": split,
        "limit": source.splits[split],
        "selection": "prefix",
        "datasets_version": version("datasets"),
    }
    target = cache_root / "subsets" / fingerprint(identity)
    with artifact_lock(target):
        if target.exists():
            return target, verify_artifact(target, identity)
        if offline:
            raise ValueError(f"Offline subset cache missing for {source.name}/{split}: {target}.")
        try:
            verification = provider.verify(source)
            with staged_directory(target) as stage:
                records_path = stage / "records.jsonl"
                count = 0
                with records_path.open("x", encoding="utf-8") as stream:
                    # Limit raw records par hai; rejects ke badle extra data download nahi karna.
                    with closing(provider.records(source, split)) as records:
                        for record in islice(records, source.splits[split]):
                            projected = (
                                {source.text_column: record[source.text_column]}
                                if isinstance(record, dict) and source.text_column in record
                                else record
                            )
                            stream.write(canonical_json(projected) + "\n")
                            count += 1
                manifest = {
                    "identity": identity,
                    "verification": verification,
                    "records": count,
                    "exhausted_before_limit": count < source.splits[split],
                    "files": {
                        "records.jsonl": {
                            "sha256": file_sha256(records_path),
                            "bytes": records_path.stat().st_size,
                        }
                    },
                }
                write_json(stage / "manifest.json", manifest)
        except Exception as exc:
            raise RuntimeError(f"Acquisition failed for {source.source_id}/{split}: {exc}") from exc
    return target, manifest


# Dataset preparation
"""Global exact deduplication, document splits, and atomic corpus manifests."""

import hashlib
import json
import platform
import sqlite3
import unicodedata
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from transformer.configuration.settings import DataPreparationConfig, DatasetSourceConfig
from transformer.configuration.settings import PathConfig

SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


def document_split(source: DatasetSourceConfig, official_split: str, content_hash: str) -> str:
    random_key = hashlib.sha256(f"{source.seed}:{content_hash}".encode()).digest()
    score = int.from_bytes(random_key[:8], "big")
    if official_split == "test":
        return "test"
    if official_split == "validation":
        return "test" if score < int(source.validation_test_fraction * 2**64) else "validation"
    if official_split == "train":
        return "validation" if score < int(source.train_validation_fraction * 2**64) else "train"
    raise ValueError(f"Unsupported official split: {official_split}.")


def implementation_identity(config: DataPreparationConfig) -> dict:
    module_dir = Path(__file__).parent
    code = {path.name: file_sha256(path) for path in sorted(module_dir.glob("*.py"))}
    config_path = module_dir.parent / "configuration/settings.py"
    code["configuration/settings.py"] = file_sha256(config_path)
    return {
        "pipeline_version": 1,
        "implementation_sha256": fingerprint(code),
        "unicode_database_version": unicodedata.unidata_version,
        "datasets_version": version("datasets"),
        "configuration": json.loads(canonical_json(config.to_dict())),
    }


@dataclass(frozen=True, slots=True)
class PreparationResult:
    directory: Path
    manifest: dict
    reused: bool


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA cache_size = -8192")
    connection.execute("PRAGMA temp_store = FILE")
    connection.executescript("""
        CREATE TABLE stories (
            hash TEXT PRIMARY KEY, text TEXT NOT NULL, split TEXT NOT NULL,
            source TEXT NOT NULL
        );
        CREATE TABLE origins (
            hash TEXT NOT NULL, source TEXT NOT NULL, official_split TEXT NOT NULL,
            source_index INTEGER NOT NULL, proposed_split TEXT NOT NULL
        );
        CREATE INDEX origins_hash ON origins(hash);
    """)
    return connection


def _ingest(connection, source, split, raw_path, max_characters):
    rejected = Counter()
    counts = {
        "scanned": 0,
        "accepted_before_deduplication": 0,
        "duplicate_occurrences": 0,
        "duplicates_within_source": 0,
        "duplicates_across_sources": 0,
    }
    for index, record in enumerate(iter_jsonl(raw_path)):
        counts["scanned"] += 1
        cleaned = clean_record(record, source.text_column, max_characters)
        if cleaned.rejection:
            rejected[cleaned.rejection] += 1
            continue
        counts["accepted_before_deduplication"] += 1
        digest = cleaned.content_hash
        proposed = document_split(source, split, digest)
        existing = connection.execute(
            "SELECT split FROM stories WHERE hash = ?", (digest,)
        ).fetchone()
        if existing:
            counts["duplicate_occurrences"] += 1
            prior_sources = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT source FROM origins WHERE hash = ?", (digest,)
                )
            }
            counts["duplicates_within_source"] += int(source.name in prior_sources)
            counts["duplicates_across_sources"] += int(bool(prior_sources - {source.name}))
            # Held-out copy ko priority do; training copy ko kabhi evaluation se leak mat hone do.
            if SPLIT_PRIORITY[proposed] > SPLIT_PRIORITY[existing[0]]:
                connection.execute(
                    "UPDATE stories SET split = ?, source = ? WHERE hash = ?",
                    (proposed, source.name, digest),
                )
        else:
            connection.execute(
                "INSERT INTO stories VALUES (?, ?, ?, ?)",
                (digest, cleaned.text, proposed, source.name),
            )
        connection.execute(
            "INSERT INTO origins VALUES (?, ?, ?, ?, ?)",
            (digest, source.name, split, index, proposed),
        )
    connection.commit()
    counts["rejected"] = dict(rejected)
    return counts


def _export(connection, config, stage):
    names = ["train", "validation", "test"] + [f"train_{s.name}" for s in config.sources]
    statistics = {
        name: {"documents": 0, "total_characters": 0, "min_characters": None, "max_characters": 0}
        for name in names
    }
    with ExitStack() as stack:
        streams = {
            name: stack.enter_context((stage / f"{name}.jsonl").open("x", encoding="utf-8"))
            for name in names
        }
        for digest, text, split, source in connection.execute(
            "SELECT hash, text, split, source FROM stories ORDER BY hash"
        ):
            origins = [
                {
                    "source": row[0],
                    "official_split": row[1],
                    "source_index": row[2],
                    "proposed_split": row[3],
                }
                for row in connection.execute(
                    "SELECT source, official_split, source_index, proposed_split "
                    "FROM origins WHERE hash = ? ORDER BY source, official_split, source_index",
                    (digest,),
                )
            ]
            if split == "train" and any(origin["official_split"] != "train" for origin in origins):
                raise RuntimeError(
                    "Leakage invariant violated: official held-out story reached training."
                )
            record = {"id": digest, "text": text, "source": source, "origins": origins}
            destinations = [split]
            if split == "train":
                destinations.extend(
                    f"train_{name}" for name in sorted({origin["source"] for origin in origins})
                )
            for name in destinations:
                streams[name].write(canonical_json(record) + "\n")
                stats = statistics[name]
                length = len(text)
                stats["documents"] += 1
                stats["total_characters"] += length
                stats["min_characters"] = (
                    length
                    if stats["min_characters"] is None
                    else min(stats["min_characters"], length)
                )
                stats["max_characters"] = max(stats["max_characters"], length)
    files = {}
    for name, stats in statistics.items():
        stats["mean_characters"] = (
            stats["total_characters"] / stats["documents"] if stats["documents"] else None
        )
        path = stage / f"{name}.jsonl"
        files[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size, **stats}
    return files


def prepare_data(
    config: DataPreparationConfig,
    project_root: Path,
    paths: PathConfig | None = None,
    *,
    offline: bool = False,
    provider=None,
) -> PreparationResult:
    data_root = (paths or PathConfig()).resolve(project_root)["data_dir"]
    target = data_root / "prepared" / config.name
    cache_root = data_root / "cache"
    identity = implementation_identity(config)
    provider = provider if provider is not None else HubStorySource(cache_root / "huggingface")
    with artifact_lock(target):
        if target.exists():
            return PreparationResult(target, verify_artifact(target, identity), True)
        with staged_directory(target) as stage:
            database_path = stage / "deduplication.sqlite"
            connection = _open_database(database_path)
            acquisition = []
            try:
                for source in config.sources:
                    for split in sorted(source.splits):
                        raw_dir, raw_manifest = acquire_subset(
                            source, split, cache_root, provider, offline=offline
                        )
                        counts = _ingest(
                            connection,
                            source,
                            split,
                            raw_dir / "records.jsonl",
                            config.max_story_characters,
                        )
                        acquisition.append(
                            {
                                "source": source.name,
                                "official_split": split,
                                "cache_key": raw_dir.name,
                                "cache_manifest": raw_manifest,
                                "counts": counts,
                            }
                        )
                files = _export(connection, config, stage)
            finally:
                connection.close()
            database_path.unlink()
            totals = Counter()
            rejected = Counter()
            for entry in acquisition:
                totals.update(
                    {key: count for key, count in entry["counts"].items() if key != "rejected"}
                )
                rejected.update(entry["counts"]["rejected"])
            manifest = {
                "identity": identity,
                "cleaning": {
                    **CLEANING_POLICY,
                    "max_story_characters": config.max_story_characters,
                },
                "selection": (
                    "First N raw records per pinned official split; ordering bias retained; "
                    "rejected records are not replenished."
                ),
                "split_policy": (
                    "Seeded SHA-256 content partition; test > validation > train on conflicts; "
                    "ties retain first configured source. No official held-out story enters train."
                ),
                "deduplication_scope": (
                    "Exact normalized text across all acquired sources/splits only; "
                    "not near-duplicate or full-upstream-corpus detection."
                ),
                "counts": {**totals, "rejected": dict(rejected)},
                "sources": acquisition,
                "files": files,
                "corpus_fingerprint": fingerprint(
                    {name: meta["sha256"] for name, meta in files.items()}
                ),
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "datasets": version("datasets"),
                    "pyarrow": version("pyarrow"),
                    "transformer": version("Transformer"),
                },
            }
            write_json(stage / "manifest.json", manifest)
    return PreparationResult(target, manifest, False)
