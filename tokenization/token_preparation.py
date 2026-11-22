"""Fit the shared training vocabulary and write separate compact token streams."""

import hashlib
import json
import platform
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transformer.configuration.settings import PathConfig
from transformer.configuration.settings import TokenPreparationConfig
from transformer.data.preparation import (
    artifact_lock,
    file_sha256,
    fingerprint,
    iter_jsonl,
    staged_directory,
    verify_artifact,
    write_json,
)
from transformer.tokenization.character_tokenizer import CharacterTokenizer

TRAINING_VIEWS = ("train_tinystories", "train_simplestories")
STREAM_SPLITS = {
    "train": "train",
    "train_tinystories": "train",
    "train_simplestories": "train",
    "validation": "validation",
    "test": "test",
}
BOUNDARY_POLICY = "packed_with_eos_within_one_stream"


def select_storage_dtype(vocab_size: int, requested: str = "auto") -> np.dtype:
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("Vocabulary size must be a positive integer.")
    choices = {"uint8": np.dtype("u1"), "uint16": np.dtype("<u2"), "uint32": np.dtype("<u4")}
    if requested not in ("auto", *choices):
        raise ValueError("Token storage must use auto, uint8, uint16 or uint32.")
    candidates = choices.values() if requested == "auto" else [choices[requested]]
    for dtype in candidates:
        if vocab_size - 1 <= np.iinfo(dtype).max:
            return dtype
    raise ValueError(f"Storage dtype {requested} cannot represent all {vocab_size} token IDs.")


def _stories(path: Path, *, training: bool):
    for number, record in enumerate(iter_jsonl(path), 1):
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"Prepared story {path}, line {number} must contain string text.")
        text = record["text"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != record.get("id"):
            raise ValueError(f"Prepared story content hash mismatch in {path}, line {number}.")
        if training:
            origins = record.get("origins")
            if (
                not isinstance(origins, list)
                or not origins
                or not all(
                    isinstance(origin, dict)
                    and origin.get("official_split") == "train"
                    and origin.get("source") in ("tinystories", "simplestories")
                    for origin in origins
                )
            ):
                raise ValueError(
                    f"Disallowed or held-out origin in training story {path}, line {number}."
                )
        yield text


def fit_shared_tokenizer(
    prepared_directory: Path, manifest: dict
) -> tuple[CharacterTokenizer, dict]:
    counts = {}

    def training_texts():
        # Vocabulary sirf dono allowed training partitions se; validation/test ko mat padho.
        for stream in TRAINING_VIEWS:
            filename = f"{stream}.jsonl"
            count = 0
            for text in _stories(prepared_directory / filename, training=True):
                count += 1
                yield text
            if count != manifest["files"][filename]["documents"]:
                raise ValueError(f"Prepared document count mismatch for {filename}.")
            counts[filename] = count

    tokenizer = CharacterTokenizer.fit(training_texts())
    return tokenizer, {
        "files": {
            name: {"sha256": manifest["files"][name]["sha256"], "documents_read": count}
            for name, count in counts.items()
        },
        "validation_or_test_used_for_vocabulary": False,
        "policy": (
            "Union of allowed TinyStories and SimpleStories training characters; "
            "sorted Unicode code points."
        ),
    }


def _encode_stream(prepared: Path, stage: Path, stream: str, tokenizer, dtype) -> dict:
    token_file, offset_file = f"{stream}.bin", f"{stream}.offsets.bin"
    total = documents = unknown = 0
    with (stage / token_file).open("xb") as output, (stage / offset_file).open("xb") as offsets:
        offsets.write(struct.pack("<Q", 0))
        for text in _stories(
            prepared / f"{stream}.jsonl", training=STREAM_SPLITS[stream] == "train"
        ):
            # Har story ke baad EOS lagao; agli story ka start explicit boundary ke baad aata hai.
            ids = tokenizer.encode(
                text,
                add_eos=True,
                unknown="error" if STREAM_SPLITS[stream] == "train" else "replace",
            )
            np.asarray(ids, dtype=dtype).tofile(output)
            total += len(ids)
            documents += 1
            unknown += ids.count(tokenizer.unk_id)
            offsets.write(struct.pack("<Q", total))
    return {
        "split": STREAM_SPLITS[stream],
        "token_file": token_file,
        "offset_file": offset_file,
        "dtype": dtype.str,
        "offset_dtype": "<u8",
        "documents": documents,
        "tokens": total,
        "unknown_tokens": unknown,
    }


def read_token_manifest(
    directory: Path,
    expected_tokenizer_fingerprint: str,
    *,
    expected_dataset_fingerprint: str | None = None,
) -> dict:
    directory = Path(directory)
    if not isinstance(expected_tokenizer_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_tokenizer_fingerprint
    ):
        raise ValueError("An expected SHA-256 tokenizer fingerprint is required.")
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Token manifest must be a JSON object.")
        if type(manifest.get("format_version")) is not int or manifest["format_version"] != 1:
            raise ValueError("Unsupported token data format version.")
        if manifest["boundary_policy"] != BOUNDARY_POLICY:
            raise ValueError("Incompatible story boundary policy.")
        if manifest["tokenizer_fingerprint"] != expected_tokenizer_fingerprint:
            raise ValueError("Incompatible tokenizer fingerprint for token data.")
        if (
            expected_dataset_fingerprint is not None
            and manifest["dataset_fingerprint"] != expected_dataset_fingerprint
        ):
            raise ValueError("Incompatible prepared dataset fingerprint for token data.")
        verify_artifact(directory, manifest["identity"])
        tokenizer = CharacterTokenizer.load(
            directory / "tokenizer.json", expected_fingerprint=expected_tokenizer_fingerprint
        )
        if (
            type(manifest["vocabulary_size"]) is not int
            or manifest["vocabulary_size"] != tokenizer.vocab_size
        ):
            raise ValueError("Manifest vocabulary size disagrees with the tokenizer.")
        if (
            manifest["dataset_fingerprint"]
            != manifest["identity"]["configuration"]["dataset_fingerprint"]
        ):
            raise ValueError("Inconsistent prepared dataset fingerprint in token manifest.")
        if set(manifest["streams"]) != set(STREAM_SPLITS):
            raise ValueError("Token manifest must describe the five separate prepared streams.")
        required_files = {"tokenizer.json"}
        for stream, info in manifest["streams"].items():
            if info["split"] != STREAM_SPLITS[stream]:
                raise ValueError(f"Split mismatch for token stream {stream}.")
            if (
                info["token_file"] != f"{stream}.bin"
                or info["offset_file"] != f"{stream}.offsets.bin"
            ):
                raise ValueError(f"Incompatible token filenames for stream {stream}.")
            dtype = np.dtype(info["dtype"])
            if (
                select_storage_dtype(tokenizer.vocab_size, dtype.name).str != dtype.str
                or info["offset_dtype"] != "<u8"
            ):
                raise ValueError("Incompatible token or offset storage dtype.")
            for field in ("documents", "tokens", "unknown_tokens"):
                if type(info[field]) is not int or info[field] < 0:
                    raise ValueError(f"Invalid {field} count in token stream {stream}.")
            if (
                info["tokens"] < info["documents"]
                or info["unknown_tokens"] > info["tokens"] - info["documents"]
            ):
                raise ValueError(f"Inconsistent token counts for {stream}.")
            if (
                info["tokenizer_fingerprint"] != expected_tokenizer_fingerprint
                or info["dataset_fingerprint"] != manifest["dataset_fingerprint"]
            ):
                raise ValueError(f"Incompatible fingerprints for token stream {stream}.")
            if (directory / info["token_file"]).stat().st_size != info["tokens"] * dtype.itemsize:
                raise ValueError(f"Token file size mismatch for {stream}.")
            if (directory / info["offset_file"]).stat().st_size != (info["documents"] + 1) * 8:
                raise ValueError(f"Story offset file size mismatch for {stream}.")
            required_files.update((info["token_file"], info["offset_file"]))
        if set(manifest["files"]) != required_files:
            raise ValueError("Token manifest file inventory is incomplete or unexpected.")
        if manifest["token_data_fingerprint"] != fingerprint(
            {name: info["sha256"] for name, info in manifest["files"].items()}
        ):
            raise ValueError("Token data fingerprint mismatch.")
        return manifest
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed token manifest in {directory}: {exc}.") from exc


@dataclass(frozen=True, slots=True)
class TokenPreparationResult:
    directory: Path
    manifest: dict
    reused: bool


def prepare_tokens(
    config: TokenPreparationConfig, project_root: Path, paths: PathConfig | None = None
) -> TokenPreparationResult:
    data_root = (paths or PathConfig()).resolve(project_root)["data_dir"]
    prepared = data_root / "prepared" / config.prepared_name
    target = data_root / "tokens" / config.name
    try:
        source = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        verify_artifact(prepared, source["identity"])
        actual = fingerprint({name: info["sha256"] for name, info in source["files"].items()})
        if source["corpus_fingerprint"] != actual or actual != config.dataset_fingerprint:
            raise ValueError(
                "Incompatible prepared dataset fingerprint; check token configuration."
            )
        if not {f"{name}.jsonl" for name in STREAM_SPLITS} <= source["files"].keys():
            raise ValueError("Prepared data is missing a required training/evaluation stream.")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed prepared manifest in {prepared}: {exc}.") from exc
    identity = {
        "pipeline_version": 1,
        "configuration": config.to_dict(),
        "prepared_manifest_sha256": file_sha256(prepared / "manifest.json"),
        "implementation_sha256": fingerprint(
            {
                name: file_sha256(Path(__file__).parent / name)
                for name in ("character_tokenizer.py", "token_preparation.py")
            }
        ),
    }
    with artifact_lock(target):
        if target.exists():
            cached = verify_artifact(target, identity)
            manifest = read_token_manifest(
                target,
                cached["tokenizer_fingerprint"],
                expected_dataset_fingerprint=config.dataset_fingerprint,
            )
            return TokenPreparationResult(target, manifest, True)
        with staged_directory(target) as stage:
            tokenizer, fit_record = fit_shared_tokenizer(prepared, source)
            dtype = select_storage_dtype(tokenizer.vocab_size, config.storage_dtype)
            tokenizer.save(stage / "tokenizer.json")
            streams = {}
            for stream in STREAM_SPLITS:
                info = _encode_stream(prepared, stage, stream, tokenizer, dtype)
                if info["documents"] != source["files"][f"{stream}.jsonl"]["documents"]:
                    raise ValueError(f"Prepared document count mismatch for {stream}.")
                streams[stream] = {
                    **info,
                    "source_file_sha256": source["files"][f"{stream}.jsonl"]["sha256"],
                    "tokenizer_fingerprint": tokenizer.fingerprint,
                    "dataset_fingerprint": config.dataset_fingerprint,
                }
            files = {
                path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                for path in sorted(stage.iterdir())
            }
            manifest = {
                "format_version": 1,
                "identity": identity,
                "boundary_policy": BOUNDARY_POLICY,
                "tokenizer_fingerprint": tokenizer.fingerprint,
                "dataset_fingerprint": config.dataset_fingerprint,
                "vocabulary_size": tokenizer.vocab_size,
                "vocabulary_fit": fit_record,
                "prepared_sources": source["identity"]["configuration"]["sources"],
                "streams": streams,
                "files": files,
                "token_data_fingerprint": fingerprint(
                    {name: info["sha256"] for name, info in files.items()}
                ),
                "environment": {"python": platform.python_version(), "numpy": np.__version__},
            }
            write_json(stage / "manifest.json", manifest)
            read_token_manifest(
                stage,
                tokenizer.fingerprint,
                expected_dataset_fingerprint=config.dataset_fingerprint,
            )
    return TokenPreparationResult(target, manifest, False)
