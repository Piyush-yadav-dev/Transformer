# Sampler
"""CPU window ordering with no worker prefetch or hidden iterator RNG consumption."""

import torch


class WindowSampler:
    def __init__(self, size: int, seed: int, shuffle: bool) -> None:
        if type(size) is not int or size < 1:
            raise ValueError("Training sampler requires at least one window.")
        self.size, self.shuffle = size, shuffle
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch, self.cursor = 0, 0
        self.order = self._order()

    def _order(self) -> torch.Tensor:
        return (
            torch.randperm(self.size, generator=self.generator)
            if self.shuffle
            else torch.arange(self.size)
        )

    def take(self, count: int) -> list[int]:
        indices = []
        while len(indices) < count:
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
                self.order = self._order()
            end = min(self.size, self.cursor + count - len(indices))
            indices.extend(self.order[self.cursor : end].tolist())
            self.cursor = end
        return indices

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order.clone(),
            "generator": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != set(self.state_dict()):
            raise ValueError("Unsupported or incomplete sampler state.")
        if state["size"] != self.size or state["shuffle"] != self.shuffle:
            raise ValueError("Incompatible sampler size or shuffle policy.")
        if (
            type(state["epoch"]) is not int
            or state["epoch"] < 0
            or type(state["cursor"]) is not int
            or not 0 <= state["cursor"] <= self.size
        ):
            raise ValueError("Invalid sampler position.")
        order = state["order"]
        if (
            not isinstance(order, torch.Tensor)
            or order.dtype != torch.int64
            or order.shape != (self.size,)
            or not torch.equal(order.sort().values, torch.arange(self.size))
        ):
            raise ValueError("Invalid sampler permutation.")
        self.generator.set_state(state["generator"])
        self.order, self.epoch, self.cursor = order.clone(), state["epoch"], state["cursor"]


# Token dataset
"""Read fixed next-token windows without mapping the token corpus onto an accelerator."""

import mmap
import os
import struct
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from transformer.configuration.settings import _integer
from transformer.configuration.settings import DataLoaderConfig
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.tokenization.token_preparation import read_token_manifest


class TokenWindowDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        stream: str,
        tokenizer_fingerprint: str,
        *,
        context_length: int = 256,
        dataset_fingerprint: str | None = None,
    ) -> None:
        _integer(context_length, "context_length", 1, 1_000_000)
        self.directory = Path(directory).resolve()
        manifest = read_token_manifest(
            self.directory, tokenizer_fingerprint, expected_dataset_fingerprint=dataset_fingerprint
        )
        if stream not in manifest["streams"]:
            raise ValueError(
                f"Unknown token stream {stream!r}; choose one of {sorted(manifest['streams'])}."
            )
        self.stream = stream
        self.info = manifest["streams"][stream]
        self.context_length = context_length
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.dataset_fingerprint = manifest["dataset_fingerprint"]
        self.token_count = self.info["tokens"]
        self.dtype = np.dtype(self.info["dtype"])
        self.path = self.directory / self.info["token_file"]
        self._mapping = self._tokens = self._pid = None
        tokenizer = CharacterTokenizer.load(
            self.directory / "tokenizer.json", expected_fingerprint=tokenizer_fingerprint
        )
        self._validate_tokens(tokenizer)
        if self.token_count < context_length + 1:
            raise ValueError(
                f"Stream {stream} has {self.token_count} tokens; context_length={context_length} "
                f"requires at least {context_length + 1}. Use a shorter context or more data; "
                "no padding or split concatenation is performed."
            )
        self.window_count = (self.token_count - 1) // context_length
        self.dropped_tail_pairs = (self.token_count - 1) % context_length

    def _validate_tokens(self, tokenizer: CharacterTokenizer) -> None:
        eos_count = unknown_count = 0
        with self.path.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                ids = np.frombuffer(chunk, dtype=self.dtype)
                if ids.size and int(ids.max()) >= tokenizer.vocab_size:
                    raise ValueError(f"Out-of-vocabulary token ID in stream {self.stream}.")
                eos_count += int(np.count_nonzero(ids == tokenizer.eos_id))
                unknown_count += int(np.count_nonzero(ids == tokenizer.unk_id))
        if eos_count != self.info["documents"] or unknown_count != self.info["unknown_tokens"]:
            raise ValueError(f"Special-token counts disagree with the manifest for {self.stream}.")
        with (
            (self.directory / self.info["offset_file"]).open("rb") as offsets,
            self.path.open("rb") as tokens,
        ):
            previous = struct.unpack("<Q", offsets.read(8))[0]
            if previous != 0:
                raise ValueError("Story offsets must start at zero.")
            for _ in range(self.info["documents"]):
                end = struct.unpack("<Q", offsets.read(8))[0]
                if not previous < end <= self.token_count:
                    raise ValueError("Story offsets must increase within the token stream.")
                tokens.seek((end - 1) * self.dtype.itemsize)
                if (
                    np.frombuffer(tokens.read(self.dtype.itemsize), dtype=self.dtype)[0]
                    != tokenizer.eos_id
                ):
                    raise ValueError("Each story must end with its explicit EOS token.")
                previous = end
            if previous != self.token_count:
                raise ValueError("Final story offset does not match the token count.")

    def __len__(self) -> int:
        return self.window_count

    def _ensure_mapping(self) -> None:
        if self._tokens is None or self._pid != os.getpid():
            self.close()
            with self.path.open("rb") as handle:
                self._mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._tokens = np.ndarray((self.token_count,), dtype=self.dtype, buffer=self._mapping)
            self._pid = os.getpid()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self):
            raise IndexError(f"Window index must be between 0 and {len(self) - 1}.")
        self._ensure_mapping()
        start = int(index) * self.context_length
        # EOS ke across window chal sakti hai, par doosri dataset split file tak kabhi nahi.
        window = np.array(
            self._tokens[start : start + self.context_length + 1], dtype=np.int64, copy=True
        )
        # Target ek token aage hai: x=tokens[:-1], y=tokens[1:]. Dono CPU int64 tensors hain.
        tensor = torch.from_numpy(window)
        return tensor[:-1].clone(), tensor[1:].clone()

    def close(self) -> None:
        self._tokens = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        self._pid = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.update(_mapping=None, _tokens=None, _pid=None)
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_data_loader(dataset: TokenWindowDataset, config: DataLoaderConfig) -> DataLoader:
    if dataset.context_length != config.context_length:
        raise ValueError("Dataset and loader context lengths must match.")
    if config.drop_last and len(dataset) < config.batch_size:
        raise ValueError(
            "drop_last would produce no batches; reduce batch_size or disable drop_last."
        )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        generator=generator,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
        pin_memory=False,
        persistent_workers=False,
        multiprocessing_context="spawn" if config.num_workers else None,
        prefetch_factor=1 if config.num_workers else None,
    )


# Source provenance
"""Audit existing story partitions and attribute packed target tokens to their source."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from transformer.data.preparation import file_sha256, fingerprint, iter_jsonl, verify_artifact
from transformer.data.preparation import CLEANING_POLICY, clean_record
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.tokenization.token_preparation import read_token_manifest

SOURCES = ("tinystories", "simplestories")


def record_source(record):
    origins = record.get("origins")
    if (
        not isinstance(origins, list)
        or not origins
        or any(not isinstance(o, dict) for o in origins)
    ):
        raise ValueError("Prepared story requires source provenance.")
    names = {origin.get("source") for origin in origins if isinstance(origin, dict)}
    if len(names) != 1 or not names <= set(SOURCES):
        raise ValueError("Ambiguous or unsupported source ownership; attribution refused.")
    source = names.pop()
    if record.get("source", source) != source:
        raise ValueError("Story owner disagrees with its recorded origins.")
    return source


@dataclass
class SourceProvenance:
    directory: Path
    stream: str
    offsets: np.ndarray
    sources: np.ndarray
    story_ids: tuple[str, ...]
    tokenizer_fingerprint: str
    dataset_fingerprint: str
    context_length: int

    @property
    def window_count(self):
        return (int(self.offsets[-1]) - 1) // self.context_length

    def target_counts(self):
        counts = np.zeros((self.window_count, len(SOURCES)), dtype=np.int64)
        for start in range(0, self.window_count, 64):
            end = min(start + 64, self.window_count)
            # Target shift ke baad source gino; poori mixed window ko ek source mat do.
            positions = np.arange(start * self.context_length + 1, end * self.context_length + 1)
            owners = self.sources[np.searchsorted(self.offsets[1:], positions, side="right")]
            for number in range(len(SOURCES)):
                counts[start:end, number] = (
                    (owners == number).reshape(-1, self.context_length).sum(axis=1)
                )
        return counts

    def description(self):
        counts = self.target_counts()
        totals = counts.sum(axis=0)
        lengths = np.diff(self.offsets)
        return {
            "stream": self.stream,
            "documents": len(self.story_ids),
            "stored_tokens": int(self.offsets[-1]),
            "eligible_windows": self.window_count,
            "eligible_target_tokens": int(totals.sum()),
            "source_documents": {
                name: int((self.sources == i).sum()) for i, name in enumerate(SOURCES)
            },
            "source_stored_tokens": {
                name: int(lengths[self.sources == i].sum()) for i, name in enumerate(SOURCES)
            },
            "source_eligible_target_tokens": dict(zip(SOURCES, map(int, totals), strict=True)),
            "source_eligible_target_fraction": dict(
                zip(SOURCES, map(float, totals / totals.sum()), strict=True)
            ),
            "mixed_source_windows": int((np.count_nonzero(counts, axis=1) > 1).sum()),
            "attribution": (
                "Each predicted token belongs to its story's source; EOS belongs to the ending "
                "story. Input-only tokens are not training exposure."
            ),
            "provenance_fingerprint": fingerprint(
                {
                    "stream": self.stream,
                    "story_ids": self.story_ids,
                    "offsets": self.offsets.tolist(),
                    "sources": self.sources.tolist(),
                    "tokenizer": self.tokenizer_fingerprint,
                    "dataset": self.dataset_fingerprint,
                    "context_length": self.context_length,
                }
            ),
        }


def audit_partitions(root, config):
    token_dir = Path(root) / config.token_data_dir
    token_manifest = read_token_manifest(
        token_dir,
        config.tokenizer_fingerprint,
        expected_dataset_fingerprint=config.dataset_fingerprint,
    )
    if token_manifest["token_data_fingerprint"] != config.token_data_fingerprint:
        raise ValueError("Unexpected token-data fingerprint.")
    prepared_name = token_manifest["identity"]["configuration"]["prepared_name"]
    prepared = token_dir.parent.parent / "prepared" / prepared_name
    manifest = json.loads((prepared / "manifest.json").read_text())
    verify_artifact(prepared, manifest["identity"])
    if (
        file_sha256(prepared / "manifest.json")
        != token_manifest["identity"]["prepared_manifest_sha256"]
    ):
        raise ValueError("Prepared manifest changed since token preparation.")
    if manifest["corpus_fingerprint"] != config.dataset_fingerprint:
        raise ValueError("Prepared dataset identity mismatch.")
    if any(manifest.get("cleaning", {}).get(k) != v for k, v in CLEANING_POLICY.items()):
        raise ValueError("Prepared cleaning policy differs from the paired experiment.")
    tokenizer = CharacterTokenizer.load(
        token_dir / "tokenizer.json", expected_fingerprint=config.tokenizer_fingerprint
    )
    seen, counts, views = {}, {}, {}
    for stream in ("train", "validation", "test", "train_tinystories", "train_simplestories"):
        ids, owners = [], []
        offsets = np.fromfile(token_dir / f"{stream}.offsets.bin", dtype="<u8")
        info = token_manifest["streams"][stream]
        with (token_dir / info["token_file"]).open("rb") as tokens:
            for number, record in enumerate(iter_jsonl(prepared / f"{stream}.jsonl")):
                source = record_source(record)
                cleaned = clean_record(record, "text", manifest["cleaning"]["max_story_characters"])
                if (
                    cleaned.text != record.get("text")
                    or cleaned.content_hash != record.get("id")
                    or cleaned.rejection
                ):
                    raise ValueError(f"Invalid cleaned story identity in {stream}.")
                identity = record["id"]
                if stream in ("train", "validation", "test"):
                    if identity in seen:
                        raise ValueError(
                            f"Duplicate story or split overlap: {seen[identity]} and {stream}."
                        )
                    seen[identity] = stream
                if stream.startswith("train") and any(
                    o.get("official_split") != "train" for o in record["origins"]
                ):
                    raise ValueError("Official held-out story found in training.")
                if (
                    number + 1 >= len(offsets)
                    or int(offsets[number]) != tokens.tell() // np.dtype(info["dtype"]).itemsize
                ):
                    raise ValueError("Story offsets do not match prepared record order.")
                length = int(offsets[number + 1] - offsets[number])
                # Test text is inspected only for integrity/overlap, never scored or fitted.
                expected = tokenizer.encode(
                    record["text"],
                    add_eos=True,
                    unknown="error" if stream.startswith("train") else "replace",
                )
                actual = np.frombuffer(
                    tokens.read(length * np.dtype(info["dtype"]).itemsize), dtype=info["dtype"]
                )
                if length != len(expected) or not np.array_equal(actual, expected):
                    raise ValueError(
                        "Token content/offset provenance disagrees with prepared stories."
                    )
                ids.append(identity)
                owners.append(SOURCES.index(source))
            if len(ids) != info["documents"] or len(offsets) != len(ids) + 1 or tokens.read(1):
                raise ValueError("Prepared/token document inventory mismatch.")
        if len(set(ids)) != len(ids):
            raise ValueError("Repeated story within a prepared view.")
        provenance = SourceProvenance(
            token_dir,
            stream,
            offsets,
            np.asarray(owners, dtype=np.uint8),
            tuple(ids),
            config.tokenizer_fingerprint,
            config.dataset_fingerprint,
            config.model.context_length,
        )
        views[stream] = provenance
        counts[stream] = dict(Counter(SOURCES[i] for i in owners))
    for i, source in enumerate(SOURCES):
        expected = [
            identity
            for identity, owner in zip(
                views["train"].story_ids, views["train"].sources, strict=True
            )
            if owner == i
        ]
        if list(views[f"train_{source}"].story_ids) != expected:
            raise ValueError("Source training view differs from the combined training partition.")
    return views, {
        "status": "passed",
        "exact_story_overlap_across_splits": 0,
        "counts": counts,
        "cleaning": manifest["cleaning"],
        "boundary_policy": token_manifest["boundary_policy"],
        "prepared_manifest_sha256": file_sha256(prepared / "manifest.json"),
        "token_manifest_sha256": file_sha256(token_dir / "manifest.json"),
        "limitations": (
            "Exact normalized-content check over acquired subsets; near-duplicate or "
            "upstream-unseen overlap is not ruled out."
        ),
        "test_content_scored_or_used_to_fit": False,
    }


def source_exposure(provenance, config, tokens, sampler_state=None):
    if tokens < 0 or tokens > config.total_tokens or tokens % provenance.context_length:
        raise ValueError("Exposure requires a valid committed token count.")
    counts = provenance.target_counts()
    sampler = WindowSampler(provenance.window_count, config.seed, config.shuffle)
    windows = tokens // provenance.context_length
    totals = np.zeros(len(SOURCES), dtype=np.int64)
    visited, mixed, digest = set(), 0, hashlib.sha256()
    remaining = windows
    while remaining:
        indices = sampler.take(min(config.microbatch_size, remaining))
        rows = counts[indices]
        totals += rows.sum(axis=0)
        mixed += int((np.count_nonzero(rows, axis=1) > 1).sum())
        visited.update(indices)
        digest.update(np.asarray(indices, dtype="<u8").tobytes())
        remaining -= len(indices)
    state = sampler.state_dict()
    if sampler_state is not None:
        if set(state) != set(sampler_state) or any(
            not torch.equal(value, sampler_state[name])
            if isinstance(value, torch.Tensor)
            else value != sampler_state[name]
            for name, value in state.items()
        ):
            raise ValueError("Observed checkpoint sampler does not match reconstructed exposure.")
    return {
        "training_target_tokens": int(totals.sum()),
        "sampled_windows": windows,
        "unique_windows": len(visited),
        "repeated_window_draws": windows - len(visited),
        "mixed_source_window_draws": mixed,
        "sampler_epoch": state["epoch"],
        "sampler_cursor": state["cursor"],
        "source_target_tokens": dict(zip(SOURCES, map(int, totals), strict=True)),
        "source_target_fraction": {
            name: float(totals[i] / tokens) if tokens else 0.0 for i, name in enumerate(SOURCES)
        },
        "sampled_window_order_sha256": digest.hexdigest(),
        "checkpoint_sampler_state_verified": sampler_state is not None,
        "method": (
            "Exact replay of the seeded sampler, checked against all committed sampler fields; "
            "counts assigned by predicted token position, including EOS."
        ),
    }
