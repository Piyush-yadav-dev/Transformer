# Project config
"""Phase 1 configuration and versioned JSON snapshots."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

DevicePreference = Literal["auto", "cpu", "mps"]
DEVICE_CHOICES = ("auto", "cpu", "mps")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: DevicePreference = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or self.device not in DEVICE_CHOICES:
            raise ValueError("runtime.device must be one of: auto, cpu, mps.")


def _validate_relative_directory(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(
            f"paths.{name} must be a non-empty relative directory without outer spaces."
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or "\\" in value
        or "\x00" in value
        or ".." in path.parts
        or not path.parts
    ):
        raise ValueError(
            f"paths.{name} must be project-relative, using '/' without '..' or an absolute path."
        )


@dataclass(frozen=True, slots=True)
class PathConfig:
    data_dir: str = "data"
    runs_dir: str = "runs"

    def __post_init__(self) -> None:
        _validate_relative_directory("data_dir", self.data_dir)
        _validate_relative_directory("runs_dir", self.runs_dir)

    def resolve(self, project_root: Path) -> dict[str, Path]:
        """Resolve paths against an explicit root without creating directories."""
        root = Path(project_root).resolve()
        return {name: root / value for name, value in asdict(self).items()}


def _configuration_object(value: object, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} settings: {', '.join(sorted(map(str, unknown)))}.")
    return value


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    schema_version: int = 1
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be the integer 1.")
        if not isinstance(self.runtime, RuntimeConfig):
            raise ValueError("runtime must be a RuntimeConfig object.")
        if not isinstance(self.paths, PathConfig):
            raise ValueError("paths must be a PathConfig object.")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: object) -> "ProjectConfig":
        data = _configuration_object(value, "configuration", {"schema_version", "runtime", "paths"})
        runtime = _configuration_object(data.get("runtime", {}), "runtime", {"device"})
        paths = _configuration_object(data.get("paths", {}), "paths", {"data_dir", "runs_dir"})
        return cls(
            schema_version=data.get("schema_version", 1),
            runtime=RuntimeConfig(**runtime),
            paths=PathConfig(**paths),
        )

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc.msg} (line {exc.lineno}).") from exc
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        """Write a snapshot; existing files are never overwritten."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Purana snapshot overwrite mat karo; recovery ke liye original settings chahiye.
        with destination.open("x", encoding="utf-8") as snapshot:
            snapshot.write(self.to_json())


# Data config
"""Validated, revision-pinned settings for bounded document preparation."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path



def _integer(value: object, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")


def _fraction(value: object, name: str) -> None:
    if type(value) not in (int, float) or not 0 <= value < 1:
        raise ValueError(f"{name} must be a number from 0 (inclusive) to 1 (exclusive).")


@dataclass(frozen=True, slots=True)
class DatasetSourceConfig:
    name: str
    source_id: str
    revision: str
    configuration: str
    text_column: str
    license: str
    splits: dict[str, int]
    seed: int = 42
    train_validation_fraction: float = 0.0
    validation_test_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name):
            raise ValueError("Source name must be lowercase letters, digits or underscores.")
        if not isinstance(self.source_id, str) or not re.fullmatch(
            r"[\w-]+/[\w.-]+", self.source_id
        ):
            raise ValueError("source_id must identify an explicit owner/dataset repository.")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(
                "revision must be a full 40-character lowercase commit SHA, not a branch."
            )
        for name in ("configuration", "text_column", "license"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.splits, dict) or not self.splits:
            raise ValueError("splits must map official split names to positive record limits.")
        for split, limit in self.splits.items():
            if split not in ("train", "validation", "test"):
                raise ValueError("Supported official split names are train, validation and test.")
            _integer(limit, f"{self.name}.{split} limit", 1, 1_000_000)
        if "train" not in self.splits or not ({"validation", "test"} & self.splits.keys()):
            raise ValueError(
                "Each source must include train and at least one official held-out split."
            )
        _integer(self.seed, "seed", 0, 2**32 - 1)
        _fraction(self.train_validation_fraction, "train_validation_fraction")
        _fraction(self.validation_test_fraction, "validation_test_fraction")
        if "validation" not in self.splits and self.train_validation_fraction == 0:
            raise ValueError(
                "Sources without validation require a positive train_validation_fraction."
            )
        if "test" not in self.splits and self.validation_test_fraction == 0:
            raise ValueError("Sources without test require a positive validation_test_fraction.")
        if "validation" in self.splits and self.train_validation_fraction != 0:
            raise ValueError("Use official validation rather than deriving it from training.")
        if "test" in self.splits and self.validation_test_fraction != 0:
            raise ValueError("Use official test rather than subdividing official validation.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DataPreparationConfig:
    name: str
    sources: tuple[DatasetSourceConfig, ...]
    schema_version: int = 1
    max_story_characters: int = 100_000

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Data configuration schema_version must be the integer 1.")
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]{0,79}", self.name
        ):
            raise ValueError("Preparation name must be a portable lowercase directory name.")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("sources must be a non-empty tuple of DatasetSourceConfig objects.")
        if not all(isinstance(source, DatasetSourceConfig) for source in self.sources):
            raise ValueError("Every source must be a DatasetSourceConfig object.")
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("Source names must be unique.")
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("Dataset repositories must not be repeated under different names.")
        _integer(self.max_story_characters, "max_story_characters", 1, 1_000_000)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "DataPreparationConfig":
        data = _configuration_object(
            value,
            "data configuration",
            {"name", "sources", "schema_version", "max_story_characters"},
        ).copy()
        if "name" not in data or not isinstance(data.get("sources"), list):
            raise ValueError("Data configuration requires name and a sources list.")
        sources = []
        fields = set(DatasetSourceConfig.__dataclass_fields__)
        required = {
            "name",
            "source_id",
            "revision",
            "configuration",
            "text_column",
            "license",
            "splits",
        }
        for value in data.pop("sources"):
            source = _configuration_object(value, "source", fields)
            if missing := required - source.keys():
                raise ValueError(f"Missing source settings: {', '.join(sorted(missing))}.")
            sources.append(DatasetSourceConfig(**source))
        return cls(sources=tuple(sources), **data)

    @classmethod
    def load(cls, path: Path) -> "DataPreparationConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid data configuration JSON in {path}, line {exc.lineno}."
            ) from exc


# Transformer config
"""Validated architecture settings, independent of dataset and optimizer settings."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path



@dataclass(frozen=True, slots=True)
class TransformerConfig:
    vocab_size: int
    num_layers: int = 4
    num_heads: int = 4
    embedding_dim: int = 256
    context_length: int = 256
    feed_forward_expansion: int = 4
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initialization_std: float = 0.02
    tie_embeddings: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "num_layers",
            "num_heads",
            "embedding_dim",
            "context_length",
            "feed_forward_expansion",
        ):
            value = getattr(self, name)
            minimum = 2 if name == "vocab_size" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        for name in ("dropout", "layer_norm_epsilon", "initialization_std"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            if name == "dropout":
                if not 0 <= value < 1:
                    raise ValueError("dropout must satisfy 0 <= dropout < 1.")
            elif value <= 0:
                raise ValueError(f"{name} must be positive.")
        if type(self.tie_embeddings) is not bool:
            raise ValueError("tie_embeddings must be boolean.")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Transformer schema_version must be the integer 1.")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: object) -> "TransformerConfig":
        data = _configuration_object(
            value, "Transformer configuration", set(cls.__dataclass_fields__)
        )
        if "vocab_size" not in data:
            raise ValueError(
                "Transformer configuration requires vocab_size from the frozen tokenizer."
            )
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> "TransformerConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Transformer JSON in {path}, line {exc.lineno}.") from exc

    def save(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as output:
            output.write(self.to_json())


# Token data config
"""Separate token preparation identity from changeable batch-loading settings."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path



def _load_settings(path: Path, cls):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}, line {exc.lineno}.") from exc
    values = _configuration_object(payload, cls.__name__, set(cls.__dataclass_fields__))
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError(f"Missing required {cls.__name__} settings: {exc}.") from exc


@dataclass(frozen=True, slots=True)
class TokenPreparationConfig:
    name: str
    prepared_name: str
    dataset_fingerprint: str
    storage_dtype: str = "auto"
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field in ("name", "prepared_name"):
            value = getattr(self, field)
            if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value):
                raise ValueError(f"{field} must be a portable lowercase directory name.")
        if not isinstance(self.dataset_fingerprint, str) or not re.fullmatch(
            r"[a-f0-9]{64}", self.dataset_fingerprint
        ):
            raise ValueError("dataset_fingerprint must be a full SHA-256 hexadecimal fingerprint.")
        if self.storage_dtype not in ("auto", "uint8", "uint16", "uint32"):
            raise ValueError("storage_dtype must be auto, uint8, uint16 or uint32.")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Token preparation schema_version must be the integer 1.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> "TokenPreparationConfig":
        return _load_settings(path, cls)


@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    context_length: int = 256
    batch_size: int = 8
    shuffle: bool = True
    seed: int = 42
    num_workers: int = 0
    drop_last: bool = False

    def __post_init__(self) -> None:
        _integer(self.context_length, "context_length", 1, 1_000_000)
        _integer(self.batch_size, "batch_size", 1, 65_536)
        _integer(self.seed, "seed", 0, 2**32 - 1)
        _integer(self.num_workers, "num_workers", 0, 2)
        if type(self.shuffle) is not bool or type(self.drop_last) is not bool:
            raise ValueError("shuffle and drop_last must be boolean.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> "DataLoaderConfig":
        return _load_settings(path, cls)


# Training config
"""Immutable training budget, data identities, and optimizer settings."""

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path



@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run_id: str
    token_data_dir: str
    tokenizer_path: str
    tokenizer_fingerprint: str
    dataset_fingerprint: str
    token_data_fingerprint: str
    model: TransformerConfig
    train_stream: str = "train_tinystories"
    device: str = "cpu"
    seed: int = 42
    microbatch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_updates: int | None = None
    max_training_tokens: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    warmup_updates: int = 0
    minimum_lr_ratio: float = 0.1
    evaluation_interval: int = 100
    checkpoint_interval: int = 100
    max_validation_batches: int = 8
    console_interval: int = 10
    shuffle: bool = True
    num_workers: int = 0
    cpu_threads: int = 2
    deterministic_algorithms: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]{0,79}", self.run_id
        ):
            raise ValueError("run_id must be a portable lowercase directory name.")
        for name in ("token_data_dir", "tokenizer_path"):
            _validate_relative_directory(name, getattr(self, name))
        for name in ("tokenizer_fingerprint", "dataset_fingerprint", "token_data_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                raise ValueError(f"{name} must be a full SHA-256 fingerprint.")
        if not isinstance(self.model, TransformerConfig):
            raise ValueError("model must be a TransformerConfig.")
        if self.train_stream not in ("train_tinystories", "train"):
            raise ValueError("train_stream must be train_tinystories (Model A) or train (Model B).")
        RuntimeConfig(self.device)
        _integer(self.seed, "seed", 0, 2**32 - 1)
        for name in (
            "microbatch_size",
            "gradient_accumulation_steps",
            "evaluation_interval",
            "checkpoint_interval",
            "max_validation_batches",
            "console_interval",
            "cpu_threads",
        ):
            _integer(getattr(self, name), name, 1, 2**31 - 1)
        if (self.max_updates is None) == (self.max_training_tokens is None):
            raise ValueError("Set exactly one of max_updates or max_training_tokens.")
        for name in ("max_updates", "max_training_tokens"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name, 1, 2**63 - 1)
        if (
            self.max_training_tokens is not None
            and self.max_training_tokens % self.model.context_length
        ):
            raise ValueError("max_training_tokens must be a multiple of model.context_length.")
        _integer(self.warmup_updates, "warmup_updates", 0, self.total_updates - 1)
        for name in ("learning_rate", "weight_decay", "gradient_clip_norm", "minimum_lr_ratio"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < 0 or (name in ("learning_rate", "gradient_clip_norm") and value == 0):
                raise ValueError(
                    f"{name} must be {'positive' if name != 'weight_decay' else 'nonnegative'}."
                )
        if not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError("minimum_lr_ratio must be between zero and one.")
        for name in ("shuffle", "deterministic_algorithms"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean.")
        if type(self.num_workers) is not int or self.num_workers != 0:
            raise ValueError(
                "Training supports num_workers=0 only; prefetched sampler recovery is unsupported."
            )
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Training schema_version must be the integer 1.")

    @property
    def tokens_per_update(self) -> int:
        return self.microbatch_size * self.gradient_accumulation_steps * self.model.context_length

    @property
    def total_updates(self) -> int:
        if self.max_updates is not None:
            return self.max_updates
        return (self.max_training_tokens + self.tokens_per_update - 1) // self.tokens_per_update

    @property
    def total_tokens(self) -> int:
        return (
            self.max_training_tokens
            if self.max_training_tokens is not None
            else (self.max_updates * self.tokens_per_update)
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "TrainingConfig":
        data = dict(
            _configuration_object(value, "training configuration", set(cls.__dataclass_fields__))
        )
        if "model" not in data:
            raise ValueError("Training configuration requires model settings.")
        data["model"] = TransformerConfig.from_dict(data["model"])
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"Missing required training settings: {exc}.") from exc

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid training JSON in {path}, line {exc.lineno}.") from exc

    def save(self, path: Path) -> None:
        with Path(path).open("x", encoding="utf-8") as output:
            json.dump(self.to_dict(), output, indent=2, sort_keys=True)
            output.write("\n")


# Benchmark config
"""Bounded hardware experiments; architecture and effective batch stay fixed."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path



@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    output_dir: str
    training: TrainingConfig
    microbatch_sizes: tuple[int, ...] = (1, 2, 4)
    effective_batch_size: int = 8
    warmup_updates: int = 4
    measured_updates: int = 8
    sustained_updates: int = 120
    case_max_seconds: float = 45.0
    sustained_max_seconds: float = 60.0
    validation_windows: int = 16
    evaluation_interval: int = 4
    checkpoint_interval: int = 4
    max_process_rss_bytes: int = 4 * 1024**3
    max_mps_driver_bytes: int = 4 * 1024**3
    max_output_bytes: int = 512 * 1024**2
    minimum_system_free_percent: float = 20.0
    near_best_fraction: float = 0.9

    def __post_init__(self):
        _validate_relative_directory("benchmark_output_dir", self.output_dir)
        if not isinstance(self.training, TrainingConfig):
            raise ValueError("training must be a TrainingConfig.")
        if not isinstance(self.microbatch_sizes, tuple) or not 1 <= len(self.microbatch_sizes) <= 4:
            raise ValueError("Benchmark requires one to four microbatch sizes.")
        if any(type(size) is not int or not 1 <= size <= 8 for size in self.microbatch_sizes):
            raise ValueError("Benchmark microbatch sizes must be integers from 1 through 8.")
        if tuple(sorted(set(self.microbatch_sizes))) != self.microbatch_sizes:
            raise ValueError("Microbatch sizes must be unique and increasing.")
        limits = {
            "effective_batch_size": (1, 32),
            "warmup_updates": (1, 10),
            "measured_updates": (4, 32),
            "sustained_updates": (8, 200),
            "validation_windows": (1, 128),
            "evaluation_interval": (1, 100),
            "checkpoint_interval": (1, 100),
            "max_process_rss_bytes": (1, 2**40),
            "max_mps_driver_bytes": (1, 2**40),
            "max_output_bytes": (1, 2**40),
        }
        for name, (minimum, maximum) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
        if any(
            self.effective_batch_size % size or self.validation_windows % size
            for size in self.microbatch_sizes
        ):
            raise ValueError(
                "Effective batch and validation windows must divide by every microbatch size."
            )
        for name, minimum, maximum in (
            ("case_max_seconds", 1, 120),
            ("sustained_max_seconds", 1, 120),
            ("minimum_system_free_percent", 0, 100),
            ("near_best_fraction", 0.5, 1),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"Invalid benchmark {name}.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(
            _configuration_object(value, "benchmark configuration", set(cls.__dataclass_fields__))
        )
        if "training" not in data:
            raise ValueError("Benchmark requires training settings and artifact identities.")
        data["training"] = TrainingConfig.from_dict(data["training"])
        if "microbatch_sizes" in data:
            if not isinstance(data["microbatch_sizes"], list):
                raise ValueError("microbatch_sizes must be a JSON list.")
            data["microbatch_sizes"] = tuple(data["microbatch_sizes"])
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"Missing benchmark settings: {exc}.") from exc

    @classmethod
    def load(cls, path: Path):
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid benchmark JSON: {exc}.") from exc


# Comparison protocol
"""Frozen controls for the TinyStories baseline and the later matched-token comparison."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path



@dataclass(frozen=True, slots=True)
class ComparisonProtocol:
    model_a: TrainingConfig
    model_b: TrainingConfig
    document_path: str
    sample_updates: tuple[int, ...]
    prompts: tuple[str, ...]
    max_new_tokens: int = 128
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.model_a, TrainingConfig) or not isinstance(
            self.model_b, TrainingConfig
        ):
            raise ValueError("Both comparison runs require validated training configurations.")
        if self.model_a.train_stream != "train_tinystories" or self.model_b.train_stream != "train":
            raise ValueError(
                "Model A must use TinyStories only; Model B must use the combined stream."
            )
        if self.model_a.run_id == self.model_b.run_id:
            raise ValueError("Comparison runs require distinct run IDs.")
        for name, value in self.model_a.to_dict().items():
            if name not in ("run_id", "train_stream") and self.model_b.to_dict()[name] != value:
                raise ValueError(f"Unmatched comparison setting: {name}.")
        if self.model_a.max_training_tokens is None:
            raise ValueError(
                "The comparison must declare an explicit matched training-token budget."
            )
        _validate_relative_directory("document_path", self.document_path)
        if (
            not isinstance(self.sample_updates, tuple)
            or not 1 <= len(self.sample_updates) <= 6
            or any(
                type(n) is not int or not 1 <= n <= self.model_a.total_updates
                for n in self.sample_updates
            )
            or tuple(sorted(set(self.sample_updates))) != self.sample_updates
            or self.sample_updates[-1] != self.model_a.total_updates
        ):
            raise ValueError(
                "Sample updates must increase, stay within budget, and include the final update."
            )
        if (
            not isinstance(self.prompts, tuple)
            or not 1 <= len(self.prompts) <= 5
            or any(not isinstance(p, str) or not p.strip() or len(p) > 256 for p in self.prompts)
            or len(set(self.prompts)) != len(self.prompts)
        ):
            raise ValueError(
                "Specify one to five unique nonempty prompts, at most 256 characters each."
            )
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 128:
            raise ValueError("Diagnostic continuations are limited to 1–128 new tokens.")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported comparison protocol version.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(
            _configuration_object(value, "comparison protocol", set(cls.__dataclass_fields__))
        )
        for name in ("model_a", "model_b"):
            if name not in data:
                raise ValueError(f"Comparison protocol requires {name}.")
            data[name] = TrainingConfig.from_dict(data[name])
        for name in ("prompts", "sample_updates"):
            if not isinstance(data.get(name), list):
                raise ValueError(f"{name} must be a JSON list.")
            data[name] = tuple(data[name])
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"Incomplete comparison protocol: {exc}.") from exc

    @classmethod
    def load(cls, path: Path):
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid comparison JSON: {exc}.") from exc


# Evaluation config
"""Bounded, serializable controls for the frozen paired held-out evaluation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path



@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    comparison_protocol: str
    document_path: str
    output_directory: str
    device: str = "cpu"
    batch_size: int = 4
    document_stride: int = 128
    benchmark_repetitions: int = 3
    benchmark_warmups: int = 1
    benchmark_new_tokens: int = 128
    schema_version: int = 1

    def __post_init__(self):
        for name in ("comparison_protocol", "document_path", "output_directory"):
            _validate_relative_directory(name, getattr(self, name))
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"device must be one of {DEVICE_CHOICES}.")
        for name, limit in (
            ("batch_size", 16),
            ("document_stride", 256),
            ("benchmark_repetitions", 10),
            ("benchmark_warmups", 3),
            ("benchmark_new_tokens", 128),
            ("schema_version", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{name} must be an integer in [1, {limit}].")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = _configuration_object(
            json.loads(Path(path).read_text()), "evaluation", set(cls.__dataclass_fields__)
        )
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"Incomplete evaluation configuration: {exc}") from exc


# Generation config
"""Explicit modes and bounded input/output for reusable single-prompt inference."""

import math
from dataclasses import asdict, dataclass

MAX_NEW_TOKENS = 1024
MAX_PROMPT_CHARACTERS = 16384


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    mode: str = "greedy"
    max_new_tokens: int = 128
    temperature: float | None = None
    top_k: int | None = None
    seed: int | None = None
    unknown_policy: str = "error"

    def __post_init__(self):
        if self.mode not in ("greedy", "sample"):
            raise ValueError("mode must be 'greedy' or 'sample'.")
        if type(self.max_new_tokens) is not int or not 0 <= self.max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be an integer in [0, {MAX_NEW_TOKENS}].")
        if self.unknown_policy not in ("error", "replace"):
            raise ValueError("unknown_policy must be 'error' or 'replace'.")
        if self.temperature is not None and (
            type(self.temperature) not in (float, int) or not math.isfinite(self.temperature)
        ):
            raise ValueError("temperature must be finite.")
        if self.mode == "greedy":
            if self.temperature not in (None, 0) or self.top_k is not None or self.seed is not None:
                raise ValueError(
                    "Greedy mode uses argmax: temperature 0/omitted, no top_k or seed."
                )
        elif self.temperature is None or self.temperature <= 0:
            raise ValueError("Sample mode requires a finite temperature > 0.")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or omitted.")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("seed must be an integer in [0, 2**63 - 1] or omitted.")

    def validate_vocabulary(self, size):
        if self.top_k is not None and self.top_k > size:
            raise ValueError(f"top_k={self.top_k} exceeds vocabulary size {size}.")

    def to_dict(self):
        return asdict(self)


def validate_prompt(prompt):
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Prompt must be a nonempty string; this tokenizer has no BOS token.")
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError(f"Prompt exceeds the {MAX_PROMPT_CHARACTERS}-character input limit.")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in prompt):
        raise ValueError("Prompt contains an invalid Unicode surrogate code point.")
