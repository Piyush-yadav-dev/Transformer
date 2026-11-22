# Model registry
"""An explicit project-owned allowlist; request model IDs never resolve arbitrary paths."""

import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from transformer.configuration.settings import (
    _configuration_object,
    _validate_relative_directory,
)

REGISTRY_PATH = "configs/inference/model_registry.json"


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Registry SHA-256 values must contain 64 lowercase hexadecimal digits.")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: str
    sha256: str

    def __post_init__(self):
        _validate_relative_directory("artifact", self.path)
        if Path(self.path).as_posix() != self.path:
            raise ValueError("Registry artifact paths must be canonical project-relative paths.")
        _sha(self.sha256)


@dataclass(frozen=True, slots=True)
class ModelVersion:
    model_id: str
    kind: str
    comparison_member: str
    run_id: str
    checkpoint: ArtifactReference
    configuration: ArtifactReference
    tokenizer: ArtifactReference
    metrics: ArtifactReference
    tokenizer_fingerprint: str
    checkpoint_update: int

    def __post_init__(self):
        for name in ("model_id", "run_id"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(
                r"[a-z0-9][a-z0-9_-]{0,79}", getattr(self, name)
            ):
                raise ValueError(f"Registry {name} must be a portable lowercase identifier.")
        if (self.kind, self.comparison_member) not in (
            ("baseline", "model_a"),
            ("combined", "model_b"),
            ("combined", "extended_combined"),
            ("combined", "story_continuation"),
            ("english", "english"),
        ):
            raise ValueError("Registry kind/member must identify a supported model version.")
        for name, prefix in (
            ("checkpoint", f"runs/{self.run_id}/checkpoints/"),
            ("configuration", f"runs/{self.run_id}/"),
            (
                "tokenizer",
                "data/models/"
                if self.comparison_member == "story_continuation"
                else "data/tokens/",
            ),
            ("metrics", "artifacts/"),
        ):
            ref = getattr(self, name)
            if not isinstance(ref, ArtifactReference) or not ref.path.startswith(prefix):
                raise ValueError(f"Registry {name} must stay inside {prefix}.")
        if not self.checkpoint.path.endswith(".pt"):
            raise ValueError("Registered checkpoints must be local .pt project artifacts.")
        _sha(self.tokenizer_fingerprint)
        if type(self.checkpoint_update) is not int or self.checkpoint_update < 1:
            raise ValueError("checkpoint_update must be a positive trained update.")

    def to_dict(self):
        return asdict(self)


class ModelRegistry:
    """The root and checked-in catalog are trusted operator configuration, not request input."""

    def __init__(self, project_root):
        self.root = Path(project_root).resolve(strict=True)
        try:
            with self._open_local(REGISTRY_PATH, limit=1 << 20) as source:
                raw = source.read()
            self.fingerprint = hashlib.sha256(raw).hexdigest()
            data = _configuration_object(
                json.loads(raw), "model registry", {"schema_version", "models"}
            )
            if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
                raise ValueError("Unsupported model registry schema_version.")
            if not isinstance(data.get("models"), list) or not data["models"]:
                raise ValueError("Model registry must contain an explicit nonempty model list.")
            models = {}
            for row in data["models"]:
                values = dict(
                    _configuration_object(
                        row, "model version", set(ModelVersion.__dataclass_fields__)
                    )
                )
                for key in ("checkpoint", "configuration", "tokenizer", "metrics"):
                    values[key] = ArtifactReference(
                        **_configuration_object(values[key], key, {"path", "sha256"})
                    )
                version = ModelVersion(**values)
                if version.model_id in models:
                    raise ValueError(f"Duplicate stable model ID: {version.model_id}.")
                models[version.model_id] = version
            self._models = MappingProxyType(models)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Cannot load trusted project registry {REGISTRY_PATH}: {exc}"
            ) from exc

    def resolve(self, model_id):
        if not isinstance(model_id, str) or model_id not in self._models:
            raise ValueError(
                f"Unknown model ID {model_id!r}. Available IDs: {', '.join(self._models)}."
            )
        return self._models[model_id]

    def list_models(self):
        return [v.to_dict() for v in self._models.values()]

    def local_path(self, relative):
        _validate_relative_directory("artifact", relative)
        path = self.root
        for part in Path(relative).parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"Symbolic links are not trusted registry artifacts: {relative}.")
        if not path.is_file() or not path.resolve(strict=True).is_relative_to(self.root):
            raise ValueError(f"Missing regular local project artifact: {relative}.")
        return path

    @contextmanager
    def _open_local(self, relative, *, limit):
        path = self.local_path(relative)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError(f"Artifact is not a bounded regular file: {relative}.")
            yield source

    @contextmanager
    def verified_file(self, reference, *, limit=512 << 20):
        try:
            with self._open_local(reference.path, limit=limit) as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != reference.sha256:
                    raise ValueError(f"SHA-256 mismatch: {reference.path}.")
                source.seek(0)
                # Hash aur deserialize ek hi open file se; path dobara kholna zaroori nahi.
                yield source
        except OSError as exc:
            raise ValueError(f"Cannot read registered artifact {reference.path}: {exc}") from exc

    def read_json(self, reference):
        with self.verified_file(reference, limit=16 << 20) as source:
            try:
                return json.load(source)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Malformed registered JSON {reference.path}: {exc}") from exc


# Sampling
"""Stable top-k selection and isolated CPU-generator sampling from model logits."""

import torch


def select_next_token(logits, settings, generator=None):
    # MPS float64 support nahi karta: pehle CPU transfer, phir probability precision badhao.
    values = logits.detach().to(device="cpu").to(dtype=torch.float64)
    if values.ndim != 1 or not values.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError("Next-token logits must be a finite nonempty vector.")
    settings.validate_vocabulary(values.numel())
    if settings.mode == "greedy":
        return int(values.argmax().item())
    if generator is None or generator.device.type != "cpu":
        raise ValueError("Sampling requires a private CPU torch.Generator.")
    # Equal scores par chhota token ID pehle: top-k mein exactly k IDs rahenge.
    indices = torch.argsort(values, descending=True, stable=True)
    if settings.top_k is not None:
        indices = indices[: settings.top_k]
    values = values[indices]
    probabilities = torch.softmax((values - values.max()) / settings.temperature, dim=0)
    chosen = torch.multinomial(probabilities, 1, generator=generator)
    return int(indices[chosen.item()].item())


# Checkpoint loading
"""Reconstruct only registered, checksum-pinned local Transformer checkpoints."""

import pickle
from dataclasses import dataclass

import torch

from transformer.configuration.settings import TrainingConfig
from transformer.data.preparation import canonical_json
from transformer.model import TransformerLanguageModel
from transformer.tokenization.character_tokenizer import CharacterTokenizer


@dataclass
class LoadedModel:
    version: object
    model: TransformerLanguageModel
    tokenizer: CharacterTokenizer
    metadata: dict


def _load_story_model(registry, version, device):
    from transformer.inference.continuation_model import add_adapters, load_base

    config = registry.read_json(version.configuration)
    record = registry.read_json(version.metrics)
    if (
        config["model_id"] != version.model_id
        or record["selected_update"] != version.checkpoint_update
        or version.tokenizer.sha256 != version.tokenizer_fingerprint
        or record["adapter_sha256"] != version.checkpoint.sha256
    ):
        raise ValueError("Story training evidence does not match the registry.")
    base_files = config["base_files"]
    base = registry.local_path(version.tokenizer.path).parent
    if set(base_files) != {path.name for path in base.iterdir() if path.is_file()}:
        raise ValueError("Story base files differ from the pinned inventory.")
    from transformer.inference.service import ArtifactReference

    for name, digest in base_files.items():
        path = (base.relative_to(registry.root) / name).as_posix()
        with registry.verified_file(
            ArtifactReference(path, digest),
            limit=(4 << 30) if name.endswith(".safetensors") else (16 << 20),
        ):
            pass
    model, tokenizer = load_base(base, device)
    parameters = add_adapters(model)
    with registry.verified_file(version.checkpoint) as source:
        adapter = torch.load(source, map_location="cpu", weights_only=True)
    if set(adapter) != set(parameters):
        raise ValueError("Missing or unexpected story adapter parameters.")
    with torch.no_grad():
        for name, param in parameters.items():
            value = adapter[name]
            if (
                value.shape != param.shape
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid story adapter tensor.")
            param.copy_(value)
    model.eval().requires_grad_(False)
    tokenizer.fingerprint = version.tokenizer_fingerprint
    return LoadedModel(
        version,
        model,
        tokenizer,
        {
            **version.to_dict(),
            "device": str(device),
            "architecture": config["architecture"],
            "parameter_count": sum(param.numel() for param in model.parameters()),
            "dtype": str(model.dtype).removeprefix("torch."),
            "instruction": config.get("instruction"),
        },
    )


def load_registered_model(registry, version, device):
    try:
        if version.comparison_member == "story_continuation":
            return _load_story_model(registry, version, device)
        config = TrainingConfig.from_dict(registry.read_json(version.configuration))
        raw_tokenizer = registry.read_json(version.tokenizer)
        if not isinstance(raw_tokenizer, dict) or not isinstance(
            raw_tokenizer.get("characters"), list
        ):
            raise ValueError("Malformed registered character tokenizer.")
        tokenizer = CharacterTokenizer(tuple(raw_tokenizer["characters"]))
        if (
            canonical_json(raw_tokenizer) != canonical_json(tokenizer.to_dict())
            or tokenizer.fingerprint != version.tokenizer_fingerprint
            or tokenizer.fingerprint != config.tokenizer_fingerprint
        ):
            raise ValueError("Incompatible tokenizer fingerprint or format metadata.")
        if tokenizer.vocab_size != config.model.vocab_size:
            raise ValueError("Tokenizer vocabulary size differs from saved architecture.")
        expected_stream = (
            "train_tinystories" if version.kind in {"baseline", "english"} else "train"
        )
        if config.run_id != version.run_id or config.train_stream != expected_stream:
            raise ValueError("Registry identity differs from saved run configuration.")
        metrics = registry.read_json(version.metrics)
        record = metrics["runs"][version.comparison_member]
        if (
            metrics["identity"]["tokenizer_fingerprint"] != tokenizer.fingerprint
            or record["checkpoint"]["sha256"] != version.checkpoint.sha256
            or record["configuration"] != config.to_dict()
            or record["selected_update"] != version.checkpoint_update
        ):
            raise ValueError("Registered metrics do not describe this checkpoint/configuration.")
        validation_only = version.comparison_member in {"english", "extended_combined"}
        if validation_only and metrics.get("scope") != "validation_and_diagnostic_samples":
            raise ValueError("Extended model requires a validation and diagnostic record.")
        packed = {} if validation_only else metrics["scores"][version.comparison_member]["packed"]
        with registry.verified_file(version.checkpoint) as source:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or type(payload.get("format_version")) is not int
            or payload["format_version"] != 1
        ):
            raise ValueError("Unsupported project checkpoint format.")
        if payload["identity"]["configuration"] != config.to_dict():
            raise ValueError("Checkpoint architecture/training configuration is incompatible.")
        if (
            payload["position"]["updates"] != version.checkpoint_update
            or payload["position"]["tokens"] != record["selected_training_tokens"]
        ):
            raise ValueError("Checkpoint training position differs from registry/metrics.")
        state = payload["model"]
        if (
            not isinstance(state, dict)
            or not state
            or any(
                not isinstance(t, torch.Tensor)
                or t.dtype != torch.float32
                or not bool(torch.isfinite(t).all())
                for t in state.values()
            )
        ):
            raise ValueError("Model weights must be finite float32 tensors.")
        if config.model.tie_embeddings and not torch.equal(
            state["embeddings.token.weight"], state["vocabulary_projection.weight"]
        ):
            raise ValueError("Tied embedding/output weights disagree in checkpoint.")
        # Constructor RNG ko caller se alag rakho; saved weights hi inference mein use honge.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = TransformerLanguageModel(config.model).float()
        model.load_state_dict(state, strict=True)
        parameter_count = model.trainable_parameter_count()
        if parameter_count != record["parameter_count"]:
            raise ValueError("Parameter count differs from registered evaluation record.")
        model.requires_grad_(False)
        model.eval()
        del payload, state
        model.to(device)
        return LoadedModel(
            version,
            model,
            tokenizer,
            {
                **version.to_dict(),
                "architecture": config.model.to_dict(),
                "parameter_count": parameter_count,
                "device": str(device),
                "dtype": "float32",
                "selected_training_tokens": record["selected_training_tokens"],
                "selection_rule": record["selection_rule"],
                "held_out_primary": {
                    k: v for k, v in packed.items() if k in ("combined", "sources")
                },
            },
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        raise ValueError(f"Cannot load registered model {version.model_id}: {exc}") from exc


# Generation service
"""One-model cache and a synchronous frontend-independent inference service."""

import copy
import gc
import secrets
import threading
import time
from contextlib import contextmanager

import torch

from transformer.configuration.settings import GenerationConfig, validate_prompt
from transformer.runtime.device import select_device
from transformer.training.engine import synchronize

_RUNTIME_LOCK = threading.RLock()


@contextmanager
def _runtime(cpu_threads):
    # PyTorch thread/dtype settings process-wide hain; service calls ko serialize karo.
    with _RUNTIME_LOCK:
        threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
        try:
            torch.set_num_threads(cpu_threads)
            torch.set_default_dtype(torch.float32)
            yield
        finally:
            torch.set_default_dtype(dtype)
            torch.set_num_threads(threads)


class GenerationService:
    """Reuse one instance per application process; calls and switches are serialized."""

    def __init__(self, project_root, *, device="auto", cpu_threads=2):
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
            raise ValueError("cpu_threads must be an integer in [1, 8].")
        self.registry = ModelRegistry(project_root)
        self.selection = select_device(device)
        self.cpu_threads = cpu_threads
        self._lock = threading.RLock()
        self._loaded = None
        self._file_stamps = None

    def list_models(self):
        return self.registry.list_models()

    def cache_info(self):
        with self._lock:
            return {
                "capacity": 1,
                "cached_models": int(self._loaded is not None),
                "model_id": self._loaded.version.model_id if self._loaded else None,
                "device": str(self.selection.device),
            }

    def _stamps(self, version):
        result = []
        for ref in (version.checkpoint, version.configuration, version.tokenizer, version.metrics):
            info = self.registry.local_path(ref.path).stat()
            result.append(
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            )
        return tuple(result)

    def _unload(self):
        if self._loaded is not None:
            synchronize(self.selection.device)
            self._loaded = None
            self._file_stamps = None
            gc.collect()
            if self.selection.device.type == "mps":
                torch.mps.empty_cache()

    def unload(self):
        with self._lock, _runtime(self.cpu_threads):
            self._unload()

    close = unload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _load(self, model_id):
        version = self.registry.resolve(model_id)
        if self._loaded is not None and self._loaded.version == version:
            try:
                if self._stamps(version) == self._file_stamps:
                    return True, 0.0
            except (ValueError, OSError):
                self._unload()
                raise
        started = time.perf_counter()
        # Switch se pehle purana model release karo; do MPS models cache mein mat rakho.
        self._unload()
        self._loaded = load_registered_model(self.registry, version, self.selection.device)
        try:
            self._file_stamps = self._stamps(version)
            synchronize(self.selection.device)
        except (ValueError, RuntimeError, OSError):
            self._unload()
            raise
        return False, time.perf_counter() - started

    def load(self, model_id):
        """Optional prewarm; returns JSON metadata, never a mutable model reference."""
        with self._lock, _runtime(self.cpu_threads):
            cache_hit, seconds = self._load(model_id)
            return {
                **copy.deepcopy(self._loaded.metadata),
                "cache_hit": cache_hit,
                "model_load_seconds": seconds,
                "registry_fingerprint": self.registry.fingerprint,
            }

    def generate(self, model_id, prompt, settings=None):
        settings = GenerationConfig() if settings is None else settings
        if not isinstance(settings, GenerationConfig):
            raise ValueError("settings must be a GenerationConfig object.")
        validate_prompt(prompt)
        with self._lock, _runtime(self.cpu_threads):
            cache_hit, load_seconds = self._load(model_id)
            loaded = self._loaded
            settings.validate_vocabulary(loaded.tokenizer.vocab_size)
            synchronize(self.selection.device)
            started = time.perf_counter()
            generated = []
            effective_seed = None
            generator = None
            if settings.mode == "sample":
                effective_seed = (
                    settings.seed if settings.seed is not None else secrets.randbits(63)
                )
                generator = torch.Generator(device="cpu").manual_seed(effective_seed)
            story_model = loaded.version.comparison_member == "story_continuation"
            if story_model:
                from transformer.inference.continuation_model import INSTRUCTION, continue_opening

                continuation, generated, stop_reason, prompt_tokens, capacity = continue_opening(
                    loaded.model,
                    loaded.tokenizer,
                    prompt,
                    settings,
                    generator,
                    instruction=loaded.metadata.get("instruction") or INSTRUCTION,
                )
                unknown_count = 0
            else:
                encoded = loaded.tokenizer.encode(prompt, unknown=settings.unknown_policy)
                unknown_count = encoded.count(loaded.tokenizer.unk_id)
                prompt_tokens = len(encoded)
                capacity = loaded.model.config.context_length
                context = encoded[-capacity:]
                stop_reason = "max_new_tokens"
                with torch.inference_mode():
                    for _ in range(settings.max_new_tokens):
                        x = torch.tensor([context], dtype=torch.int64, device=self.selection.device)
                        logits = loaded.model(x).logits[0, -1]
                        token = select_next_token(logits, settings, generator)
                        generated.append(token)
                        if token == loaded.tokenizer.eos_id:
                            stop_reason = "eos"
                            break
                        # Learned positions har cropped window mein 0..L-1 se restart hote hain.
                        context = (context + [token])[-capacity:]
                continuation = loaded.tokenizer.decode(generated, skip_eos=True)
            synchronize(self.selection.device)
            elapsed = time.perf_counter() - started
            return {
                "model_id": model_id,
                "model_kind": loaded.version.kind,
                "checkpoint": {
                    "path": loaded.version.checkpoint.path,
                    "sha256": loaded.version.checkpoint.sha256,
                    "update": loaded.version.checkpoint_update,
                },
                "registry_fingerprint": self.registry.fingerprint,
                "tokenizer_fingerprint": loaded.tokenizer.fingerprint,
                "prompt": prompt,
                "continuation": continuation,
                "generated_token_ids": generated,
                "generated_token_count": len(generated),
                "stop_reason": stop_reason,
                "sampling_settings": {
                    **settings.to_dict(),
                    "temperature": 0 if settings.mode == "greedy" else settings.temperature,
                    "effective_seed": effective_seed,
                    "sampling_device": "cpu",
                    "stop_at_eos": True,
                },
                "prompt_token_count": prompt_tokens,
                "unknown_prompt_tokens": unknown_count,
                "initial_prompt_tokens_truncated": max(0, prompt_tokens - capacity),
                "context_length": capacity,
                "position_policy": (
                    "Instruction and opening are encoded together; reuse KV cache."
                    if story_model
                    else "Keep last context_length IDs; recompute positions 0..L-1 each step; "
                    "no KV cache."
                ),
                "special_token_policy": (
                    "EOS counts as a generated token but is not rendered; UNK renders as <unk>."
                ),
                "device": str(self.selection.device),
                "device_reason": self.selection.reason,
                "runtime": {
                    "torch": str(torch.__version__),
                    "dtype": loaded.metadata.get("dtype", "float32"),
                    "cpu_threads": self.cpu_threads,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                },
                "cache_hit": cache_hit,
                "model_load_seconds": load_seconds,
                "generation_duration_seconds": elapsed,
                "timing_definition": (
                    "Generation includes prompt encoding, forward passes, transfers, sampling, "
                    "decoding and device synchronization. Excludes model loading and lock wait; "
                    "load time includes eviction/verification on a cache miss."
                ),
                "reproducibility": (
                    "Private CPU RNG; reproducible with fixed seed, model, inputs and "
                    "runtime/device. "
                    "Cross-device/release bitwise equivalence is not guaranteed."
                ),
            }
