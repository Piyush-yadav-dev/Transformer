# Analytics
"""Bounded, read-only projections of registry-linked experiment evidence.

No checkpoint deserialization, arbitrary request paths, model allocation or training.
"""
import hashlib
import json
import math

from transformer.configuration.settings import TransformerConfig

LIMIT = 16 << 20


def number(value, *, integer=False, minimum=0):
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError("Expected a numeric metric.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum:
        raise ValueError("Metric is nonfinite or out of range.")
    return value


def parse_log(raw):
    """Retain valid rows; report rejected lines and gaps rather than inventing zeros."""
    rows, warnings = [], []
    previous = 0
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected an object")
            if row.get("event") != "update":
                continue
            update = number(row["optimizer_update"], integer=True, minimum=1)
            tokens = number(row["tokens_processed"], integer=True, minimum=1)
            loss = number(row["training_loss"])
            validation = row.get("validation_loss")
            if validation is not None:
                number(validation)
            if update <= previous or (rows and tokens <= rows[-1]["tokens"]):
                raise ValueError("Duplicate or out-of-order position")
            if len(rows) >= 10000:
                warnings.append("Chart limited to the first 10,000 valid updates; later records are omitted.")
                break
            rows.append({"update": update, "tokens": tokens, "loss": loss,
                         "validation_loss": validation, "gap_before": update != previous + 1})
            previous = update
        except (ValueError, KeyError, TypeError):
            warnings.append(f"Log line {line_number}: malformed, nonfinite or nonmonotonic update omitted.")
    if not rows:
        warnings.append("No valid training updates available.")
    if any(row["gap_before"] for row in rows):
        warnings.append("Training updates contain gaps; the chart does not join across them.")
    return rows, warnings


def _json(raw):
    def invalid(_):
        raise ValueError("Nonfinite JSON value")
    return json.loads(raw, parse_constant=invalid)


def _read(registry, path, evidence, expected=None):
    with registry._open_local(path, limit=LIMIT) as source:
        raw = source.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError("Artifact exceeds read limit")
    digest = hashlib.sha256(raw).hexdigest()
    evidence.append({"path": path, "sha256": digest, "matches_saved_hash": digest == expected if expected else None})
    if expected and digest != expected:
        raise ValueError("Artifact differs from the saved evaluation snapshot")
    return raw


def _metric_sources(scores):
    result = {}
    for name, row in scores["sources"].items():
        loss = number(row["cross_entropy_nats"])
        count = number(row["target_tokens"], integer=True, minimum=1)
        perplexity = row.get("perplexity")
        if perplexity is not None:
            number(perplexity, minimum=1)
            if not math.isclose(math.log(perplexity), loss, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Perplexity does not match exp(mean loss)")
        result[name] = {"loss": loss, "perplexity": perplexity, "target_tokens": count}
    return result


def _extended_model(registry, version, config, tokenizer, evaluation, run, evidence):
    if (
        evaluation.get("scope") != "validation_and_diagnostic_samples"
        or run["configuration"] != config
        or config["run_id"] != version.run_id
        or config["train_stream"] != "train"
        or run["checkpoint"]["sha256"] != version.checkpoint.sha256
        or run["selected_update"] != version.checkpoint_update
        or evaluation["identity"]["tokenizer_fingerprint"] != version.tokenizer_fingerprint
    ):
        raise ValueError("Combined checkpoint evidence does not match the registry.")
    selected = run["selected_validation"]
    if selected["update"] != version.checkpoint_update:
        raise ValueError("Validation evidence describes a different checkpoint.")
    warnings, training = [], []
    try:
        raw = _read(registry, f"runs/{version.run_id}/metrics.jsonl", evidence)
        training, warnings = parse_log(raw)
    except (OSError, ValueError):
        warnings.append("Training journal is unavailable.")
    validation = [
        {"update": row["update"], "tokens": row["tokens"], "loss": row["validation_loss"]}
        for row in training if row["validation_loss"] is not None
    ]
    return {
        "model_id": version.model_id,
        "run_id": version.run_id,
        "kind": version.kind,
        "status": "available",
        "scope": evaluation["scope"],
        "architecture": config["model"],
        "tokenizer_type": tokenizer["kind"],
        "parameter_count": number(run["parameter_count"], integer=True, minimum=1),
        "training_tokens": number(run["selected_training_tokens"], integer=True, minimum=1),
        "checkpoint_update": version.checkpoint_update,
        "validation_loss": number(selected["loss"]),
        "diagnostics": evaluation["diagnostics"],
        "training": training,
        "validation": validation,
        "warnings": warnings,
    }


def _model(registry, version, shared_registered):
    evidence, warnings = [], []
    def registered(reference):
        key = (reference.path, reference.sha256)
        if key not in shared_registered:
            value = _json(_read(registry, reference.path, evidence, reference.sha256))
            shared_registered[key] = (value, dict(evidence[-1]))
        else:
            value, provenance = shared_registered[key]
            evidence.append(dict(provenance))
        return value

    config = registered(version.configuration)
    if version.comparison_member == "story_continuation":
        record = registered(version.metrics)
        if (record["selected_update"] != version.checkpoint_update
                or record["adapter_sha256"] != version.checkpoint.sha256
                or config["model_id"] != version.model_id):
            raise ValueError("Story checkpoint evidence does not match the registry.")
        return {
            "model_id": version.model_id, "run_id": version.run_id,
            "kind": version.kind, "status": "available", "scope": "story_continuation",
            "architecture": config["architecture"], "base_model": config["base_model"],
            "validation_before": record["validation_before"],
            "validation_loss": record["validation_after"],
            "checkpoint_update": version.checkpoint_update,
            "trainable_parameters": record["trainable_parameters"],
            "diagnostics": [{"prompt": row["opening"], "continuation": row["continuation"]}
                            for row in record["samples"]],
            "warnings": [],
        }
    architecture = config["model"]
    TransformerConfig(**architecture)
    tokenizer = registered(version.tokenizer)
    if (tokenizer["fingerprint"] != version.tokenizer_fingerprint
            or config["tokenizer_fingerprint"] != version.tokenizer_fingerprint
            or tokenizer["kind"] != "character"
            or len(tokenizer["characters"]) + len(tokenizer["special_tokens"]) != architecture["vocab_size"]):
        raise ValueError("Configuration/tokenizer identity mismatch")
    evaluation = registered(version.metrics)
    run = evaluation["runs"][version.comparison_member]
    if version.comparison_member == "extended_combined":
        return _extended_model(registry, version, config, tokenizer, evaluation, run, evidence)
    if (run["configuration"] != config or run["run_id"] != version.run_id
            or run["checkpoint"]["sha256"] != version.checkpoint.sha256
            or evaluation["identity"]["tokenizer_fingerprint"] != version.tokenizer_fingerprint
            or evaluation["test_results_used_for_selection"] is not False
            or evaluation["status"] != "completed"):
        raise ValueError("Evaluation identity or selection evidence mismatch")
    saved = run["run_files_before"]
    base = f"runs/{version.run_id}/"
    # Logs remain inspectable when interrupted; metrics from other artifacts stay independent.
    log_path = base + "metrics.jsonl"
    try:
        raw = _read(registry, log_path, evidence)
        points, issues = parse_log(raw)
        warnings.extend(issues)
        evidence[-1]["matches_saved_hash"] = hashlib.sha256(raw).hexdigest() == saved.get("metrics.jsonl")
        if not evidence[-1]["matches_saved_hash"]:
            warnings.append("Training log differs from the frozen evaluation snapshot; chart is current, unverified log evidence.")
        if not points or points[-1]["update"] != version.checkpoint_update:
            warnings.append("Training log is partial or extends beyond the registered checkpoint.")
    except (OSError, ValueError, UnicodeError):
        points = []
        warnings.append("Training log is missing, unreadable or too large.")
    validation = []
    try:
        history = _json(_read(registry, base + "validation_history.json", evidence,
                              saved["validation_history.json"]))["history"]
        if history != run["validation_history"]:
            raise ValueError("Validation history mismatch")
        for row in history:
            validation.append({"update": number(row["optimizer_update"], integer=True, minimum=1),
                               "tokens": number(row["tokens_processed"], integer=True, minimum=1),
                               "loss": number(row["validation_loss"])})
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        validation = []
        warnings.append("Validation history is missing, malformed or differs from saved evidence.")
    duration = None
    timing = run.get("training_timing_definition")
    try:
        summary = _json(_read(registry, base + "training_summary.json", evidence,
                              saved["training_summary.json"]))
        duration = number(summary["training_session_wall_seconds"])
        if duration != run["training_session_wall_seconds"]:
            raise ValueError("Training duration mismatch")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        duration = None
        warnings.append("Training duration unavailable: missing or inconsistent run summary.")

    manifest = None
    try:
        manifest = _json(_read(registry, config["token_data_dir"] + "/manifest.json", evidence))
        if (manifest["token_data_fingerprint"] != config["token_data_fingerprint"]
                or manifest["dataset_fingerprint"] != config["dataset_fingerprint"]
                or manifest["tokenizer_fingerprint"] != version.tokenizer_fingerprint
                or manifest["prepared_sources"] != run["prepared_sources"]):
            raise ValueError("Dataset manifest mismatch")
        manifest = {"sources": manifest["prepared_sources"], "boundary_policy": manifest["boundary_policy"],
                    "vocabulary_policy": manifest["vocabulary_fit"]["policy"],
                    "fingerprint": manifest["dataset_fingerprint"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        manifest = None
        warnings.append("Dataset manifest unavailable or inconsistent; source details omitted.")

    sources = _metric_sources(evaluation["scores"][version.comparison_member]["packed"])
    exposure = run["training_exposure"]["source_target_tokens"]
    for count in exposure.values():
        number(count, integer=True)
    actual_tokens = number(run["selected_training_tokens"], integer=True, minimum=1)
    if sum(exposure.values()) != actual_tokens:
        raise ValueError("Source exposures do not sum to selected training tokens")
    benchmark = evaluation.get("generation_benchmark", {})
    throughput = None
    try:
        benchmark_row = benchmark["summaries"][version.comparison_member]
        throughput = number(benchmark_row["pooled_tokens_per_second"])
        if not math.isclose(throughput, benchmark_row["new_tokens"] / benchmark_row["elapsed_seconds"]):
            raise ValueError("Throughput mismatch")
    except (ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError):
        throughput = None
        warnings.append("Generation throughput unavailable or inconsistent.")
    return {"model_id": version.model_id, "kind": version.kind, "run_id": version.run_id,
            "status": "available", "architecture": architecture, "tokenizer_type": tokenizer["kind"],
            "parameter_count": number(run["parameter_count"], integer=True, minimum=1),
            "token_budget": config.get("max_training_tokens"), "update_budget": config.get("max_updates"),
            "selection_rule": run["selection_rule"], "checkpoint_update": version.checkpoint_update,
            "training_tokens": actual_tokens, "exposure": exposure, "training_seconds": duration,
            "timing_definition": timing, "generation_tokens_per_second": throughput,
            "benchmark_definition": benchmark.get("definition"), "evaluation_environment": evaluation["environment"],
            "test_sources": sources, "training": points, "validation": validation,
            "manifest": manifest, "evidence": evidence, "warnings": warnings,
            "compatibility": {"evaluation_sha256": version.metrics.sha256,
                              "tokenizer": version.tokenizer_fingerprint,
                              "dataset": config["dataset_fingerprint"], "architecture": architecture,
                              "budget": config.get("max_training_tokens"), "selection": run["selection_rule"]}}


def load_analytics(root):
    # Lazy registry import keeps shell construction independent of inference and torch.
    from transformer.inference.service import ModelRegistry
    try:
        registry = ModelRegistry(root)
    except (OSError, ValueError):
        return {"models": [], "compatible": False, "warnings": ["Experiment registry unavailable. Restore the project registry and reload."]}
    models = []
    # Sirf is request mein verified JSON share karo; next request files dobara padhega.
    shared_registered = {}
    for item in registry.list_models():
        if item["kind"] == "english":
            continue
        version = registry.resolve(item["model_id"])
        try:
            models.append(_model(registry, version, shared_registered))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
            models.append({"model_id": version.model_id, "run_id": version.run_id, "kind": version.kind,
                           "status": "unavailable", "warnings": ["Registered experiment evidence is missing, malformed or incompatible. Restore its original artifacts."]})
    valid = [m for m in models if m["status"] == "available"]
    compatible = (len(valid) == 2 and {m["kind"] for m in valid} == {"baseline", "combined"}
                  and valid[0]["compatibility"] == valid[1]["compatibility"]
                  and valid[0]["training_tokens"] == valid[1]["training_tokens"]
                  and {k:v["target_tokens"] for k,v in valid[0]["test_sources"].items()}
                  == {k:v["target_tokens"] for k,v in valid[1]["test_sources"].items()})
    return {"models": models, "compatible": compatible,
            "warnings": [] if compatible or any(m.get("scope") for m in valid) else ["A paired comparison is unavailable: two compatible registered runs and matching evaluation evidence are required."]}


# Schemas
"""Strict public API types; project paths and internal tensors never cross this boundary."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from transformer.configuration.settings import (
    MAX_NEW_TOKENS,
    MAX_PROMPT_CHARACTERS,
    GenerationConfig,
    validate_prompt,
)

CONTRACT_VERSION = "2.0.0"
ModelID = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")]
SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
TokenLimit = Annotated[int, Field(ge=0, le=MAX_NEW_TOKENS)]
Seed = Annotated[str, Field(pattern=r"^(0|[1-9][0-9]{0,18})$")]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class GreedySettings(StrictSchema):
    mode: Literal["greedy"]
    max_new_tokens: TokenLimit = 128
    unknown_policy: Literal["error", "replace"] = "error"


class SampleSettings(StrictSchema):
    mode: Literal["sample"]
    max_new_tokens: TokenLimit = 128
    temperature: Annotated[float, Field(gt=0)]
    top_k: Annotated[int, Field(ge=1)] | None = None
    seed: Seed | None = None
    unknown_policy: Literal["error", "replace"] = "error"

    @field_validator("seed")
    @classmethod
    def seed_in_backend_range(cls, value):
        if value is not None and int(value) >= 2**63:
            raise ValueError("seed must be at most 9223372036854775807.")
        return value


class GenerationRequest(StrictSchema):
    model_id: ModelID
    prompt: Annotated[str, Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)]
    settings: Annotated[GreedySettings | SampleSettings, Field(discriminator="mode")]

    @field_validator("prompt")
    @classmethod
    def valid_opening(cls, value):
        validate_prompt(value)
        if not value.strip():
            raise ValueError("Write an opening or choose an example to begin.")
        return value

    def generation_config(self):
        values = self.settings.model_dump()
        if values.get("seed") is not None:
            values["seed"] = int(values["seed"])
        return GenerationConfig(**values)


class CheckpointIdentity(StrictSchema):
    sha256: SHA256
    update: Annotated[int, Field(ge=1)]


class EffectiveSettings(StrictSchema):
    mode: Literal["greedy", "sample"]
    max_new_tokens: TokenLimit
    temperature: Annotated[float, Field(ge=0)]
    top_k: int | None
    seed: Seed | None
    effective_seed: Seed | None
    unknown_policy: Literal["error", "replace"]
    stop_at_eos: Literal[True]


class GenerationResult(StrictSchema):
    request_id: str
    prompt: str
    continuation: str
    model_id: ModelID
    model_kind: Literal["baseline", "combined", "english"]
    checkpoint: CheckpointIdentity
    sampling_settings: EffectiveSettings
    generated_token_count: TokenLimit
    stop_reason: Literal["eos", "max_new_tokens"]
    generation_duration_seconds: Annotated[float, Field(ge=0)]
    model_load_seconds: Annotated[float, Field(ge=0)]
    initial_prompt_tokens_truncated: Annotated[int, Field(ge=0)]
    unknown_prompt_tokens: Annotated[int, Field(ge=0)]
    context_length: Annotated[int, Field(ge=1)]
    tokenizer_fingerprint: SHA256
    registry_fingerprint: SHA256
    device: Literal["cpu", "mps"]
    cache_hit: bool


class ModelSummary(StrictSchema):
    model_id: ModelID
    kind: Literal["baseline", "combined", "english"]
    display_name: str
    availability: Literal["registered", "unavailable"]
    unavailable_reason: str | None
    checkpoint: CheckpointIdentity
    tokenizer_fingerprint: SHA256


class ModelCatalog(StrictSchema):
    models: list[ModelSummary]
    registry_fingerprint: SHA256


class ServiceStatus(StrictSchema):
    application: Literal["Transformer"] = "Transformer"
    phase: Literal[15] = 15
    interface: Literal["connected_generate"] = "connected_generate"
    generation_enabled: bool
    busy: bool
    contract_version: Literal["2.0.0"] = CONTRACT_VERSION
    device: Literal["cpu", "mps"] | None


class ErrorDetail(StrictSchema):
    code: Literal[
        "invalid_request",
        "model_not_found",
        "artifact_unavailable",
        "busy",
        "inference_failed",
        "service_unavailable",
        "request_too_large",
        "origin_not_allowed",
    ]
    message: str
    retryable: bool
    field: str | None = None
    request_id: str


class ErrorResponse(StrictSchema):
    error: ErrorDetail


# Http boundary
"""Bound JSON request bodies and return safe, traceable API errors."""

from uuid import uuid4

from fastapi.responses import JSONResponse


MAX_REQUEST_BYTES = 256 * 1024


class APIError(Exception):
    def __init__(self, status, code, message, *, retryable=False, field=None):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.retryable, self.field = retryable, field


def error_response(error, request_id):
    return JSONResponse(
        ErrorResponse(
            error=ErrorDetail(
                code=error.code,
                message=error.message,
                retryable=error.retryable,
                field=error.field,
                request_id=request_id,
            )
        ).model_dump(),
        status_code=error.status,
    )


class RequestBoundary:
    def __init__(self, app, development_origin=None):
        self.app = app
        self.development_origin = development_origin

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def identified_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode()),
                ]
            await send(message)

        async def reject(error):
            adapter = getattr(scope["app"].state, "inference", None)
            if adapter is not None:
                adapter.record("request_rejected", request_id=request_id, code=error.code)
            await error_response(error, request_id)(scope, receive, identified_send)

        if scope["path"] == "/api/v1/generations" and scope["method"] == "POST":
            headers = dict(scope["headers"])
            origin = headers.get(b"origin")
            allowed = {f"{scope['scheme']}://{headers.get(b'host', b'').decode('latin1')}"}
            if self.development_origin:
                allowed.add(self.development_origin)
            if origin is not None and origin.decode("latin1") not in allowed:
                return await reject(
                    APIError(
                        403, "origin_not_allowed", "Use the local application tab to generate text."
                    )
                )
            if (
                headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
                != b"application/json"
            ):
                return await reject(
                    APIError(415, "invalid_request", "Send the request as application/json.")
                )
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                    return await reject(
                        APIError(
                            413,
                            "request_too_large",
                            "This request is too large. Shorten the opening and try again.",
                            field="prompt",
                        )
                    )
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            replayed = False

            async def bounded_receive():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            return await self.app(scope, bounded_receive, identified_send)
        await self.app(scope, receive, identified_send)


# Inference adapter
"""One application-owned service, immediate busy responses, and private technical logs."""

import json
import logging
import re
import threading
import traceback
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path



class InferenceAdapter:
    def __init__(self, project_root, *, device="auto"):
        self.service = None
        self._job = threading.Lock()
        directory = Path(project_root) / "logs/web"
        directory.mkdir(parents=True, exist_ok=True)
        self._handler = RotatingFileHandler(
            directory / "inference.jsonl",
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        self._logger = logging.Logger("transformer.web", level=logging.INFO)
        self._logger.addHandler(self._handler)
        try:
            from transformer.inference import GenerationService

            self.service = GenerationService(project_root, device=device)
            self.record("service_started", device=str(self.service.selection.device))
        except Exception:
            self.record("service_start_failed", traceback=traceback.format_exc())

    def record(self, event, **details):
        self._logger.info(
            json.dumps(
                {
                    "time_utc": datetime.now(UTC).isoformat(),
                    "event": event,
                    **details,
                }
            )
        )

    def status(self):
        return ServiceStatus(
            generation_enabled=self.service is not None,
            busy=self._job.locked(),
            device=str(self.service.selection.device) if self.service else None,
        )

    def require_service(self):
        if self.service is None:
            raise APIError(
                503,
                "service_unavailable",
                "The local inference service could not start. Check logs/web/inference.jsonl, "
                "restore the model registry or device settings, and restart the application.",
                retryable=True,
            )
        return self.service

    def catalog(self):
        service = self.require_service()
        rows = []
        for entry in service.list_models():
            available = True
            try:
                for name in ("checkpoint", "tokenizer", "configuration", "metrics"):
                    service.registry.local_path(entry[name]["path"])
            except (ValueError, OSError):
                available = False
            rows.append(
                ModelSummary(
                    model_id=entry["model_id"],
                    kind=entry["kind"],
                    display_name="Story continuation · TinyStories + SimpleStories"
                    if entry["comparison_member"] == "story_continuation"
                    else {
                        "baseline": "TinyStories",
                        "combined": "TinyStories + SimpleStories",
                        "english": "TinyStories · Extended training",
                    }[entry["kind"]],
                    availability="registered" if available else "unavailable",
                    unavailable_reason=None
                    if available
                    else (
                        "Registered model files are missing or inaccessible. "
                        "Restore them and check the service again."
                    ),
                    checkpoint={
                        "sha256": entry["checkpoint"]["sha256"],
                        "update": entry["checkpoint_update"],
                    },
                    tokenizer_fingerprint=entry["tokenizer_fingerprint"],
                )
            )
        return ModelCatalog(models=rows, registry_fingerprint=service.registry.fingerprint)

    def generate(self, request, request_id):
        service = self.require_service()
        try:
            service.registry.resolve(request.model_id)
        except ValueError:
            raise APIError(
                404,
                "model_not_found",
                "This model is not in the local registry. "
                "Check the service and choose an available model.",
                field="model_id",
            ) from None
        settings = request.generation_config()
        # Worker lock rakhta hai: browser disconnect hone par bhi doosra GPU job na chale.
        if not self._job.acquire(blocking=False):
            raise APIError(
                409,
                "busy",
                "Another generation is running on this local service. "
                "Wait for it to finish, then try again.",
                retryable=True,
            )
        try:
            self.record("generation_started", request_id=request_id, model_id=request.model_id)
            raw = service.generate(request.model_id, request.prompt, settings)
            values = {key: raw[key] for key in GenerationResult.model_fields if key != "request_id"}
            values["request_id"] = request_id
            values["checkpoint"] = {key: raw["checkpoint"][key] for key in ("sha256", "update")}
            sampling = dict(raw["sampling_settings"])
            sampling.pop("sampling_device")
            for key in ("seed", "effective_seed"):
                sampling[key] = str(sampling[key]) if sampling[key] is not None else None
            values["sampling_settings"] = sampling
            result = GenerationResult(**values)
            self.record(
                "generation_completed",
                request_id=request_id,
                model_id=result.model_id,
                checkpoint_sha256=result.checkpoint.sha256,
                tokens=result.generated_token_count,
                generation_seconds=result.generation_duration_seconds,
                load_seconds=result.model_load_seconds,
                cache_hit=result.cache_hit,
            )
            return result
        except Exception as exc:
            self.record(
                "generation_failed",
                request_id=request_id,
                model_id=request.model_id,
                exception=type(exc).__name__,
                traceback=traceback.format_exc(),
            )
            text = str(exc)
            unknown = re.fullmatch(r"Unknown character U\+([0-9A-F]+) at position ([0-9]+)\.", text)
            if isinstance(exc, ValueError) and unknown:
                raise APIError(
                    422,
                    "invalid_request",
                    f"The model does not know character U+{unknown[1]} "
                    f"at position {unknown[2]} (counting from zero). "
                    "Remove or replace that character and try again.",
                    field="prompt",
                ) from None
            if isinstance(exc, ValueError) and re.fullmatch(
                r"top_k=[0-9]+ exceeds vocabulary size [0-9]+\.", text
            ):
                raise APIError(422, "invalid_request", text, field="settings.top_k") from None
            if isinstance(exc, OSError) or (
                isinstance(exc, ValueError)
                and text.startswith(
                    (
                        "Cannot load registered model",
                        "Missing regular local project artifact",
                        "Symbolic links are not trusted",
                        "Cannot read registered artifact",
                    )
                )
            ):
                raise APIError(
                    503,
                    "artifact_unavailable",
                    "This model's checkpoint, tokenizer or metadata could not be verified. "
                    "Restore the registered files, or choose another model. "
                    "Technical details are in logs/web/inference.jsonl.",
                    retryable=True,
                    field="model_id",
                ) from None
            raise APIError(
                500,
                "inference_failed",
                "Generation could not finish. Your opening is preserved. Try again; "
                "if it repeats, check logs/web/inference.jsonl and restart the application.",
                retryable=True,
            ) from None
        finally:
            self._job.release()

    def close(self):
        with self._job:
            if self.service:
                self.service.close()
            self.record("service_stopped")
            self._handler.close()


# Application
"""Local Generate interface and a thin adapter over the existing inference service."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool


DEVELOPMENT_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")
ASSETS = Path(__file__).parent.parent / "frontend"


def create_app(
    *, project_root=None, device="auto", development_origin: str | None = None
) -> FastAPI:
    if development_origin is not None and development_origin not in DEVELOPMENT_ORIGINS:
        raise ValueError(f"Development origin must be one of {DEVELOPMENT_ORIGINS}.")
    root = Path.cwd() if project_root is None else Path(project_root)
    if device not in ("auto", "cpu", "mps"):
        raise ValueError("device must be auto, cpu or mps.")

    @asynccontextmanager
    async def lifespan(app):
        app.state.inference = await run_in_threadpool(InferenceAdapter, root, device=device)
        try:
            yield
        finally:
            await run_in_threadpool(app.state.inference.close)

    app = FastAPI(
        title="Transformer local inference",
        version=CONTRACT_VERSION,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.inference = None
    app.add_middleware(RequestBoundary, development_origin=development_origin)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[development_origin] if development_origin else [],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def local_headers(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
            "font-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(APIError)
    async def known_error(request: Request, exc: APIError):
        request.app.state.inference.record(
            "request_rejected", request_id=request.state.request_id, code=exc.code
        )
        return error_response(exc, request.state.request_id)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        issue = exc.errors()[0]
        allowed_fields = {
            "model_id",
            "prompt",
            "settings",
            "mode",
            "temperature",
            "top_k",
            "seed",
            "max_new_tokens",
            "unknown_policy",
        }
        field = ".".join(str(key) for key in issue["loc"] if key in allowed_fields) or None
        message = "Check the request format. " + issue["msg"]
        request.app.state.inference.record(
            "request_rejected",
            request_id=request.state.request_id,
            code="invalid_request",
            field=field,
            validation_type=issue["type"],
        )
        return error_response(
            APIError(422, "invalid_request", message, field=field), request.state.request_id
        )

    @app.get("/api/v1/status", response_model=ServiceStatus)
    def service_status(request: Request):
        return request.app.state.inference.status()

    failures = {code: {"model": ErrorResponse} for code in (403, 404, 409, 413, 415, 422, 500, 503)}

    @app.get(
        "/api/v1/models", response_model=ModelCatalog, responses={503: {"model": ErrorResponse}}
    )
    def models(request: Request):
        return request.app.state.inference.catalog()

    @app.post("/api/v1/generations", response_model=GenerationResult, responses=failures)
    async def generate(payload: GenerationRequest, request: Request):
        return await run_in_threadpool(
            request.app.state.inference.generate, payload, request.state.request_id
        )

    @app.get("/api/v1/analytics", response_model=dict)
    def analytics():
        from transformer.web.application import load_analytics
        return load_analytics(root)

    @app.get("/", include_in_schema=False)
    def shell():
        return FileResponse(ASSETS / "index.html")

    @app.get("/assets/favicon.svg", include_in_schema=False)
    def favicon():
        from fastapi.responses import Response

        return Response('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="3" fill="#245f56"/><path d="M7 8h18v5h-2v-3h-5v15h3v2H11v-2h3V10H9v3H7z" fill="#fcfbf7"/></svg>\n', media_type="image/svg+xml")

    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
    return app
