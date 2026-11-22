"""Offline API integration uses clearly synthetic local weights, never real experiments."""

import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("fastapi")

from tests.web.test_application_shell import request, server_for  # noqa: E402
from transformer.web.application import create_app  # noqa: E402
from transformer.web.application import MAX_REQUEST_BYTES  # noqa: E402


@pytest.fixture
def connected(inference_fixture):
    app = create_app(project_root=inference_fixture, device="cpu")
    with server_for(app) as url:
        yield app, url


def opening(**changes):
    return {
        "model_id": "fixture-baseline-v1",
        "prompt": "abcd",
        "settings": {
            "mode": "greedy",
            "max_new_tokens": 4,
        },
        **changes,
    }


def generate(url, payload):
    status, headers, body = request(url, "/api/v1/generations", method="POST", data=payload)
    return status, headers, json.loads(body)


def test_catalog_generation_and_switching_are_real_and_path_free(connected):
    app, url = connected
    _, _, body = request(url, "/api/v1/models")
    catalog = json.loads(body)
    assert len(catalog["models"]) == 2
    assert all(row["availability"] == "registered" for row in catalog["models"])
    service = app.state.inference.service
    assert service.cache_info()["cached_models"] == 0
    for model in catalog["models"]:
        status, headers, result = generate(url, opening(model_id=model["model_id"]))
        assert status == 200, result
        assert result["checkpoint"] == model["checkpoint"]
        assert result["tokenizer_fingerprint"] == model["tokenizer_fingerprint"]
        assert result["registry_fingerprint"] == catalog["registry_fingerprint"]
        assert result["request_id"] == headers["X-Request-ID"]
        assert result["generated_token_count"] <= 4
        assert isinstance(result["continuation"], str)
        assert "path" not in result["checkpoint"] and "generated_token_ids" not in result
        assert service.cache_info()["cached_models"] == 1
        assert service.cache_info()["model_id"] == model["model_id"]
    assert generate(url, opening(model_id=model["model_id"]))[2]["cache_hit"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        opening(prompt=""),
        opening(prompt=" \n"),
        opening(prompt="a" * 16385),
        opening(prompt="a\ud800"),
        opening(prompt=12),
        opening(checkpoint="/tmp/model.pt"),
        opening(settings={"mode": "greedy", "temperature": 1}),
        opening(settings={"mode": "sample", "temperature": 0}),
        opening(settings={"mode": "sample", "temperature": float("nan")}),
        opening(settings={"mode": "sample", "temperature": float("inf")}),
        opening(settings={"mode": "sample", "temperature": True}),
        opening(settings={"mode": "sample", "temperature": 1, "top_k": True}),
        opening(settings={"mode": "sample", "temperature": 1, "top_k": 0}),
        opening(settings={"mode": "sample", "temperature": 1, "seed": "9223372036854775808"}),
        opening(settings={"mode": "sample", "temperature": 1, "seed": 42}),
        opening(settings={"mode": "greedy", "max_new_tokens": True}),
        opening(settings={"mode": "greedy", "max_new_tokens": "128"}),
        opening(settings={"mode": "greedy", "max_new_tokens": 1025}),
        opening(model_id="../checkpoints/model.pt"),
    ],
)
def test_invalid_requests_are_bounded_safe_and_do_not_load(connected, payload):
    app, url = connected
    status, _, result = generate(url, payload)
    assert status == 422, result
    assert result["error"]["code"] == "invalid_request"
    assert result["error"]["message"] and "Traceback" not in str(result)
    assert app.state.inference.service.cache_info()["cached_models"] == 0


def test_model_specific_validation_and_recovery(connected):
    _, url = connected
    assert generate(url, opening(model_id="unknown-version"))[0] == 404
    status, _, result = generate(url, opening(prompt="ab☃"))
    assert status == 422 and "U+2603" in result["error"]["message"]
    assert result["error"]["field"] == "prompt"
    assert (
        generate(url, opening(settings={"mode": "sample", "temperature": 1, "top_k": 7}))[0] == 422
    )
    assert generate(url, opening())[0] == 200


def test_private_seed_roundtrip_and_zero_output(connected):
    _, url = connected
    payload = opening(
        settings={
            "mode": "sample",
            "temperature": 0.8,
            "seed": "9223372036854775807",
            "max_new_tokens": 5,
        }
    )
    status, _, first = generate(url, payload)
    second = generate(url, payload)[2]
    assert status == 200 and first["continuation"] == second["continuation"]
    assert first["sampling_settings"]["effective_seed"] == "9223372036854775807"
    zero = generate(url, opening(settings={"mode": "greedy", "max_new_tokens": 0}))[2]
    assert zero["continuation"] == "" and zero["generated_token_count"] == 0


@pytest.mark.parametrize("artifact", ["checkpoint", "tokenizer"])
def test_missing_artifact_and_logs_are_useful(inference_fixture, artifact):
    registry = json.loads((inference_fixture / "configs/inference/model_registry.json").read_text())
    (inference_fixture / registry["models"][0][artifact]["path"]).unlink()
    with server_for(create_app(project_root=inference_fixture, device="cpu")) as url:
        _, _, body = request(url, "/api/v1/models")
        assert json.loads(body)["models"][0]["availability"] == "unavailable"
        status, _, result = generate(url, opening())
        assert status == 503 and result["error"]["code"] == "artifact_unavailable"
        assert str(inference_fixture) not in str(result)
    events = [
        json.loads(line)
        for line in (inference_fixture / "logs/web/inference.jsonl").read_text().splitlines()
    ]
    failure = next(row for row in events if row["event"] == "generation_failed")
    assert "traceback" in failure and failure["request_id"] == result["error"]["request_id"]


def test_corrupt_checkpoint_is_not_exposed_to_browser(inference_fixture):
    registry = json.loads((inference_fixture / "configs/inference/model_registry.json").read_text())
    (inference_fixture / registry["models"][0]["checkpoint"]["path"]).write_bytes(b"bad checkpoint")
    with server_for(create_app(project_root=inference_fixture, device="cpu")) as url:
        status, _, result = generate(url, opening())
        assert status == 503 and result["error"]["code"] == "artifact_unavailable"
        assert "SHA-256 mismatch" not in str(result)


def test_request_transport_limits_and_origin_policy(connected):
    app, url = connected
    assert generate(url, b"{")[0] == 422
    assert generate(url, b"x" * (MAX_REQUEST_BYTES + 1))[0] == 413
    assert (
        request(
            url,
            "/api/v1/generations",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "text/plain"},
        )[0]
        == 415
    )
    assert (
        request(
            url,
            "/api/v1/generations",
            method="POST",
            data=opening(),
            headers={"Origin": "https://untrusted.example"},
        )[0]
        == 403
    )
    assert (
        request(url, "/api/v1/generations", method="POST", data=opening(), headers={"Origin": url})[
            0
        ]
        == 200
    )
    assert app.state.inference.service.cache_info()["cached_models"] == 1


def test_busy_requests_and_disconnect_do_not_release_running_job(connected, monkeypatch):
    app, url = connected
    entered, release = threading.Event(), threading.Event()
    original = app.state.inference.service.generate

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(app.state.inference.service, "generate", held)
    host, port = url.removeprefix("http://").split(":")
    client = socket.create_connection((host, int(port)))
    data = json.dumps(opening()).encode()
    client.sendall(
        (
            f"POST /api/v1/generations HTTP/1.1\r\nHost: {host}:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n"
        ).encode()
        + data
    )
    try:
        assert entered.wait(5)
        client.close()
        status, _, body = generate(url, opening())
        assert status == 409 and body["error"]["code"] == "busy"
        _, _, body = request(url, "/api/v1/status")
        assert json.loads(body)["busy"] is True
    finally:
        release.set()
        client.close()
    with ThreadPoolExecutor() as pool:
        assert pool.submit(app.state.inference._job.acquire, timeout=5).result()
        app.state.inference._job.release()
    assert generate(url, opening())[0] == 200


def test_unexpected_failure_is_logged_and_retry_recovers(connected, monkeypatch):
    app, url = connected
    original = app.state.inference.service.generate

    def failed(*args, **kwargs):
        raise RuntimeError("technical-only private details /private/example")

    monkeypatch.setattr(app.state.inference.service, "generate", failed)
    status, _, result = generate(url, opening())
    assert status == 500 and "private" not in str(result)
    assert app.state.inference.status().busy is False
    monkeypatch.setattr(app.state.inference.service, "generate", original)
    assert generate(url, opening())[0] == 200


def test_extended_model_loads_with_validation_only_evidence(inference_fixture):
    from transformer.data.preparation import file_sha256

    registry = inference_fixture / "configs/inference/model_registry.json"
    catalog = json.loads(registry.read_text())
    baseline = catalog["models"][0]
    original_metrics = json.loads((inference_fixture / baseline["metrics"]["path"]).read_text())
    metrics = {
        "scope": "validation_and_diagnostic_samples",
        "identity": original_metrics["identity"],
        "runs": {"english": original_metrics["runs"]["model_a"]},
    }
    relative = "artifacts/extended-validation.json"
    record = inference_fixture / relative
    record.write_text(json.dumps(metrics))
    model_id = "fixture-extended-v1"
    catalog["models"].append(
        {
            **baseline,
            "model_id": model_id,
            "kind": "english",
            "comparison_member": "english",
            "metrics": {"path": relative, "sha256": file_sha256(record)},
        }
    )
    registry.write_text(json.dumps(catalog))
    with server_for(create_app(project_root=inference_fixture, device="cpu")) as url:
        status, _, result = generate(url, opening(model_id=model_id))
        assert status == 200, result
        assert result["model_kind"] == "english"
        assert result["model_id"] == model_id
        _, _, body = request(url, "/api/v1/analytics")
        assert model_id not in {model["model_id"] for model in json.loads(body)["models"]}


def test_combined_validation_record_is_the_only_served_model(inference_fixture):
    from transformer.data.preparation import file_sha256

    registry = inference_fixture / "configs/inference/model_registry.json"
    catalog = json.loads(registry.read_text())
    combined = next(row for row in catalog["models"] if row["kind"] == "combined")
    original = json.loads((inference_fixture / combined["metrics"]["path"]).read_text())
    run = original["runs"]["model_b"]
    run["selected_validation"] = {"update": combined["checkpoint_update"], "loss": 1.25}
    path = inference_fixture / "artifacts/combined-validation.json"
    path.write_text(
        json.dumps(
            {
                "scope": "validation_and_diagnostic_samples",
                "identity": original["identity"],
                "runs": {"extended_combined": run},
                "diagnostics": [{"prompt": "abcd", "continuation": "Fixture only."}],
            }
        )
    )
    combined["comparison_member"] = "extended_combined"
    combined["metrics"] = {
        "path": path.relative_to(inference_fixture).as_posix(),
        "sha256": file_sha256(path),
    }
    catalog["models"] = [combined]
    registry.write_text(json.dumps(catalog))
    with server_for(create_app(project_root=inference_fixture, device="cpu")) as url:
        _, _, raw = request(url, "/api/v1/models")
        assert [model["model_id"] for model in json.loads(raw)["models"]] == [combined["model_id"]]
        status, _, result = generate(url, opening(model_id=combined["model_id"]))
        assert status == 200, result
        _, _, raw = request(url, "/api/v1/analytics")
        result = json.loads(raw)
        assert not result["compatible"]
        assert len(result["models"]) == 1
        assert result["models"][0]["validation_loss"] == 1.25
