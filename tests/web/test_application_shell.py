"""Loopback HTTP checks for the optional shell; no models or external services are used."""

import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="The local web shell is an optional extra.")
uvicorn = pytest.importorskip("uvicorn")

from transformer.web.application import create_app  # noqa: E402


@contextmanager
def server_for(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    address = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("Local test server did not start within five seconds.")
        yield address
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive(), "Local test server failed to stop."


@pytest.fixture(scope="module")
def local_url(tmp_path_factory):
    with server_for(
        create_app(project_root=tmp_path_factory.mktemp("web-shell"), device="cpu")
    ) as address:
        yield address


def request(url, path, *, method="GET", headers=None, data=None):
    headers = dict(headers or {})
    if data is not None:
        headers.setdefault("Content-Type", "application/json")
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
    req = urllib.request.Request(url + path, method=method, headers=headers, data=data)
    # Local test transport ignores proxy environment variables.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        result = opener.open(req, timeout=5)
    except urllib.error.HTTPError as exc:
        result = exc
    with result:
        return result.status, result.headers, result.read()


def test_shell_status_is_not_an_inference_readiness_claim(local_url):
    status, _, body = request(local_url, "/api/v1/status")
    assert status == 200
    assert json.loads(body) == {
        "application": "Transformer",
        "phase": 15,
        "interface": "connected_generate",
        "generation_enabled": False,
        "contract_version": "2.0.0",
        "busy": False,
        "device": None,
    }
    _, _, body = request(local_url, "/openapi.json")
    assert set(json.loads(body)["paths"]) == {
        "/api/v1/status",
        "/api/v1/models",
        "/api/v1/generations",
        "/api/v1/analytics",
    }


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/assets/styles.css",
        "/assets/application.js",
        "/assets/favicon.svg",
    ],
)
def test_only_packaged_public_assets_are_served(local_url, path):
    status, headers, body = request(local_url, path)
    assert status == 200 and body
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


@pytest.mark.parametrize(
    "path,method",
    [
        ("/runs", "GET"),
        ("/fixtures/fixture_app.js", "GET"),
        ("/fixtures/foundations.html", "GET"),
        ("/assets/fixture_app.js", "GET"),
        ("/__ui_tests__", "GET"),
        ("/assets/%2e%2e/application.py", "GET"),
        ("/contracts/inference.openapi.json", "GET"),
    ],
)
def test_private_and_planned_routes_are_not_exposed(local_url, path, method):
    assert request(local_url, path, method=method)[0] == 404


def test_default_origin_and_host_policy(local_url):
    headers = {"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST"}
    status, response_headers, _ = request(
        local_url, "/api/v1/status", method="OPTIONS", headers=headers
    )
    assert status == 400 and "access-control-allow-origin" not in response_headers
    assert request(local_url, "/", headers={"Host": "untrusted.example"})[0] == 400


def test_development_origin_is_exact_and_opt_in(tmp_path):
    origin = "http://127.0.0.1:5173"
    with server_for(create_app(project_root=tmp_path, development_origin=origin)) as url:
        headers = {
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        status, response_headers, _ = request(
            url, "/api/v1/status", method="OPTIONS", headers=headers
        )
        assert status == 200 and response_headers["access-control-allow-origin"] == origin
        assert "access-control-allow-credentials" not in response_headers
        headers["Origin"] = origin + ".untrusted.example"
        assert request(url, "/api/v1/status", method="OPTIONS", headers=headers)[0] == 400
    with pytest.raises(ValueError, match="Development origin"):
        create_app(development_origin="*")


def test_startup_does_not_import_inference_or_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from transformer.web.application import create_app; create_app(); "
            "assert 'torch' not in sys.modules; assert 'transformer.inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_launcher_rejects_invalid_port():
    result = subprocess.run(
        [sys.executable, "-m", "transformer.web", "--port", "80"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2 and "1024 and 65535" in result.stderr


def test_planned_contract_has_resolvable_references_and_matches_backend_limits():
    import transformer.web
    from transformer.configuration.settings import MAX_NEW_TOKENS, MAX_PROMPT_CHARACTERS

    contract = json.loads(
        (Path(transformer.web.__file__).parent / "contracts/inference.openapi.json").read_text()
    )
    schemas = contract["components"]["schemas"]

    def validate_references(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].startswith("#/components/schemas/")
                assert value["$ref"].rsplit("/", 1)[1] in schemas
            for child in value.values():
                validate_references(child)
        elif isinstance(value, list):
            for child in value:
                validate_references(child)

    validate_references(contract)
    assert (
        schemas["GenerationRequest"]["properties"]["prompt"]["maxLength"] == MAX_PROMPT_CHARACTERS
    )
    for name in ("GreedySettings", "SampleSettings"):
        assert schemas[name]["properties"]["max_new_tokens"]["maximum"] == MAX_NEW_TOKENS
    assert (
        contract["paths"]["/api/v1/generations"]["post"]["x-implementation-status"]
        == "implemented-phase15"
    )
