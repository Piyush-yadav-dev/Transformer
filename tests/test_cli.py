"""Exercise installed entry points and user-facing errors."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from transformer.cli import main
from transformer.configuration.settings import ProjectConfig, RuntimeConfig


def run_cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "transformer", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_module_command_runs_from_outside_project(tmp_path):
    result = run_cli("diagnose", "--device", "cpu", "--check", "--json", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["selected_device"] == "cpu"
    assert report["isolated_environment"] is True
    assert report["tensor_check"] == "passed"


def test_installed_console_entry_point(tmp_path):
    executable = Path(sys.executable).with_name("transformer")
    result = subprocess.run(
        [str(executable), "diagnose", "--device", "auto", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    expected = (
        "mps" if torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu"
    )
    assert json.loads(result.stdout)["selected_device"] == expected


def test_invalid_device_reports_choices():
    result = run_cli("diagnose", "--device", "invalid")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert all(choice in result.stderr for choice in ("auto", "cpu", "mps"))
    assert "Traceback" not in result.stderr


def test_cli_configuration_override_and_snapshot(tmp_path):
    source = tmp_path / "config.json"
    destination = tmp_path / "effective.json"
    ProjectConfig().save(source)
    result = run_cli(
        "diagnose",
        "--config",
        str(source),
        "--device",
        "cpu",
        "--save-config",
        str(destination),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    assert ProjectConfig.load(destination).runtime == RuntimeConfig(device="cpu")
    assert ProjectConfig.load(source).runtime == RuntimeConfig(device="auto")


@pytest.mark.parametrize("contents", ['{"runtime": {"device": "invalid"}}', '{"runtime":', "[]"])
def test_bad_config_is_reported_without_traceback(tmp_path, contents):
    config = tmp_path / "bad.json"
    config.write_text(contents, encoding="utf-8")
    result = run_cli("diagnose", "--config", str(config))
    assert result.returncode == 2
    assert "transformer: error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_existing_cli_snapshot_is_preserved(tmp_path):
    snapshot = tmp_path / "existing.json"
    snapshot.write_text("original content", encoding="utf-8")
    result = run_cli("diagnose", "--save-config", str(snapshot))
    assert result.returncode == 2
    assert "File exists" in result.stderr
    assert snapshot.read_text(encoding="utf-8") == "original content"


def test_missing_config_has_helpful_error(tmp_path):
    result = run_cli("diagnose", "--config", str(tmp_path / "missing.json"))
    assert result.returncode == 2
    assert "missing.json" in result.stderr
    assert "Traceback" not in result.stderr


def test_unavailable_explicit_mps_is_cli_error(monkeypatch, capsys):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(SystemExit) as error:
        main(["diagnose", "--device", "mps"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "MPS was explicitly requested" in captured.err
    assert "--device cpu" in captured.err
