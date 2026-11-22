"""Validate configuration boundaries and portable snapshot recovery."""

import json
from pathlib import Path

import pytest

from transformer.configuration.settings import PathConfig, ProjectConfig, RuntimeConfig


def test_configuration_round_trip_and_explicit_path_resolution(tmp_path):
    config = ProjectConfig(
        runtime=RuntimeConfig(device="cpu"),
        paths=PathConfig(data_dir="datasets/raw", runs_dir="experiments/runs"),
    )
    snapshot = tmp_path / "records" / "config.json"
    config.save(snapshot)
    assert ProjectConfig.load(snapshot) == config
    assert ProjectConfig.from_dict(json.loads(config.to_json())) == config
    assert config.to_dict()["runtime"]["device"] == "cpu"
    assert config.paths.resolve(tmp_path) == {
        "data_dir": tmp_path / "datasets/raw",
        "runs_dir": tmp_path / "experiments/runs",
    }
    assert not (tmp_path / "datasets").exists()
    assert not (tmp_path / "experiments").exists()


def test_snapshot_refuses_to_overwrite_existing_file(tmp_path):
    snapshot = tmp_path / "config.json"
    original = ProjectConfig()
    original.save(snapshot)
    with pytest.raises(FileExistsError):
        ProjectConfig(runtime=RuntimeConfig(device="cpu")).save(snapshot)
    assert ProjectConfig.load(snapshot) == original


@pytest.mark.parametrize("device", ["cuda", "MPS", "", None, 1, True, []])
def test_invalid_device_preference_is_actionable(device):
    with pytest.raises(ValueError, match="runtime.device must be one of: auto, cpu, mps"):
        RuntimeConfig(device=device)


@pytest.mark.parametrize(
    "directory",
    [
        "",
        " ",
        " data",
        "/tmp/data",
        "../data",
        "data/../runs",
        "C:/data",
        "C:\\data",
        ".",
        "x\x00y",
        None,
        2,
    ],
)
def test_invalid_paths_are_rejected(directory):
    with pytest.raises(ValueError, match="paths.data_dir"):
        PathConfig(data_dir=directory)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ([], "configuration must be a JSON object"),
        ({"unknown": 1}, "Unknown configuration settings: unknown"),
        ({"runtime": []}, "runtime must be a JSON object"),
        ({"runtime": {"devcie": "cpu"}}, "Unknown runtime settings: devcie"),
        ({"paths": None}, "paths must be a JSON object"),
        ({"paths": {"run_dir": "runs"}}, "Unknown paths settings: run_dir"),
        ({"schema_version": 2}, "schema_version must be the integer 1"),
        ({"schema_version": True}, "schema_version must be the integer 1"),
        ({"schema_version": 1.0}, "schema_version must be the integer 1"),
    ],
)
def test_invalid_configuration_objects_are_rejected(payload, error):
    with pytest.raises(ValueError, match=error):
        ProjectConfig.from_dict(payload)


def test_nested_constructor_requires_validated_configuration():
    with pytest.raises(ValueError, match="runtime must be a RuntimeConfig"):
        ProjectConfig(runtime={"device": "cpu"})
    with pytest.raises(ValueError, match="paths must be a PathConfig"):
        ProjectConfig(paths={"data_dir": "data"})


def test_malformed_json_has_file_and_line_context(tmp_path):
    snapshot = tmp_path / "broken.json"
    snapshot.write_text('{"runtime":', encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid JSON in .*broken.json:.*line 1"):
        ProjectConfig.load(snapshot)


def test_tracked_default_matches_program_defaults():
    root = Path(__file__).resolve().parents[1]
    assert ProjectConfig.load(root / "configs/default.json") == ProjectConfig()
