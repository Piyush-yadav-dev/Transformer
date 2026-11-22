"""Exercise installed inference commands from a separate working directory."""

import json
import subprocess
import sys

import pytest

from transformer.cli import main


def test_installed_cli_lists_and_generates(inference_fixture, tmp_path):
    for command in (
        ["list-models"],
        [
            "generate",
            "--model",
            "fixture-baseline-v1",
            "--prompt",
            "abcd",
            "--device",
            "cpu",
            "--mode",
            "sample",
            "--temperature",
            "0.8",
            "--top-k",
            "3",
            "--seed",
            "42",
            "--max-new-tokens",
            "4",
            "--json",
        ],
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "transformer",
                *command,
                "--project-root",
                str(inference_fixture),
            ],
            cwd=tmp_path.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        if command[0] == "list-models":
            assert len(data) == 2
        else:
            assert data["model_id"] == "fixture-baseline-v1"
            assert 1 <= data["generated_token_count"] <= 4
            assert data["sampling_settings"]["effective_seed"] == 42


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--model", "../../bad.pt"], "Unknown model ID"),
        (["--max-new-tokens", "1025"], "max_new_tokens"),
        (["--mode", "sample", "--temperature", "0"], "temperature"),
        (["--prompt", ""], "nonempty"),
    ],
)
def test_cli_invalid_inputs_are_actionable(inference_fixture, capsys, extra, match):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "generate",
                "--project-root",
                str(inference_fixture),
                "--model",
                "fixture-baseline-v1",
                "--prompt",
                "abcd",
                "--device",
                "cpu",
                *extra,
            ]
        )
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert match in captured.err and "Traceback" not in captured.err


def test_cli_explicit_unavailable_mps_is_not_cpu_fallback(inference_fixture, capsys, monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "generate",
                "--project-root",
                str(inference_fixture),
                "--model",
                "fixture-baseline-v1",
                "--prompt",
                "abcd",
                "--device",
                "mps",
            ]
        )
    assert error.value.code == 2
    assert "MPS was explicitly requested" in capsys.readouterr().err
