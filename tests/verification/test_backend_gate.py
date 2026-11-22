"""Offline integration and gate logic; these fixture models never certify real availability."""

import json
from pathlib import Path

import pytest

from transformer.verification import checks as backend_gate
from transformer.verification.checks import check_failure_cases
from transformer.verification.checks import run_fixture_pipeline


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    pytest.importorskip("matplotlib", reason="Full release workflow requires the evaluation extra.")
    root = tmp_path_factory.mktemp("backend") / "synthetic_project"
    result = run_fixture_pipeline(root)
    failures = check_failure_cases(root, root.parent / "negative_checks")
    return root, result, failures


def test_full_preparation_to_inference_and_exact_recovery(integrated):
    root, result, _ = integrated
    assert result["status"] == "passed"
    assert result["synthetic_fixture_only"]
    assert result["exact_cpu_resume_matches_uninterrupted"]
    assert result["training_only_vocabulary_verified"]
    assert result["data_counts"]["scanned"] == 26
    assert result["data_counts"]["duplicate_occurrences"] == 4
    assert sum(result["data_counts"]["rejected"].values()) == 4
    assert len(result["generations"]) == 2
    for row in result["generations"]:
        assert row["model_id"].startswith("fixture-")
        assert row["generated_token_count"] <= 8
    assert (root / "artifacts/fixture-evaluation/results.json").is_file()


def test_fixture_rerun_protects_existing_artifacts(integrated):
    root, _, _ = integrated
    before = (root / "fixture_verification.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_fixture_pipeline(root)
    assert before == (root / "fixture_verification.json").read_bytes()


def test_meaningful_failure_messages(integrated):
    _, _, results = integrated
    assert len(results) == 8
    assert all(r["status"] == "pass" and r["message"] for r in results)
    assert {r["name"] for r in results} >= {
        "corrupt_checkpoint",
        "incomplete_checkpoint",
        "tokenizer_mismatch",
    }


def test_missing_real_artifacts_cannot_be_replaced_by_fixture_success(
    tmp_path, integrated, monkeypatch
):
    _, summary, failures = integrated
    monkeypatch.setattr(backend_gate, "run_fixture_pipeline", lambda _: summary)
    monkeypatch.setattr(backend_gate, "check_failure_cases", lambda *_: failures)
    monkeypatch.setattr(backend_gate, "verify_test_report", lambda _: {"tests": 1, "passed": 1})
    report = backend_gate.run_backend_gate(tmp_path, tmp_path / "artifacts/gate")
    assert report["status"] == "incomplete"
    assert [c["status"] for c in report["checks"]] == ["pass", "pass", "pass", "fail"]
    assert report["checks"][-1]["name"] == "real_artifacts"
    assert json.loads((tmp_path / "artifacts/gate/verification.json").read_text()) == report


@pytest.mark.parametrize("location", ["runs/do-not-overwrite", "artifacts", "../outside"])
def test_gate_output_must_be_isolated(tmp_path, location):
    with pytest.raises(ValueError, match="new directory inside"):
        backend_gate.run_backend_gate(tmp_path, tmp_path / location)


def test_gate_output_cannot_overwrite_or_follow_links(tmp_path):
    output = tmp_path / "artifacts/gate"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        backend_gate.run_backend_gate(tmp_path, output)
    link = tmp_path / "artifacts/link"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        backend_gate.run_backend_gate(tmp_path, link)


def test_missing_suite_evidence_cannot_pass(tmp_path):
    path = tmp_path / "suite.xml"
    path.write_text('<testsuites><testsuite tests="0"/></testsuites>')
    with pytest.raises(ValueError, match="missing"):
        backend_gate.verify_test_report(path)


def test_failed_integration_keeps_incomplete_report_and_skips_dependent_checks(
    tmp_path, monkeypatch
):
    def fail(_):
        raise ValueError("fixture pipeline failed")

    monkeypatch.setattr(backend_gate, "run_fixture_pipeline", fail)
    report = backend_gate.run_backend_gate(tmp_path, tmp_path / "artifacts/failed-gate")
    assert report["status"] == "incomplete"
    assert [c["status"] for c in report["checks"]] == ["fail", "fail", "skip", "fail"]
    assert Path(tmp_path / "artifacts/failed-gate/verification.json").is_file()
