"""Serial, fresh-process measurements and a conservative unexecuted recommendation."""

import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from transformer.benchmarking.worker import hardware_identity
from transformer.configuration.settings import BenchmarkConfig
from transformer.data.preparation import file_sha256, write_json


def select_candidate(cases, near_best_fraction):
    """Pick the smallest safe microbatch close to the best measured pipeline rate."""
    compute_batches = {
        case["microbatch_size"]
        for case in cases
        if case.get("mode") == "compute" and case["status"] == "completed"
    }
    candidates = [
        case
        for case in cases
        if case.get("mode") == "pipeline"
        and case["status"] == "completed"
        and case.get("measured_iterations", 0) >= 4
        and case["microbatch_size"] in compute_batches
    ]
    if not candidates:
        return None
    threshold = max(case["tokens_per_second"] for case in candidates) * near_best_fraction
    return min(
        (case for case in candidates if case["tokens_per_second"] >= threshold),
        key=lambda case: case["microbatch_size"],
    )


def estimate_runtime(case, proposal, warmup_rows):
    """Project measured work, charging the proposed evaluation/checkpoint cadence."""
    count = case["measured_iterations"]
    val_times, save_times = case["validation_event_seconds"], case["checkpoint_event_seconds"]
    val = statistics.mean(val_times) if val_times else 0.0
    save = statistics.mean(save_times) if save_times else 0.0
    base = max(
        case["mean_core_update_seconds"],
        case["mean_boundary_update_seconds"] - (sum(val_times) + sum(save_times)) / count,
    )
    updates = proposal.total_updates
    validations = math.ceil(updates / proposal.evaluation_interval)
    checkpoints = (
        updates // proposal.evaluation_interval
        + updates // proposal.checkpoint_interval
        - updates // math.lcm(proposal.evaluation_interval, proposal.checkpoint_interval)
        + int(
            updates % proposal.evaluation_interval != 0
            and updates % proposal.checkpoint_interval != 0
        )
    )
    cold_extra = max(
        0.0,
        sum(row["boundary_seconds"] for row in warmup_rows)
        - len(warmup_rows) * case["mean_boundary_update_seconds"],
    )
    seconds = (
        case["setup_to_ready_seconds"]
        + cold_extra
        + updates * base
        + validations * val
        + checkpoints * save
    )
    return {
        "scheduled_updates": updates,
        "training_target_tokens": proposal.total_tokens,
        "measured_training_and_housekeeping_seconds_per_update": base,
        "mean_validation_event_seconds": val,
        "mean_checkpoint_event_seconds": save,
        "proposed_validation_events": validations,
        "proposed_checkpoint_events": checkpoints,
        "observed_setup_seconds": case["setup_to_ready_seconds"],
        "estimated_extra_cold_warmup_seconds": cold_extra,
        "estimated_seconds_at_proposed_cadence": seconds,
        "seconds_at_benchmark_cadence_excluding_startup": updates
        * case["mean_boundary_update_seconds"],
        "limitations": "Linear projection, not a duration guarantee: longer runs, background load, "
        "power state, memory pressure, and validation/checkpoint costs can alter throughput.",
    }


def run_benchmark(config: BenchmarkConfig, root: Path):
    root = Path(root).resolve()
    output = root / config.output_dir
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "benchmark_configuration.json"
    write_json(snapshot, config.to_dict())
    hardware = hardware_identity()
    write_json(output / "hardware.json", hardware)
    package = Path(__file__).resolve().parents[1]
    write_json(
        output / "implementation_hashes.json",
        {
            str(path.relative_to(package)): file_sha256(path)
            for path in sorted(package.rglob("*.py"))
        },
    )
    cases, stopped = [], None

    def execute(batch, mode, sustained=False):
        name = f"{'sustained' if sustained else mode}-mb{batch}"
        command = [
            sys.executable,
            "-m",
            "transformer.benchmarking.worker",
            "--config",
            str(snapshot),
            "--project-root",
            str(root),
            "--batch",
            str(batch),
            "--mode",
            mode,
        ]
        if sustained:
            command.append("--sustained")
        budget = config.sustained_max_seconds if sustained else config.case_max_seconds
        print(
            f"Benchmark {name}: warmup {config.warmup_updates}, wall budget {budget:g}s",
            file=sys.stderr,
        )
        environment = dict(
            os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1"
        )
        with (
            (output / f"{name}.stdout.log").open("w") as stdout,
            (output / f"{name}.stderr.log").open("w") as stderr,
        ):
            launched = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=budget + 60,
                    check=False,
                )
                code = completed.returncode
            except subprocess.TimeoutExpired:
                return {
                    "case": name,
                    "status": "worker_timeout",
                    "reason": "Startup plus bounded case timeout exceeded.",
                }
        summary = output / name / "summary.json"
        if not summary.is_file():
            return {
                "case": name,
                "status": "failed",
                "reason": f"Worker exit {code}; see stderr log.",
            }
        result = json.loads(summary.read_text())
        result["subprocess_wall_seconds"] = time.perf_counter() - launched
        if code != 0 and result["status"] == "completed":
            result.update(status="failed", reason=f"Unexpected worker exit {code}")
        write_json(output / f"{name}.json", result)
        return result

    for batch in config.microbatch_sizes:
        for mode in ("compute", "pipeline"):
            case = execute(batch, mode)
            cases.append(case)
            if case["status"] != "completed":
                stopped = f"Stopped search after {case['case']}: {case['status']}"
                break
            if (
                sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
                > config.max_output_bytes
            ):
                stopped = "Stopped search at configured artifact disk budget."
                break
        if stopped:
            break
    write_json(output / "case_results.json", cases)
    selected = select_candidate(cases, config.near_best_fraction)
    sustained, estimate, proposed = None, None, None
    if selected is not None:
        batch = selected["microbatch_size"]
        if stopped is None:
            sustained = execute(batch, "pipeline", sustained=True)
            if (
                sustained["status"] not in ("completed", "time_limit")
                or sustained.get("measured_iterations", 0) < 8
            ):
                stopped = "Sustained check incomplete; review its measurements before training."
        # Naya configuration sirf recommendation hai; ise yahan train nahi karte.
        proposal = replace(
            config.training,
            run_id="phase7-proposed-model-a-unexecuted",
            device=selected["device"],
            microbatch_size=batch,
            gradient_accumulation_steps=config.effective_batch_size // batch,
            max_validation_batches=config.validation_windows // batch,
        )
        proposed = "proposed_training.json"
        proposal.save(output / proposed)
        rows = [
            json.loads(line)
            for line in (output / selected["case"] / "iterations.jsonl").read_text().splitlines()
        ]
        estimate = estimate_runtime(
            selected, proposal, [row for row in rows if row["phase"] == "warmup"]
        )
    report = {
        "kind": "bounded_hardware_benchmark",
        "not_model_quality_evaluation": True,
        "status": "completed" if selected and stopped is None else "incomplete",
        "hardware": hardware,
        "search_stop_reason": stopped,
        "cases": cases,
        "selection_rule": (
            f"Smallest microbatch within {config.near_best_fraction:g} of best measured "
            "pipeline rate, after all resource checks."
        ),
        "selected_microbatch_size": selected["microbatch_size"] if selected else None,
        "sustained": sustained,
        "proposed_training_configuration": proposed,
        "runtime_estimate": estimate,
        "m2_specific_conclusions": "measured_on_target"
        if hardware["matches_target_m2_air_16gb"]
        else "pending_target_hardware",
        "production_training_started": False,
    }
    write_json(output / "summary.json", report)
    return report
