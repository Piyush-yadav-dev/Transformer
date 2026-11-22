"""Maintenance commands. Run maintenance.py --help to list available operations."""

def benchmark_analytics():
    """Paired read-only analytics benchmark; baseline is a trusted local source snapshot."""
    import argparse
    import hashlib
    import importlib.util
    import json
    import platform
    import statistics
    import time
    from pathlib import Path
    from transformer.web import application as analytics

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("analytics_baseline", args.baseline)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    root = Path(__file__).resolve().parents[1]
    expected = before.load_analytics(root)
    assert expected == analytics.load_analytics(root)
    identities = {ref["path"]: ref["sha256"] for m in expected["models"] for ref in m["evidence"]}
    for _ in range(2):
        before.load_analytics(root)
        analytics.load_analytics(root)
    measurements = []
    for trial in range(11):
        order = [("before", before), ("after", analytics)]
        if trial % 2:
            order.reverse()
        for label, module in order:
            start = time.perf_counter()
            result = module.load_analytics(root)
            elapsed = time.perf_counter() - start
            assert result == expected
            measurements.append({"trial": trial, "implementation": label, "seconds": elapsed})
    for path, digest in identities.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    medians = {label: statistics.median(r["seconds"] for r in measurements if r["implementation"] == label)
               for label in ("before", "after")}
    report = {"environment": {"platform": platform.platform(), "python": platform.python_version()},
              "definition": "Synchronous analytics projection; two warmups per path; 11 paired trials, alternating order; warm OS file cache; includes reading, hashing, parsing and validation, excludes HTTP serialization.",
              "generation": "Not measured: model loading and generation are unchanged and not invoked.",
              "median_seconds": medians, "reduction_percent": 100 * (1 - medians["after"] / medians["before"]),
              "identical_output": True, "unchanged_source_hashes": identities,
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              "measurements": measurements}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"medians": medians, "reduction_percent": report["reduction_percent"]}))


def export_story_charts():
    """Extract measured chart data for the active story-continuation checkpoint."""

    import json
    import re
    from pathlib import Path

    import numpy as np
    from safetensors import safe_open
    from transformers import AutoTokenizer

    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs/inference/model_registry.json").read_text())
    (version,) = registry["models"]
    if version["comparison_member"] != "story_continuation":
        raise ValueError("The active model is not a story continuation checkpoint.")
    base = (root / version["tokenizer"]["path"]).parent
    log = root / "artifacts" / version["run_id"] / "training-console.log"
    record = json.loads((root / version["metrics"]["path"]).read_text())
    train, validation = [], []
    for line in log.read_text().splitlines():
        if match := re.fullmatch(r"update (\d+)/\d+ loss ([\d.]+) grad ([\d.]+)", line):
            update, loss, gradient = match.groups()
            train.append({"update": int(update), "loss": float(loss), "gradient": float(gradient)})
        elif match := re.fullmatch(r"validation ([\d.]+)", line):
            validation.append({"update": train[-1]["update"], "loss": float(match[1])})
    if not train or not validation or train[-1]["update"] != record["updates"]:
        raise ValueError("Training log does not describe the completed run.")
    (selected,) = [row for row in validation if row["update"] == record["selected_update"]]
    if abs(selected["loss"] - record["validation_after"]) > 0.00005:
        raise ValueError("Selected validation loss differs from registry evidence.")
    selected["loss"] = record["validation_after"]
    validation.insert(0, {"update": 0, "loss": record["validation_before"]})

    groups = {
        "Animals": ["bird", "cat", "dog", "fox", "rabbit", "kitten"],
        "Places": ["garden", "school", "forest", "river", "library", "village"],
        "Objects": ["ribbon", "key", "box", "boat", "book", "door"],
        "Feelings": ["happy", "sad", "afraid", "angry", "excited", "lonely"],
    }
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    points, vectors = [], []
    (weight_file,) = base.glob("*.safetensors")
    with safe_open(weight_file, framework="pt", device="cpu") as weights:
        matrix = weights.get_slice("model.embed_tokens.weight")
        for group, words in groups.items():
            for word in words:
                ids = tokenizer.encode(" " + word, add_special_tokens=False)
                if len(ids) == 1:
                    points.append({"word": word, "group": group, "token_id": ids[0]})
                    vectors.append(matrix[ids[0] : ids[0] + 1].float().numpy()[0])
        width = matrix.get_shape()[1]
    vectors = np.asarray(vectors, dtype=np.float64)
    vectors -= vectors.mean(axis=0)
    _, singular, axes = np.linalg.svd(vectors, full_matrices=False)
    projection = vectors @ axes[:2].T
    for point, coords in zip(points, projection, strict=True):
        point.update(x=float(coords[0]), y=float(coords[1]))
    data = {
        "model_id": version["model_id"],
        "update": version["checkpoint_update"],
        "train": train,
        "validation": validation,
        "embedding": {
            "dimensions": width,
            "variance": (singular[:2] ** 2 / (singular**2).sum()).tolist(),
            "points": points,
        },
    }
    output = root / "frontend/application.js"
    text = output.read_text()
    start = text.index("// BEGIN MEASURED CHART DATA")
    end = text.index("// END MEASURED CHART DATA", start)
    output.write_text(
        text[:start] + "// BEGIN MEASURED CHART DATA\nconst chartData = "
        + json.dumps(data, indent=2, allow_nan=False) + ";\n" + text[end:]
    )

    print(
        f"{len(train)} training points, {len(validation)} validation points, {len(points)} embeddings"
    )


def preview_ui_fixtures():
    """Serve synthetic UI fixtures separately from the normal application, on loopback only."""

    import argparse
    from pathlib import Path

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    from transformer.web.application import ASSETS


    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--port", type=int, default=8001)
        args = parser.parse_args()
        if not 1024 <= args.port <= 65535:
            parser.error("--port must be between 1024 and 65535.")
        fixtures = Path(__file__).resolve().parents[1] / "tests/web/ui_fixtures"
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.get("/")
        def index():
            return FileResponse(fixtures / "index.html")

        app.mount("/assets", StaticFiles(directory=ASSETS))
        app.mount("/fixtures", StaticFiles(directory=fixtures))
        uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)

    main()


def verify_analytics():
    """Read-only audit of every numeric analytics projection against saved sources."""
    import hashlib
    import json
    from pathlib import Path

    import torch

    from transformer.configuration.settings import TransformerConfig
    from transformer.model.transformer import TransformerLanguageModel
    from transformer.web.application import load_analytics

    root = Path(__file__).resolve().parents[1]
    data = load_analytics(root)
    assert data["compatible"], data["warnings"]
    report = {"phase": 17, "status": "passed", "checks": [], "source_sha256": {}}
    registry = json.loads((root / "configs/inference/model_registry.json").read_text())
    for model in data["models"]:
        assert model["status"] == "available" and not model["warnings"], model
        version = next(v for v in registry["models"] if v["model_id"] == model["model_id"])
        config = json.loads((root / version["configuration"]["path"]).read_text())
        evaluation = json.loads((root / version["metrics"]["path"]).read_text())
        run = evaluation["runs"][version["comparison_member"]]
        base = root / "runs" / version["run_id"]
        summary = json.loads((base / "training_summary.json").read_text())
        history = json.loads((base / "validation_history.json").read_text())["history"]
        logs = [json.loads(line) for line in (base / "metrics.jsonl").read_text().splitlines()]
        logs = [r for r in logs if r["event"] == "update"]
        assert model["architecture"] == config["model"]
        assert model["token_budget"] == config["max_training_tokens"]
        assert model["checkpoint_update"] == run["selected_update"]
        assert model["selection_rule"] == run["selection_rule"]
        assert model["training_tokens"] == summary["training_tokens"] == run["selected_training_tokens"]
        assert model["training_seconds"] == summary["training_session_wall_seconds"]
        assert model["exposure"] == run["training_exposure"]["source_target_tokens"]
        assert model["tokenizer_type"] == json.loads((root/version["tokenizer"]["path"]).read_text())["kind"]
        manifest = json.loads((root/config["token_data_dir"]/"manifest.json").read_text())
        assert model["manifest"]["sources"] == manifest["prepared_sources"]
        assert model["manifest"]["boundary_policy"] == manifest["boundary_policy"]
        assert model["manifest"]["vocabulary_policy"] == manifest["vocabulary_fit"]["policy"]
        assert len(logs) == len(model["training"])
        for raw,point in zip(logs,model["training"],strict=True):
            assert (point["update"],point["tokens"],point["loss"]) == (raw["optimizer_update"],raw["tokens_processed"],raw["training_loss"])
        for raw,point in zip(history,model["validation"],strict=True):
            assert (point["update"],point["tokens"],point["loss"]) == (raw["optimizer_update"],raw["tokens_processed"],raw["validation_loss"])
        for source,row in model["test_sources"].items():
            raw=evaluation["scores"][version["comparison_member"]]["packed"]["sources"][source]
            assert row == {"loss":raw["cross_entropy_nats"],"perplexity":raw["perplexity"],"target_tokens":raw["target_tokens"]}
        assert model["generation_tokens_per_second"] == evaluation["generation_benchmark"]["summaries"][version["comparison_member"]]["pooled_tokens_per_second"]
        with torch.random.fork_rng():
            network=TransformerLanguageModel(TransformerConfig(**config["model"]))
            assert network.trainable_parameter_count() == model["parameter_count"]
            assert len(network.blocks) == config["model"]["num_layers"]
            del network
        for name, expected in evaluation["identity"]["source_hashes"].items():
            if not name.startswith("model/"):
                continue
            path = root / name
            if not path.is_file():
                raise ValueError("Historical source layout changed; audit against the original source snapshot.")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == expected
        for ref in model["evidence"]:
            digest=hashlib.sha256((root/ref["path"]).read_bytes()).hexdigest()
            assert digest == ref["sha256"]
            report["source_sha256"][ref["path"]]=digest
        report["checks"].append({"model_id":model["model_id"],"training_points":len(logs),
                                "validation_points":len(history),"parameter_count":model["parameter_count"],
                                "overview_metrics_provenance_and_architecture":"passed"})
    (root/"artifacts/phase17-source-verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"models":len(report["checks"]),"source_files":len(report["source_sha256"])}))


def verify_backend():
    """Run the backend gate using a new artifact directory and full-suite JUnit evidence."""

    import argparse
    import json
    from pathlib import Path

    from transformer.verification.checks import run_backend_gate


    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--project-root", type=Path, default=Path.cwd())
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--test-report", type=Path, required=True)
        parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
        args = parser.parse_args()
        report = run_backend_gate(
            args.project_root, args.output, device=args.device, test_report=args.test_report
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.output / "verification.json"),
                    "checks": [
                        {k: v for k, v in c.items() if k != "details"} for c in report["checks"]
                    ],
                },
                indent=2,
            )
        )
        return 0 if report["status"] == "passed" else 1

    main()


def verify_inference():
    """Bounded real-checkpoint verification; never trains, rewrites runs, or downloads data."""

    import argparse
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    import torch

    from transformer.benchmarking.worker import MemoryMonitor, hardware_identity
    from transformer.data.preparation import file_sha256
    from transformer.inference import GenerationConfig, GenerationService


    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--project-root", type=Path, default=Path.cwd())
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
        args = parser.parse_args()
        root = args.project_root.resolve()
        if args.output.exists():
            raise FileExistsError(f"Verification output exists: {args.output}")
        run_dirs = [
            root / "runs/phase8-model-a-tinystories-20261119-v1",
            root / "runs/phase9-model-b-combined-20261119-v1",
        ]

        def originals():
            paths = [p for directory in run_dirs for p in directory.rglob("*") if p.is_file()]
            paths += [p for p in (root / "artifacts/phase10-comparison-v1").rglob("*") if p.is_file()]
            return {str(p.relative_to(root)): file_sha256(p) for p in paths}

        before = originals()
        monitor = MemoryMonitor(torch.device(args.device))
        report = {
            "started_at_utc": datetime.now(UTC).isoformat(),
            "hardware": hardware_identity(),
            "memory_before": monitor.snapshot(),
            "models": {},
            "switch_memory": [],
            "registry_sha256": file_sha256(root / "configs/inference/model_registry.json"),
        }
        cpu_rng = torch.get_rng_state().clone()
        mps_rng = torch.mps.get_rng_state().clone() if args.device == "mps" else None
        ids = ("transformer-tinystories-v1", "transformer-combined-v1")
        with GenerationService(root, device=args.device) as service:
            for model_id, run in zip(ids, run_dirs, strict=True):
                archived = json.loads((run / "diagnostics/samples-update-001000.json").read_text())
                outputs = []
                for prior in archived["samples"]:
                    row = service.generate(
                        model_id, prior["prompt"], GenerationConfig(max_new_tokens=128)
                    )
                    assert row["generated_token_ids"] == prior["generated_ids"]
                    outputs.append(row)
                options = GenerationConfig(
                    mode="sample", temperature=0.8, top_k=20, seed=1234, max_new_tokens=64
                )
                sampled = service.generate(model_id, "Tom found a small box under a tree.", options)
                repeated = service.generate(model_id, sampled["prompt"], options)
                assert sampled["generated_token_ids"] == repeated["generated_token_ids"]
                long_prompt = (archived["samples"][0]["prompt"] + " ") * 8
                longer = service.generate(model_id, long_prompt, GenerationConfig(max_new_tokens=16))
                cropped = service.generate(
                    model_id, long_prompt[-256:], GenerationConfig(max_new_tokens=16)
                )
                assert longer["generated_token_ids"] == cropped["generated_token_ids"]
                assert longer["initial_prompt_tokens_truncated"] > 0
                empty = service.generate(model_id, "a", GenerationConfig(max_new_tokens=0))
                assert empty["generated_token_count"] == 0
                assert not service._loaded.model.training
                assert all(
                    not p.requires_grad and p.grad is None for p in service._loaded.model.parameters()
                )
                report["models"][model_id] = {
                    "metadata": service.load(model_id),
                    "archived_greedy_ids_match": True,
                    "greedy_outputs": outputs,
                    "sampled_output": sampled,
                    "seeded_sample_reproduced": True,
                    "long_prompt_suffix_equivalence": True,
                    "long_prompt_output": longer,
                    "zero_token_result": empty,
                    "parameters_frozen_and_no_gradients": True,
                }
            for cycle in range(3):
                for model_id in ids:
                    service.generate(model_id, "The little bird", GenerationConfig(max_new_tokens=8))
                    report["switch_memory"].append(
                        {
                            "cycle": cycle,
                            "model_id": model_id,
                            "cache": service.cache_info(),
                            **monitor.snapshot(),
                        }
                    )
                    assert service.cache_info()["cached_models"] == 1
            service.unload()
            assert service.cache_info()["cached_models"] == 0
            report["memory_after_unload"] = monitor.snapshot()
            if args.device == "mps":
                live = [r["mps_live_tensor_bytes"] for r in report["switch_memory"]]
                assert max(live) - min(live) < 1024**2
                assert (
                    report["memory_after_unload"]["mps_live_tensor_bytes"]
                    <= report["memory_before"]["mps_live_tensor_bytes"]
                )
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if mps_rng is not None:
            assert torch.equal(mps_rng, torch.mps.get_rng_state())
        assert originals() == before
        report.update(
            status="passed",
            original_run_and_evaluation_files_unchanged=True,
            original_files_verified=len(before),
            cpu_and_mps_rng_preserved=True,
            completed_at_utc=datetime.now(UTC).isoformat(),
            memory_definition="Sampled process RSS, lifetime peak RSS, MPS live tensors and "
            "driver allocations are distinct overlapping counters, not total system memory.",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "output": str(args.output),
                    "original_files_verified": len(before),
                },
                indent=2,
            )
        )

    main()


def verify_web_smoke():
    """Bounded real-registry HTTP smoke check; run against an already-started local server."""
    import json
    import subprocess
    import urllib.request
    from pathlib import Path
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8002)
    parser.add_argument("--server-pid",type=int,required=True)
    args=parser.parse_args()
    base=f"http://127.0.0.1:{args.port}"
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def get(path):
        with opener.open(base+path,timeout=30) as r:return json.load(r)
    catalog=get("/api/v1/models")
    ids=[m["model_id"] for m in catalog["models"]]
    records=[]
    for model_id in [ids[0]]*4+[ids[1]]*4+[ids[0]]*2:
        payload={"model_id":model_id,"prompt":"Tom found a small box under a tree.",
                 "settings":{"mode":"greedy","max_new_tokens":16,"unknown_policy":"error"}}
        req=urllib.request.Request(base+"/api/v1/generations",data=json.dumps(payload).encode(),
                                  headers={"Content-Type":"application/json"})
        with opener.open(req,timeout=60) as r:result=json.load(r)
        assert result["model_id"]==model_id
        assert isinstance(result["continuation"],str)
        assert 0<=result["generated_token_count"]<=16
        assert result["sampling_settings"]["max_new_tokens"]==16
        assert result["checkpoint"]["sha256"]==next(m["checkpoint"]["sha256"] for m in catalog["models"] if m["model_id"]==model_id)
        rss=int(subprocess.check_output(["ps","-o","rss=","-p",str(args.server_pid)],text=True).strip())*1024
        records.append({k:result[k] for k in ["model_id","cache_hit","model_load_seconds","generation_duration_seconds","generated_token_count","device"]}|{"process_rss_bytes":rss})
    for i in [1,2,3,5,6,7,9]:
        assert records[i]["cache_hit"] and records[i]["model_load_seconds"]==0
    assert not records[4]["cache_hit"] and not records[8]["cache_hit"]
    Path("artifacts/phase19-resources.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps({"requests":len(records),"rss_min":min(r["process_rss_bytes"] for r in records),"rss_max":max(r["process_rss_bytes"] for r in records),"rss_final":records[-1]["process_rss_bytes"]}))


def main():
    import sys

    commands = {
        'benchmark-analytics': benchmark_analytics,
        'export-story-charts': export_story_charts,
        'preview-ui-fixtures': preview_ui_fixtures,
        'verify-analytics': verify_analytics,
        'verify-backend': verify_backend,
        'verify-inference': verify_inference,
        'verify-web-smoke': verify_web_smoke,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: maintenance.py COMMAND [options]\n\nCommands:\n  " + "\n  ".join(commands))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    commands[sys.argv.pop(1)]()


if __name__ == "__main__":
    main()
