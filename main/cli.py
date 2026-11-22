"""Command-line entry points for the Transformer backend foundation."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from transformer import __version__
from transformer.configuration.settings import DEVICE_CHOICES, ProjectConfig, RuntimeConfig
from transformer.runtime.device import select_device
from transformer.runtime.device import collect_diagnostics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transformer", description="Transformer backend utilities."
    )
    parser.add_argument("--version", action="version", version=f"Transformer {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)
    diagnostic = subcommands.add_parser(
        "diagnose", help="Report the environment and selected device."
    )
    diagnostic.add_argument("--config", type=Path, help="Read a JSON configuration file.")
    diagnostic.add_argument(
        "--device", choices=DEVICE_CHOICES, help="Override the configured device."
    )
    diagnostic.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    diagnostic.add_argument(
        "--check", action="store_true", help="Run a tiny tensor and gradient check."
    )
    diagnostic.add_argument(
        "--save-config", type=Path, help="Save effective settings to a new JSON file."
    )
    prepare = subcommands.add_parser(
        "prepare-data", help="Acquire and prepare bounded story subsets."
    )
    prepare.add_argument(
        "--config", type=Path, required=True, help="Revision-pinned data configuration."
    )
    prepare.add_argument("--project-root", type=Path, default=Path.cwd())
    prepare.add_argument(
        "--project-config", type=Path, help="Optional Phase 1 project path settings."
    )
    prepare.add_argument(
        "--offline", action="store_true", help="Require verified local subset caches."
    )
    tokens = subcommands.add_parser(
        "prepare-tokens", help="Freeze the shared tokenizer and encode prepared stories."
    )
    tokens.add_argument(
        "--config", type=Path, required=True, help="Token preparation configuration."
    )
    tokens.add_argument("--project-root", type=Path, default=Path.cwd())
    tokens.add_argument("--project-config", type=Path, help="Optional project path settings.")
    check = subcommands.add_parser(
        "check-token-data", help="Read a bounded number of CPU token batches without training."
    )
    check.add_argument(
        "--tokens", type=Path, required=True, help="Prepared token artifact directory."
    )
    check.add_argument(
        "--tokenizer",
        type=Path,
        help="Expected tokenizer file; defaults to the artifact tokenizer.",
    )
    check.add_argument("--loader-config", type=Path, help="Data-loader configuration JSON.")
    check.add_argument("--stream", default="train", help="One named training or evaluation stream.")
    check.add_argument("--batches", type=int, default=2, help="Maximum batches to inspect (1–100).")
    model_check = subcommands.add_parser(
        "check-model", help="Run one synthetic forward/backward check without optimizer updates."
    )
    model_check.add_argument(
        "--config", type=Path, required=True, help="Transformer configuration."
    )
    model_check.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    model_check.add_argument("--seed", type=int, default=42)
    train = subcommands.add_parser(
        "train", help="Run or resume explicitly budgeted local training."
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--project-root", type=Path, default=Path.cwd())
    train.add_argument("--project-config", type=Path)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report budget without updates or run files.",
    )
    train.add_argument(
        "--stop-after-updates",
        type=int,
        help="Stop at this absolute update without changing the schedule budget.",
    )
    for command, help_text in (
        ("training-smoke", "Run a bounded prepared-data save/reload/resume diagnostic."),
        ("overfit-check", "Try learning one synthetic window within a strict update budget."),
        ("benchmark-training", "Measure a small bounded hardware matrix in fresh processes."),
    ):
        diagnostic_run = subcommands.add_parser(command, help=help_text)
        diagnostic_run.add_argument("--config", type=Path, required=True)
        diagnostic_run.add_argument("--project-root", type=Path, default=Path.cwd())
        if command != "benchmark-training":
            diagnostic_run.add_argument("--output", type=Path, required=True)
        if command == "overfit-check":
            diagnostic_run.add_argument("--max-seconds", type=float, default=30.0)
    baseline = subcommands.add_parser(
        "train-baseline", help="Run Model A under a frozen matched-token protocol."
    )
    baseline.add_argument("--protocol", type=Path, required=True)
    baseline.add_argument("--project-root", type=Path, default=Path.cwd())
    baseline.add_argument("--resume", action="store_true")
    combined = subcommands.add_parser(
        "train-combined", help="Prepare or train the frozen Model B mixture."
    )
    combined.add_argument("--protocol", type=Path, required=True)
    combined.add_argument("--plan", type=Path, required=True)
    combined.add_argument("--output", type=Path, required=True)
    combined.add_argument("--project-root", type=Path, default=Path.cwd())
    combined.add_argument("--prepare-only", action="store_true")
    combined.add_argument("--resume", action="store_true")
    evaluation = subcommands.add_parser(
        "evaluate-comparison", help="Evaluate frozen selected checkpoints on held-out text."
    )
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--project-root", type=Path, default=Path.cwd())
    models = subcommands.add_parser("list-models", help="List explicit trusted project model IDs.")
    models.add_argument("--project-root", type=Path, default=Path.cwd())
    generation = subcommands.add_parser(
        "generate", help="Generate from one registered local model."
    )
    generation.add_argument("--project-root", type=Path, default=Path.cwd())
    generation.add_argument(
        "--model", required=True, help="Stable ID from list-models; never a file path."
    )
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    generation.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    generation.add_argument("--max-new-tokens", type=int, default=128)
    generation.add_argument("--temperature", type=float)
    generation.add_argument("--top-k", type=int)
    generation.add_argument("--seed", type=int)
    generation.add_argument("--unknown-policy", choices=("error", "replace"), default="error")
    generation.add_argument("--json", action="store_true", help="Include full generation metadata.")
    args = parser.parse_args(argv)

    try:
        if args.command == "list-models":
            from transformer.inference import ModelRegistry

            print(json.dumps(ModelRegistry(args.project_root).list_models(), indent=2))
            return 0
        if args.command == "generate":
            from transformer.inference import GenerationConfig, GenerationService

            settings = GenerationConfig(
                mode=args.mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
                unknown_policy=args.unknown_policy,
            )
            with GenerationService(args.project_root, device=args.device) as service:
                result = service.generate(args.model, args.prompt, settings)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
            else:
                print(result["continuation"])
                print(
                    f"[{result['model_id']} | {result['generated_token_count']} new tokens | "
                    f"{result['stop_reason']} | {result['generation_duration_seconds']:.3f}s "
                    f"generation; load {result['model_load_seconds']:.3f}s]"
                )
            return 0
        if args.command == "evaluate-comparison":
            from transformer.configuration.settings import EvaluationConfig
            from transformer.evaluation.comparison import run_comparison

            result = run_comparison(EvaluationConfig.load(args.config), args.project_root)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if args.command == "train-combined":
            from transformer.configuration.settings import ComparisonProtocol
            from transformer.training.experiments import run_combined

            result = run_combined(
                ComparisonProtocol.load(args.protocol),
                args.project_root,
                args.output,
                args.plan,
                resume=args.resume,
                prepare_only=args.prepare_only,
            )
            print(json.dumps(result, indent=2, allow_nan=False))
            return 130 if result["status"] == "interrupted" else 0
        if args.command == "train-baseline":
            from transformer.configuration.settings import ComparisonProtocol
            from transformer.training.experiments import run_baseline

            result = run_baseline(
                ComparisonProtocol.load(args.protocol), args.project_root, resume=args.resume
            )
            print(json.dumps(result, indent=2, allow_nan=False))
            return 130 if result["status"] == "interrupted" else 0
        if args.command == "benchmark-training":
            from transformer.benchmarking.hardware_benchmark import run_benchmark
            from transformer.configuration.settings import BenchmarkConfig

            result = run_benchmark(BenchmarkConfig.load(args.config), args.project_root)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["status"] == "completed" else 2
        if args.command in ("training-smoke", "overfit-check"):
            from transformer.benchmarking.worker import run_overfit, run_smoke
            from transformer.configuration.settings import TrainingConfig

            config = TrainingConfig.load(args.config)
            if args.command == "training-smoke":
                result = run_smoke(config, args.project_root, args.output)
            else:
                result = run_overfit(
                    config, args.project_root, args.output, max_seconds=args.max_seconds
                )
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["status"] == "passed" else 2
        if args.command == "train":
            from transformer.configuration.settings import TrainingConfig
            from transformer.training.engine import run_training

            project = (
                ProjectConfig.load(args.project_config) if args.project_config else ProjectConfig()
            )
            result = run_training(
                TrainingConfig.load(args.config),
                args.project_root,
                project.paths,
                resume=args.resume,
                stop_after_updates=args.stop_after_updates,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, indent=2, allow_nan=False))
            return 130 if result["status"] == "interrupted" else 0
        if args.command == "check-model":
            from transformer.configuration.settings import TransformerConfig
            from transformer.runtime.device import check_model

            report = check_model(TransformerConfig.load(args.config), args.device, args.seed)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "prepare-tokens":
            from transformer.configuration.settings import TokenPreparationConfig
            from transformer.tokenization.token_preparation import prepare_tokens

            project = (
                ProjectConfig.load(args.project_config) if args.project_config else ProjectConfig()
            )
            result = prepare_tokens(
                TokenPreparationConfig.load(args.config), args.project_root, project.paths
            )
            print(
                json.dumps(
                    {
                        "directory": str(result.directory),
                        "reused": result.reused,
                        "tokenizer_fingerprint": result.manifest["tokenizer_fingerprint"],
                        "dataset_fingerprint": result.manifest["dataset_fingerprint"],
                        "vocabulary_size": result.manifest["vocabulary_size"],
                        "streams": {
                            name: {
                                key: info[key]
                                for key in ("documents", "tokens", "dtype", "unknown_tokens")
                            }
                            for name, info in result.manifest["streams"].items()
                        },
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "check-token-data":
            from itertools import islice

            from transformer.configuration.settings import DataLoaderConfig
            from transformer.data.loading import (
                TokenWindowDataset,
                create_data_loader,
            )
            from transformer.tokenization.character_tokenizer import CharacterTokenizer

            if not 1 <= args.batches <= 100:
                raise ValueError("--batches must be between 1 and 100.")
            settings = (
                DataLoaderConfig.load(args.loader_config)
                if args.loader_config
                else DataLoaderConfig()
            )
            tokenizer = CharacterTokenizer.load(args.tokenizer or args.tokens / "tokenizer.json")
            with TokenWindowDataset(
                args.tokens,
                args.stream,
                tokenizer.fingerprint,
                context_length=settings.context_length,
            ) as dataset:
                loader = create_data_loader(dataset, settings)
                shapes = []
                for inputs, targets in islice(loader, args.batches):
                    if (
                        inputs.dtype != targets.dtype
                        or inputs.device.type != "cpu"
                        or targets.device.type != "cpu"
                    ):
                        raise RuntimeError("Unexpected token batch dtype or device.")
                    shapes.append(
                        {
                            "input": list(inputs.shape),
                            "target": list(targets.shape),
                            "dtype": str(inputs.dtype),
                            "device": str(inputs.device),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "stream": args.stream,
                            "windows": len(dataset),
                            "dropped_tail_pairs": dataset.dropped_tail_pairs,
                            "loader_configuration": settings.to_dict(),
                            "tokenizer_fingerprint": tokenizer.fingerprint,
                            "batches": shapes,
                        },
                        indent=2,
                    )
                )
            return 0
        if args.command == "prepare-data":
            from transformer.configuration.settings import DataPreparationConfig
            from transformer.data.preparation import prepare_data

            project = (
                ProjectConfig.load(args.project_config) if args.project_config else ProjectConfig()
            )
            result = prepare_data(
                DataPreparationConfig.load(args.config),
                args.project_root,
                project.paths,
                offline=args.offline,
            )
            print(
                json.dumps(
                    {
                        "directory": str(result.directory),
                        "reused": result.reused,
                        "corpus_fingerprint": result.manifest["corpus_fingerprint"],
                        "documents": {
                            name: info["documents"]
                            for name, info in result.manifest["files"].items()
                        },
                        "counts": result.manifest["counts"],
                    },
                    indent=2,
                )
            )
            return 0
        config = ProjectConfig.load(args.config) if args.config else ProjectConfig()
        if args.device is not None:
            config = replace(config, runtime=RuntimeConfig(device=args.device))
        selection = select_device(config.runtime.device)
        report = collect_diagnostics(selection, check=args.check)
        if args.save_config:
            config.save(args.save_config)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"transformer: error: {exc}\n")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("Transformer environment diagnostics")
        for name, value in report.to_dict().items():
            label = (
                name.replace("_", " ")
                .capitalize()
                .replace("Pytorch", "PyTorch")
                .replace("Mps", "MPS")
            )
            print(f"{label}: {value}")
    return 0
