"""Register the reviewed story adapter as the app's sole active model."""

import argparse
import hashlib
import json
from pathlib import Path

from transformer.inference.service import load_registered_model
from transformer.inference.service import (
    REGISTRY_PATH,
    ArtifactReference,
    ModelRegistry,
    ModelVersion,
)
from transformer.training.engine import atomic_json

RUN = Path("runs/story-continuation-v1")
BASE = Path("data/models/qwen-story-base")
MODEL_ID = "transformer-story-continuation-v1"


def reference(path):
    with path.open("rb") as source:
        return ArtifactReference(path.as_posix(), hashlib.file_digest(source, "sha256").hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--larger-base", action="store_true")
    args = parser.parse_args()
    run = Path("runs/story-continuation-v2") if args.larger_base else RUN
    base = Path("data/models/qwen-story-base-1.5b") if args.larger_base else BASE
    model_id = "transformer-story-continuation-v2" if args.larger_base else MODEL_ID
    model_name = "Qwen2.5-1.5B-Instruct" if args.larger_base else "Qwen2.5-0.5B-Instruct"
    revision = (
        (base / "revision.txt").read_text().strip()
        if args.larger_base
        else "7ae557604adf67be50417f59c2c2f167def9a775"
    )
    results = json.loads((run / "results.json").read_text())
    if not results["samples"] or results["validation_after"] >= results["validation_before"]:
        raise ValueError("Story checkpoint lacks successful validation and sample evidence.")
    review = json.loads((run / "continuity-review.json").read_text()) if args.larger_base else None
    adapter = reference(run / "checkpoints/adapter.pt")
    tokenizer = reference(base / "tokenizer.json")
    base_config = json.loads((base / "config.json").read_text())
    config_path = run / "configuration.json"
    atomic_json(
        config_path,
        {
            "model_id": model_id,
            "base_model": model_name,
            "base_revision": revision,
            **({"instruction": review["instruction"]} if review else {}),
            "base_files": {
                path.name: reference(path).sha256 for path in base.iterdir() if path.is_file()
            },
            "architecture": {"layers": base_config["num_hidden_layers"]},
        },
    )
    record = Path("artifacts") / run.name / "registration.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    if review:
        results["continuity_review"] = review
        updated = {row["opening"]: row for row in review["samples"]}
        results["samples"] = [updated.get(row["opening"], row) for row in results["samples"]]
    atomic_json(record, {**results, "adapter_sha256": adapter.sha256})
    registry = ModelRegistry(".")
    version = ModelVersion(
        model_id=model_id,
        kind="combined",
        comparison_member="story_continuation",
        run_id=run.name,
        checkpoint=adapter,
        configuration=reference(config_path),
        tokenizer=tokenizer,
        metrics=reference(record),
        tokenizer_fingerprint=tokenizer.sha256,
        checkpoint_update=results["selected_update"],
    )
    load_registered_model(registry, version, "mps")
    path = Path(REGISTRY_PATH)
    old = json.loads(path.read_text())
    archive = record.with_name("previous-registry.json")
    if not archive.exists():
        atomic_json(archive, old)
    atomic_json(path, {**old, "models": [version.to_dict()]})


if __name__ == "__main__":
    main()
