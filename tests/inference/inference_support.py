"""Helpers for deliberately corrupting synthetic, disposable loader fixtures."""

import json
from pathlib import Path

import torch

from transformer.data.preparation import file_sha256


def rewrite_checkpoint(root, mutate):
    """Re-pin an intentionally invalid synthetic payload to reach deeper compatibility checks."""
    path = root / "configs/inference/model_registry.json"
    registry = json.loads(path.read_text())
    entry = registry["models"][0]
    checkpoint = root / entry["checkpoint"]["path"]
    payload = torch.load(checkpoint, weights_only=True)
    mutate(payload)
    torch.save(payload, checkpoint)
    entry["checkpoint"]["sha256"] = file_sha256(checkpoint)
    metric_path = root / entry["metrics"]["path"]
    metrics = json.loads(metric_path.read_text())
    metrics["runs"]["model_a"]["checkpoint"]["sha256"] = entry["checkpoint"]["sha256"]
    metric_path.write_text(json.dumps(metrics))
    for row in registry["models"]:
        row["metrics"]["sha256"] = file_sha256(metric_path)
    path.write_text(json.dumps(registry))


def read_registry(root):
    path = Path(root) / "configs/inference/model_registry.json"
    return path, json.loads(path.read_text())
