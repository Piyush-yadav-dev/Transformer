"""Fine-tune on story-opening/continuation pairs and save raw held-out samples."""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from transformer.configuration.settings import GenerationConfig
from transformer.inference.continuation_model import (
    add_adapters,
    continue_opening,
    load_base,
    opening_ids,
)

BASE = Path("data/models/qwen-story-base")
DATA = Path("data/prepared/english-stories-expanded-v1")
RUN = Path("runs/story-continuation-v1")
SOURCES = ("tinystories", "simplestories")
PROMPTS = (
    "Tom opened a small box and heard a quiet tapping inside.",
    "The little bird found a blue ribbon beside the old oak tree.",
    "Mia heard her friend crying behind the garden gate.",
    "When the lights went out, Noah heard his dog scratching at the kitchen door.",
    "Lena kept a tiny paper boat in her pocket on the first day of school.",
    "A fox woke up to find fresh footprints all around his den.",
)


def examples(split, size):
    paths = (
        [DATA / f"train_{name}.jsonl" for name in SOURCES]
        if split == "train"
        else [DATA / f"{split}.jsonl"]
    )
    found = {name: [] for name in SOURCES}
    for path in paths:
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                name = row["source"]
                if name not in found or len(found[name]) >= size:
                    continue
                parts = re.split(r"(?<=[.!?])\s+", row["text"].strip(), maxsplit=1)
                if len(parts) == 2 and 25 <= len(parts[0]) <= 180 and 100 <= len(parts[1]) <= 850:
                    found[name].append(
                        {"opening": parts[0], "target": parts[1], "id": row["id"], "source": name}
                    )
                if all(len(items) == size for items in found.values()):
                    break
    if any(len(items) != size for items in found.values()):
        raise RuntimeError(f"Insufficient examples in {split}")
    return [row for pair in zip(*(found[name] for name in SOURCES), strict=True) for row in pair]


def loss(model, tokenizer, row):
    prompt = opening_ids(tokenizer, row["opening"])
    target = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
    x = torch.tensor([prompt + target[:-1]], device=model.device)
    hidden = model.model(x, use_cache=False).last_hidden_state[:, len(prompt) - 1 :, :]
    logits = model.lm_head(hidden).float()
    return F.cross_entropy(logits[0], torch.tensor(target, device=model.device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--larger-base", action="store_true")
    args = parser.parse_args()
    run = Path("runs/story-continuation-v2") if args.larger_base else RUN
    base = Path("data/models/qwen-story-base-1.5b") if args.larger_base else BASE
    if args.updates < 2 or args.updates % 2:
        parser.error("--updates must be a positive even number")
    train, validation, test = (
        examples("train", args.updates // 2),
        examples("validation", 4),
        examples("test", 3),
    )
    assert len({row["id"] for row in train + validation + test}) == len(train + validation + test)
    run.mkdir(parents=True, exist_ok=False)
    (run / "examples.json").write_text(
        json.dumps({"train": train, "validation": validation, "test": test}, indent=2)
    )
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model, tokenizer = load_base(base, "mps")
    adapters = add_adapters(model)
    optimizer = torch.optim.AdamW(adapters.values(), lr=0.0002)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()

    def check_validation():
        model.eval()
        with torch.no_grad():
            return sum(float(loss(model, tokenizer, row)) for row in validation) / len(validation)

    before = check_validation()
    print(f"Initial validation {before:.4f}", flush=True)
    best, selected = float("inf"), 0
    for step, row in enumerate(train, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        value = loss(model, tokenizer, row)
        if not torch.isfinite(value):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapters.values(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step % 8 == 0:
            print(
                f"update {step}/{len(train)} loss {float(value.detach()):.4f} "
                f"grad {float(norm):.3f}",
                flush=True,
            )
        if step % 24 == 0 or step == len(train):
            score = check_validation()
            print(f"validation {score:.4f}", flush=True)
            if score < best:
                best, selected = score, step
                torch.save(
                    {name: param.detach().cpu() for name, param in adapters.items()},
                    checkpoints / "adapter.pt",
                )
    saved = torch.load(checkpoints / "adapter.pt", weights_only=True)
    with torch.no_grad():
        for name, param in adapters.items():
            param.copy_(saved[name])
    model.eval()
    settings = GenerationConfig(
        mode="sample", temperature=0.65, top_k=20, max_new_tokens=250, seed=42
    )
    rows = []
    for prompt in [*PROMPTS, *(row["opening"] for row in test)]:
        text, *_ = continue_opening(
            model, tokenizer, prompt, settings, torch.Generator().manual_seed(42)
        )
        rows.append({"opening": prompt, "continuation": text})
        print(json.dumps(rows[-1]), flush=True)
    (run / "results.json").write_text(
        json.dumps(
            {
                "updates": len(train),
                "selected_update": selected,
                "validation_before": before,
                "validation_after": best,
                "trainable_parameters": sum(p.numel() for p in adapters.values()),
                "samples": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
