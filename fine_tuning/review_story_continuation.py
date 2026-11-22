"""Review the larger model on the same prompt set with a focus on continuity."""

import json
from pathlib import Path

import torch

from transformer.configuration.settings import GenerationConfig
from transformer.inference.continuation_model import add_adapters, continue_opening, load_base

INSTRUCTION = (
    "Continue the user's story in simple English. Begin with what happens right after "
    "the opening. Explain the sound, object, or surprise in the opening early in the story. "
    "Keep the same characters and situation. Add one small event and a satisfying ending. "
    "Write two or three short paragraphs. Do not repeat the opening or address the reader."
)


def main():
    run = Path("runs/story-continuation-v2")
    model, tokenizer = load_base("data/models/qwen-story-base-1.5b", "mps")
    parameters = add_adapters(model)
    adapter = torch.load(run / "checkpoints/adapter.pt", weights_only=True)
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(adapter[name])
    model.eval().requires_grad_(False)
    torch.set_num_threads(2)
    results = json.loads((run / "results.json").read_text())
    samples = []
    settings = GenerationConfig(
        mode="sample", temperature=0.65, top_k=20, max_new_tokens=250, seed=42
    )
    for row in results["samples"][:6]:
        prompt = row["opening"]
        text, _, stop, *_ = continue_opening(
            model,
            tokenizer,
            prompt,
            settings,
            torch.Generator().manual_seed(42),
            instruction=INSTRUCTION,
        )
        sample = {"opening": prompt, "continuation": text, "stop_reason": stop}
        samples.append(sample)
        print(json.dumps(sample), flush=True)
    (run / "continuity-review.json").write_text(
        json.dumps(
            {
                "instruction": INSTRUCTION,
                "sampling": settings.to_dict(),
                "samples": samples,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
