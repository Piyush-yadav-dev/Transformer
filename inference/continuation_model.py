"""Local prompt-conditioned model and lightweight attention adapters."""

import torch
from torch import nn

INSTRUCTION = (
    "Continue the supplied story opening in simple English. "
    "Keep its characters and situation. Write only the continuation, "
    "with a clear ending in two to four short paragraphs."
)


def opening_ids(tokenizer, opening, instruction=INSTRUCTION):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": instruction}, {"role": "user", "content": opening}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )


class AttentionAdapter(nn.Module):
    """Train a small weight update while the pretrained projection stays frozen."""

    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x.float())).to(x.dtype)


def add_adapters(model):
    model.requires_grad_(False)
    for layer in model.model.layers[-8:]:
        layer.self_attn.v_proj = AttentionAdapter(layer.self_attn.v_proj)
        layer.self_attn.o_proj = AttentionAdapter(layer.self_attn.o_proj)
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def load_base(path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float16 if str(device) == "mps" else torch.float32,
    ).to(device)
    return model, tok


@torch.inference_mode()
def continue_opening(model, tokenizer, prompt, settings, generator, *, instruction=INSTRUCTION):
    device = model.device
    if settings.max_new_tokens >= 400:
        # Bade budget par chhoti summary ke bajay story ko scenes mein badhao.
        words = int(settings.max_new_tokens * 0.6)
        instruction += (
            f" Develop a longer story of approximately {words} words, within the output limit."
            " Use several paragraphs with dialogue, connected events, and a clear ending."
            " Resolve the situation in the opening. Avoid rushing to the ending or repeating events."
        )
    full = opening_ids(tokenizer, prompt, instruction)
    capacity = 2048
    if len(full) > capacity:
        raise ValueError("Keep the opening below 2,048 subword tokens.")
    x = torch.tensor([full], device=device)
    cache, generated = None, []
    eos = model.generation_config.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or [tokenizer.eos_token_id])
    for _ in range(settings.max_new_tokens):
        result = model(x, cache=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        logits = result.logits[0, -1].float()
        if settings.mode == "greedy":
            token = int(logits.argmax().item())
        else:
            if settings.top_k:
                values, indices = logits.topk(settings.top_k)
            else:
                values, indices = logits, None
            probs = torch.softmax(values.cpu() / settings.temperature, dim=0)
            choice = int(torch.multinomial(probs, 1, generator=generator).item())
            token = choice if indices is None else int(indices[choice].item())
        generated.append(token)
        if token in stop_ids:
            break
        x = torch.tensor([[token]], device=device)
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    prefix = prompt.strip()
    if text.startswith(prefix) and text[len(prefix) : len(prefix) + 1].isspace():
        text = text[len(prefix) :].lstrip()
    return (
        text,
        generated,
        "eos" if generated and generated[-1] in stop_ids else "max_new_tokens",
        len(full),
        capacity,
    )
