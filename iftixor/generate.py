import torch

from .config import GPTConfig
from .model import GPT
from .tokenizer import CharTokenizer


def load_model(checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    tokenizer = CharTokenizer(checkpoint["stoi"], checkpoint["itos"])
    return model, tokenizer


@torch.no_grad()
def generate_text(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cpu",
) -> str:
    prompt = prompt or "\n"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())
