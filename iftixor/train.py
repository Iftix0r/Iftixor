import argparse
import time
from pathlib import Path

import torch

from .config import GPTConfig
from .dataset import get_batch
from .model import GPT
from .notify import notify_admin
from .tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser(description="Iftixor til modelini noldan o'qitish (CPU uchun mo'ljallangan)")
    parser.add_argument("--data", default="data/corpus.txt")
    parser.add_argument("--out", default="checkpoints/iftixor.pt")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vocab hajmi: {tokenizer.vocab_size} | Parametrlar soni: {n_params:,} | Qurilma: {device}")
    notify_admin(
        f"Iftixor o'qitish boshlandi.\n"
        f"Qadamlar: {args.steps} | Parametrlar: {n_params:,}"
    )

    start = time.time()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data, args.block_size, args.batch_size, device)
                _, val_loss = model(xv, yv)
            model.train()
            elapsed = time.time() - start
            percent = 100 * step / args.steps
            eta_min = (elapsed / step) * (args.steps - step) / 60
            print(f"step {step}/{args.steps} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f} | {elapsed:.1f}s")
            notify_admin(
                f"Iftixor o'qitilmoqda: {step}/{args.steps} ({percent:.1f}%)\n"
                f"Train loss: {loss.item():.4f} | Val loss: {val_loss.item():.4f}\n"
                f"O'tgan vaqt: {elapsed / 60:.1f} daq | Taxminiy qolgan: {eta_min:.1f} daq"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        out_path,
    )
    print(f"Model saqlandi: {out_path}")
    notify_admin(f"Iftixor o'qitildi va saqlandi: {out_path}\nJami vaqt: {(time.time() - start) / 60:.1f} daqiqa")


if __name__ == "__main__":
    main()
