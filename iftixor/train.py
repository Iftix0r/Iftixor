import argparse
import time
from pathlib import Path

import torch

from .config import GPTConfig
from .dataset import get_batch
from .model import GPT
from .notify import notify_admin
from .tokenizer import CharTokenizer


def save_checkpoint(out_path: Path, model: GPT, optimizer: torch.optim.Optimizer, config: GPTConfig,
                     tokenizer: CharTokenizer, step: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.__dict__,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "step": step,
        },
        tmp_path,
    )
    tmp_path.replace(out_path)


def main():
    parser = argparse.ArgumentParser(description="Iftixor til modelini noldan o'qitish (CPU uchun mo'ljallangan)")
    parser.add_argument("--data", nargs="+", default=["data/corpus.txt"],
                         help="Bitta yoki bir nechta matn fayli (barchasi birlashtirilib o'qitiladi)")
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
    parser.add_argument("--resume", action="store_true",
                         help="--out'dagi mavjud checkpointdan davom ettirish (masalan, jarayon uzilib qolgandan keyin)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"
    out_path = Path(args.out)

    texts = []
    for data_path in args.data:
        p = Path(data_path)
        if p.exists():
            texts.append(p.read_text(encoding="utf-8"))
        else:
            print(f"Ogohlantirish: {p} topilmadi, o'tkazib yuborildi.")
    text = "\n".join(texts)

    start_step = 0
    if args.resume and out_path.exists():
        checkpoint = torch.load(out_path, map_location=device)
        tokenizer = CharTokenizer(checkpoint["stoi"], checkpoint["itos"])
        config = GPTConfig(**checkpoint["config"])
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint.get("step", 0)
        print(f"Checkpointdan davom ettirilmoqda: {out_path} (qadam {start_step} dan)")
    else:
        tokenizer = CharTokenizer.from_text(text)
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

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vocab hajmi: {tokenizer.vocab_size} | Parametrlar soni: {n_params:,} | Qurilma: {device}")
    if start_step == 0:
        notify_admin(
            f"Iftixor o'qitish boshlandi.\n"
            f"Qadamlar: {args.steps} | Parametrlar: {n_params:,}"
        )

    start = time.time()
    for step in range(start_step + 1, args.steps + 1):
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
            eta_min = (elapsed / (step - start_step)) * (args.steps - step) / 60
            print(f"step {step}/{args.steps} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f} | {elapsed:.1f}s")
            notify_admin(
                f"Iftixor o'qitilmoqda: {step}/{args.steps} ({percent:.1f}%)\n"
                f"Train loss: {loss.item():.4f} | Val loss: {val_loss.item():.4f}\n"
                f"O'tgan vaqt: {elapsed / 60:.1f} daq | Taxminiy qolgan: {eta_min:.1f} daq"
            )
            save_checkpoint(out_path, model, optimizer, config, tokenizer, step)

    print(f"Model saqlandi: {out_path}")
    notify_admin(f"Iftixor o'qitildi va saqlandi: {out_path}\nJami vaqt: {(time.time() - start) / 60:.1f} daqiqa")


if __name__ == "__main__":
    main()
