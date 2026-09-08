import argparse

from .generate import generate_text, load_model


def main():
    parser = argparse.ArgumentParser(description="Iftixor modeli bilan terminalda sinov suhbati")
    parser.add_argument("--checkpoint", default="checkpoints/iftixor.pt")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)
    print("Iftixor bilan suhbat boshlandi. Chiqish uchun 'exit' deb yozing.\n")

    history = ""
    while True:
        user_input = input("Siz: ")
        if user_input.strip().lower() == "exit":
            break

        history += f"Foydalanuvchi: {user_input}\nIftixor:"
        generated = generate_text(
            model, tokenizer, history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        reply = generated[len(history):].split("Foydalanuvchi:")[0].strip() or "..."
        print(f"Iftixor: {reply}\n")
        history += f" {reply}\n"


if __name__ == "__main__":
    main()
