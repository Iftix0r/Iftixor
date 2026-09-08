import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "Siz o'zbek tilida sun'iy intellekt uchun o'quv ma'lumoti (dataset) yaratish "
    "yordamchisisiz. Iftixor ismli kichik til modelini o'qitish uchun namunaviy "
    "suhbatlar yozasiz. Iftixor — do'stona, foydali, qisqa va aniq javob beradigan "
    "yordamchi. Har bir namunani ANIQ shu formatda yozing, boshqa hech narsa "
    "qo'shmang:\n\n"
    "Foydalanuvchi: <savol yoki xabar>\n"
    "Iftixor: <javob>\n\n"
    "Turli mavzularda yozing: kundalik hayot, ta'lim, texnologiya, salomlashish, "
    "O'zbekiston, umumiy bilim, maslahat so'rash, hazil. Har bir namunani bo'sh "
    "qator bilan ajrating. Takrorlanmaydigan, tabiiy o'zbek tilida yozing."
)


def generate_batch(client: OpenAI, model: str, batch_size: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{batch_size} ta yangi, bir-biriga o'xshamagan namuna yozing."},
        ],
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI yordamida Iftixor uchun o'zbek tilida sun'iy o'quv ma'lumoti (dataset) yaratish"
    )
    parser.add_argument("--count", type=int, default=200, help="Jami nechta namuna (savol-javob) yaratish")
    parser.add_argument("--batch-size", type=int, default=20, help="Har bir so'rovda nechta namuna so'raladi")
    parser.add_argument("--out", default="data/synthetic.txt")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Xatolik: .env faylida OPENAI_API_KEY topilmadi.")

    client = OpenAI(api_key=api_key)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    with out_path.open("a", encoding="utf-8") as f:
        while generated < args.count:
            batch = min(args.batch_size, args.count - generated)
            text = generate_batch(client, args.model, batch)
            f.write(text + "\n\n")
            f.flush()
            generated += batch
            print(f"{generated}/{args.count} namuna yaratildi")

    print(f"Tayyor: {out_path}")


if __name__ == "__main__":
    main()
