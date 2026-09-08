import os
import sys

from dotenv import load_dotenv

from iftixor.bot import build_application


def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints/iftixor.pt")

    if not token:
        sys.exit("Xatolik: .env faylida TELEGRAM_BOT_TOKEN topilmadi.")
    if not os.path.exists(checkpoint_path):
        sys.exit(
            f"Xatolik: model checkpoint topilmadi ({checkpoint_path}).\n"
            "Avval modelni o'qiting: python -m iftixor.train"
        )

    application = build_application(token, checkpoint_path)
    print("Iftixor bot ishga tushdi. To'xtatish uchun Ctrl+C.")
    application.run_polling()


if __name__ == "__main__":
    main()
