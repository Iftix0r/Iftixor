import asyncio
import os
import sys

from dotenv import load_dotenv

from iftixor.bot import build_application

# Python 3.12+ da asyncio.get_event_loop() joriy loop bo'lmasa endi xato
# qaytaradi (avval avtomatik yaratardi). python-telegram-bot 21.x hali ham
# shu funksiyaga tayanadi, shuning uchun run_polling() dan oldin loop'ni
# qo'lda o'rnatib qo'yamiz.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


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
