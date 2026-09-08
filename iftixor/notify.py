import json
import os
import urllib.request

from dotenv import load_dotenv

load_dotenv()

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
_ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")


def notify_admin(text: str) -> None:
    """Adminga Telegram orqali xabar yuboradi. Token/chat_id sozlanmagan bo'lsa jim o'tkazib yuboradi."""
    if not _TOKEN or not _ADMIN_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": _ADMIN_CHAT_ID, "text": text}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=10)
    except Exception as exc:
        print(f"Adminga xabar yuborilmadi: {exc}")
