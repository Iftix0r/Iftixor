import logging
import os
import subprocess
import sys
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .generate import generate_text, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iftixor.bot")

MAX_HISTORY_CHARS = 800
DEFAULT_RETRAIN_STEPS = 1500
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "conversations.txt"

WELCOME_TEXT = (
    "Salom! Men Iftixor — noldan o'qitilayotgan tajriba AI modeliman.\n"
    "Hozircha kichik hajmda o'qiganman, shuning uchun javoblarim cheklangan bo'lishi mumkin.\n"
    "/reset — suhbat tarixini tozalash."
)


def _is_admin(update: Update) -> bool:
    admin_chat_id = os.getenv("ADMIN_CHAT_ID")
    return bool(admin_chat_id) and str(update.effective_chat.id) == str(admin_chat_id)


def _log_conversation(user_text: str, reply: str) -> None:
    CONVERSATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONVERSATIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(f"Foydalanuvchi: {user_text}\nIftixor: {reply}\n\n")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["history"] = ""
    await update.message.reply_text(WELCOME_TEXT)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["history"] = ""
    await update.message.reply_text("Suhbat tarixi tozalandi.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model = context.bot_data["model"]
    tokenizer = context.bot_data["tokenizer"]

    history = context.chat_data.get("history", "")
    history += f"Foydalanuvchi: {update.message.text}\nIftixor:"
    if len(history) > MAX_HISTORY_CHARS:
        history = history[-MAX_HISTORY_CHARS:]

    generated = generate_text(model, tokenizer, history, max_new_tokens=150, temperature=0.8, top_k=40)
    reply = generated[len(history):].split("Foydalanuvchi:")[0].strip() or "..."

    context.chat_data["history"] = history + f" {reply}\n"
    _log_conversation(update.message.text, reply)
    await update.message.reply_text(reply)


async def retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("Bu buyruq faqat admin uchun.")
        return

    process = context.bot_data.get("retrain_process")
    if process is not None and process.poll() is None:
        await update.message.reply_text("Qayta o'qitish allaqachon ketmoqda, kuting.")
        return

    steps = DEFAULT_RETRAIN_STEPS
    if context.args:
        try:
            steps = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Qadamlar soni butun son bo'lishi kerak, masalan: /retrain 2000")
            return

    data_args = [str(PROJECT_ROOT / "data" / "corpus.txt")]
    if CONVERSATIONS_PATH.exists():
        data_args.append(str(CONVERSATIONS_PATH))

    checkpoint_path = context.bot_data["checkpoint_path"]
    log_path = PROJECT_ROOT / "retrain.log"

    cmd = [
        sys.executable, "-m", "iftixor.train",
        "--data", *data_args,
        "--out", checkpoint_path,
        "--steps", str(steps),
        "--batch-size", "16",
    ]
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    context.bot_data["retrain_process"] = process

    await update.message.reply_text(
        f"Qayta o'qitish boshlandi ({steps} qadam, {len(data_args)} ta fayl asosida).\n"
        f"Progress shu chatga avtomatik keladi. Tugagach /reload buyrug'ini yuboring."
    )


async def reload_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("Bu buyruq faqat admin uchun.")
        return

    checkpoint_path = context.bot_data["checkpoint_path"]
    try:
        model, tokenizer = load_model(checkpoint_path)
    except Exception as exc:
        await update.message.reply_text(f"Model yuklab bo'lmadi: {exc}")
        return

    context.bot_data["model"] = model
    context.bot_data["tokenizer"] = tokenizer
    await update.message.reply_text("Yangi model xotiraga yuklandi.")


def build_application(token: str, checkpoint_path: str) -> Application:
    model, tokenizer = load_model(checkpoint_path)

    application = Application.builder().token(token).build()
    application.bot_data["model"] = model
    application.bot_data["tokenizer"] = tokenizer
    application.bot_data["checkpoint_path"] = checkpoint_path

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("retrain", retrain))
    application.add_handler(CommandHandler("reload", reload_model))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return application
