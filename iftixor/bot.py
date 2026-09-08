import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .generate import generate_text, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iftixor.bot")

MAX_HISTORY_CHARS = 800

WELCOME_TEXT = (
    "Salom! Men Iftixor — noldan o'qitilayotgan tajriba AI modeliman.\n"
    "Hozircha kichik hajmda o'qiganman, shuning uchun javoblarim cheklangan bo'lishi mumkin.\n"
    "/reset — suhbat tarixini tozalash."
)


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
    await update.message.reply_text(reply)


def build_application(token: str, checkpoint_path: str) -> Application:
    model, tokenizer = load_model(checkpoint_path)

    application = Application.builder().token(token).build()
    application.bot_data["model"] = model
    application.bot_data["tokenizer"] = tokenizer

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return application
