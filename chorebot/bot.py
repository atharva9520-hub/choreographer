"""Telegram bot: reads every group message, replies, handles ✅/❌ buttons, posts the weekly plan and daily reminders."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections import deque
from datetime import time as dtime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler,
                          filters)

from .assistant import Assistant
from .llm import make_llm
from .logic import ChoreBook, monday_of
from .storage import SheetsStore

log = logging.getLogger("chorebot")
UNDO_WORDS = {"undo", "undo it", "undo that", "revert", "cancel that"}

HELP = """Hi! Just talk normally in this group, no special format needed. For example:
• I'm Atharva  (everyone says this once)
• Yuvraj did mopping and Ari vacuumed
• We'll clean the bathroom every other week
• Assign Yuvraj and Ari mopping every week
• Trash is whenever it's full, we take turns
• I'm out of town till Sunday
• Who did the mopping? / Whose turn is trash? / What do I have this week?
• Oops, undo that
Settings changes ask for a ✅ first."""


class ChoreBot:
    def __init__(self, assistant: Assistant, allowed_chat: str | None = None):
        self.a = assistant
        self.allowed_chat = allowed_chat
        self.pending: dict[str, list] = {}       # button id -> tool calls awaiting ✅
        self.undo_map: dict[int, list[str]] = {}  # bot message id -> log ids
        self.history: deque[str] = deque(maxlen=12)  # recent chat lines, so follow-ups make sense

    def _allowed(self, update: Update) -> bool:
        return not self.allowed_chat or str(update.effective_chat.id) == self.allowed_chat

    async def help(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
        if self._allowed(update):
            await update.effective_message.reply_text(HELP)

    async def on_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        msg = update.effective_message
        if not msg or not msg.text or not self._allowed(update):
            return
        book = self.a.book
        if book.setting("chat_id") != str(msg.chat_id):
            await asyncio.to_thread(book.set_setting, "chat_id", str(msg.chat_id))

        reply_to = msg.reply_to_message
        from_bot = reply_to and reply_to.from_user and reply_to.from_user.id == ctx.bot.id
        if from_bot and msg.text.strip().lower() in UNDO_WORDS:
            ids = self.undo_map.pop(reply_to.message_id, [])
            text = await asyncio.to_thread(self.a.undo, ids) if ids else "I can only undo my recent ✅ messages."
            await msg.reply_text(text)
            return

        name = await asyncio.to_thread(book.member_by_telegram, msg.from_user.id) or msg.from_user.first_name
        context = list(self.history)
        self.history.append(f"{name}: {msg.text[:300]}")
        try:
            res = await asyncio.to_thread(self.a.handle_message, msg.text, msg.from_user.id,
                                          reply_to.text if reply_to and reply_to.text else "", context)
        except Exception as e:  # noqa: BLE001
            log.exception("failed to handle message")
            await msg.reply_text(f"⚠️ Sorry, something went wrong on my side ({type(e).__name__}: {str(e)[:120]}). "
                                 "Try again in a bit.")
            return
        if res.text or res.pending_text:
            self.history.append("Bot: " + (res.text or res.pending_text)[:300].replace("\n", " | "))
        if res.text:
            sent = await msg.reply_text(res.text)
            if res.log_ids:
                self.undo_map[sent.message_id] = res.log_ids
        if res.pending:
            pid = uuid.uuid4().hex[:10]
            self.pending[pid] = res.pending
            buttons = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Yes", callback_data=f"ok:{pid}"),
                                             InlineKeyboardButton("❌ No", callback_data=f"no:{pid}")]])
            await msg.reply_text(res.pending_text, reply_markup=buttons)

    async def on_button(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        action, pid = q.data.split(":", 1)
        sender = await asyncio.to_thread(self.a.book.member_by_telegram, q.from_user.id)
        if sender is None:
            await q.answer("Introduce yourself first (\"I'm <name>\").", show_alert=True)
            return
        await q.answer()
        calls = self.pending.pop(pid, None)
        if calls is None:
            await q.edit_message_text("This request expired. Please send it again.")
            return
        if action == "no":
            await q.edit_message_text(f"❌ Cancelled by {sender}.")
            return
        text = await asyncio.to_thread(self.a.apply_confirmed, calls, sender)
        await q.edit_message_text(f"✅ Confirmed by {sender}\n{text}")

    # ---------- scheduled posts ----------
    async def weekly_plan(self, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = self.a.book.setting("chat_id")
        if not chat_id:
            return
        next_monday = monday_of(self.a.today()) + timedelta(weeks=1)
        await asyncio.to_thread(self.a.book.make_plan, next_monday)
        text = await asyncio.to_thread(self.a.book.week_summary, next_monday)
        await ctx.bot.send_message(chat_id, "Next week's plan:\n" + text)

    async def daily(self, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = self.a.book.setting("chat_id")
        if chat_id:
            text = await asyncio.to_thread(self.a.schedule_text, "today")
            await ctx.bot.send_message(chat_id, "Good morning! " + text)


def _hhmm(value: str, tz: ZoneInfo) -> dtime:
    h, m = value.split(":")
    return dtime(int(h), int(m), tzinfo=tz)


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    tz_name = os.environ.get("TIMEZONE", "America/Chicago")
    tz = ZoneInfo(tz_name)

    store = SheetsStore(os.environ["GOOGLE_SHEET_ID"], os.environ.get("GOOGLE_CREDENTIALS_FILE", "service_account.json"))
    assistant = Assistant(ChoreBook(store), make_llm(), tz_name)
    bot = ChoreBot(assistant, os.environ.get("ALLOWED_CHAT_ID") or None)

    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(CommandHandler(["start", "help"], bot.help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_message))
    app.add_handler(CallbackQueryHandler(bot.on_button))
    # python-telegram-bot: days 0=Sunday ... 6=Saturday
    app.job_queue.run_daily(bot.weekly_plan, _hhmm(os.environ.get("WEEKLY_PLAN_TIME", "19:00"), tz), days=(0,))
    app.job_queue.run_daily(bot.daily, _hhmm(os.environ.get("REMINDER_TIME", "08:00"), tz))
    log.info("Chore bot running")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
