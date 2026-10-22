"""One-time: create the tabs + headers in your Google Sheet. Run: python setup_sheet.py"""
import os

from dotenv import load_dotenv

from chorebot.storage import SheetsStore

load_dotenv()
store = SheetsStore(os.environ["GOOGLE_SHEET_ID"], os.environ.get("GOOGLE_CREDENTIALS_FILE", "service_account.json"))
store.ensure_tabs()
print("Sheet ready:", [w.title for w in store.book.worksheets()])
