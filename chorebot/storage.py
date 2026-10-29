"""Storage: Google Sheets (real) and in-memory (tests). Every tab is a list of dict rows."""
from __future__ import annotations

import copy
import time

import gspread

# Tab name -> column headers. The Sheets workbook must have exactly these tabs.
SCHEMA: dict[str, list[str]] = {
    "Members": ["name", "telegram_id", "order"],
    "Chores": ["name", "type", "frequency", "members", "next_up", "active"],
    "Tasks": ["id", "date", "chore", "assigned_to", "status", "done_by"],
    "Busy": ["id", "member", "start", "end", "active"],
    "Log": ["id", "timestamp", "sender", "message", "action", "details", "undone"],
    "Settings": ["key", "value"],
    "Messages": ["timestamp", "sender", "text", "tool_calls", "reply"],
}


def _clean(row: dict, tab: str) -> dict:
    # Sheets may return numbers for numeric-looking cells; keep everything as strings.
    return {c: ("" if row.get(c) is None else str(row.get(c))) for c in SCHEMA[tab]}


class MemoryStore:
    def __init__(self):
        self.tabs: dict[str, list[dict]] = {t: [] for t in SCHEMA}

    def rows(self, tab: str) -> list[dict]:
        return copy.deepcopy(self.tabs[tab])

    def append(self, tab: str, row: dict) -> None:
        self.tabs[tab].append(_clean(row, tab))

    def update(self, tab: str, key: str, value: str, changes: dict) -> bool:
        for r in self.tabs[tab]:
            if r[key] == str(value):
                r.update({k: str(v) for k, v in changes.items()})
                return True
        return False


class SheetsStore:
    """Google Sheets backend. Reads a whole tab per call (fine for 3 people)."""

    CACHE_SECONDS = 60  # manual edits in the sheet show up within a minute

    def __init__(self, sheet_id: str, credentials_file: str):
        self.book = gspread.service_account(filename=credentials_file).open_by_key(sheet_id)
        self._ws = {}
        self._cache: dict[str, tuple[float, list[dict]]] = {}  # keeps us under the Sheets API read quota

    def ws(self, tab: str):
        if tab not in self._ws:
            try:
                self._ws[tab] = self.book.worksheet(tab)
            except gspread.exceptions.WorksheetNotFound:  # tab added in a newer version: create it
                ws = self.book.add_worksheet(title=tab, rows=1000, cols=len(SCHEMA[tab]))
                ws.append_row(SCHEMA[tab])
                self._ws[tab] = ws
        return self._ws[tab]

    def _rows(self, tab: str) -> list[dict]:
        hit = self._cache.get(tab)
        if hit and time.monotonic() - hit[0] < self.CACHE_SECONDS:
            return hit[1]
        records = self.ws(tab).get_all_records(expected_headers=SCHEMA[tab])
        rows = [_clean(r, tab) for r in records]
        self._cache[tab] = (time.monotonic(), rows)
        return rows

    def rows(self, tab: str) -> list[dict]:
        return copy.deepcopy(self._rows(tab))

    def append(self, tab: str, row: dict) -> None:
        row = _clean(row, tab)
        self.ws(tab).append_row([row[c] for c in SCHEMA[tab]], value_input_option="RAW",
                                table_range="A1")
        if tab in self._cache:  # keep the cache current without re-reading the tab
            self._cache[tab][1].append(row)

    def update(self, tab: str, key: str, value: str, changes: dict) -> bool:
        cols = SCHEMA[tab]
        for i, r in enumerate(self._rows(tab), start=2):  # row 1 is the header
            if r[key] == str(value):
                self.ws(tab).batch_update([
                    {"range": gspread.utils.rowcol_to_a1(i, cols.index(k) + 1), "values": [[str(v)]]}
                    for k, v in changes.items()], value_input_option="RAW")
                r.update({k: str(v) for k, v in changes.items()})
                return True
        return False

    def ensure_tabs(self) -> None:
        """Create missing tabs with headers (used by setup_sheet.py)."""
        existing = {w.title for w in self.book.worksheets()}
        for tab, cols in SCHEMA.items():
            if tab not in existing:
                ws = self.book.add_worksheet(title=tab, rows=1000, cols=len(cols))
                ws.append_row(cols)
            else:
                ws = self.book.worksheet(tab)
                if ws.row_values(1) != cols:
                    ws.update(range_name="A1", values=[cols])
        if "Sheet1" in existing:  # remove the default empty tab
            try:
                self.book.del_worksheet(self.book.worksheet("Sheet1"))
            except gspread.exceptions.APIError:
                pass
