"""SheetsStore against a fake worksheet (checks row/cell addressing without Google)."""
from chorebot.logic import ChoreBook
from chorebot.storage import SCHEMA, SheetsStore


class FakeWS:
    def __init__(self, cols):
        self.grid = [list(cols)]
        self.reads = 0

    def get_all_records(self, expected_headers):
        self.reads += 1
        h = self.grid[0]
        return [dict(zip(h, r)) for r in self.grid[1:]]

    def append_row(self, values, **_):
        self.grid.append(list(values))

    def batch_update(self, data, **_):
        import gspread
        for d in data:
            r, c = gspread.utils.a1_to_rowcol(d["range"])
            self.grid[r - 1][c - 1] = d["values"][0][0]


def make_store():
    s = SheetsStore.__new__(SheetsStore)
    s._ws = {t: FakeWS(c) for t, c in SCHEMA.items()}
    s._cache = {}
    return s


def test_sheets_store_roundtrip():
    s = make_store()
    b = ChoreBook(s)
    for i, n in enumerate(["Atharva", "Yuvraj", "Arihant"]):
        b.register_member(n, 10 + i)
    b.add_chore("trash", "turn")
    b.mark_done("Atharva", "trash", __import__("datetime").date(2026, 10, 5), "Atharva", "")
    # the underlying grid was really updated, not only the cache
    chores = s._ws["Chores"].grid
    assert chores[1][chores[0].index("next_up")] == "Yuvraj"
    s._cache.clear()
    assert b.find_chore("trash")["next_up"] == "Yuvraj"
    assert s._ws["Members"].reads <= 3
