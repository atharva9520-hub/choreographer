from datetime import date, timedelta

import pytest

from chorebot.logic import ChoreBook
from chorebot.storage import MemoryStore

MON = date(2026, 10, 5)  # a Monday


@pytest.fixture
def book():
    b = ChoreBook(MemoryStore())
    for i, n in enumerate(["Atharva", "Yuvraj", "Arihant"]):
        b.register_member(n, 100 + i)
    return b


def open_tasks(b, who=None):
    return [t for t in b.s.rows("Tasks") if t["status"] == "open" and (who is None or t["assigned_to"] == who)]


def test_register_and_fuzzy_names(book):
    assert book.member_by_telegram(101) == "Yuvraj"
    assert book.resolve_member("yuv") == "Yuvraj"
    assert book.resolve_member("arihnt") == "Arihant"
    with pytest.raises(ValueError):
        book.resolve_member("Bob")


def test_rest_rotates_weekly(book):
    rests = [book.rest_member(MON + timedelta(weeks=i)) for i in range(6)]
    assert rests[:3] == rests[3:]
    assert sorted(rests[:3]) == ["Arihant", "Atharva", "Yuvraj"]


def test_set_rest(book):
    book.set_rest("Arihant", MON)
    assert book.rest_member(MON) == "Arihant"
    assert book.rest_member(MON + timedelta(weeks=1)) == "Atharva"


def test_plan_never_assigns_rest_person_and_is_balanced(book):
    book.add_chore("dishwasher", "scheduled", "FREQ=DAILY")
    book.add_chore("mop", "scheduled", "FREQ=WEEKLY;BYDAY=SA")
    book.add_chore("vacuum", "scheduled", "FREQ=WEEKLY;BYDAY=WE,SU")
    for w in range(6):
        monday = MON + timedelta(weeks=w)
        rest = book.rest_member(monday)
        tasks = book.make_plan(monday)
        assert len(tasks) == 10
        assert all(t["assigned_to"] != rest for t in tasks)
        counts = {}
        for t in tasks:
            counts[t["assigned_to"]] = counts.get(t["assigned_to"], 0) + 1
        assert sorted(counts.values()) == [5, 5]


def test_plan_respects_excluded_members(book):
    book.add_chore("mop", "scheduled", "FREQ=DAILY", members=["Atharva", "Arihant"])
    for w in range(3):
        for t in book.make_plan(MON + timedelta(weeks=w)):
            assert t["assigned_to"] in ("Atharva", "Arihant")


def test_replan_keeps_done_tasks(book):
    book.add_chore("dishwasher", "scheduled", "FREQ=DAILY")
    book.make_plan(MON)
    who = open_tasks(book)[0]["assigned_to"]
    book.mark_done(who, "dishwasher", MON, "x", "")
    book.make_plan(MON)
    tasks = [t for t in book.s.rows("Tasks") if t["status"] in ("open", "done")]
    assert len(tasks) == 7


def test_turn_based_strict_rotation(book):
    book.add_chore("trash", "turn", first_up="Yuvraj")
    chore = lambda: book.find_chore("trash")  # noqa: E731
    assert chore()["next_up"] == "Yuvraj"
    book.mark_done("Yuvraj", "trash", MON, "Yuvraj", "")
    assert chore()["next_up"] == "Arihant"
    book.mark_done("Arihant", "trash", MON, "Arihant", "")
    assert chore()["next_up"] == "Atharva"
    # out of turn: rotation continues from whoever did it
    book.mark_done("Yuvraj", "trash", MON, "Yuvraj", "")
    assert chore()["next_up"] == "Arihant"


def test_turn_based_ignores_rest_week_and_respects_pool(book):
    book.add_chore("trash", "turn", members=["Atharva", "Arihant"])
    assert book.find_chore("trash")["next_up"] == "Atharva"
    book.mark_done("Atharva", "trash", MON, "a", "")
    assert book.find_chore("trash")["next_up"] == "Arihant"
    book.mark_done("Arihant", "trash", MON, "a", "")
    assert book.find_chore("trash")["next_up"] == "Atharva"


def test_whose_turn_skips_busy(book):
    book.add_chore("trash", "turn", first_up="Yuvraj")
    book.set_busy("Yuvraj", MON, MON + timedelta(days=3), "x", "")
    assert book.whose_turn(book.find_chore("trash"), MON) == "Arihant"


def test_mark_done_by_someone_else_covers(book):
    book.add_chore("mop", "scheduled", "FREQ=WEEKLY;BYDAY=SA")
    task = book.make_plan(MON)[0]
    other = next(n for n in book.member_names() if n not in (task["assigned_to"], book.rest_member(MON)))
    reply, _ = book.mark_done(other, "mop", MON + timedelta(days=5), "x", "")
    assert "covering" in reply
    assert not open_tasks(book)


def test_adhoc_chore(book):
    reply, _ = book.mark_done("Atharva", "cleaned fridge", MON, "Atharva", "")
    assert "one-off" in reply
    assert book.s.rows("Tasks")[0]["done_by"] == "Atharva"


def test_busy_reassigns_and_undo_restores(book):
    book.add_chore("dishwasher", "scheduled", "FREQ=DAILY")
    book.make_plan(MON)
    rest = book.rest_member(MON)
    worker = next(n for n in book.member_names() if n != rest)
    before = {t["id"]: t["assigned_to"] for t in open_tasks(book)}
    _, log_id = book.set_busy(worker, MON, MON + timedelta(days=6), worker, "busy all week")
    assert not open_tasks(book, worker)
    # cover goes to the other active person, not the resting one
    assert all(t["assigned_to"] != rest for t in open_tasks(book))
    book.undo(log_id)
    assert {t["id"]: t["assigned_to"] for t in open_tasks(book)} == before
    assert not book.busy_on(worker, MON)


def test_undo_mark_done_restores_turn(book):
    book.add_chore("trash", "turn")
    _, log_id = book.mark_done("Atharva", "trash", MON, "a", "")
    assert book.find_chore("trash")["next_up"] == "Yuvraj"
    assert book.undo(log_id) == "↩️ Undone."
    assert book.find_chore("trash")["next_up"] == "Atharva"
    assert book.undo(log_id) == "Already undone."


def test_update_chore_include_exclude(book):
    book.add_chore("dishwasher", "scheduled", "FREQ=DAILY")
    book.update_chore("dishwasher", exclude=["Yuvraj"])
    assert book.pool(book.find_chore("dishwasher")) == ["Atharva", "Arihant"]
    book.update_chore("dishwasher", include=["Yuvraj"])
    assert book.find_chore("dishwasher")["members"] == ""


def test_bad_frequency_rejected(book):
    with pytest.raises(ValueError):
        book.add_chore("mop", "scheduled", "every saturday")


def test_covering_earns_a_lighter_week(book):
    book.add_chore("dishwasher", "scheduled", "FREQ=DAILY")
    week1 = book.make_plan(MON)
    rest1 = book.rest_member(MON)
    helper = next(n for n in book.member_names() if n != rest1)
    # helper does every task in week 1, including the other person's
    for t in week1:
        book.mark_done(helper, "dishwasher", date.fromisoformat(t["date"]), helper, "")
    nxt = MON + timedelta(weeks=1)
    week2 = book.make_plan(nxt)
    if helper != book.rest_member(nxt):
        mine = sum(t["assigned_to"] == helper for t in week2)
        assert mine < len(week2) / 2


def test_no_balancing_when_disabled(book):
    book.compensate = False
    book.add_chore("dishwasher", "scheduled", "FREQ=DAILY")
    tasks = book.make_plan(MON)
    counts = sorted([sum(t["assigned_to"] == n for t in tasks) for n in book.member_names()])
    assert counts == [0, 3, 4]
