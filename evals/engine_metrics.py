"""Simulate a year in the apartment to measure the scheduling engine (no LLM, no network).

    python -m evals.engine_metrics            # 52 weeks, seed 7
    python -m evals.engine_metrics 104 3      # weeks, seed

Every week: build the plan, randomly mark people busy, then have people log chores (sometimes someone
else's, sometimes turn-based chores out of turn). Reports fairness and rule violations.
"""
from __future__ import annotations

import random
import statistics
import sys
import time
from datetime import date, timedelta

from chorebot.logic import ChoreBook
from chorebot.storage import MemoryStore

PEOPLE = ["Atharva", "Yuvraj", "Ari"]
START = date(2026, 1, 5)  # a Monday


def simulate(weeks: int = 52, seed: int = 7, compensate: bool = True) -> dict:
    rng = random.Random(seed)
    b = ChoreBook(MemoryStore(), compensate=compensate)
    for i, p in enumerate(PEOPLE):
        b.register_member(p, i + 1)
    b.add_chore("dishwasher", "scheduled", "FREQ=DAILY")
    b.add_chore("mop", "scheduled", "FREQ=WEEKLY;BYDAY=SA")
    b.add_chore("vacuum", "scheduled", "FREQ=WEEKLY;BYDAY=WE,SU")
    b.add_chore("bathroom", "scheduled", "FREQ=WEEKLY;INTERVAL=2;BYDAY=SU", members=["Atharva", "Yuvraj"])
    b.add_chore("trash", "turn")

    week_gaps, plan_ms = [], []
    rest_violations = excluded_violations = busy_violations = 0
    busy_tasks = busy_moved = 0
    turn_checks = turn_ok = 0
    done_by = {p: 0 for p in PEOPLE}
    assigned = {p: 0 for p in PEOPLE}
    turn_done = {p: 0 for p in PEOPLE}

    for w in range(weeks):
        monday = START + timedelta(weeks=w)
        t0 = time.perf_counter()
        tasks = b.make_plan(monday)
        plan_ms.append((time.perf_counter() - t0) * 1000)
        rest = b.rest_member(monday)

        # plan quality: rest person and excluded people never assigned when someone else could do it
        counts = {p: 0 for p in PEOPLE}
        for t in tasks:
            counts[t["assigned_to"]] += 1
            assigned[t["assigned_to"]] += 1
            pool = b.pool(b.find_chore(t["chore"]))
            if t["assigned_to"] not in pool:
                excluded_violations += 1
            if t["assigned_to"] == rest and any(p != rest for p in pool):
                rest_violations += 1
        active = [p for p in PEOPLE if p != rest]
        week_gaps.append(abs(counts[active[0]] - counts[active[1]]))
        # covering is decided up front from a separate stream so both engines see the same week
        cover_rng = random.Random(seed * 1000 + w)

        # someone gets busy for 1-3 days about 30% of weeks
        if rng.random() < 0.3:
            who = rng.choice(active)
            start = monday + timedelta(days=rng.randint(0, 4))
            end = start + timedelta(days=rng.randint(0, 2))
            before = [t for t in b.s.rows("Tasks") if t["assigned_to"] == who and t["status"] == "open"
                      and start.isoformat() <= t["date"] <= end.isoformat()]
            busy_tasks += len(before)
            b.set_busy(who, start, end, who, "busy")
            after = {t["id"]: t for t in b.s.rows("Tasks")}
            for t in before:
                new = after[t["id"]]["assigned_to"]
                if new != who:
                    busy_moved += 1
                    if b.busy_on(new, date.fromisoformat(t["date"])):
                        busy_violations += 1

        # people log the week's chores; 15% of the time someone else covers
        for t in [t for t in b.s.rows("Tasks") if t["status"] == "open"
                  and monday.isoformat() <= t["date"] <= (monday + timedelta(days=6)).isoformat()]:
            doer = t["assigned_to"]
            if cover_rng.random() < 0.15:
                doer = cover_rng.choice([p for p in PEOPLE if p != doer])
            b.mark_done(doer, t["chore"], date.fromisoformat(t["date"]), doer, "done")
            done_by[doer] += 1

        # trash 2-4 times a week; usually whoever's turn, sometimes someone else
        for _ in range(rng.randint(2, 4)):
            chore = b.find_chore("trash")
            expected_next = chore["next_up"]
            doer = expected_next if rng.random() < 0.8 else rng.choice(PEOPLE)
            b.mark_done(doer, "trash", monday + timedelta(days=rng.randint(0, 6)), doer, "trash")
            turn_done[doer] += 1
            turn_checks += 1
            pool = b.pool(chore)
            if b.find_chore("trash")["next_up"] == pool[(pool.index(doer) + 1) % len(pool)]:
                turn_ok += 1

    total = sum(done_by.values())
    mean = total / len(PEOPLE)
    return {
        "weeks": weeks,
        "scheduled_tasks": total,
        "rest_violations": rest_violations,
        "excluded_member_violations": excluded_violations,
        "weekly_gap_mean": statistics.mean(week_gaps),
        "weekly_gap_max": max(week_gaps),
        "weeks_balanced_within_1": sum(g <= 1 for g in week_gaps) / len(week_gaps),
        "yearly_share": {p: done_by[p] / total for p in PEOPLE},
        "yearly_spread_pct": (max(done_by.values()) - min(done_by.values())) / mean,
        "assigned_share": {p: assigned[p] / sum(assigned.values()) for p in PEOPLE},
        "busy_tasks": busy_tasks,
        "busy_reassigned": busy_moved / busy_tasks if busy_tasks else 1.0,
        "busy_reassigned_to_busy_person": busy_violations,
        "turn_rotation_correct": turn_ok / turn_checks if turn_checks else 1.0,
        "turn_share": {p: turn_done[p] / sum(turn_done.values()) for p in PEOPLE},
        "plan_ms_p50": statistics.median(plan_ms),
    }


def compare(weeks: int = 52, seeds=range(1, 21)) -> None:
    """Average yearly spread of work done, with and without evening out covering, over many seeds."""
    for comp in (False, True):
        runs = [simulate(weeks, s, comp) for s in seeds]
        spread = statistics.mean(r["yearly_spread_pct"] for r in runs)
        worst = max(r["yearly_spread_pct"] for r in runs)
        gap = statistics.mean(r["weekly_gap_mean"] for r in runs)
        print(f"  {'with' if comp else 'without'} cover balancing: yearly spread of work done "
              f"mean {spread:.1%}, worst {worst:.1%} · weekly assignment gap {gap:.2f} ({len(runs)} simulated years)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
        return
    weeks = int(sys.argv[1]) if len(sys.argv) > 1 else 52
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    m = simulate(weeks, seed)
    print(f"Simulated {m['weeks']} weeks, {m['scheduled_tasks']} scheduled tasks (seed {seed})")
    print(f"  Rest-week violations:            {m['rest_violations']}")
    print(f"  Excluded-member violations:      {m['excluded_member_violations']}")
    print(f"  Weekly load gap (active pair):   mean {m['weekly_gap_mean']:.2f}, max {m['weekly_gap_max']} tasks")
    print(f"  Weeks balanced within 1 task:    {m['weeks_balanced_within_1']:.0%}")
    print(f"  Yearly share of assignments:     " + ", ".join(f"{p} {s:.1%}" for p, s in m['assigned_share'].items()))
    print(f"  Yearly share of work done:       " + ", ".join(f"{p} {s:.1%}" for p, s in m['yearly_share'].items()))
    print(f"  Yearly spread (max-min / mean):  {m['yearly_spread_pct']:.1%}")
    print(f"  Busy-period tasks reassigned:    {m['busy_reassigned']:.0%} of {m['busy_tasks']}"
          f" (to someone also busy: {m['busy_reassigned_to_busy_person']})")
    print(f"  Turn rotation correct:           {m['turn_rotation_correct']:.0%}")
    print(f"  Turn-based share:                " + ", ".join(f"{p} {s:.1%}" for p, s in m['turn_share'].items()))
    print(f"  Weekly plan build time (p50):    {m['plan_ms_p50']:.1f} ms")


if __name__ == "__main__":
    main()
