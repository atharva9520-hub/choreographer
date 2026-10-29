"""Benchmark the language layer: accuracy, false triggers, latency and tokens, per model and prompt variant.

    python -m evals.run_evals                                   # model from .env, full prompt
    python -m evals.run_evals --models groq,gemini              # compare providers (default model each)
    python -m evals.run_evals --models groq:openai/gpt-oss-20b,groq:llama-3.3-70b-versatile
    python -m evals.run_evals --variants full,no_examples,no_history   # ablations
    python -m evals.run_evals --limit 20                        # quick smoke run

Writes evals/results/<timestamp>.json (every call) and evals/results/<timestamp>.md (the report).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dateutil.rrule import rrulestr

from chorebot.assistant import _replace_me
from chorebot.llm import LLM, ToolCall, build_system_prompt, format_message
from chorebot.logic import RRULE_START, ChoreBook
from chorebot.storage import MemoryStore

from .cases import CASES, Case

VARIANTS = {
    "full": {"examples": True, "history": True},
    "no_examples": {"examples": False, "history": True},
    "no_history": {"examples": True, "history": False},
}


def make_house() -> ChoreBook:
    book = ChoreBook(MemoryStore())
    for tg, name in enumerate(["Atharva", "Yuvraj", "Ari"], start=1):
        book.register_member(name, tg)
    book.add_chore("dishwasher", "scheduled", "FREQ=DAILY")
    book.add_chore("mop", "scheduled", "FREQ=WEEKLY;BYDAY=SA")
    book.add_chore("vacuum", "scheduled", "FREQ=WEEKLY;BYDAY=WE,SU")
    book.add_chore("trash", "turn")
    return book


def resolve(value, today):
    """Turn date helpers inside expected args into real dates."""
    if callable(value):
        return value(today)
    if isinstance(value, list):
        return [resolve(v, today) for v in value]
    if isinstance(value, tuple):
        return tuple(resolve(v, today) for v in value)
    if isinstance(value, dict):
        return {k: resolve(v, today) for k, v in value.items()}
    return value


def eval_today():
    return datetime.now(ZoneInfo("America/Chicago")).date()


# ---------------- comparing an answer to the expected one ----------------
def _same_freq(a: str, b: str) -> bool:
    try:
        ra = rrulestr(a.upper().removeprefix("RRULE:"), dtstart=RRULE_START)
        rb = rrulestr(b.upper().removeprefix("RRULE:"), dtstart=RRULE_START)
        return list(ra[:20]) == list(rb[:20])
    except Exception:  # noqa: BLE001
        return False


def _member(book, x):
    try:
        return book.resolve_member(str(x))
    except ValueError:
        return str(x).strip().title()


def _same(key: str, want, got, book: ChoreBook, today) -> bool:
    want = resolve(want, today)
    if got is None:
        return False
    if key == "frequency":
        return _same_freq(want, got)
    if key in ("member", "next_up"):
        return _member(book, got) == _member(book, want)
    if key in ("chore", "name"):
        g, w = book.find_chore(str(got)), book.find_chore(str(want))
        if g or w:
            return bool(g and w and g["name"] == w["name"])
        g, w = str(got).lower(), str(want).lower()
        return w in g or g in w  # new chore: 'plants' ~ 'water plants'
    if key in ("people", "add_people", "remove_people", "names"):
        return sorted(_member(book, x) for x in got) == sorted(_member(book, x) for x in want)
    if key in ("start", "end", "date"):
        return str(got)[:10] == want
    if key == "entries":
        return len(got) == len(want) and all(
            any(all(_same(k, v, g.get(k), book, today) for k, v in w.items()) for g in got) for w in want)
    return str(got).lower() == str(want).lower()


def matches(expected: list, calls: list[ToolCall], book: ChoreBook, today) -> bool:
    if not expected:
        return not calls
    if [t for t, _ in expected] == ["reply"]:
        return bool(calls) and all(c.name == "reply" for c in calls)
    acting = [c for c in calls if c.name != "reply"]
    if len(acting) != len(expected):
        return False
    return all(any(c.name == tool and all(_same(k, v, c.args.get(k), book, today) for k, v in args.items())
                   for c in acting) for tool, args in expected)


def check(case_or_expected, calls, book, today) -> bool:
    if isinstance(case_or_expected, Case):
        options = [case_or_expected.expected] + case_or_expected.also_ok
    else:
        options = [case_or_expected]
    return any(matches(opt, calls, book, today) for opt in options)


# ---------------- running ----------------
@dataclass
class Row:
    model: str
    variant: str
    category: str
    sender: str
    text: str
    passed: bool
    calls: list = field(default_factory=list)
    error: str = ""
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class DailyLimit(Exception):
    """The provider's per-day quota is used up; stop this model and resume tomorrow."""


def _call_with_retries(llm: LLM, message: str, prompt: str, max_tries: int = 6):
    """Retry per-minute rate limits and overloads; give up at once on per-day limits.
    Returns (calls, latency of the successful attempt)."""
    import re

    for attempt in range(max_tries):
        start = time.perf_counter()
        try:
            calls = llm.parse(message, prompt)
            return calls, time.perf_counter() - start
        except Exception as e:  # noqa: BLE001
            text = str(e)
            if "per day" in text or "RPD" in text or "TPD" in text:
                raise DailyLimit(text[:160]) from e
            retryable = any(k in text for k in ("429", "503", "500", "rate limit", "Rate limit", "overloaded",
                                                "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if not retryable or attempt == max_tries - 1:
                raise
            m = re.search(r"try again in ([0-9.]+)s", text)
            wait = float(m.group(1)) + 1 if m else min(60, 5 * 2 ** attempt)
            print(f"   (rate limited, waiting {wait:.0f}s)")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run(llm: LLM, cases=CASES, pause: float = 0.0, verbose: bool = True, variant: str = "full",
        model_name: str = "model", done: dict | None = None, on_row=None) -> list[Row]:
    """Score every case. `done` maps (model, variant, text) -> Row from an earlier run to reuse."""
    v = VARIANTS[variant]
    book = make_house()
    today = eval_today()
    chores = [book.describe(c) for c in book.chores()]
    rows = []
    stopped = ""
    for case in cases:
        key = (model_name, variant, case.text)
        if done and key in done:
            rows.append(done[key])
            continue
        if stopped:
            rows.append(Row(model_name, variant, case.category, case.sender, case.text, False, [], "skipped: " + stopped))
            continue
        prompt = build_system_prompt(today, case.sender, book.member_names(), chores, examples=v["examples"])
        message = format_message(case.sender, case.text, case.history if v["history"] else None)
        err, latency = "", 0.0
        try:
            calls, latency = _call_with_retries(llm, message, prompt)
        except DailyLimit as e:
            stopped = "daily limit reached"
            print(f"   Daily limit reached for {model_name}: {e}\n   Re-run later with --resume to finish.")
            rows.append(Row(model_name, variant, case.category, case.sender, case.text, False, [], "skipped: " + stopped))
            continue
        except Exception as e:  # noqa: BLE001
            calls, err = [], f"{type(e).__name__}: {e}"[:200]
        calls = [ToolCall(c.name, _replace_me(c.args, case.sender)) for c in calls]  # same as the live bot
        ok = not err and check(case, calls, book, today)
        tin, tout = getattr(llm, "last_usage", (0, 0)) or (0, 0)
        row = Row(model_name, variant, case.category, case.sender, case.text, ok,
                  [{"name": c.name, "args": c.args} for c in calls], err, round(latency, 3), tin, tout)
        rows.append(row)
        if on_row:
            on_row(row)
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'} [{model_name} · {variant}] {case.sender}: {case.text}"
                  + (f"   -> {err or [(c.name, c.args) for c in calls]}" if not ok else ""))
        time.sleep(pause)
    return rows


def scored(rows: list[Row]) -> list[Row]:
    """Rows that got an answer from the model (API errors and skipped cases are reported, not scored)."""
    return [r for r in rows if not r.error]


def score(rows: list[Row]) -> float:
    rows = scored(rows)
    return sum(r.passed for r in rows) / len(rows) if rows else 0.0


# ---------------- report ----------------
def _pct(x):
    return f"{x:.1%}"


def summarize(rows: list[Row]) -> dict:
    all_rows, rows = rows, scored(rows)
    chat = [r for r in rows if r.category == "chitchat"]
    action = [r for r in rows if r.category != "chitchat"]
    lat = sorted(r.latency_s for r in rows)
    return {
        "n": len(rows),
        "total": len(all_rows),
        "accuracy": score(rows),
        "action_accuracy": score(action),
        # bot acted (or spoke) on ordinary chat
        "false_trigger_rate": (sum(not r.passed for r in chat) / len(chat)) if chat else 0.0,
        # chore message the bot ignored completely
        "missed_rate": (sum(1 for r in action if not r.calls and not r.error) / len(action)) if action else 0.0,
        "errors": sum(1 for r in all_rows if r.error and not r.error.startswith("skipped")),
        "skipped": sum(1 for r in all_rows if r.error.startswith("skipped")),
        "p50_latency_s": statistics.median(lat) if lat else 0.0,
        "p95_latency_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
        "avg_tokens_in": statistics.mean(r.tokens_in for r in rows) if rows else 0,
        "avg_tokens_out": statistics.mean(r.tokens_out for r in rows) if rows else 0,
    }


def report(all_rows: list[Row]) -> str:
    groups: dict[tuple, list[Row]] = {}
    for r in all_rows:
        groups.setdefault((r.model, r.variant), []).append(r)
    cats = sorted({r.category for r in all_rows})
    lines = [f"# Choreographer language benchmark — {datetime.now():%Y-%m-%d %H:%M}",
             f"{len({r.text for r in all_rows})} held-out messages · {len(cats)} categories", "",
             "| Model | Prompt | Scored | Accuracy | Action acc. | False triggers | Missed | p50 latency "
             "| p95 latency | Tokens in/out | API errors |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, v), rows in groups.items():
        s = summarize(rows)
        lines.append(f"| {m} | {v} | {s['n']}/{s['total']} | {_pct(s['accuracy'])} | {_pct(s['action_accuracy'])} | "
                     f"{_pct(s['false_trigger_rate'])} | {_pct(s['missed_rate'])} | {s['p50_latency_s']:.2f}s | "
                     f"{s['p95_latency_s']:.2f}s | {s['avg_tokens_in']:.0f}/{s['avg_tokens_out']:.0f} | {s['errors']} |")
    lines += ["", "## Accuracy by category", "",
              "| Category | " + " | ".join(f"{m} · {v}" for m, v in groups) + " |",
              "|---|" + "---|" * len(groups)]
    for c in cats:
        cells = []
        for rows in groups.values():
            sub = scored([r for r in rows if r.category == c])
            cells.append(f"{_pct(score(sub))} ({sum(r.passed for r in sub)}/{len(sub)})" if sub else "—")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "Accuracy counts only messages the model answered; API errors and cases skipped after a daily "
              "limit are listed under Scored/API errors instead of counted as wrong.", "", "## Failures", ""]
    for (m, v), rows in groups.items():
        fails = [r for r in scored(rows) if not r.passed]
        lines.append(f"**{m} · {v}** — {len(fails)} failed")
        for r in fails:
            got = ", ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in r.calls) or "nothing"
            lines.append(f"- [{r.category}] {r.sender}: “{r.text}” → {got}")
        lines.append("")
    return "\n".join(lines)


def build_model(spec: str) -> tuple[str, LLM]:
    """'groq', 'gemini', 'groq:openai/gpt-oss-20b', 'gemini:gemini-3.5-flash'."""
    from chorebot.llm import GeminiLLM, OpenAICompatibleLLM, make_llm

    if spec == "env":
        llm = make_llm()
        return getattr(llm, "model", "env"), llm
    provider, _, model = spec.partition(":")
    if provider == "gemini":
        llm = GeminiLLM(model=model or None)
        llm.models = [llm.model]  # no silent fallback while benchmarking
    else:
        llm = OpenAICompatibleLLM(provider, model or None)
        llm.client = llm.client.with_options(max_retries=0)  # we retry ourselves and time one attempt
    return f"{provider}:{llm.model}", llm


def main():
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="env")
    ap.add_argument("--variants", default="full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=2.5, help="seconds between calls (free-tier rate limits)")
    ap.add_argument("--resume", default="", help="results .json from an earlier run: reuse its answered cases")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    done: dict = {}
    if args.resume:
        path = Path(args.resume)
        for d in json.loads(path.read_text()):
            r = Row(**d)
            if not r.error:
                done[(r.model, r.variant, r.text)] = r
        stamp = path.stem
        print(f"Resuming {path.name}: {len(done)} answered cases reused")
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out / f"{stamp}.json"

    cases = CASES[: args.limit] if args.limit else CASES
    all_rows: list[Row] = []

    def save(extra: list[Row] = ()):  # write after every answer so a stop never loses work
        rows = all_rows + list(extra)
        keep = {(r.model, r.variant, r.text): r for r in done.values()}
        keep.update({(r.model, r.variant, r.text): r for r in rows})
        json_path.write_text(json.dumps([r.__dict__ for r in keep.values()], indent=1, ensure_ascii=False))

    for spec in args.models.split(","):
        name, llm = build_model(spec.strip())
        for variant in args.variants.split(","):
            current: list[Row] = []

            def on_row(r, current=current):
                current.append(r)
                save(current)

            all_rows += run(llm, cases, args.pause, not args.quiet, variant.strip(), name, done, on_row)
            save()

    md = report(all_rows)
    (out / f"{stamp}.md").write_text(md)
    print("\n" + md.split("## Failures")[0])
    print(f"Full report: evals/results/{stamp}.md")
    if any(r.error.startswith("skipped") for r in all_rows):
        print(f"Some cases were skipped (daily limit). Finish later with:\n"
              f"  python -m evals.run_evals --models {args.models} --variants {args.variants} "
              f"--resume evals/results/{stamp}.json")


if __name__ == "__main__":
    main()
