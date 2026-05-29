"""
Iteration-history log for the autoresearch loop.

Each accepted-or-rejected variant proposal produces one record. Records are
stored two ways:

* ``result/history.jsonl`` — append-only, one JSON object per line. Source
  of truth. Easy to parse, never edit by hand.
* ``result/history.md`` — regenerated from the JSONL on every append. The
  human-readable view shown to Claude on the next iteration so it sees
  what's already been tried.

Schema of one record::

    {
        "iteration":            int,    # 0 for baseline, then 1, 2, …
        "timestamp":            str,    # ISO-8601 UTC
        "summary":              str,    # short description of the change,
                                        # mirrors autoresearcher.md's
                                        # 30-word summary
        "threshold":            float|None,  # per-iteration accept threshold
                                             # (None for baseline)
        "ll_before":            float|None,  # champion's logloss_by_user
        "ll_after":             float,       # this iteration's logloss_by_user
        "improvement":          float|None,  # ll_before − ll_after
        "complexity_before":    int|None,    # champion's complexity score
        "complexity_after":     int,         # this iteration's complexity score
        "complexity_pct_change": float|None, # 100 * (after − before) / before
        "status":               str,    # "champion" | "accepted" | "rejected"
        "reason":               str,    # free text
    }

Public API
----------
* :func:`append_iteration(record, ...)` — adds one record and regenerates
  the Markdown view in one shot.
* :func:`load_history(...)` — read all records (returns ``list[dict]``).
* :func:`render_markdown(records)` — pure formatter, doesn't touch disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_DIR = Path("result")
JSONL_NAME = "history.jsonl"
MARKDOWN_NAME = "history.md"

_REQUIRED_FIELDS = (
    "iteration",
    "summary",
    "ll_after",
    "complexity_after",
    "status",
    "reason",
)
_VALID_STATUSES = {"champion", "accepted", "rejected"}


def append_iteration(
    record: dict,
    *,
    dir_: Path = DEFAULT_DIR,
) -> None:
    """
    Append one iteration record to history.jsonl and regenerate history.md.

    Mutates the record in place to fill in ``timestamp`` (UTC ISO-8601) if
    missing and ``improvement`` / ``complexity_pct_change`` from the
    before/after pairs when they can be derived. Validates required fields
    and status.

    Idempotent over content: this function appends unconditionally. The
    loop driver is responsible for not double-recording the same iteration.
    """
    for f in _REQUIRED_FIELDS:
        if f not in record:
            raise ValueError(f"history record missing required field {f!r}")
    if record["status"] not in _VALID_STATUSES:
        raise ValueError(
            f"history status must be one of {_VALID_STATUSES}, "
            f"got {record['status']!r}"
        )

    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    if record.get("improvement") is None and record.get("ll_before") is not None:
        record["improvement"] = record["ll_before"] - record["ll_after"]
    if (
        record.get("complexity_pct_change") is None
        and record.get("complexity_before") not in (None, 0)
    ):
        before = record["complexity_before"]
        after = record["complexity_after"]
        record["complexity_pct_change"] = 100.0 * (after - before) / before

    dir_.mkdir(parents=True, exist_ok=True)
    jsonl_path = dir_ / JSONL_NAME
    md_path = dir_ / MARKDOWN_NAME

    with jsonl_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record) + "\n")

    md_path.write_text(render_markdown(load_history(dir_=dir_)), encoding="utf-8")


def load_history(*, dir_: Path = DEFAULT_DIR) -> list[dict]:
    """Read every record from history.jsonl. Empty list if file is missing."""
    jsonl_path = dir_ / JSONL_NAME
    if not jsonl_path.exists():
        return []
    out = []
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def render_markdown(records: Iterable[dict]) -> str:
    """Render a list of records as a Markdown report. Pure, no disk I/O."""
    rows = list(records)
    lines: list[str] = [
        "# FSRS-7 autoresearch — iteration history",
        "",
        f"_{len(rows)} record(s). Generated from `{JSONL_NAME}` — do not edit by hand._",
        "",
        "| # | Time (UTC) | Thresh. | LL before | LL after | Δ LL | Cx before | Cx after | Δ Cx % | Status | Summary |",
        "|--:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {iteration} | {ts} | {thresh} | {llb} | {lla} | "
            "{imp} | {cxb} | {cxa} | {dcx} | {status} | {summary} |".format(
                iteration=r.get("iteration", "?"),
                ts=_fmt_timestamp(r.get("timestamp")),
                summary=_escape_pipes(r.get("summary", "")),
                thresh=_fmt_float(r.get("threshold"), 4),
                llb=_fmt_float(r.get("ll_before"), 5),
                lla=_fmt_float(r.get("ll_after"), 5),
                imp=_fmt_float(r.get("improvement"), 5, sign=True),
                cxb=_fmt_int(r.get("complexity_before")),
                cxa=_fmt_int(r.get("complexity_after")),
                dcx=_fmt_float(r.get("complexity_pct_change"), 2, sign=True, suffix="%"),
                status=r.get("status", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt_float(
    v: Optional[float],
    decimals: int,
    *,
    sign: bool = False,
    suffix: str = "",
) -> str:
    if v is None:
        return "—"
    if sign and v >= 0:
        return f"+{v:.{decimals}f}{suffix}"
    return f"{v:.{decimals}f}{suffix}"


def _fmt_int(v: Optional[int]) -> str:
    if v is None:
        return "—"
    return f"{int(v):,}"


def _fmt_timestamp(ts: Optional[str]) -> str:
    if not ts:
        return "—"
    # Trim subsecond precision and trailing Z/offset for column readability.
    if "." in ts:
        ts = ts.split(".", 1)[0]
    if ts.endswith("+00:00"):
        ts = ts[:-6] + "Z"
    return ts


def _escape_pipes(s: str) -> str:
    return s.replace("|", "\\|")
