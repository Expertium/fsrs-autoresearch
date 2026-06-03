#!/usr/bin/env python3
"""
Automated hyperparameter tuner for the FSRS-7 autoresearch loop.

The interesting part of the loop is the model-structure search; nudging numeric
*training* hyperparameters (learning rate, Adam betas, L2 strength, recency
weighting) up and down is mechanical and boring. This automates it. The compute
operating point (n_epoch, batch_size) is handled separately by the Pareto grid
below — it is NOT part of the coordinate descent.

What it does
------------
A greedy **coordinate-descent** local search over the *fine* training HPs (LR,
Adam betas, L2 strength, recency weighting). For each it tries a step up and a
step down (multiplicative for LR/L2, on the ``1 - beta`` scale for the betas),
keeps whichever lowers ``logloss_by_user``, and re-probes the improving ones in
later rounds. Params that don't improve are frozen ("light" mode → typically
~14-20 benchmark runs/pass). Training is deterministic (``seed`` fixed), so every
delta is real and reproducible — no averaging needed.

**The compute operating point — ``n_epoch`` and ``batch_size`` — is NOT tuned
here.** Those are a speed/log-loss trade-off, not a pure-loss knob (a loss-only
search always pushes batch down, ignoring the speed cost), so they live in a
separate, rarely-run **epoch x batch Pareto grid** (``--epoch-batch-grid``). The
gold standard is ``(n_epoch=8, batch_size=512)``; the grid sweeps the 20 combos of
``n_epoch in {5,8,12,16,30}`` x ``batch_size in {128,256,512,1024}`` and re-anchors
to any cell that Pareto-dominates the gold standard (no worse on log loss OR
compute time, strictly better on >=1). Each cell is measured with an Adam
sqrt LR-batch-scaled learning rate for fairness; the winner is then fine-tuned by
the regular pass. n_epoch is driven per-cell via the ``FSRS_N_EPOCHS`` env var and
persisted by editing config.py's env-default; ``compute_seconds`` (train+eval wall
time) comes from diagnostics.json (emitted by run.py).

Most fine knobs live in ``fsrs_v7_constants.py``; **BATCH_SIZE / N_EPOCHS live in
``src/main/config.py``**, so this tuner edits *two* files (see ``read_texts`` /
``HParam.file``). Each trial edits a numeric literal *in place*. That never
changes the AST node count, so the complexity score is unaffected — which is
exactly why a hyperparameter tweak only has to clear the ``+0.0001`` floor
threshold, with no complexity gate to worry about. (Changing BATCH_SIZE / N_EPOCHS
rebuilds only the cheap batch-size-dependent cache arrays — ``num_training_steps``
and the batch permutation — via their own cache manifests; the expensive tensor
cache is reused. Verified to take effect, not silently masked.)

Fully autonomous: when the best config found beats the champion by ≥ threshold
it commits the new champion (constants + config + diagnostics + history) and tags
it; otherwise it restores the champion and records a rejected pass. It owns one
iteration number per pass (read from ``history.jsonl``).

Run it from the **host** (it shells out to ``docker compose`` for each
benchmark and to ``git`` for commits). A pass is many ~70-150 s runs, so launch
it in the background:

    python -m src.autoresearch.hp_tune                   # full auto pass (fine HPs)
    python -m src.autoresearch.hp_tune --epoch-batch-grid  # re-anchor n_epoch x batch
    python src/autoresearch/hp_tune.py --dry-run         # validate file edits only
    python src/autoresearch/hp_tune.py --no-commit       # search, leave best, no git
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
CONSTANTS = REPO / "src" / "main" / "fsrs" / "fsrs_v7_constants.py"
CONFIG = REPO / "src" / "main" / "config.py"
DIAGNOSTICS = REPO / "result" / "diagnostics.json"
HISTORY_JSONL = REPO / "result" / "history.jsonl"
SUMMARY = REPO / "result" / "hp_tune_last.json"
GRID_SUMMARY = REPO / "result" / "epoch_batch_grid.json"

# ── epoch × batch_size Pareto grid (the "gold standard" re-anchor, run rarely) ──
# A speed-aware OUTER search over the compute operating point, distinct from the
# per-5-iter LR/betas/L2 coordinate descent. The gold standard is (n_epoch=8,
# batch_size=512); a cell replaces it only if it Pareto-dominates it — no worse on
# either axis (log loss, train+eval compute time) and strictly better on >=1. The
# fine HPs are conditional on (epoch, batch), so this runs FIRST and the regular
# tuner re-tunes LR/betas/L2 on the winner afterward; for fairness each cell is
# measured with an Adam sqrt LR-batch-scaled learning rate (the winner is then
# fine-tuned precisely). batch_size is OWNED here and was removed from the regular
# coordinate descent so the loss-only pass can't move it for log loss at a speed cost.
GOLD_EPOCH, GOLD_BATCH = 8, 512
GRID_EPOCHS = [5, 8, 12, 16, 30]
GRID_BATCHES = [128, 256, 512, 1024]
SPEED_TOL = 0.03   # fractional compute-time noise band: within +/-3% counts as "same speed"
LL_TOL = 1e-5      # log-loss tie band: sits above the measured ~4e-6 by_user GPU-noise
                   # floor (training is NOT bit-exact — float reduction non-associativity)

# The files this tuner may edit: constants.py holds the 6 fine knobs (regular
# pass); config.py holds BATCH_SIZE / N_EPOCHS, edited ONLY by epoch_batch_grid().
# Both are validated with ast.parse before any benchmark runs.
FILES = {"constants": CONSTANTS, "config": CONFIG}

# Tracked paths the REGULAR pass touches on accept (commit) / reject (revert). The
# regular coordinate descent only edits constants.py now (batch_size moved to the
# grid); config.py is kept here as a defensive no-op (unchanged by the regular pass,
# so `git add` / `git checkout` on it do nothing). The grid does its own bookkeeping.
EDITED_PATHS = ["src/main/fsrs/fsrs_v7_constants.py", "src/main/config.py"]

BENCH_CMD = ["docker", "compose", "--progress", "quiet", "run", "--rm",
             "srs-benchmark", "bash", "src/main/run.sh"]
APPEND_CMD = ["docker", "compose", "--progress", "quiet", "run", "--rm", "-T",
              "srs-benchmark", "python", "-c",
              "import sys, json; from src.autoresearch.history import "
              "append_iteration; append_iteration(json.loads(sys.stdin.read()))"]

TRAILER = "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
EPS = 1e-6  # minimum by_user decrease counted as a real improvement during search

_NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


# ── editing the numeric literals ─────────────────────────────────────────────
def _fmt(v: float) -> str:
    """Compact literal capped at 4 decimal places (user constraint 2026-06-03:
    every committed non-integer hyperparameter has <=4 digits after the decimal
    point). Integers stay integer-formatted (1024.0 -> '1024'); rounding also
    strips float-repr noise (0.0300000000002 -> '0.03'). Candidates are rounded
    too (search granularity = 1e-4), which is ample for LR/L2/betas/recency."""
    v = round(float(v), 4)
    if v == int(v):
        return str(int(v))
    return f"{v:.4f}".rstrip("0")  # guaranteed <=4 dp, no trailing zeros


def read_texts() -> dict[str, str]:
    """Read every editable file's source, keyed by short name."""
    return {name: path.read_text(encoding="utf-8") for name, path in FILES.items()}


def write_texts(texts: dict[str, str]) -> None:
    """Write every editable file back to disk (unchanged files are rewritten
    identically, which git treats as a no-op)."""
    for name, path in FILES.items():
        path.write_text(texts[name], encoding="utf-8")


def get_scalar(text: str, name: str) -> float:
    m = re.search(rf"^{re.escape(name)}\b[^=\n]*=\s*({_NUM})", text, re.MULTILINE)
    if not m:
        raise ValueError(f"could not read scalar {name!r}")
    return float(m.group(1))


def set_scalar(text: str, name: str, value: float) -> str:
    pat = re.compile(rf"^({re.escape(name)}\b[^=\n]*=\s*){_NUM}", re.MULTILINE)
    new, n = pat.subn(rf"\g<1>{_fmt(value)}", text, count=1)
    if n != 1:
        raise ValueError(f"could not set scalar {name!r} (matched {n}x)")
    return new


def get_betas(text: str) -> tuple[float, float]:
    m = re.search(rf"^BETAS\b[^=\n]*=\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)",
                  text, re.MULTILINE)
    if not m:
        raise ValueError("could not read BETAS")
    return float(m.group(1)), float(m.group(2))


def set_betas(text: str, b1: float, b2: float) -> str:
    pat = re.compile(rf"^(BETAS\b[^=\n]*=\s*)\(\s*{_NUM}\s*,\s*{_NUM}\s*\)",
                     re.MULTILINE)
    new, n = pat.subn(rf"\g<1>({_fmt(b1)}, {_fmt(b2)})", text, count=1)
    if n != 1:
        raise ValueError(f"could not set BETAS (matched {n}x)")
    return new


# N_EPOCHS lives in config.py as `int(os.environ.get("FSRS_N_EPOCHS", "8"))`. The
# grid persists the winning epoch count by editing the *default* literal "8" (the
# sweep itself drives it via the FSRS_N_EPOCHS env var, no file edit). Only this
# integer-in-the-env-default is touched — the env-override mechanism is preserved.
_NEPOCH_RE = re.compile(r'(FSRS_N_EPOCHS["\']\s*,\s*["\'])(\d+)(["\'])')


def get_n_epochs(config_text: str) -> int:
    m = _NEPOCH_RE.search(config_text)
    if not m:
        raise ValueError("could not read N_EPOCHS default from config.py")
    return int(m.group(2))


def set_n_epochs(config_text: str, n: int) -> str:
    new, k = _NEPOCH_RE.subn(rf"\g<1>{int(n)}\g<3>", config_text, count=1)
    if k != 1:
        raise ValueError(f"could not set N_EPOCHS default (matched {k}x)")
    return new


# ── hyperparameter spec ──────────────────────────────────────────────────────
@dataclass
class HParam:
    name: str
    get: Callable[[dict[str, str]], float]
    put: Callable[[dict[str, str], float], dict[str, str]]
    kind: str        # "mul" perturbs v; "beta" perturbs (1 - v)
    step: float
    lo: float
    hi: float
    active: bool = True
    file: str = "constants"  # which FILES entry this knob's ast.parse check targets

    def candidates(self, v: float) -> list[float]:
        # NOTE: only "mul" (LR/L2/recency) and "beta" (Adam betas) knobs exist here.
        # batch_size used to be a discrete "batch" knob; it's no longer tuned in this
        # coordinate descent — the epoch_batch_grid() Pareto search owns it.
        if self.kind == "mul":
            raw = [v * self.step, v / self.step]
        elif self.kind == "beta":
            raw = [1.0 - (1.0 - v) * self.step, 1.0 - (1.0 - v) / self.step]
        else:
            raise ValueError(self.kind)
        out = []
        for c in raw:
            c = float(_fmt(c))
            if self.lo <= c <= self.hi and abs(c - v) > 1e-9:
                out.append(c)
        return out


def build_hparams() -> list[HParam]:
    hps = [
        HParam("LR",
               lambda ts: get_scalar(ts["constants"], "LR"),
               lambda ts, v: {**ts, "constants": set_scalar(ts["constants"], "LR", v)},
               "mul", 1.5, 1e-3, 0.3),
        HParam("PENALTY_W_L2",
               lambda ts: get_scalar(ts["constants"], "PENALTY_W_L2"),
               lambda ts, v: {**ts, "constants": set_scalar(ts["constants"], "PENALTY_W_L2", v)},
               "mul", 1.5, 0.02, 8.0),
        HParam("BETA1",
               lambda ts: get_betas(ts["constants"])[0],
               lambda ts, v: {**ts, "constants": set_betas(ts["constants"], v, get_betas(ts["constants"])[1])},
               "beta", 1.5, 0.4, 0.98),
        HParam("BETA2",
               lambda ts: get_betas(ts["constants"])[1],
               lambda ts, v: {**ts, "constants": set_betas(ts["constants"], get_betas(ts["constants"])[0], v)},
               "beta", 1.5, 0.4, 0.999),
    ]
    # iter-65: recency-weighting knobs replaced the iter-52 per-group LR multipliers
    # (dropped for a single global LR). RECENCY_C0 = floor weight on the oldest
    # reviews; RECENCY_EXP = sharpness of the ramp toward the newest. gradient_weight
    # = C0 + (1-C0)*ord_frac^EXP, so the newest-review weight stays pinned at 1 for
    # any C0 (no separate C1 knob). Both are scalars in constants.py.
    hps.append(HParam("RECENCY_C0",
                      lambda ts: get_scalar(ts["constants"], "RECENCY_C0"),
                      lambda ts, v: {**ts, "constants": set_scalar(ts["constants"], "RECENCY_C0", v)},
                      "mul", 1.5, 0.01, 0.5))
    hps.append(HParam("RECENCY_EXP",
                      lambda ts: get_scalar(ts["constants"], "RECENCY_EXP"),
                      lambda ts, v: {**ts, "constants": set_scalar(ts["constants"], "RECENCY_EXP", v)},
                      "mul", 1.5, 1.0, 12.0))
    # NOTE: BATCH_SIZE used to be tuned here, but it's a speed/log-loss trade-off,
    # not a pure log-loss knob — the loss-only coordinate descent would always push
    # it down (smaller batch = lower loss) while ignoring the speed cost. It now
    # belongs to the epoch_batch_grid() Pareto search (run --epoch-batch-grid), which
    # owns the (n_epoch, batch_size) compute operating point. This regular pass tunes
    # the fine HPs (LR/betas/L2/recency) at that fixed operating point.
    return hps


# ── running the benchmark ────────────────────────────────────────────────────
def run_benchmark() -> tuple[float, int, float]:
    t0 = time.time()
    proc = subprocess.run(BENCH_CMD, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"benchmark exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-1500:]}\n"
            f"--- stderr ---\n{proc.stderr[-1500:]}"
        )
    diag = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    by_user = float(diag["logloss"]["by_user"])
    complexity = int(sum(f["score"] for f in diag["complexity"]["files"]))
    return by_user, complexity, time.time() - t0


def run_benchmark_grid(extra_env: dict[str, str] | None = None) -> tuple[float, float]:
    """One benchmark for the epoch/batch grid; returns (by_user, compute_seconds).

    ``extra_env`` (e.g. ``{"FSRS_N_EPOCHS": "12"}``) is injected into the container
    via ``docker compose run -e``. compute_seconds is the train+eval wall time
    emitted by run.py into diagnostics.json — the grid's speed axis."""
    cmd = ["docker", "compose", "--progress", "quiet", "run", "--rm"]
    for k, v in (extra_env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["srs-benchmark", "bash", "src/main/run.sh"]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"benchmark exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-1500:]}\n"
            f"--- stderr ---\n{proc.stderr[-1500:]}"
        )
    diag = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    by_user = float(diag["logloss"]["by_user"])
    secs = float(diag.get("compute_seconds") or 0.0)
    return by_user, secs


# ── epoch × batch_size Pareto grid ───────────────────────────────────────────
def _scaled_lr(committed_lr: float, committed_batch: float, batch: float) -> float:
    """Adam sqrt LR-batch scaling, anchored at the committed (LR, batch). Gives each
    grid cell a roughly-right LR for a fair comparison; the winner is fine-tuned after."""
    return round(committed_lr * math.sqrt(batch / committed_batch), 4)


def _grid_table(cells: list[dict], gold: dict | None) -> str:
    """ASCII table of the grid: rows = n_epoch, cols = batch_size, each cell
    'by_user / seconds'. The gold standard cell is marked with '*'."""
    head = "epoch\\batch | " + " | ".join(f"{b:>16}" for b in GRID_BATCHES)
    out = [head, "-" * len(head)]
    by_key = {(c["epoch"], c["batch"]): c for c in cells}
    for e in GRID_EPOCHS:
        parts = []
        for b in GRID_BATCHES:
            c = by_key.get((e, b))
            if c is None or c.get("by_user") is None:
                parts.append(f"{'FAIL':>16}")
            else:
                star = "*" if (gold and e == GOLD_EPOCH and b == GOLD_BATCH) else " "
                parts.append(f"{c['by_user']:.5f}/{c['seconds']:>5.0f}s{star}")
        out.append(f"{e:>11} | " + " | ".join(parts))
    return "\n".join(out)


def epoch_batch_grid() -> None:
    """Measure the (8,512) gold standard FIRST, then sweep the 19 OTHER (n_epoch x
    batch_size) candidates (skipping (8,512) — no point comparing it to itself),
    judging each against the gold reference. Picks the Pareto operating point and
    leaves the winner's (batch, n_epoch, scaled LR) on disk for review. Does NOT
    commit or record history — re-anchoring the champion is a deliberate step the
    caller finalises after the follow-up fine-tune. Total runs = 1 gold + 19 = 20."""
    base_texts = read_texts()
    for text in base_texts.values():
        ast.parse(text)
    committed_lr = get_scalar(base_texts["constants"], "LR")
    committed_batch = get_scalar(base_texts["config"], "BATCH_SIZE")
    committed_epoch = get_n_epochs(base_texts["config"])
    n_candidates = len(GRID_EPOCHS) * len(GRID_BATCHES) - 1
    print(f"[grid] committed operating point: epoch={committed_epoch} "
          f"batch={committed_batch:g} LR={committed_lr:g}", flush=True)
    print(f"[grid] gold standard = (epoch {GOLD_EPOCH}, batch {GOLD_BATCH}); measured "
          f"first, then {n_candidates} candidates judged against it (sqrt LR-batch scaling).",
          flush=True)

    def run_cell(epoch: int, batch: int) -> dict:
        """Edit batch + sqrt-scaled LR, run with FSRS_N_EPOCHS=epoch; return the cell."""
        lr = _scaled_lr(committed_lr, committed_batch, batch)
        trial = {**base_texts,
                 "config": set_scalar(base_texts["config"], "BATCH_SIZE", batch)}
        trial["constants"] = set_scalar(base_texts["constants"], "LR", lr)
        ast.parse(trial["config"]); ast.parse(trial["constants"])
        write_texts(trial)
        t0 = time.time()
        try:
            ll, secs = run_benchmark_grid({"FSRS_N_EPOCHS": str(epoch)})
        except Exception as e:  # noqa: BLE001 — one bad cell shouldn't kill the grid
            print(f"[grid] epoch {epoch:>2} batch {batch:>4}: FAILED ({e})", flush=True)
            return {"epoch": epoch, "batch": batch, "lr": lr,
                    "by_user": None, "seconds": None, "error": str(e)}
        print(f"[grid] epoch {epoch:>2} batch {batch:>4} (LR {lr:g}): "
              f"by_user={ll:.6f}  compute={secs:.1f}s  ({time.time() - t0:.0f}s wall)",
              flush=True)
        return {"epoch": epoch, "batch": batch, "lr": lr, "by_user": ll, "seconds": secs}

    cells: list[dict] = []
    try:
        # 1. GOLD STANDARD FIRST — the reference every candidate is judged against.
        print(f"[grid] measuring gold standard (epoch {GOLD_EPOCH}, batch {GOLD_BATCH}) "
              f"first ...", flush=True)
        gold = run_cell(GOLD_EPOCH, GOLD_BATCH)
        cells.append(gold)
        if gold["by_user"] is None:
            print("[grid] ABORT: gold-standard cell failed — no reference, no decision.",
                  flush=True)
            GRID_SUMMARY.write_text(json.dumps({"cells": cells, "gold": None}, indent=2),
                                    encoding="utf-8")
            return
        g_ll, g_s = gold["by_user"], gold["seconds"]
        print(f"[grid] GOLD: by_user={g_ll:.6f}  compute={g_s:.1f}s  "
              f"(the {n_candidates} candidates are compared to this)\n", flush=True)

        def dominates(c: dict) -> bool:
            if c["by_user"] is None:
                return False
            not_worse = (c["by_user"] <= g_ll + LL_TOL) and (c["seconds"] <= g_s * (1 + SPEED_TOL))
            strictly_better = (c["by_user"] < g_ll - LL_TOL) or (c["seconds"] < g_s * (1 - SPEED_TOL))
            return not_worse and strictly_better

        # 2. The candidates — every combo EXCEPT the gold standard itself.
        for epoch in GRID_EPOCHS:
            for batch in GRID_BATCHES:
                if (epoch, batch) == (GOLD_EPOCH, GOLD_BATCH):
                    continue  # no point comparing (8,512) to itself
                c = run_cell(epoch, batch)
                cells.append(c)
                if c["by_user"] is not None:
                    dl, ds = c["by_user"] - g_ll, 100 * (c["seconds"] - g_s) / g_s
                    if dominates(c):
                        tag = "DOMINATES gold"
                    elif c["by_user"] > g_ll + LL_TOL and c["seconds"] > g_s * (1 + SPEED_TOL):
                        tag = "worse on both"
                    else:
                        tag = "mixed (trade-off)"
                    print(f"        vs gold: d_loss={dl:+.6f}  d_speed={ds:+.1f}%  -> {tag}",
                          flush=True)
    finally:
        write_texts(base_texts)  # restore committed config before deciding/persisting

    ok = [c for c in cells if c.get("by_user") is not None]
    print("\n[grid] results (by_user / compute_seconds; * = gold standard):", flush=True)
    print(_grid_table(cells, gold), flush=True)

    dominators = [c for c in ok if dominates(c)]
    # Pick the operating point: among cells NOT slower than gold, the lowest log loss
    # (tiebreak: fastest). Gold is in the pool, so if no candidate dominates it, gold
    # wins and we stay put. This captures "lower loss at <= gold speed" and, on a loss
    # tie, "faster at the same loss" — exactly the user's at-least-one-axis rule.
    not_slower = [c for c in ok if c["seconds"] <= g_s * (1 + SPEED_TOL)]
    winner = min(not_slower, key=lambda c: (c["by_user"], c["seconds"]))
    improved = (winner["epoch"], winner["batch"]) != (GOLD_EPOCH, GOLD_BATCH) and dominates(winner)

    print(f"\n[grid] gold (8,512): by_user={g_ll:.6f}  compute={g_s:.1f}s", flush=True)
    if dominators:
        print(f"[grid] {len(dominators)} candidate(s) Pareto-dominate gold:", flush=True)
        for c in sorted(dominators, key=lambda c: (c["by_user"], c["seconds"])):
            print(f"        epoch {c['epoch']:>2} batch {c['batch']:>4}: "
                  f"by_user={c['by_user']:.6f} ({c['by_user'] - g_ll:+.6f})  "
                  f"compute={c['seconds']:.1f}s ({100 * (c['seconds'] - g_s) / g_s:+.1f}%)",
                  flush=True)
    else:
        print("[grid] no candidate Pareto-dominates gold — operating point stays at (8,512).",
              flush=True)

    # Persist the winner's operating point (batch, n_epoch default, sqrt-scaled LR).
    new_cfg = set_scalar(base_texts["config"], "BATCH_SIZE", winner["batch"])
    new_cfg = set_n_epochs(new_cfg, winner["epoch"])
    new_const = set_scalar(base_texts["constants"], "LR", winner["lr"])
    final = {"config": new_cfg, "constants": new_const}
    ast.parse(final["config"]); ast.parse(final["constants"])
    write_texts(final)
    print(f"\n[grid] new operating point: epoch={winner['epoch']} batch={winner['batch']} "
          f"LR={winner['lr']:g}  (by_user={winner['by_user']:.6f}, compute={winner['seconds']:.1f}s)"
          f"{'  [Pareto win over gold]' if improved else '  [= gold standard]'}", flush=True)

    # Refresh diagnostics.json to match the persisted operating point (the grid loop
    # left it pointing at the last cell). Uses the committed N_EPOCHS default now.
    print("[grid] re-running winner to refresh diagnostics ...", flush=True)
    ll2, secs2 = run_benchmark_grid()
    if abs(ll2 - winner["by_user"]) > 1e-6:
        print(f"[grid] WARNING: winner re-check by_user {ll2:.6f} != {winner['by_user']:.6f}",
              flush=True)

    GRID_SUMMARY.write_text(json.dumps({
        "gold": {"epoch": GOLD_EPOCH, "batch": GOLD_BATCH,
                 "by_user": g_ll, "seconds": g_s},
        "committed_before": {"epoch": committed_epoch, "batch": committed_batch,
                             "lr": committed_lr},
        "winner": winner, "improved_over_gold": improved,
        "dominators": dominators, "cells": cells,
        "speed_tol": SPEED_TOL, "ll_tol": LL_TOL,
    }, indent=2), encoding="utf-8")

    print(f"\n[grid] wrote {GRID_SUMMARY.relative_to(REPO)}. Operating point left on disk "
          f"(uncommitted).", flush=True)
    print("[grid] NEXT: review, then run the regular pass to fine-tune LR/betas/L2 at "
          "this operating point, and record the combined re-anchor as one iteration.",
          flush=True)


def time_noise(n: int = 5) -> None:
    """Measure the noise floor on the grid's SPEED axis. Runs the gold config
    (GOLD_EPOCH, GOLD_BATCH, sqrt-scaled LR) n+1 times — 1 warm-up (discarded; it
    pays the one-time batch_perm cache rebuild + GPU clock cold-start) then n measured
    runs — and reports the compute_seconds spread. log loss is deterministic, so only
    time carries noise; this spread is what SPEED_TOL must cover. Restores config after.
    Does not commit or touch git."""
    base_texts = read_texts()
    for text in base_texts.values():
        ast.parse(text)
    committed_lr = get_scalar(base_texts["constants"], "LR")
    committed_batch = get_scalar(base_texts["config"], "BATCH_SIZE")
    lr = _scaled_lr(committed_lr, committed_batch, GOLD_BATCH)
    trial = {**base_texts,
             "config": set_scalar(base_texts["config"], "BATCH_SIZE", GOLD_BATCH)}
    trial["constants"] = set_scalar(base_texts["constants"], "LR", lr)
    ast.parse(trial["config"]); ast.parse(trial["constants"])
    print(f"[noise] gold config: epoch={GOLD_EPOCH} batch={GOLD_BATCH} LR={lr:g}; "
          f"1 warm-up + {n} measured runs.", flush=True)

    secs: list[float] = []
    lls: list[float] = []
    try:
        write_texts(trial)
        for i in range(n + 1):
            ll, s = run_benchmark_grid({"FSRS_N_EPOCHS": str(GOLD_EPOCH)})
            label = "warm-up (discarded)" if i == 0 else f"run {i}/{n}"
            print(f"[noise] {label:<20} compute={s:7.2f}s  by_user={ll:.7f}", flush=True)
            if i > 0:
                secs.append(s); lls.append(ll)
    finally:
        write_texts(base_texts)  # restore committed config

    mean = sum(secs) / len(secs)
    var = sum((x - mean) ** 2 for x in secs) / len(secs)
    sd = var ** 0.5
    lo, hi = min(secs), max(secs)
    cv = sd / mean if mean else 0.0
    rng_frac = (hi - lo) / mean if mean else 0.0
    ll_lo, ll_hi = min(lls), max(lls)
    ll_spread = ll_hi - ll_lo
    suggested = max(0.02, round(1.5 * rng_frac, 3))
    print(f"\n[noise] compute_seconds over {n} runs:  mean={mean:.2f}s  std={sd:.3f}s  "
          f"min={lo:.2f}s  max={hi:.2f}s", flush=True)
    print(f"[noise] CV (std/mean) = {100 * cv:.2f}%   range/mean = {100 * rng_frac:.2f}%",
          flush=True)
    print(f"[noise] log-loss spread across runs: {ll_spread:.2e} (min {ll_lo:.7f}, "
          f"max {ll_hi:.7f}) — NOT bit-exact: GPU floating-point non-associativity in "
          f"parallel reductions. This is the by_user noise floor; well below the grid's "
          f"LL_TOL={LL_TOL:g} and the 1e-4 acceptance floor.", flush=True)
    print(f"[noise] current SPEED_TOL = {SPEED_TOL} ({100 * SPEED_TOL:.0f}%); suggested "
          f">= {suggested} (~1.5x observed range/mean, floor 2%) so time noise can't fake "
          f"a speed win. Current value is safe.", flush=True)
    NOISE_SUMMARY = REPO / "result" / "time_noise_last.json"
    NOISE_SUMMARY.write_text(json.dumps({
        "epoch": GOLD_EPOCH, "batch": GOLD_BATCH, "lr": lr, "n": n,
        "compute_seconds": secs, "by_user": lls,
        "mean": mean, "std": sd, "min": lo, "max": hi, "cv": cv,
        "range_frac": rng_frac,
        "logloss_min": ll_lo, "logloss_max": ll_hi, "logloss_spread": ll_spread,
        "current_speed_tol": SPEED_TOL, "suggested_speed_tol": suggested,
    }, indent=2), encoding="utf-8")
    print(f"[noise] wrote {NOISE_SUMMARY.relative_to(REPO)}.", flush=True)


# ── git + history ────────────────────────────────────────────────────────────
def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, check=check)


def tree_clean() -> bool:
    return git("status", "--porcelain").stdout.strip() == ""


def next_iteration() -> int:
    if not HISTORY_JSONL.exists():
        return 0
    last = -1
    for line in HISTORY_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = max(last, int(json.loads(line)["iteration"]))
    return last + 1


def append_history(record: dict) -> None:
    proc = subprocess.run(APPEND_CMD, cwd=str(REPO), input=json.dumps(record),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"history append failed:\n{proc.stderr[-1500:]}")


def refresh_plot() -> None:
    """Regenerate result/history_plot.png from history via the human's
    plot_history.py (run as a subprocess — never edited). Non-fatal: a plot
    failure must not block recording/committing an iteration."""
    try:
        proc = subprocess.run(
            [sys.executable, str(REPO / "src" / "autoresearch" / "plot_history.py")],
            cwd=str(REPO), capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"[hp_tune] {proc.stdout.strip() or 'history plot refreshed'}", flush=True)
        else:
            print(f"[hp_tune] plot refresh skipped (rc={proc.returncode}): "
                  f"{(proc.stderr or proc.stdout)[-400:]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hp_tune] plot refresh error (skipped): {e}", flush=True)


# ── the search ───────────────────────────────────────────────────────────────
def search(rounds: int, max_runs: int) -> dict:
    base_texts = read_texts()
    for name, text in base_texts.items():
        ast.parse(text)  # sanity: every editable file is valid before we touch it
    hparams = build_hparams()
    orig = {hp.name: hp.get(base_texts) for hp in hparams}

    runs = 0
    print("[hp_tune] baseline run ...", flush=True)
    base_ll, base_cx, dt = run_benchmark()
    runs += 1
    print(f"[hp_tune] baseline by_user={base_ll:.6f} complexity={base_cx} ({dt:.0f}s)",
          flush=True)

    best_texts, best_ll = base_texts, base_ll
    trials: list[dict] = []

    for r in range(rounds):
        improved = False
        for hp in hparams:
            if not hp.active or runs >= max_runs:
                continue
            cur = hp.get(best_texts)
            results = []  # (ll, cand, texts)
            for cand in hp.candidates(cur):
                if runs >= max_runs:
                    break
                trial_texts = hp.put(best_texts, cand)
                try:
                    ast.parse(trial_texts[hp.file])
                    if abs(hp.get(trial_texts) - cand) > 1e-9:
                        raise ValueError("round-trip mismatch")
                except Exception as e:  # noqa: BLE001
                    print(f"[hp_tune] skip {hp.name} -> {cand:g}: bad edit ({e})",
                          flush=True)
                    continue
                write_texts(trial_texts)
                ll, _, dt = run_benchmark()
                runs += 1
                delta = base_ll - ll
                better = ll < best_ll - EPS
                trials.append({"round": r, "hp": hp.name, "from": cur, "to": cand,
                               "by_user": ll, "delta_vs_baseline": delta})
                print(f"[hp_tune] r{r} {hp.name}: {cur:g} -> {cand:g}  "
                      f"by_user={ll:.6f}  d_vs_base={delta:+.6f}"
                      f"{'  <-- new best' if better else ''}  ({dt:.0f}s)",
                      flush=True)
                results.append((ll, cand, trial_texts))
            if not results:
                hp.active = False
                continue
            results.sort(key=lambda x: x[0])
            cand_ll, _, cand_texts = results[0]
            if cand_ll < best_ll - EPS:
                best_ll, best_texts = cand_ll, cand_texts
                improved = True
                # keep hp active so a later round can step it further
            else:
                hp.active = False  # locally optimal at this granularity
        if not improved or runs >= max_runs:
            break

    # Leave the best config on disk and refresh diagnostics to match it.
    write_texts(best_texts)
    print("[hp_tune] re-running best config to refresh diagnostics ...", flush=True)
    final_ll, final_cx, dt = run_benchmark()
    runs += 1
    if abs(final_ll - best_ll) > 1e-6:
        print(f"[hp_tune] WARNING: best re-check {final_ll:.6f} != {best_ll:.6f} "
              f"(non-determinism?)", flush=True)
    final = {hp.name: hp.get(best_texts) for hp in hparams}
    changes = {n: [orig[n], final[n]] for n in orig if abs(orig[n] - final[n]) > 1e-12}
    return {
        "baseline_by_user": base_ll,
        "best_by_user": best_ll,
        "final_recheck_by_user": final_ll,
        "improvement": base_ll - best_ll,
        "complexity": final_cx,
        "changes": changes,
        "n_runs": runs,
        "trials": trials,
    }


# ── dry run: validate the regex editing without GPU/git ──────────────────────
def dry_run() -> None:
    base_texts = read_texts()
    ok = True
    try:
        for hp in build_hparams():
            cur = hp.get(base_texts)
            cands = hp.candidates(cur)
            details = []
            for c in cands:
                trial = hp.put(base_texts, c)
                try:
                    ast.parse(trial[hp.file])
                    rt = hp.get(trial)
                    assert abs(rt - c) < 1e-9, f"round-trip {rt} != {c}"
                    details.append(f"{c:g} OK")
                except Exception as e:  # noqa: BLE001
                    ok = False
                    details.append(f"{c:g} FAIL({e})")
            print(f"[dry-run] {hp.name} ({hp.file}): current={cur:g}  "
                  f"candidates=[{', '.join(details) or 'none'}]")
        # betas coupling sanity
        b1, b2 = get_betas(base_texts["constants"])
        t = set_betas(base_texts["constants"], 0.77, 0.93)
        assert get_betas(t) == (0.77, 0.93)
        print(f"[dry-run] BETAS read=({b1:g}, {b2:g})  set/read round-trip OK")
    finally:
        write_texts(base_texts)
        restored = read_texts() == base_texts
        print(f"[dry-run] editable files restored unchanged: {restored}")
    print(f"[dry-run] {'ALL EDITS VALID' if ok else 'SOME EDITS FAILED'}")


# ── accept / reject + commit ─────────────────────────────────────────────────
def _changes_str(changes: dict) -> str:
    return ", ".join(f"{k} {a:g}->{b:g}" for k, (a, b) in changes.items()) or "none"


def finalize(res: dict, threshold: float) -> None:
    n = next_iteration()
    imp = res["improvement"]
    cx = res["complexity"]
    base_ll = res["baseline_by_user"]
    best_ll = res["best_by_user"]
    changes = res["changes"]
    cs = _changes_str(changes)
    accept = imp >= threshold and bool(changes)

    # cadence marker — machine-local and gitignored, so it is written for the
    # next run's cadence check but NOT git-added (adding it would fail with exit 1).
    (REPO / "result" / ".last_hptune_iter").write_text(str(n), encoding="utf-8")

    knobs = "LR/betas/L2/recency C0+EXP"
    if accept:
        summary = (f"AUTO hyperparameter tune (coordinate descent over training "
                   f"hyperparameters ({knobs})): {cs}.")
        comment = (f"Automated tuning pass. Improvement +{imp:.6f} >= threshold "
                   f"{threshold} over {res['n_runs']} runs. Changes: {cs}. "
                   f"logloss_by_user {base_ll:.6f} -> {best_ll:.6f}. Numbers-only, "
                   f"complexity unchanged. Hyperparameter optima drift after "
                   f"structural changes; this recaptures it.")
        append_history({
            "iteration": n, "summary": summary, "threshold": threshold,
            "ll_before": base_ll, "ll_after": best_ll,
            "complexity_before": cx, "complexity_after": cx,
            "status": "accepted", "comment": comment,
        })
        refresh_plot()
        git("add", *EDITED_PATHS,
            "result/diagnostics.json", "result/diagnostics.md",
            "result/history.jsonl", "result/history.md",
            "result/history_plot.png")
        msg = (f"iter {n} accepted: hyperparameter auto-tune ({cs})\n\n"
               f"Automated coordinate-descent pass over training hyperparameters "
               f"(LR, Adam betas, L2 strength, recency weighting).\n"
               f"LL_by_user {base_ll:.5f} -> {best_ll:.5f} (+{imp:.6f} >= {threshold} "
               f"thresh) over {res['n_runs']} runs. Numbers-only, complexity {cx} "
               f"unchanged. New champion.\n\n{TRAILER}")
        git("commit", "-m", msg)
        git("tag", f"iter-{n}-hp-tune")
        print(f"[hp_tune] ACCEPTED as iter {n}: {cs}  (+{imp:.6f})", flush=True)
    else:
        # restore champion (covers the near-miss case where the files changed)
        git("checkout", "HEAD", "--", *EDITED_PATHS,
            "result/diagnostics.json", "result/diagnostics.md")
        summary = (f"AUTO hyperparameter tune (coordinate descent over training "
                   f"hyperparameters ({knobs})): best candidate {cs}.")
        comment = (f"Automated tuning pass over {res['n_runs']} runs found no config "
                   f"clearing threshold {threshold}. Best: {cs}, improvement "
                   f"+{imp:.6f}. logloss_by_user {base_ll:.6f} -> {best_ll:.6f}. "
                   f"Below threshold; reverted to champion (current hyperparameters "
                   f"at/near their optimum).")
        append_history({
            "iteration": n, "summary": summary, "threshold": threshold,
            "ll_before": base_ll, "ll_after": best_ll,
            "complexity_before": cx, "complexity_after": cx,
            "status": "rejected", "comment": comment,
        })
        refresh_plot()
        git("add", "result/history.jsonl", "result/history.md",
            "result/history_plot.png")
        msg = (f"iter {n} rejected: hyperparameter auto-tune (no improvement >= "
               f"{threshold})\n\nAutomated coordinate-descent pass over LR / Adam "
               f"betas / L2 strength / recency weighting. Best: {cs}, "
               f"LL_by_user {base_ll:.5f} -> {best_ll:.5f} (+{imp:.6f} < {threshold} "
               f"thresh). Reverted to champion.\n\n{TRAILER}")
        git("commit", "-m", msg)
        git("tag", f"iter-{n}-hp-tune-rejected")
        print(f"[hp_tune] REJECTED as iter {n}: best {cs} (+{imp:.6f} < {threshold})",
              flush=True)


def main() -> None:
    try:  # Windows consoles default to cp1252; keep our stdout UTF-8-safe
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3,
                    help="max coordinate-descent rounds (default 3)")
    ap.add_argument("--max-runs", type=int, default=34,
                    help="hard cap on benchmark runs per pass (default 34; sized "
                         "for the 6 fine knobs: LR, L2, BETA1/2, RECENCY_C0, "
                         "RECENCY_EXP — batch_size/n_epoch are tuned by "
                         "--epoch-batch-grid, not here)")
    ap.add_argument("--threshold", type=float, default=0.0001,
                    help="acceptance threshold on logloss_by_user (default 1e-4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate constants.py + config.py edits only; no benchmark/git")
    ap.add_argument("--no-commit", action="store_true",
                    help="run the search and leave the best config on disk, "
                         "but do not touch git/history")
    ap.add_argument("--epoch-batch-grid", action="store_true",
                    help="run the 20-combo n_epoch x batch_size Pareto grid (the "
                         "speed-aware gold-standard re-anchor over the compute "
                         "operating point) INSTEAD of the LR/betas/L2 coordinate "
                         "descent; leaves the winning (batch, n_epoch, scaled LR) on "
                         "disk for review, does not commit.")
    ap.add_argument("--time-noise", type=int, nargs="?", const=5, default=None,
                    metavar="N",
                    help="measure the compute_seconds noise floor: run the gold config "
                         "(8,512) N times (default 5, + 1 warm-up) and report the spread "
                         "+ a suggested SPEED_TOL. No commit, no model change.")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    if args.time_noise is not None:
        time_noise(args.time_noise)
        return

    if args.epoch_batch_grid:
        if not tree_clean():
            sys.exit("[hp_tune] ABORT: git working tree is not clean — the grid leaves "
                     "the new operating point on disk for review; commit or stash first.")
        epoch_batch_grid()
        return

    if not args.no_commit and not tree_clean():
        sys.exit("[hp_tune] ABORT: git working tree is not clean — refusing to "
                 "auto-commit on top of uncommitted changes.")

    t0 = time.time()
    try:
        res = search(args.rounds, args.max_runs)
        SUMMARY.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\n[hp_tune] search done in {time.time() - t0:.0f}s, "
              f"{res['n_runs']} runs.", flush=True)
        print(f"[hp_tune] baseline={res['baseline_by_user']:.6f}  "
              f"best={res['best_by_user']:.6f}  "
              f"improvement={res['improvement']:+.6f}", flush=True)
        print(f"[hp_tune] changes: {_changes_str(res['changes'])}", flush=True)

        if args.no_commit:
            print("[hp_tune] --no-commit: best config left on disk; not committing.",
                  flush=True)
            return

        finalize(res, args.threshold)
    except BaseException as e:  # noqa: BLE001 — restore a clean tree on any failure
        git("checkout", "HEAD", "--", *EDITED_PATHS,
            "result/diagnostics.json", "result/diagnostics.md",
            "result/history.jsonl", "result/history.md", check=False)
        print(f"[hp_tune] ERROR ({type(e).__name__}): {e}", flush=True)
        print("[hp_tune] restored tracked files to HEAD before exiting.", flush=True)
        raise


if __name__ == "__main__":
    main()
