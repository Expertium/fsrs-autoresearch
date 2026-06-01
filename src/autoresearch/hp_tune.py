#!/usr/bin/env python3
"""
Automated hyperparameter tuner for the FSRS-7 autoresearch loop.

The interesting part of the loop is the model-structure search; nudging numeric
*training* hyperparameters (learning rate, Adam betas, L2 strength, recency
weighting, and the training batch size) up and down is mechanical and boring.
This automates it.

What it does
------------
A greedy **coordinate-descent** local search. For each hyperparameter it tries a
step up and a step down (multiplicative for LR/L2, on the ``1 - beta`` scale for
the Adam betas, and a discrete x2 / /2 snapped to a multiple of 128 for the batch
size), keeps whichever lowers ``logloss_by_user``, and re-probes the improving
ones in later rounds. Params that don't improve are frozen ("light" mode →
typically ~14-20 benchmark runs/pass). Training is deterministic (``seed`` fixed),
so every delta is real and reproducible — no averaging needed.

Most knobs live in ``fsrs_v7_constants.py``; **BATCH_SIZE lives in
``src/main/config.py``**, so this tuner edits *two* files (see ``read_texts`` /
``HParam.file``). Each trial edits a numeric literal *in place*. That never
changes the AST node count, so the complexity score is unaffected — which is
exactly why a hyperparameter tweak only has to clear the ``+0.0001`` floor
threshold, with no complexity gate to worry about. (Changing BATCH_SIZE rebuilds
only the cheap batch-size-dependent cache arrays — ``num_training_steps`` and the
batch permutation — via their own cache manifests; the expensive tensor cache is
reused. Verified to take effect, not silently masked.)

Fully autonomous: when the best config found beats the champion by ≥ threshold
it commits the new champion (constants + config + diagnostics + history) and tags
it; otherwise it restores the champion and records a rejected pass. It owns one
iteration number per pass (read from ``history.jsonl``).

Run it from the **host** (it shells out to ``docker compose`` for each
benchmark and to ``git`` for commits). A pass is many ~70-150 s runs, so launch
it in the background:

    python -m src.autoresearch.hp_tune              # full auto pass
    python src/autoresearch/hp_tune.py --dry-run    # validate file edits only
    python src/autoresearch/hp_tune.py --no-commit  # search, leave best, no git
"""
from __future__ import annotations

import argparse
import ast
import json
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

# The files this tuner may edit, keyed by the short name an HParam references in
# its `file` field. Both are validated with ast.parse before any benchmark runs.
FILES = {"constants": CONSTANTS, "config": CONFIG}

# Tracked paths touched on accept (commit) / reject (revert). config.py is here
# because the BATCH_SIZE knob edits it; including it when it's unchanged is a
# harmless no-op for both `git add` and `git checkout`.
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
    """Compact literal without float-repr noise (0.0300000000002 -> 0.03,
    1024.0 -> 1024)."""
    return f"{float(v):.6g}"


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


# ── hyperparameter spec ──────────────────────────────────────────────────────
@dataclass
class HParam:
    name: str
    get: Callable[[dict[str, str]], float]
    put: Callable[[dict[str, str], float], dict[str, str]]
    kind: str        # "mul" perturbs v; "beta" perturbs (1 - v); "batch" is discrete
    step: float
    lo: float
    hi: float
    active: bool = True
    file: str = "constants"  # which FILES entry this knob's ast.parse check targets

    def candidates(self, v: float) -> list[float]:
        if self.kind == "mul":
            raw = [v * self.step, v / self.step]
        elif self.kind == "beta":
            raw = [1.0 - (1.0 - v) * self.step, 1.0 - (1.0 - v) / self.step]
        elif self.kind == "batch":
            # Discrete: multiplicative step, snapped to a multiple of 128 (the CUDA
            # kernel block size) and kept integer-valued. The lo/hi bounds keep it
            # in the OOM-safe [512, 2048] range; from 1024 the neighbours are 512
            # and 2048 (both already multiples of 128).
            raw = [round(v * self.step / 128.0) * 128.0,
                   round(v / self.step / 128.0) * 128.0]
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
    # BATCH_SIZE lives in config.py (not constants.py) and is discrete. Kept LAST so
    # that within a coordinate-descent round all the constants.py knobs run first at
    # a fixed batch size (no cache rebuild between them); only the batch_size trials
    # trigger a (cheap) num_training_steps / batch_perm rebuild.
    hps.append(HParam("BATCH_SIZE",
                      lambda ts: get_scalar(ts["config"], "BATCH_SIZE"),
                      lambda ts, v: {**ts, "config": set_scalar(ts["config"], "BATCH_SIZE", v)},
                      "batch", 2.0, 512, 2048, file="config"))
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

    # cadence marker (committed below so the tree stays clean for the next run)
    (REPO / "result" / ".last_hptune_iter").write_text(str(n), encoding="utf-8")

    knobs = "LR/betas/L2/recency C0+EXP/batch_size"
    if accept:
        summary = (f"AUTO hyperparameter tune (coordinate descent over training "
                   f"hyperparameters ({knobs})): {cs}. Numbers-only, complexity unchanged.")
        reason = (f"Automated tuning pass. Improvement +{imp:.6f} >= threshold "
                  f"{threshold} over {res['n_runs']} runs. Changes: {cs}. "
                  f"logloss_by_user {base_ll:.6f} -> {best_ll:.6f}. Hyperparameter "
                  f"optima drift after structural changes; this recaptures it.")
        append_history({
            "iteration": n, "summary": summary, "threshold": threshold,
            "ll_before": base_ll, "ll_after": best_ll,
            "complexity_before": cx, "complexity_after": cx,
            "status": "accepted", "reason": reason,
        })
        git("add", *EDITED_PATHS,
            "result/diagnostics.json", "result/diagnostics.md",
            "result/history.jsonl", "result/history.md",
            "result/.last_hptune_iter")
        msg = (f"iter {n} accepted: hyperparameter auto-tune ({cs})\n\n"
               f"Automated coordinate-descent pass over training hyperparameters "
               f"(LR, Adam betas, L2 strength, recency weighting, batch size).\n"
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
                   f"hyperparameters ({knobs})): best found {cs}, below threshold.")
        reason = (f"Automated tuning pass over {res['n_runs']} runs found no config "
                  f"clearing threshold {threshold}. Best: {cs}, improvement "
                  f"+{imp:.6f}. logloss_by_user {base_ll:.6f} -> {best_ll:.6f}. "
                  f"Reverted to champion; the current hyperparameters are at/near "
                  f"their optimum.")
        append_history({
            "iteration": n, "summary": summary, "threshold": threshold,
            "ll_before": base_ll, "ll_after": best_ll,
            "complexity_before": cx, "complexity_after": cx,
            "status": "rejected", "reason": reason,
        })
        git("add", "result/history.jsonl", "result/history.md",
            "result/.last_hptune_iter")
        msg = (f"iter {n} rejected: hyperparameter auto-tune (no improvement >= "
               f"{threshold})\n\nAutomated coordinate-descent pass over LR / Adam "
               f"betas / L2 strength / recency weighting / batch size. Best: {cs}, "
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
                         "for the 7-knob set: LR, L2, BETA1/2, RECENCY_C0, "
                         "RECENCY_EXP, BATCH_SIZE)")
    ap.add_argument("--threshold", type=float, default=0.0001,
                    help="acceptance threshold on logloss_by_user (default 1e-4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate constants.py + config.py edits only; no benchmark/git")
    ap.add_argument("--no-commit", action="store_true",
                    help="run the search and leave the best config on disk, "
                         "but do not touch git/history")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
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
