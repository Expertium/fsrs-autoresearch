#!/usr/bin/env python3
r"""
central_diff_init_w.py
======================
Unified Adam + central-difference meta-optimizer for the two FSRS-7 "default"
parameter roles, adapted to the *fsrs-autoresearch* repo (custom CUDA/Enzyme
FSRS-7: 36 params, docker benchmark, 3000 users, result/diagnostics.json).

This MERGES the two srs-benchmark-gru scripts
(central_diff_default.py / central_diff_recency.py) into one file that runs BOTH
phases back to back, --default first.  Per the originals, the only per-phase
differences are the benchmark *command* and the *number of (meta) steps* — here:

  PHASE 1  "default"  -> USER-FACING DEFAULT parameters
      FSRS_N_EPOCHS=0  : no per-user SGD; the 36 params are applied directly to
      every user.  Optimizes the static global default that minimizes
      logloss_by_user with zero training.  (step-0 baseline ~0.3458 on the
      iter-54 champion default.)

  PHASE 2  "recency"  -> BEST STARTING POINT for per-user SGD
      FSRS_N_EPOCHS=8  : the normal recency-weighted 8-epoch per-user SGD, with
      these 36 params as the init.  Optimizes the SGD *starting point* that
      minimizes post-training logloss_by_user.  (step-0 baseline ~0.3211, the
      current champion.)

NOTE (2026-06-02): the "split" hypothesis was REJECTED — see the comment above
PHASES.  The recency/SGD-init phase moved log loss only within the GPU-noise
floor, so the user-facing 'default' phase is now the sole live workflow; the
'recency' phase is kept only for reference and is gated off by default.

Mechanics
---------
* Each eval writes the candidate into FSRS7_DEFAULT_35_VALUES in
  src/main/fsrs/fsrs_v7_constants.py, then shells out to
      docker compose --progress quiet run --rm -e FSRS_N_EPOCHS=<e> \
          srs-benchmark bash src/main/run.sh
  and reads result/diagnostics.json -> logloss.by_user (the primary metric, the
  one the autoresearch loop accepts on).
* Gradient via central differences: 2*N evals/step (N=36), then 1 eval at the
  Adam-updated point => 2*N+1 = 73 evals per meta-step.
* Bounds = FSRS_MIN_VALUES / FSRS_MAX_VALUES (read live from the repo so the
  script self-syncs if bounds move).  Ordering constraints mirror FSRS7's clipper
  (src/main/fsrs/fsrs_v7_helpers.apply_parameter_clipper):
      w0 <= w1 <= w2 <= w3      (initial stabilities ordered by rating)
      w28 >= w27                (base2 >= base1)
  The pre-dual-trace decay1<=decay2 chain was DROPPED in iter-54, so it is NOT
  enforced here (independent fast/slow curves).
* CRASH-RESUMABLE: an *atomic* checkpoint JSON is written after every meta-step
  (write-temp + os.replace, so a power loss never leaves a half-written file);
  re-running resumes from it.  constants.py is backed up once and restored on
  EVERY exit (normal, exception, Ctrl-C) — the script never permanently mutates
  the champion.  The tuned parameters live in each phase's checkpoint JSON under
  "best_params"; wiring them into the model (the user-facing/default split) is a
  deliberate, separate, human-reviewed step.

Run from the HOST (needs `docker` + git-bash on PATH; same environment hp_tune.py
uses).  N_EPOCHS is controlled via the FSRS_N_EPOCHS env override added to
src/main/config.py (default stays 8, so the autoresearch loop is unaffected):

    python src/autoresearch/central_diff_init_w.py

Stop any time with Ctrl-C; the champion constants.py is restored and the next run
resumes from the last checkpoint.
"""

import atexit
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
import warnings
from typing import Callable, List, Union

import numpy as np

np.random.seed(42)
warnings.filterwarnings("ignore")

# ============================================================================
# REPO WIRING  (edit REPO_DIR if you move the checkout)
# ============================================================================

REPO_DIR       = pathlib.Path(__file__).resolve().parents[2]
CONSTANTS_PATH = REPO_DIR / "src" / "main" / "fsrs" / "fsrs_v7_constants.py"
DIAG_PATH      = REPO_DIR / "result" / "diagnostics.json"
OUTPUT_DIR     = REPO_DIR / "result" / "init_w_metaopt"
BACKUP_PATH    = OUTPUT_DIR / "fsrs_v7_constants.champion_backup.py"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# META-OPTIMIZER CONFIGURATION
# ============================================================================

# Adam (meta) hyperparameters — identical to the user's two source scripts.
LR    = 5e-3   # Adam learning rate
BETA1 = 0.9    # Adam first-moment decay
BETA2 = 0.999  # Adam second-moment decay
EPS   = 1e-8   # Adam epsilon
H     = 5e-3   # central-difference step size

EVAL_RETRIES = 2   # retry a benchmark eval this many times on a transient failure

# The two phases, run in order.  max_steps is a SOFT CAP: every meta-step is
# checkpointed, so you can Ctrl-C and resume, or just stop when satisfied.
#   default eval  ~8 s (warm) -> ~9 min/step (73 evals) -> 100 steps ~ 15 h
#   recency eval ~100 s        -> ~2 h/step   (73 evals) ->  30 steps ~ 61 h (~2.5 d)
# 3000-user evals are far less noisy than the 30-user mini-runs, so the Adam
# descent is smooth — raise/lower these caps freely (every step is checkpointed).
#
# ---------------------------------------------------------------------------
# 2026-06-02 — "SPLIT" HYPOTHESIS REJECTED.  The original premise was that the
# best *SGD starting point* (the 'recency' phase) should be optimized SEPARATELY
# from the user-facing default to squeeze out extra log loss.  In practice the
# recency phase moved logloss_by_user only within the ~1e-5 GPU-noise floor over
# ~12 meta-steps (essentially flat): post-8-epoch loss is insensitive to the SGD
# starting point, so a separately-tuned init buys nothing.  We therefore abandon
# the split — there is ONE default tuple, optimized by the 'default' (user-facing,
# no-training) phase.  The 'recency' phase is KEPT for reference / future use but
# stays gated off (its RUN_RECENCY sentinel) and is not part of the live workflow.
# ---------------------------------------------------------------------------
PHASES = [
    dict(
        name="default",
        n_epochs=0,
        max_steps=200,   # 2026-06-02: 100 -> 200 (run 100 MORE user-facing-default steps)
        approx_eval_secs=8,
        checkpoint=OUTPUT_DIR / "FSRS_7_central_diff_default_results.json",
        plot=OUTPUT_DIR / "loss_default.png",
        title="USER-FACING DEFAULT  (no per-user SGD, FSRS_N_EPOCHS=0)",
    ),
    dict(
        name="recency",
        n_epochs=8,
        max_steps=30,   # KEPT but gated off (RUN_RECENCY); split hypothesis rejected 2026-06-02 — see note above PHASES
        approx_eval_secs=100,
        checkpoint=OUTPUT_DIR / "FSRS_7_central_diff_recency_results.json",
        plot=OUTPUT_DIR / "loss_recency.png",
        title="BEST SGD STARTING POINT  (8-epoch recency SGD, FSRS_N_EPOCHS=8)",
        ytick=1e-4,  # recency range is tiny (~1e-5); fixed 0.0001 y-ticks keep it readable
    ),
]

# ============================================================================
# READ / WRITE the champion default tuple in constants.py
# ============================================================================

def _load_constants_module():
    """Exec constants.py standalone to read its tuples WITHOUT importing the
    torch-heavy src.main.fsrs package __init__.  constants.py has no imports, so
    a bare exec is safe (see history.py for the same importlib trick)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_fsrs_v7_constants_standalone", CONSTANTS_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replace_default_values(new_values: List[float]) -> None:
    """Rewrite the `FSRS7_DEFAULT_35_VALUES = ( ... )` tuple in constants.py.

    The tuple spans many lines and some have inline comments containing a literal
    ')', so we match `( ... \n)` non-greedily with DOTALL — the only `\n)` in the
    block is the closing paren (comment parens are '<text>)\n', never '\n)')."""
    src = CONSTANTS_PATH.read_text(encoding="utf-8")
    body = (
        "FSRS7_DEFAULT_35_VALUES = (\n"
        + "".join(f"    {float(v)!r},\n" for v in new_values)
        + ")"
    )
    new_src, n = re.subn(
        r"FSRS7_DEFAULT_35_VALUES\s*=\s*\(.*?\n\)",
        lambda _m: body,  # callable => no backslash/group interpretation in body
        src,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError(
            f"replace_default_values: expected 1 match, got {n} in {CONSTANTS_PATH}"
        )
    CONSTANTS_PATH.write_text(new_src, encoding="utf-8")


# ============================================================================
# constants.py backup / restore  (champion is never permanently mutated)
# ============================================================================

def _ensure_backup() -> None:
    if not BACKUP_PATH.exists():
        # First launch: snapshot the current champion constants.py.
        shutil.copy2(CONSTANTS_PATH, BACKUP_PATH)
        print(f"Backed up champion constants -> {BACKUP_PATH.name}")
    else:
        # Resuming (possibly after a crash that left a candidate on disk):
        # restore the pristine champion before doing anything else.
        shutil.copy2(BACKUP_PATH, CONSTANTS_PATH)
        print(f"Resume: restored champion constants from {BACKUP_PATH.name}")


def _restore_constants() -> None:
    if BACKUP_PATH.exists():
        shutil.copy2(BACKUP_PATH, CONSTANTS_PATH)


# ============================================================================
# Evaluate a candidate via the docker benchmark
# ============================================================================

def evaluate(params: List[float], n_epochs: int) -> float:
    """Write params into constants.py, run the benchmark at FSRS_N_EPOCHS=n_epochs,
    return logloss_by_user from result/diagnostics.json."""
    replace_default_values(params)
    before_mtime = DIAG_PATH.stat().st_mtime if DIAG_PATH.exists() else 0.0

    cmd = [
        "docker", "compose", "--progress", "quiet", "run", "--rm",
        "-e", f"FSRS_N_EPOCHS={n_epochs}",
        "srs-benchmark", "bash", "src/main/run.sh",
    ]

    last_err = ""
    for attempt in range(EVAL_RETRIES + 1):
        start = time.perf_counter()
        result = subprocess.run(
            cmd, cwd=str(REPO_DIR), capture_output=True, text=True
        )
        dur = time.perf_counter() - start
        fresh = DIAG_PATH.exists() and DIAG_PATH.stat().st_mtime > before_mtime
        if result.returncode == 0 and fresh:
            diag = json.loads(DIAG_PATH.read_text(encoding="utf-8"))
            return float(diag["logloss"]["by_user"])
        last_err = (result.stderr or result.stdout or "")[-2000:]
        print(
            f"      ! eval failed (attempt {attempt + 1}/{EVAL_RETRIES + 1}, "
            f"rc={result.returncode}, fresh={fresh}, {dur:.0f}s); retrying in 5s"
        )
        time.sleep(5)

    raise RuntimeError(
        f"Benchmark failed after {EVAL_RETRIES + 1} attempts (epochs={n_epochs}).\n"
        f"Last stderr/stdout tail:\n{last_err}"
    )


# ============================================================================
# Bounds + ordering constraints (mirror FSRS7ParameterClipper)
# ============================================================================

def clip_params(params, bounds):
    clipped = np.copy(params)
    for i, (lo, hi) in enumerate(bounds):
        clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


def check_constraints(params):
    """Ordering constraints enforced by apply_parameter_clipper:
        w0 <= w1 <= w2 <= w3   (initial stabilities)
        w28 >= w27             (base2 >= base1)
    (decay1<=decay2 was dropped in iter-54.)"""
    return (
        params[0] <= params[1] <= params[2] <= params[3]
        and params[28] >= params[27]
    )


# ============================================================================
# Adam optimizer with central-difference gradients  (faithful to the originals)
# ============================================================================

class AdamCentralDiff:
    """Adam, gradients estimated by central differences (2*n evals / gradient)."""

    def __init__(
        self,
        params: Union[List[float], np.ndarray],
        bounds: List[tuple],
        constraints_fn: Callable[[np.ndarray], bool],
        lr: float = 5e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        h: float = 5e-3,
    ):
        self.n = len(params)
        self.params = np.array(params, dtype=float)
        self.bounds = bounds
        self.constraints_fn = constraints_fn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.h = h

        self.m = np.zeros(self.n)   # first moment
        self.v = np.zeros(self.n)   # second moment
        self.t = 0                  # step counter (bias correction)
        self.counteval = 0          # total function evaluations

    def compute_gradient(self, eval_fn: Callable) -> "tuple[np.ndarray, list]":
        """Central-difference gradient at self.params. Both perturbed points are
        clipped to bounds first; the denominator uses the realized step so the
        estimate stays correct at saturated bounds."""
        grad = np.zeros(self.n)
        eval_log = []

        for i in range(self.n):
            p_plus = self.params.copy()
            p_minus = self.params.copy()
            p_plus[i] = np.clip(self.params[i] + self.h, self.bounds[i][0], self.bounds[i][1])
            p_minus[i] = np.clip(self.params[i] - self.h, self.bounds[i][0], self.bounds[i][1])

            f_plus = eval_fn(p_plus.tolist())
            f_minus = eval_fn(p_minus.tolist())
            self.counteval += 2

            denom = p_plus[i] - p_minus[i]
            grad[i] = (f_plus - f_minus) / denom if denom != 0.0 else 0.0

            eval_log.append({
                "param_index": i,
                "params_plus": p_plus.tolist(),
                "loss_plus": f_plus,
                "params_minus": p_minus.tolist(),
                "loss_minus": f_minus,
            })
            print(f"    dim {i:2d}  f+={f_plus:.6f}  f-={f_minus:.6f}  grad={grad[i]:+.6f}")

        return grad, eval_log

    def step(self, grad: np.ndarray) -> np.ndarray:
        """One Adam update, clipped to bounds, then ordering constraints repaired
        exactly as FSRS7ParameterClipper would (raise w1..w3, raise base2)."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        new_params = self.params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        new_params = clip_params(new_params, self.bounds)

        if not self.constraints_fn(new_params):
            new_params[0:4] = np.maximum.accumulate(new_params[0:4])   # w0<=w1<=w2<=w3
            new_params[28] = max(new_params[28], new_params[27])       # base2 >= base1
            new_params = clip_params(new_params, self.bounds)

        self.params = new_params
        return self.params


# ============================================================================
# Checkpoint IO (atomic) + plotting
# ============================================================================

def _atomic_write_json(path: pathlib.Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic on the same filesystem


def _save_plot(history, plot_path, title, ytick=None):
    """Save a loss-vs-step PNG. Matplotlib is imported lazily and failures are
    non-fatal, so a host without matplotlib still runs the optimization. If
    ``ytick`` is given, the y-axis uses that fixed tick spacing with absolute
    (non-offset) labels, so the scale is readable no matter how small the data
    range is (e.g. the recency phase, where the whole range is ~1e-5)."""
    if not history:
        return
    try:
        import math
        import matplotlib
        matplotlib.use("Agg")  # headless: never open a window during a long run
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except Exception as e:
        print(f"      (plot skipped: matplotlib unavailable: {e})")
        return
    steps = [e["step"] for e in history]
    losses = [e["loss"] for e in history]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, marker="o")
    ax.set_xlabel("meta-step")
    ax.set_ylabel("logloss_by_user")
    ax.set_title(title)
    ax.grid(True)
    if ytick:
        lo, hi = min(losses), max(losses)
        y0 = math.floor(lo / ytick) * ytick
        y1 = math.ceil(hi / ytick) * ytick
        if y1 - y0 < ytick:  # always span at least one full increment
            y1 = y0 + ytick
        ax.set_ylim(y0, y1)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


# ============================================================================
# Run one phase
# ============================================================================

def _fresh_start(optimizer, eval_fn, starting_params):
    history = []
    best_params = np.array(starting_params, dtype=float).copy()
    best_loss = eval_fn(best_params.tolist())
    print(f"  starting-point loss: {best_loss:.6f}")
    return history, best_params, best_loss, 0


def run_phase(phase: dict, starting_params: np.ndarray, bounds: List[tuple]):
    n_epochs = phase["n_epochs"]
    max_steps = phase["max_steps"]
    ckpt = phase["checkpoint"]
    eval_fn = lambda p: evaluate(p, n_epochs)

    print("\n" + "=" * 80)
    print(f"PHASE '{phase['name']}'  —  {phase['title']}")
    print("=" * 80)
    print(f"Adam lr={LR} beta1={BETA1} beta2={BETA2} eps={EPS} h={H}; "
          f"max_steps={max_steps}; evals/step={2 * len(starting_params) + 1}")

    optimizer = AdamCentralDiff(
        starting_params, bounds, check_constraints,
        lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, h=H,
    )

    if ckpt.is_file():
        try:
            with open(ckpt) as f:
                cp = json.load(f)
            optimizer.params    = np.array(cp["params"], dtype=float)
            optimizer.m         = np.array(cp["m"], dtype=float)
            optimizer.v         = np.array(cp["v"], dtype=float)
            optimizer.t         = int(cp["t"])
            optimizer.counteval = int(cp["counteval"])
            history         = cp["history"]
            best_params     = np.array(cp["best_params"], dtype=float)
            best_loss       = float(cp["best_loss"])
            completed_steps = int(cp["completed_steps"])
            print(f"Resumed from {ckpt.name}: step {completed_steps}, "
                  f"best_loss {best_loss:.6f}")
        except Exception as e:
            print(f"Could not load checkpoint ({e}); starting fresh")
            history, best_params, best_loss, completed_steps = _fresh_start(
                optimizer, eval_fn, starting_params)
    else:
        print(f"No checkpoint at {ckpt.name}; starting fresh")
        history, best_params, best_loss, completed_steps = _fresh_start(
            optimizer, eval_fn, starting_params)

    for step in range(completed_steps, max_steps):
        print(f"\n{'-' * 80}\n[{phase['name']}] Step {step + 1}/{max_steps}  "
              f"(best so far {best_loss:.6f})")
        print(f"Computing gradient via central differences "
              f"({2 * optimizer.n} evals)...")

        grad, eval_log = optimizer.compute_gradient(eval_fn)
        print(f"Gradient norm: {np.linalg.norm(grad):.6f}")

        new_params = optimizer.step(grad)
        loss = eval_fn(new_params.tolist())
        optimizer.counteval += 1

        if loss < best_loss:
            best_loss = loss
            best_params = new_params.copy()
            print(f"  ** new best: {best_loss:.6f}")

        print(f"[{phase['name']}] step {step + 1}: loss={loss:.6f}  "
              f"best={best_loss:.6f}  total_evals={optimizer.counteval}")

        history.append({
            "step": step + 1,
            "params": new_params.tolist(),
            "loss": loss,
            "grad_norm": float(np.linalg.norm(grad)),
            "eval_log": eval_log,
        })
        _atomic_write_json(ckpt, {
            "params":          optimizer.params.tolist(),
            "m":               optimizer.m.tolist(),
            "v":               optimizer.v.tolist(),
            "t":               int(optimizer.t),
            "counteval":       int(optimizer.counteval),
            "best_params":     best_params.tolist(),
            "best_loss":       float(best_loss),
            "history":         history,
            "completed_steps": step + 1,
            "phase":           phase["name"],
            "n_epochs":        n_epochs,
        })
        _save_plot(history, phase["plot"], f"{phase['name']}: logloss_by_user", ytick=phase.get("ytick"))

    print(f"\n[{phase['name']}] complete. best_loss={best_loss:.6f}")
    print(f"[{phase['name']}] best_params={[round(v, 5) for v in best_params.tolist()]}")
    return best_params, best_loss


# ============================================================================
# Main: default phase first, then recency phase
# ============================================================================

def main():
    _ensure_backup()
    atexit.register(_restore_constants)
    try:
        mod = _load_constants_module()
        start_default = np.array([float(x) for x in mod.FSRS7_DEFAULT_35_VALUES], dtype=float)
        lo = [float(x) for x in mod.FSRS_MIN_VALUES]
        hi = [float(x) for x in mod.FSRS_MAX_VALUES]
        bounds = list(zip(lo, hi))
        n = len(start_default)
        assert len(bounds) == n, f"bounds {len(bounds)} != params {n}"

        # The champion default must sit inside the box (sanity).
        for i, (p, (blo, bhi)) in enumerate(zip(start_default, bounds)):
            assert blo <= p <= bhi, f"param {i}={p} outside ({blo}, {bhi})"

        print("=" * 80)
        print("Adam + central-difference meta-optimization of FSRS-7 defaults")
        print(f"repo={REPO_DIR}  params={n}")
        print(f"champion default: {[round(x, 4) for x in start_default.tolist()]}")
        total_h = 0.0
        for ph in PHASES:
            est = ph["max_steps"] * (2 * n + 1) * ph["approx_eval_secs"] / 3600.0
            total_h += est
            print(f"  phase '{ph['name']}': <= {ph['max_steps']} steps, "
                  f"~{ph['approx_eval_secs']}s/eval  ->  ~{est:.0f} h (cap)")
        print(f"  TOTAL upper bound ~{total_h:.0f} h (both phases at full max_steps; "
              f"lower max_steps for a shorter run — every step is checkpointed).")
        print("=" * 80)

        results = {}
        for phase in PHASES:
            # Recency is GATED behind a sentinel so the run SELF-PAUSES after the
            # default phase — we do the step-2 training-recipe work (N_EPOCHS=10,
            # single LR, recency/batch hp-tune) before tuning the SGD init. To run
            # recency later: create result/init_w_metaopt/RUN_RECENCY and re-launch.
            if phase["name"] == "recency" and not (OUTPUT_DIR / "RUN_RECENCY").exists():
                print(f"\n[recency] GATED — default-only run. Create "
                      f"{OUTPUT_DIR / 'RUN_RECENCY'} and re-launch to enable recency.")
                continue
            best_params, best_loss = run_phase(phase, start_default.copy(), bounds)
            results[phase["name"]] = {"best_loss": best_loss, "best_params": best_params.tolist()}
            _atomic_write_json(OUTPUT_DIR / "summary.json", results)
            print(f"\n==== PHASE '{phase['name']}' DONE: best_loss={best_loss:.6f} ====")

        print("\n" + "=" * 80)
        print("ALL PHASES COMPLETE")
        for name, r in results.items():
            print(f"  {name:8s} best_loss={r['best_loss']:.6f}")
        print(f"  summary -> {OUTPUT_DIR / 'summary.json'}")
        print("=" * 80)

        # Clean completion: drop the backup so a future launch re-baselines off
        # the then-current champion.  (On crash/interrupt the backup is kept, so
        # the next launch restores the champion and resumes.)
        _restore_constants()
        if BACKUP_PATH.exists():
            BACKUP_PATH.unlink()
    finally:
        _restore_constants()


if __name__ == "__main__":
    main()
