#!/usr/bin/env python3
r"""
sigma_group_tune.py
===================
Central-difference Adam meta-optimizer for the L2-penalty SIGMA *shape*, by
GROUP. The L2 penalty (fsrs_v7_helpers.penalty_loss) shrinks each user's params
toward the fixed FSRS7_DEFAULT, with per-param strength set by
FSRS7_L2_SIGMA_35_VALUES (smaller sigma = stronger shrinkage). hp_tune already
optimizes the GLOBAL L2 scale (PENALTY_W_L2); this tunes the orthogonal per-GROUP
*shape* — 5 multipliers, one per param group, each scaling that group's base
sigmas:

    difficulty   w[4..6]
    long-stab    w[7..14]
    short-stab   w[15..22]
    curve        w[23..30]
    modulation   w[31..33]

(init-S w[0..3] is excluded: its sigma is 9999 = effectively no L2, and a
multiplier can't introduce regularization there.)

Unlike the default meta-opt (central_diff_init_w.py), sigma ONLY matters under
training, so this is an 8-epoch (FSRS_N_EPOCHS=8) optimization of the REAL
trained logloss_by_user — no 0-epoch proxy that can diverge (cf. iter-174). Each
eval is a full ~130 s run (the batch evaluator only helps the 0-epoch phase), so
2*5+1 = 11 evals/step ~= 24 min/step.

Mechanics mirror central_diff_init_w (reuses its AdamCentralDiff + checkpoint +
plot): each eval writes the SCALED sigma tuple into FSRS7_L2_SIGMA_35_VALUES,
shells out to the 8-epoch benchmark, reads logloss.by_user. constants.py is
backed up once and restored on EVERY exit (champion never permanently mutated);
the tuned multipliers live in the checkpoint JSON. Wiring the result in is a
deliberate, human-reviewed step (round sigmas to <=4dp, re-measure, accept iff
>= 0.0001 over the champion).

Run from the HOST:  python -m src.autoresearch.sigma_group_tune
Stop any time with Ctrl-C; constants.py is restored and the next run resumes.
"""
from __future__ import annotations

import atexit
import json
import re
import shutil
import subprocess
import time

import numpy as np

# Reuse the proven Adam/central-diff core + IO from the default meta-opt.
from src.autoresearch.central_diff_init_w import (
    AdamCentralDiff,
    REPO_DIR,
    CONSTANTS_PATH,
    DIAG_PATH,
    OUTPUT_DIR,
    EVAL_RETRIES,
    _atomic_write_json,
    _save_plot,
    _load_constants_module,
)

np.random.seed(42)

# ── group definitions (param-index ranges per multiplier) ───────────────────
GROUPS = [
    ("difficulty", list(range(4, 7))),
    ("long_stab", list(range(7, 15))),
    ("short_stab", list(range(15, 23))),
    ("curve", list(range(23, 31))),
    ("modulation", list(range(31, 34))),
]
N_GROUPS = len(GROUPS)
MULT_BOUNDS = [(0.25, 4.0)] * N_GROUPS  # scale each group's sigma 0.25x .. 4x

# meta-Adam HPs (multipliers are O(1); larger h/LR than the param meta-opt).
LR, BETA1, BETA2, EPS, H = 0.03, 0.9, 0.999, 1e-8, 0.05
MAX_STEPS = 25  # de-risk probe: enough to tell whether sigma-shape has potential
N_EPOCHS = 8

CKPT = OUTPUT_DIR / "sigma_group_results.json"
PLOT = OUTPUT_DIR / "sigma_group_loss.png"
BACKUP = OUTPUT_DIR / "fsrs_v7_constants.sigma_backup.py"


def replace_sigma_values(new_values: list[float]) -> None:
    """Rewrite the FSRS7_L2_SIGMA_35_VALUES = ( ... ) tuple in constants.py."""
    src = CONSTANTS_PATH.read_text(encoding="utf-8")
    body = (
        "FSRS7_L2_SIGMA_35_VALUES = (\n"
        + "".join(f"    {float(v)!r},\n" for v in new_values)
        + ")"
    )
    new_src, n = re.subn(
        r"FSRS7_L2_SIGMA_35_VALUES\s*=\s*\(.*?\n\)",
        lambda _m: body,
        src,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError(f"replace_sigma_values: expected 1 match, got {n}")
    CONSTANTS_PATH.write_text(new_src, encoding="utf-8")


def scaled_sigma(base_sigma: np.ndarray, multipliers) -> list[float]:
    """Apply per-group multipliers to the base sigma vector."""
    sig = base_sigma.copy()
    for (_, idxs), m in zip(GROUPS, multipliers):
        for i in idxs:
            sig[i] = base_sigma[i] * float(m)
    return sig.tolist()


def _ensure_backup() -> None:
    if not BACKUP.exists():
        shutil.copy2(CONSTANTS_PATH, BACKUP)
        print(f"Backed up champion constants -> {BACKUP.name}")
    else:
        shutil.copy2(BACKUP, CONSTANTS_PATH)
        print(f"Resume: restored champion constants from {BACKUP.name}")


def _restore() -> None:
    if BACKUP.exists():
        shutil.copy2(BACKUP, CONSTANTS_PATH)


def make_eval(base_sigma: np.ndarray):
    def evaluate(multipliers) -> float:
        replace_sigma_values(scaled_sigma(base_sigma, multipliers))
        before = DIAG_PATH.stat().st_mtime if DIAG_PATH.exists() else 0.0
        cmd = [
            "docker", "compose", "--progress", "quiet", "run", "--rm",
            "-e", f"FSRS_N_EPOCHS={N_EPOCHS}",
            "srs-benchmark", "bash", "src/main/run.sh",
        ]
        last_err = ""
        for attempt in range(EVAL_RETRIES + 1):
            t = time.perf_counter()
            r = subprocess.run(cmd, cwd=str(REPO_DIR), capture_output=True, text=True)
            dt = time.perf_counter() - t
            fresh = DIAG_PATH.exists() and DIAG_PATH.stat().st_mtime > before
            if r.returncode == 0 and fresh:
                return float(json.loads(DIAG_PATH.read_text())["logloss"]["by_user"])
            last_err = (r.stderr or r.stdout or "")[-1500:]
            print(f"      ! eval failed (try {attempt+1}, rc={r.returncode}, {dt:.0f}s); retry 5s")
            time.sleep(5)
        raise RuntimeError(f"benchmark failed after {EVAL_RETRIES+1} tries:\n{last_err}")
    return evaluate


def main() -> None:
    _ensure_backup()
    atexit.register(_restore)
    try:
        mod = _load_constants_module()
        base_sigma = np.array([float(x) for x in mod.FSRS7_L2_SIGMA_35_VALUES], dtype=float)
        eval_fn = make_eval(base_sigma)

        opt = AdamCentralDiff(
            [1.0] * N_GROUPS, MULT_BOUNDS, lambda p: True,
            lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, h=H,
        )
        history, best_mult, best_loss, done = [], np.ones(N_GROUPS), None, 0
        if CKPT.is_file():
            cp = json.load(open(CKPT))
            opt.params = np.array(cp["params"]); opt.m = np.array(cp["m"]); opt.v = np.array(cp["v"])
            opt.t = int(cp["t"]); opt.counteval = int(cp["counteval"])
            history = cp["history"]; best_mult = np.array(cp["best_params"])
            best_loss = float(cp["best_loss"]); done = int(cp["completed_steps"])
            print(f"Resumed step {done}, best_loss {best_loss:.6f}, mult {best_mult.tolist()}")
        else:
            best_loss = eval_fn([1.0] * N_GROUPS)  # multipliers=1 -> champion sigma
            best_mult = np.ones(N_GROUPS)
            print(f"baseline (mult=1) by_user = {best_loss:.6f}  [should match champion]")

        print("groups:", [g for g, _ in GROUPS], "| LR", LR, "h", H, "max_steps", MAX_STEPS)
        for step in range(done, MAX_STEPS):
            print(f"\n[sigma] step {step+1}/{MAX_STEPS} (best {best_loss:.6f}, mult {[round(x,3) for x in opt.params.tolist()]})")
            grad, _ = opt.compute_gradient(eval_fn)  # 2*N_GROUPS evals
            print(f"  grad {np.round(grad, 6).tolist()}  |grad| {np.linalg.norm(grad):.6f}")
            new = opt.step(grad)
            loss = eval_fn(new.tolist()); opt.counteval += 1
            if loss < best_loss:
                best_loss, best_mult = loss, new.copy()
                print(f"  ** new best {best_loss:.6f} @ mult {[round(x,3) for x in best_mult.tolist()]}")
            print(f"[sigma] step {step+1}: loss={loss:.6f} best={best_loss:.6f} evals={opt.counteval}")
            history.append({"step": step + 1, "mult": new.tolist(), "loss": loss,
                            "grad_norm": float(np.linalg.norm(grad))})
            _atomic_write_json(CKPT, {
                "params": opt.params.tolist(), "m": opt.m.tolist(), "v": opt.v.tolist(),
                "t": int(opt.t), "counteval": int(opt.counteval),
                "best_params": best_mult.tolist(), "best_loss": float(best_loss),
                "history": history, "completed_steps": step + 1, "groups": [g for g, _ in GROUPS],
            })
            _save_plot(history, PLOT, "sigma-group tune: logloss_by_user", ytick=0.0001)

        print(f"\n[sigma] DONE. best_loss={best_loss:.6f}  best_mult={best_mult.tolist()}")
        _restore()
        if BACKUP.exists():
            BACKUP.unlink()
    finally:
        _restore()


if __name__ == "__main__":
    main()
