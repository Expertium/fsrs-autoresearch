#!/usr/bin/env python3
r"""
joint_default_sigma_tune.py
===========================
Alternating, GATED co-optimization of the FSRS-7 user-facing DEFAULT parameters
(FSRS7_DEFAULT_35_VALUES — also the per-user init AND the L2 anchor) and the
per-parameter L2 PENALTY SIGMAS (FSRS7_L2_SIGMA_35_VALUES).

The two are coupled: the L2 anchor *is* the default vector, so the optimal sigma
*shape* depends on the defaults and vice-versa. Tuning them independently
(central_diff_init_w for defaults, sigma_group_tune for sigmas) misses that
coupling. This alternates one step of each, per CYCLE (MAX_CYCLES=25):

  1. DEFAULT step  — one Adam central-difference step on all 34 defaults against
     the cheap 0-epoch `--default` loss (lr 5e-3, batch-evaluated ~15x). That
     0-epoch loss is only a PROXY for the trained metric — it diverges from the
     8-epoch loss past ~25 steps (iter-174's inverted-U). So the step is
     *proposed* by the proxy but...
  2. RECENCY eval  — ...*gated* by the REAL 8-epoch `--recency` logloss_by_user:
     run the full benchmark with the stepped defaults + current sigmas.
  3. GATE          — commit the stepped defaults ONLY if they beat the best
     `--recency` loss seen so far; else revert. So the proxy can never drag the
     defaults past their true trained-metric peak. Once peaked, default steps
     reject and the defaults FREEZE, while sigmas keep moving — exactly the
     intended "tune L2 only on the best defaults" behavior.
  4. SIGMA step    — one Adam central-difference step on the 30 per-parameter
     sigmas (w[4..33]; S0 w[0..3] EXCLUDED — its sigma is 9999 = no L2, so a
     multiplier is meaningless there) against the 8-epoch `--recency` trained
     loss (lr 1e-2), using the committed defaults. The metric IS --recency, so
     sigmas follow the Adam trajectory; the global best is tracked.

Sigmas only matter under training, so step 4 needs full 8-epoch runs (~130 s
each, NOT batchable — each L2 config needs its own SGD trajectory). Per cycle:
2*30 sigma-gradient evals + 1 sigma-step eval + 1 default-gate eval (all 8-ep)
+ 1 batched 0-ep default-gradient ~= 2.3 h  =>  ~2.4 days for 25 cycles.

constants.py is backed up once and restored on EVERY exit; the champion is never
permanently mutated. The tuned (defaults, sigmas) live in the checkpoint under
best_defaults / best_sigma_mult. Wiring them in is a deliberate human step (apply
the multipliers to the base sigma, round to <=4 dp, re-measure, accept iff
>= 0.0001 over the champion).

Run from the HOST:  python -m src.autoresearch.joint_default_sigma_tune
Crash-resumable; Ctrl-C restores the champion and the next run resumes.
"""
from __future__ import annotations

import atexit
import json
import subprocess
import time

import numpy as np

from src.autoresearch.central_diff_init_w import (
    AdamCentralDiff,
    REPO_DIR,
    DIAG_PATH,
    OUTPUT_DIR,
    EVAL_RETRIES,
    evaluate_batch,
    replace_default_values,
    check_constraints,
    _atomic_write_json,
    _save_plot,
    _load_constants_module,
    _ensure_backup,
    _restore_constants,
)
from src.autoresearch.sigma_group_tune import replace_sigma_values

np.random.seed(42)

# ── config ──────────────────────────────────────────────────────────────────
MAX_CYCLES = 25
N_EPOCHS = 8

LR_DEFAULT, H_DEFAULT = 5e-3, 5e-3     # 0-epoch --default proxy step (gated)
LR_SIGMA, H_SIGMA = 1e-2, 0.05         # 8-epoch --recency sigma step
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

SIG_IDX = list(range(4, 34))           # 30 tunable sigmas (S0 w[0..3] excluded: sigma=9999)
N_SIG = len(SIG_IDX)
SIG_BOUNDS = [(0.25, 4.0)] * N_SIG     # per-param sigma-multiplier range

CKPT = OUTPUT_DIR / "joint_default_sigma_results.json"
PLOT = OUTPUT_DIR / "joint_default_sigma_loss.png"


def _bench(n_epochs: int) -> float:
    """Run the full benchmark at FSRS_N_EPOCHS=n_epochs (constants.py already
    written by the caller) and return logloss_by_user. Retries on transient
    failure (mirrors central_diff_init_w.evaluate's retry loop)."""
    before = DIAG_PATH.stat().st_mtime if DIAG_PATH.exists() else 0.0
    cmd = [
        "docker", "compose", "--progress", "quiet", "run", "--rm",
        "-e", f"FSRS_N_EPOCHS={n_epochs}",
        "srs-benchmark", "bash", "src/main/run.sh",
    ]
    last = ""
    for attempt in range(EVAL_RETRIES + 1):
        t = time.perf_counter()
        r = subprocess.run(cmd, cwd=str(REPO_DIR), capture_output=True, text=True)
        dt = time.perf_counter() - t
        fresh = DIAG_PATH.exists() and DIAG_PATH.stat().st_mtime > before
        if r.returncode == 0 and fresh:
            return float(json.loads(DIAG_PATH.read_text())["logloss"]["by_user"])
        last = (r.stderr or r.stdout or "")[-1500:]
        print(f"      ! bench failed (try {attempt+1}, rc={r.returncode}, {dt:.0f}s); retry 5s")
        time.sleep(5)
    raise RuntimeError(f"benchmark failed after {EVAL_RETRIES+1} tries:\n{last}")


def main() -> None:
    _ensure_backup()
    atexit.register(_restore_constants)
    try:
        mod = _load_constants_module()
        champ_def = np.array([float(x) for x in mod.FSRS7_DEFAULT_35_VALUES], dtype=float)
        base_sigma = np.array([float(x) for x in mod.FSRS7_L2_SIGMA_35_VALUES], dtype=float)
        lo = [float(x) for x in mod.FSRS_MIN_VALUES]
        hi = [float(x) for x in mod.FSRS_MAX_VALUES]
        def_bounds = list(zip(lo, hi))
        n_def = len(champ_def)
        assert len(base_sigma) == n_def, f"sigma {len(base_sigma)} != defaults {n_def}"

        # Mutable committed state shared with the eval closures.
        state = {"defaults": champ_def.copy(), "sig_mult": np.ones(n_def)}

        def write_constants(defaults, sig_mult) -> None:
            """Write BOTH tuples into constants.py: defaults verbatim, sigmas as
            base_sigma * per-param multiplier (S0 mult=1 keeps sigma=9999)."""
            replace_default_values([float(x) for x in defaults])
            replace_sigma_values((base_sigma * sig_mult).tolist())

        # --- eval closures ---------------------------------------------------
        def def_grad_eval(p):                       # single 0-ep --default eval (proxy)
            return evaluate_batch([p], 0)[0]

        def def_grad_batch(plist):                  # batched 0-ep --default evals
            return evaluate_batch(plist, 0)

        def sigma_eval(mult30):                     # 8-ep --recency on the committed defaults
            m = np.ones(n_def)
            m[SIG_IDX] = np.array(mult30, dtype=float)
            write_constants(state["defaults"], m)
            return _bench(N_EPOCHS)

        # --- optimizers ------------------------------------------------------
        opt_def = AdamCentralDiff(
            state["defaults"].tolist(), def_bounds, check_constraints,
            lr=LR_DEFAULT, beta1=BETA1, beta2=BETA2, eps=EPS, h=H_DEFAULT,
        )
        opt_sig = AdamCentralDiff(
            [1.0] * N_SIG, SIG_BOUNDS, lambda p: True,   # no ordering constraint on sigmas
            lr=LR_SIGMA, beta1=BETA1, beta2=BETA2, eps=EPS, h=H_SIGMA,
        )

        history: list = []
        best_loss = None
        best_def = state["defaults"].copy()
        best_sig = state["sig_mult"].copy()
        done = 0

        # --- resume ----------------------------------------------------------
        if CKPT.is_file():
            cp = json.load(open(CKPT))
            state["defaults"] = np.array(cp["cur_defaults"], dtype=float)
            state["sig_mult"] = np.array(cp["cur_sig_mult"], dtype=float)
            opt_def.params = np.array(cp["def_params"]); opt_def.m = np.array(cp["def_m"])
            opt_def.v = np.array(cp["def_v"]); opt_def.t = int(cp["def_t"]); opt_def.counteval = int(cp["def_counteval"])
            opt_sig.params = np.array(cp["sig_params"]); opt_sig.m = np.array(cp["sig_m"])
            opt_sig.v = np.array(cp["sig_v"]); opt_sig.t = int(cp["sig_t"]); opt_sig.counteval = int(cp["sig_counteval"])
            best_loss = float(cp["best_loss"]); best_def = np.array(cp["best_defaults"], dtype=float)
            best_sig = np.array(cp["best_sig_mult"], dtype=float)
            history = cp["history"]; done = int(cp["completed_cycles"])
            print(f"Resumed cycle {done}/{MAX_CYCLES}, best_loss {best_loss:.8f}")
        else:
            # Baseline: champion (defaults + sigmas) at 8-ep --recency. Must
            # reproduce the champion (~0.31980351); if not, a write bug exists.
            write_constants(state["defaults"], state["sig_mult"])
            best_loss = _bench(N_EPOCHS)
            best_def = state["defaults"].copy(); best_sig = state["sig_mult"].copy()
            history.append({"step": 0, "loss": best_loss, "rec_def": best_loss, "def_accepted": False})
            print(f"baseline champion --recency by_user = {best_loss:.8f}  [should match ~0.31980351]")
            _save_plot(history, PLOT, "joint default+sigma tune: logloss_by_user", ytick=1e-4)

        print(f"per-param sigma N={N_SIG} (w[4..33]); defaults N={n_def}; "
              f"LR def {LR_DEFAULT} / sigma {LR_SIGMA}; cycles {MAX_CYCLES}; "
              f"~{(2*N_SIG+2)*130/3600:.1f} h/cycle")

        for cyc in range(done, MAX_CYCLES):
            t0 = time.perf_counter()
            print(f"\n{'='*80}\n[joint] cycle {cyc+1}/{MAX_CYCLES}   best {best_loss:.8f}")

            # 1. DEFAULT step — 0-ep --default Adam step FROM the committed defaults.
            opt_def.params = state["defaults"].copy()
            print(f"  [1] default 0-epoch gradient ({2*n_def} batched evals)...")
            gd, _ = opt_def.compute_gradient(def_grad_eval, def_grad_batch)
            new_def = opt_def.step(gd)

            # 2. RECENCY eval of the stepped defaults (+ current committed sigmas).
            write_constants(new_def, state["sig_mult"])
            rec_def = _bench(N_EPOCHS)
            # 3. GATE — commit defaults iff they beat the best --recency so far.
            accepted = rec_def < best_loss
            if accepted:
                state["defaults"] = new_def.copy()
                best_loss = rec_def
                best_def = new_def.copy(); best_sig = state["sig_mult"].copy()
                print(f"  [2/3] stepped-defaults --recency {rec_def:.8f}  ** ACCEPTED -> best {best_loss:.8f}")
            else:
                opt_def.params = state["defaults"].copy()   # revert proposal
                print(f"  [2/3] stepped-defaults --recency {rec_def:.8f}  -- rejected (defaults frozen)")

            # 4. SIGMA step — 8-ep --recency Adam step on the committed defaults.
            print(f"  [4] sigma 8-epoch --recency gradient ({2*N_SIG} evals, ~{2*N_SIG*130/3600:.1f} h)...")
            gs, _ = opt_sig.compute_gradient(sigma_eval, None)
            new_sig30 = opt_sig.step(gs)
            full_mult = np.ones(n_def); full_mult[SIG_IDX] = new_sig30
            state["sig_mult"] = full_mult
            write_constants(state["defaults"], state["sig_mult"])
            rec_sig = _bench(N_EPOCHS)
            if rec_sig < best_loss:
                best_loss = rec_sig
                best_def = state["defaults"].copy(); best_sig = state["sig_mult"].copy()
                print(f"  [4] post-sigma --recency {rec_sig:.8f}  ** new best {best_loss:.8f}")
            else:
                print(f"  [4] post-sigma --recency {rec_sig:.8f}  (best {best_loss:.8f})")

            # record + checkpoint + plot (loss = end-of-cycle --recency).
            history.append({
                "step": cyc + 1, "loss": rec_sig, "rec_def": rec_def,
                "def_accepted": bool(accepted),
                "def_grad_norm": float(np.linalg.norm(gd)),
                "sig_grad_norm": float(np.linalg.norm(gs)),
            })
            _atomic_write_json(CKPT, {
                "cur_defaults": state["defaults"].tolist(),
                "cur_sig_mult": state["sig_mult"].tolist(),
                "def_params": opt_def.params.tolist(), "def_m": opt_def.m.tolist(),
                "def_v": opt_def.v.tolist(), "def_t": int(opt_def.t), "def_counteval": int(opt_def.counteval),
                "sig_params": opt_sig.params.tolist(), "sig_m": opt_sig.m.tolist(),
                "sig_v": opt_sig.v.tolist(), "sig_t": int(opt_sig.t), "sig_counteval": int(opt_sig.counteval),
                "best_loss": float(best_loss), "best_defaults": best_def.tolist(),
                "best_sig_mult": best_sig.tolist(), "history": history,
                "completed_cycles": cyc + 1, "sig_idx": SIG_IDX,
                "base_sigma": base_sigma.tolist(),
            })
            _save_plot(history, PLOT, "joint default+sigma tune: logloss_by_user", ytick=1e-4)
            print(f"[joint] cycle {cyc+1} done in {(time.perf_counter()-t0)/3600:.2f} h; "
                  f"loss={rec_sig:.8f} best={best_loss:.8f}")

        print(f"\n[joint] DONE. best_loss={best_loss:.8f}")
        print(f"  best_defaults = {[round(x,6) for x in best_def.tolist()]}")
        print(f"  best_sig_mult = {[round(x,4) for x in best_sig.tolist()]}")
        _restore_constants()
    finally:
        _restore_constants()


if __name__ == "__main__":
    main()
