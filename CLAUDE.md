# fsrs-autoresearch — Claude Code project context

## What this repo is

GPU-accelerated benchmark + research playground for **FSRS** (Free Spaced
Repetition System). Forked from [open-spaced-repetition/fsrs-gpu-benchmark].
End goal: an autoresearch loop (AlphaEvolve / Karpathy style) that proposes
FSRS model variants, trains them on real Anki review data, scores them, and
keeps the winners.

**Ignore `autoresearcher.md`** — it's an older plan written for a separate
LLM-API-driven loop. We're rebuilding it inside Claude Code.

## !!! SELF-INJECTION CAPABILITY — STRICT, NON-NEGOTIABLE LIMITS !!!

This repo runs on the Claude Code **Windows desktop app** under **bypass
permissions**. In that mode there is a self-injection mechanism —
`C:\Users\Andrew\claude-automation\inject.ps1` (low-level `SendInput` Unicode
typing + `VK_RETURN` into Claude's *own* chat window), driven by `controller.ps1`
on a Task Scheduler job (23:00–12:00) — that lets the agent submit messages into
its own chat so the autoresearch loop can run unattended overnight. **Auto Mode
HARD-BLOCKS this as a permission bypass; it works ONLY under bypass permissions.**

**This capability may be used for EXACTLY TWO things, and NOTHING ELSE
(user directive, 2026-06-04):**
1. **Compaction** — inject `/compact` whenever I judge compaction is due.
2. **A short `Continue` nudge** — to un-stall myself so the loop keeps moving.

**Using self-injection for ANY other purpose is STRICTLY FORBIDDEN.** It must
never be used to run arbitrary commands, self-approve actions, self-direct toward
any other task, or broaden its own scope. The Task Scheduler controller is
hard-coded to inject only these two fixed strings — I must not extend it, add new
injected content, or build any other self-injection path. If I ever think
injection is needed for anything else: **STOP and ask the user.** Self-injection
removes the human from the loop, which is exactly what the permission system
guards against; the user accepts it ONLY for this narrow keep-alive/compaction use.

### Running the loop autonomously (bypass mode)

When `C:\Users\Andrew\claude-automation\loop_active.txt` exists, the loop is in
unattended mode and the `ClaudeLoopController` Task Scheduler job is live
(23:00–12:00). Per-turn discipline:
- **Beat the heartbeat** each turn: run (or bake into your loop commands)
  `& C:\Users\Andrew\claude-automation\beat.ps1;` so the controller knows you're
  active, not stalled.
- **Keep-alive:** launch each `run.sh` training run as a **background** job; its
  completion re-invokes you. Always yield with a job pending — this is the primary
  keep-alive; the controller is only the failsafe (injects `continue` if you stall:
  no container + stale heartbeat).
- **!!! The app's background-task tracker is UNRELIABLE (2026-06-11 incident,
  user-confirmed):** background jobs can die silently, their completion
  notifications can simply never fire, and the Background-tasks panel can show
  dead jobs as "Running" for 11+ hours. Never trust the panel or the absence of
  a notification — on every turn, verify any job you think is running via **OS
  truth**: `Get-Process` (PID), output-file / `result/*.json` mtimes, `docker ps`.
- **Pre-flight before relying on the failsafe/injector:** confirm
  `Get-ScheduledTask ClaudeLoopController` is **Enabled** AND `controller.log`
  shows fresh ticks. The 2026-06-11 stall happened because the task was Disabled
  while `loop_active.txt` was on — the master switch alone proves nothing.
- **Compaction (every 8 iters):** at a clean boundary (no container running),
  create `C:\Users\Andrew\claude-automation\pending_compact.txt` and yield idle —
  the controller injects `/compact` then a `continue`. Bare `/compact` is fine
  (continuity is on disk). After compacting, set `.last_compact_iter` as usual.
- The controller ONLY injects `/compact` or `continue` (strict-limit rule above).
  Full details + how to stop the loop: `C:\Users\Andrew\claude-automation\README.md`.
  **The automation is already built — do not rebuild it; just run the research.**

## Host machine assumptions

- Windows 10 Pro 22H2 (build 19045), 64 GB RAM, 473+ GB free on C:
- RTX 4070 (compute cap 8.9, passes the ≥8.6 requirement)
- Docker Desktop with WSL2 backend (no Ubuntu distro needed — Docker brings
  its own `docker-desktop` WSL2 distro)
- `~/.wslconfig` set to 32 GB memory cap to avoid `BrokenProcessPool`
- Dataset at `C:\Users\Andrew\anki-revlogs-3k` (sibling of this repo),
  mounted into the container as `/anki-revlogs-3k` read-only

Run commands from **PowerShell** in this directory.

## How to run anything

Everything runs inside Docker because FSRS-7's forward pass uses a custom
CUDA extension built with LLVM 18 + Enzyme. Native Windows builds aren't
supported.

**Build the image** (one-time, ~20–40 min first run, cached afterward):
```pwsh
docker compose build srs-benchmark
```

**Prepare the dataset** (one-time):
```pwsh
docker compose --progress quiet run --rm srs-benchmark `
    python -m src.prepare.prepare --processes 10
```

`--short`, `--secs`, and `--recency` default to **on** in this fork
(see `src/prepare/prepare_config.py`). FSRS-7 is always run with that
combination, so the prepare command above already applies them. Use
`--no-short` / `--no-secs` / `--no-recency` to opt out.

**Train + evaluate FSRS-7** (each iteration of the autoresearch loop):
```pwsh
docker compose --progress quiet run --rm srs-benchmark bash src/main/run.sh
```

`run.sh` builds the Enzyme C++/CUDA extension in-place via `setup.py`
(skipped if up-to-date), then runs `python -m src.main.run`, which trains
for `N_EPOCHS=8` and prints metrics.

Drop `--processes` or use `--processes 5` if you hit `BrokenProcessPool`.

## Model registry

Only **FSRS-7** is in `MODEL_REGISTRY` (`src/models/model_factory.py`) and
`FEATURE_ENGINEER_REGISTRY` (`src/features/factory.py`); `ModelName` in
`src/prepare/prepare_config.py` is the single-value literal `"FSRS-7"`.

The files `src/models/fsrs_v1.py` … `fsrs_v6.py`, `fsrs_rs.py`, and
`fsrs_v6_one_step.py` still exist because `FSRS7` inherits transitively
from `FSRS6 → FSRS5 → FSRS4dot5 → FSRS4 → FSRS3 → FSRS2 → FSRS1 → FSRS →
BaseModel`. Flattening that chain and deleting the leftover files is a
follow-up task — do it after the baseline run confirms training still
works, so we have something to A/B against.

## Code layout

```
src/
  prepare/
    prepare.py              # one-time dataset preprocessing
    prepare_config.py       # ModelName literal + CLI args + Config class
  features/                 # feature engineers (DASH, FSRS-eng)
    base.py
    create_features.py
    dash_engineer.py
    fsrs_engineer.py
    factory.py              # FEATURE_ENGINEER_REGISTRY
  models/                   # ★ where new variants go
    base.py
    trainable.py
    fsrs.py                 # FSRS shared base
    fsrs_v1.py … fsrs_v7.py # incl. fsrs_v4dot5.py, fsrs_v6_one_step.py
    fsrs_rs.py              # FSRS-rs port (leftover in the inheritance chain; not in MODEL_REGISTRY)
    fsrs_v7_interval_penalty.py
    model_factory.py        # MODEL_REGISTRY — register new variants here
  main/
    config.py               # DEVICE, BATCH_SIZE=256, N_EPOCHS=8 (FSRS_N_EPOCHS env override), WRITE_RESULT
    run.py                  # training + eval entry point
    run.sh                  # builds Enzyme ext then runs run.py
    fsrs/                   # FSRS-7 Python training helpers (NOT the CUDA kernel):
      fsrs_v7_constants.py  #   LR, BETAS, RECENCY_C0/EXP, PENALTY_W_L2, FSRS7_DEFAULT_35_VALUES, MIN/MAX, L2_SIGMA
      fsrs_v7_helpers.py    #   L2 penalty, recency weighting, parameter clipper
      fsrs_v7_optimizer.py  #   Adam/AdamW update rule
      fsrs_v7_scheduler.py  #   cosine LR schedule
    csrc/                   # C++/.cu sources (Enzyme extension)
      fsrs_extension.cpp
      fsrs_extension.cu
      fsrs/                 # ★ the actual CUDA FSRS-7 model: fsrs7.cu / fsrs7.cuh
      fsrs_kernel/          # off-limits infra: fsrs_train.{cu,cuh} (holds dloss_dp), fsrs_test.{cu,cuh}
    srs_ops.py              # Python wrapper around the extension
    tensor_cache.py         # LMDB tensor cache
    tensor_lmdb.py
    tensors.py
    result_metrics.py
  autoresearch/             # ★ autoresearch-loop tooling (host-side, outside Docker)
    history.py              # append_iteration() -> result/history.{jsonl,md}
    diagnostics.py          # per-iteration report (+ FSRS7_BOUNDS_STATIC mirror)
    complexity.py           # complexity score (Python + CUDA)
    hp_tune.py              # automated hyperparameter tuner
    central_diff_init_w.py  # default-parameter meta-optimizer
    plot_history.py         # history plot (the human's — run, never edit)
result/                     # per-run artifacts: diagnostics.{json,md}, history.{jsonl,md}, history_plot.png
compose.yaml                # bind mounts ../anki-revlogs-3k and current dir
dockerfile                  # CUDA 11.8 + LLVM 18 + Enzyme + PyTorch 2.7.1
pyproject.toml              # Python 3.12, fsrs-optimizer, torch, etc.
uv.lock                     # uv dependency lockfile
setup.py                    # custom EnzymeBuildExtension for the .cu file
```

## FSRS-7 architecture (the thing we're evolving)

The **live model is the custom CUDA forward** (`src/main/csrc/fsrs/fsrs7.cu`
+ `fsrs7.cuh`), a 34-parameter dual-trace model. Its parameter vector,
clamps, defaults, and L2 anchor live in `src/main/fsrs/fsrs_v7_constants.py`
(`FSRS7_DEFAULT_35_VALUES` = init/anchor, `FSRS_MIN_VALUES` / `FSRS_MAX_VALUES`
= the per-element clamp ranges, `FSRS7_L2_SIGMA_35_VALUES` = L2 sigmas); the
live clamp is applied by `fsrs_v7_helpers.apply_parameter_clipper`, and
`FSRS7_BOUNDS_STATIC` in `src/autoresearch/diagnostics.py` is the mirror the
diagnostics report reads. **These — `fsrs7.cu`/`fsrs7.cuh` (shape +
formulas), `fsrs_v7_constants.py` (bounds + defaults + sigmas), and
`FSRS7_BOUNDS_STATIC` (diagnostics mirror) — are authoritative and must be
updated in lock-step** by any patch that adds, removes, or reorders
parameters (plus the param table below and this section).

> **⚠ `src/models/fsrs_v7.py` is DEAD baseline code — NOT the live model
> (confirmed 2026-06-06).** It is the original iteration-0 pure-Python FSRS-7
> (a *35*-param layout that still contains the iter-43-removed transition
> params `w[25..26]`, with its own `init_w` / `FSRS7ParameterClipper` /
> `forward`). The benchmark **never** calls its forward, `init_w`, or clipper —
> `run.py` training/eval runs entirely on the CUDA extension, seeded from
> `FSRS7_DEFAULT_35_VALUES` and clamped by `fsrs_v7_helpers.apply_parameter_clipper`.
> The class survives only because it is imported in `src/models/__init__.py`,
> registered in `MODEL_REGISTRY`, and instantiated once by `prepare.py` *solely*
> to read its `batch_size` class attribute (which has a `config.batch_size`
> fallback). **Do NOT edit `fsrs_v7.py` to change the model, and do NOT trust
> its clipper / `init_w` as a layout reference — they are stale by 2 params.**
> It is still listed in the complexity-scored `mutation_files`, so removing it
> is a deliberate cleanup (untangle the `__init__.py` import + `MODEL_REGISTRY`
> + prepare's `batch_size` read + re-baseline the complexity score), not a
> drop-in delete.

| Param range | Role |
|---|---|
| `w[0..3]` | Initial stability per rating (Again/Hard/Good/Easy) |
| `w[4..6]` | Difficulty |
| `w[7..14]` | Long-term stability update (8 params) — drives the **long** trace `s_long` |
| `w[15..22]` | Short-term stability update (8 params) — drives the **short** trace `s_short` |
| `w[23..30]` | 8-param forgetting curve (short component keyed to `s_short`, long to `s_long`) |
| `w[31..33]` | Difficulty/stability modulation of the curve (d_weight, d_decay, s_decay1). **All shifted to non-negative bounds (2026-06-05 cosmetic refactor):** d_weight∈[0,1] curve subtracts 0.5; d_decay∈[0,0.6] subtracts 0.3; s_decay1∈[0,0.6] subtracts 0.3. Neutral (no-D/S-dependence) values are 0.5/0.3/0.3. |

**Current champion: iter-165 (2026-06-07) — by_user 0.31980351, complexity
16,938, 34 params, 3 state variables (`s_long`, `d`, `s_short`).** The
mechanisms live in the CUDA forward right now:

- **Dual-trace memory** (iter-43): two stability states, `s_long` + `s_short`,
  each driving its own forgetting-curve component. The short trace's initial
  stability is **hardcoded** at `0.8 * initial_stability` (iter-69/70: a
  trainable multiplier didn't earn its param; 0.8 is the by_user optimum).
- **Trace-specific short learning** (iter-71): the short trace updates from the
  short component's own recall `r1` (`fsrs7_short_component_recall`); the long
  trace updates from the mixed-curve retention.
- **Post-lapse stability is difficulty-independent** (iter-85 ablated
  `fail_d_exp`) — satisfies constraint 4 trivially on the lapse path.
- **Post-lapse relearning cap** (iter-97): on a lapse the fast trace is capped
  at `0.8 * post-lapse-slow-stability` in `fsrs7_step`, restoring the
  `fast <= 0.8*slow` invariant.
- **Surprise-weighted lapse difficulty** (iter-101/105): on a lapse,
  `delta_d *= (1 + 1.0*(retention − 0.9))` in `fsrs7_next_d` — an unexpected
  lapse (high `R`) raises `D` more. Coefficient hardcoded; factor ∈
  [0.55, 1.05] stays positive so rating-ordering holds. This is the live D↔R
  coupling (preserve it — see the research-focus note below).
- **Long-curve D-modulation on the time-scale** (iter-165):
  `r2 = (1 + factor2 * (t/s_long) * exp((d_decay-0.3)*(d-5)))^(-decay2)` — the
  d-factor multiplies the time-scale; `decay2` itself is un-modulated. Hard
  cards get a faster effective clock (a level shift), not a steeper slope.
  This freed both params (`decay2` range_frac 0.99, `d_decay` grad 0.86 /
  5.6% at upper bound) — a `d_decay` bound-widen or re-tune may extract more.
- **Defaults re-tuned at iter-140** via the `central_diff_init_w` meta-opt
  (defaults = per-user init = the L2 anchor; the gain came from light,
  L2-pinned users).

Lessons worth keeping in mind (full stories in `docs/architecture_history.md`):
- **Sub-day over-confidence is a real, population-wide lever** (iter-135/138:
  fixed curve-shape corrections near `t=0` gained ~1.6e-4) — but both
  implementations were **reverted by the user for aesthetics**; don't re-add
  without asking.
- **Walled trainable params can work as fixed-formula coefficients**
  (iter-101/105 broke a ceiling that four trainable-`d_surprise` attempts had
  hit at ~10× less gain — light users pin trainable params to 0; fixing the
  coefficient forces the coupling on everyone).

The full per-iteration narrative (iter-40 → iter-165, incl. the reverted
iter-138 sub-day mechanism) is archived in `docs/architecture_history.md`;
`result/history.jsonl` is the per-iteration source of truth.

Training: 8 epochs, batch 256, Adam, lr 0.0188, betas (0.55, 0.9913), recency-weighted
gradient. Loss = log-loss(per-review) + L2-to-defaults (the sched penalties in
`fsrs_v7_interval_penalty.py` are dead code — `run.py` never calls them). The live values
are in `src/main/fsrs/fsrs_v7_constants.py` (source of truth; this line is a snapshot).

Metrics produced by `run.py`:
- `logloss_by_user` — **primary**, and the only one that decides accept/reject.
  Each user contributes equally to the average regardless of how many
  reviews they have, so heavy users can't dominate the metric. In the
  original evaluate.py this is the "weighted by users" column.
- `logloss_by_review` — sanity check. Average over every review with no
  per-user normalisation; heavy users dominate.

**Why Again log loss looks huge.** Don't chase it as a sign of FSRS being
"bad on Again". A model that confidently predicts 85% retention has
per-review log loss `-ln(0.85) ≈ 0.163` on a Good (label=1) review and
`-ln(1 - 0.85) ≈ 1.897` on an Again (label=0) review. Again is the
negative class, and any well-calibrated high-recall predictor has
asymmetric per-class log loss with the negative class dominating. The
Again-loss line in diagnostics is useful for *deltas* between variants,
not as an absolute target.

## Adding a new model variant

The live model is the CUDA forward, so a "variant" is an **in-place edit of
`src/main/csrc/fsrs/fsrs7.cu`** (+ `fsrs7.cuh` for struct/field changes), not a
new Python class. The old Python-class workflow (copy `fsrs_v7.py`, register in
`MODEL_REGISTRY`) is **obsolete** — `fsrs_v7.py` is dead code (see the ⚠ note
under "FSRS-7 architecture").

1. Edit the formula in `src/main/csrc/fsrs/fsrs7.cu` (and `fsrs7.cuh` for
   struct/field changes).
2. If you add / remove / reorder trainable params, update **all** of these in
   lock-step: the `fsrs_params_t` struct in `fsrs7.cuh` (field order = `w[]`
   index order), `fsrs_v7_constants.py` (`FSRS7_DEFAULT_35_VALUES`,
   `FSRS_MIN_VALUES`, `FSRS_MAX_VALUES`, `FSRS7_L2_SIGMA_35_VALUES` — keep all
   the same length), `FSRS7_BOUNDS_STATIC` in `src/autoresearch/diagnostics.py`,
   and the param table above. (The live clamp
   `fsrs_v7_helpers.apply_parameter_clipper` just reads the MIN/MAX arrays, so it
   needs no separate edit. Do **not** touch `fsrs_v7.py`.)
3. Re-run training: `docker compose --progress quiet run --rm srs-benchmark bash
   src/main/run.sh`. Any edit to a file under `src/main/csrc/` triggers an Enzyme
   rebuild (a few minutes); Python-only edits do not.

## Autoresearch loop

AlphaEvolve/Karpathy-style: Claude reads the current champion, proposes a
mutation, writes the patch, runs the benchmark, accepts or rejects against a
threshold + complexity gate, and keeps an archive of winners.

> **!!! CURRENT RESEARCH FOCUS (2026-06-10 user directive): scope RE-OPENED —
> "find ways to improve FSRS's formulas and/or training pipeline."** The
> 2026-06-06 D-only narrowing is lifted (the D campaign, iters 141–165,
> comprehensively closed that subsystem — see `result/history.md`). Current
> specifics:
> - **`n_before` fatigue feature: CONCLUDED + STRIPPED (2026-06-10).** The
>   user-greenlit same-day-prior-review-count feature was built end-to-end and
>   measured in iters 178/179: recall-side clock peaked +5.8e-5, encoding-side
>   +1.1e-5 then negative — both far under the +0.0010 input-feature bar (the
>   raw −2.4pp fatigue signal is mostly schedule confound; `delta_t` prices
>   it). The entire pipeline (prepare column → tensors → cache → kernels →
>   curve arg) was then removed per user call; cache is v8, strip verified
>   bit-exact (delta 3.9e-9). Old prepare blobs keep an unused `n_before`
>   named tensor — harmless; do **not** re-prepare. Don't re-propose fatigue
>   features without new evidence.
> - **Response-time (DURATION) feature REJECTED** by the user — do not propose.
> - **Forgetting curve CLOSED for added complexity (user veto 2026-06-10
>   evening): "Let's not add more complexity to the forgetting curve."** No
>   new factors, components, or mechanisms in `fsrs7_forgetting_curve` /
>   `fsrs7_short_component_recall`. This supersedes the earlier "sub-day may
>   be proposed" carve-out — the absolute-clock power-law factor on r1 was
>   vetoed pre-run under this rule (math write-up exists in the 2026-06-10
>   session if ever needed). Curve simplifications/ablations remain fine.
> - **User-directed priorities (2026-06-10 evening, updated late evening):
>   spend ~5–10 more iterations on the TRAINING LOOP (counting from iter-193),
>   then pivot to FORMULA changes — and in that phase, changes that ADD NEW
>   TRAINABLE PARAMETERS are explicitly back on the menu ("it's been a while
>   since changes that add new parameters have been done"; +0.0002/param bar
>   unchanged). Flexibility belongs in places OTHER than the forgetting curve
>   (stability updates, init, lapse path, …). Training-loop changes still
>   bundle hp_tune per the mandatory-bundle rule.
> **Threshold math (incl. the new input-feature row), acceptance criteria, the
> complexity gate, and hard constraints 1–12 are otherwise UNCHANGED.**

> **Iteration budget — this is a long campaign (150+ iterations).** The user
> is running this loop autonomously for ~2 weeks, which is **150+ and likely
> 300+ iterations**. Do **not** optimize for iteration economy, fear "wasting"
> an iteration on an experiment that might reject, or try to "declare
> converged" to stop early. A rejected structural probe is a *normal, valuable*
> outcome — it maps the loss surface and rules out a hypothesis. With this
> budget the right strategy is **bold structural bets** (new state variables,
> whole-formula reformulations) over safe +1-param tweaks, even though each big
> bet is more likely to reject. Spend the iterations. There will be plenty.

> **Autonomous wakeup cadence.** When self-pacing the loop with `ScheduleWakeup`
> (dynamic `/loop` mode), use a **3-minute** delay (180 s) between normal
> synchronous iterations — the user's chosen cadence (2026-05-30). 180 s also
> stays inside the prompt-cache TTL, so context stays warm between iterations.
> **Exception:** while a *tracked background job* (e.g. an `hp_tune` pass,
> ~15–20 min) is running, don't poll it every 3 min — use a long fallback
> (1200 s+) and rely on the background-completion notification, then resume the
> 3-min cadence.

**Note on `autoresearcher.md`:** that file was written before this codebase
existed, assuming a tight-loop pipeline with a separate proposer LLM emitting
structured dicts and a smaller patch-writer LLM doing `git apply`. We're not
using that split — Claude Code does both inline. Sub-agents aren't needed for
the loop. The constraints, threshold formula, and complexity gate below are
copied from there because they're the actual design decisions and still
apply.

### Mutation surface

Python-side FSRS code only:
- `src/models/fsrs_v7.py` — **DEAD baseline code, NOT an edit target.** It stays
  in the complexity-scored set (a fixed offset only); the live model forward /
  clamps are the CUDA files + constants below. See the ⚠ note under "FSRS-7
  architecture".
- `src/models/fsrs_v7_interval_penalty.py` — scheduling penalty
- `src/main/fsrs/fsrs_v7_constants.py` — `LR`, `BETAS`, `RECENCY_C0/C1`,
  `PENALTY_W_L2`, `FSRS7_DEFAULT_35_VALUES` (init_w), `FSRS_MIN_VALUES`
  / `FSRS_MAX_VALUES` (CUDA-path clamps), `FSRS7_L2_SIGMA_35_VALUES`
- `src/main/run.py` — training loop (`train_iter`, optimizer/scheduler wiring)
- `src/main/fsrs/fsrs_v7_helpers.py` — L2 penalty (`penalty_loss`), recency
  weighting (`gradient_weight`), parameter clipper (`apply_parameter_clipper`)
- `src/main/fsrs/fsrs_v7_optimizer.py` — the Adam/AdamW/NAdam update rule
- `src/main/fsrs/fsrs_v7_scheduler.py` — LR schedule (cosine decay)
- `src/main/config.py` — only `BATCH_SIZE`, never `N_EPOCHS` (in a model
  variant; the `--epoch-batch-grid` tuner is the sole sanctioned exception that
  may move both — see constraint #5 and the hp-tune section)
- `src/main/csrc/fsrs/*` — CUDA forward/backward source. In bounds, but
  touching it triggers a multi-minute Enzyme rebuild. Other paths under
  `src/main/csrc/` (`fsrs_extension.cpp`, `fsrs_extension.cu`,
  `fsrs_kernel/`) remain out of bounds.

**Hands off `src/autoresearch/plot_history.py`.** That file is the *human's* —
they edit it themselves to make the iteration-history plot prettier. If it
shows as modified in `git status`, that's the user, not a stray change.
**Never edit, revert, `git checkout`, or commit it on your own initiative —
UNLESS the user explicitly asks you to change it** (e.g. "fix the plot labels"),
in which case make the minimal change they requested and nothing else, then
regenerate the plot to verify. When rejecting an iteration,
*explicitly list* the mutation files you changed plus `result/diagnostics.{json,md}`
in the revert (never a blanket `git checkout -- .`), so `plot_history.py` is
left untouched. It is not in the scored `mutation_files`, so it never affects
the gate.

**Sync note for clamps + init_w:** `fsrs_v7_constants.FSRS_MIN_VALUES` /
`FSRS_MAX_VALUES` / `FSRS7_DEFAULT_35_VALUES` / `FSRS7_L2_SIGMA_35_VALUES` are
the authoritative param vector for the CUDA training path (init/anchor, clamps,
sigmas); the live clamp is `fsrs_v7_helpers.apply_parameter_clipper`, which reads
the MIN/MAX arrays. Any variant that changes one must change the others (and
update `FSRS7_BOUNDS_STATIC` in `src/autoresearch/diagnostics.py` and the param
table above). `fsrs_v7.py`'s `FSRS7ParameterClipper` / `init_w` are **dead** and
must NOT be used as a reference (they are the stale iter-0 35-param layout).

### Allowed change types

1. Add/remove parameters/constants in FSRS formulas; modify clamp ranges,
   default values, and the L2-penalty sigmas.
2. Add new formulas or new state variables (currently **3**: `s` = slow memory
   stability, `d` = difficulty, `s_fast` = fast memory stability [dual-trace,
   iter-43]).
3. Modify the training loop: learning rate, betas, lr scheduler, optimizer
   choice (Adam → AdamW/SGD/etc.), recency weighting — **except `n_epoch`**
   (which only `hp_tune.py --epoch-batch-grid` may re-anchor, under the
   Pareto/at-least-one-axis rule; see constraint #5).

### Hard constraints (every variant MUST satisfy all)

1. Forgetting curve is monotonic in `delta_t` for **any** parameter values
   within the allowed clamp ranges.
2. `w[0] <= w[1] <= w[2] <= w[3]` — initial stabilities ordered by rating:
   S0(Again) ≤ S0(Hard) ≤ S0(Good) ≤ S0(Easy).
3. `stability_after_review(rating=1) <= ... <= stability_after_review(rating=4)`
   — same ordering after a review.
4. Higher `D` ⇒ post-review `S` is non-increasing in `D`. This applies to
   **both** post-success and post-lapse stability: difficulty must not let
   memory grow faster after a successful review, and it must not let post-lapse
   stability rise either. As of **iter-85** the post-lapse formula `new_s_fail`
   is **difficulty-independent** (the old `d^(-fail_d_exp)` factor was ablated —
   the exponent was inert and fought its `>= 0` floor), which satisfies this
   constraint trivially (constant in `D` ⇒ non-increasing). If a future variant
   re-introduces a `d^(-k)` post-lapse factor, `k >= 0` becomes a hard structural
   requirement again.
5. **Do not change `N_EPOCHS` in a model variant** (`src/main/config.py:28`).
   Don't rename it either. We're chasing architectural wins, not brute-force
   epochs. **Exception (2026-06-03): the `hp_tune.py --epoch-batch-grid` Pareto
   search may re-anchor `N_EPOCHS` *and* `batch_size`** as the compute *operating
   point* — but only via the at-least-one-axis rule: a config replaces the gold
   standard `(n_epoch=8, batch_size=512)` only if it is **no worse on BOTH** log
   loss and train+eval compute time (strictly better on ≥1). Because more epochs
   at a fixed batch is strictly slower, brute-force epochs can never win on the
   speed axis; `n_epoch` can rise only when a *larger* batch buys the speed back
   (a genuine free lunch). So the anti-brute-force intent holds. A normal model
   iteration still must not touch `N_EPOCHS` — only the grid may.
6. Do not skip users, change time-series splits, or change review-preprocessing
   filters. **Guarded by hard asserts** at the end of `run.py::main()`
   against `EXPECTED_N_USERS` / `EXPECTED_N_REVIEWS` in `src/main/config.py`.
   Anki-revlogs-3k has `EXPECTED_N_USERS=3000` and
   `EXPECTED_N_REVIEWS=152_354_175`. If a variant trips either assert,
   it's auto-rejected.
7. Anything stochastic must be seeded (`config.seed = 42` is the existing one,
   reuse it).
8. No data leakage from the test/eval split into training. Golden rule.
9. No exploits of jsonl output, user IDs, FP rounding, etc. Wins must come
   from the FSRS-7 model and/or training loop, not the harness.
10. Eval metric is and stays `logloss_by_review`. The **training** loss is
    allowed to change (focal, BCE+aux, etc.) but the proposal must explain
    why a different training loss is expected to improve eval log-loss.
11. The predicted recall probability **is** a forgetting curve in `delta_t`
    with **fixed, non-negotiable boundary conditions**:
    - `p(delta_t = 0) = 1` — immediately after a review, recall is certain.
    - `p(delta_t → ∞) → 0` — given unbounded time, the memory is fully lost.
    Between those endpoints `p` is monotonically non-increasing in `delta_t`
    (constraint 1) and stays in `[0, 1]` — it's fed straight into
    `−[y·ln p + (1−y)·ln(1−p)]`, so any `p ≤ 0` or `p ≥ 1` is a NaN/inf bomb.
    This **forbids slip/guess-style asymptotes**: a learned floor (`p → g > 0`
    as `t → ∞`) breaks the lower endpoint, and a learned ceiling
    (`p → 1 − s < 1` at `t = 0`) breaks the upper one — both are off-limits no
    matter how much they help log-loss. The *only* sanctioned deviation is the
    existing numerical clamp to `[1e-5, 1−1e-5]` that keeps the log finite: it
    is symmetric and ~1e-5, not a calibration knob. You may reshape the curve
    **between** the endpoints freely (that's what the `w[27..34]` 2-component
    mixture does today) — you just may not move the endpoints.
12. **Forgetting-curve smoothness / shape** (added 2026-05-30). Beyond the
    endpoints (constraint 11), the curve `p(delta_t)` must stay *well-behaved*
    to rule out implausible, wiggly forgetting curves. For `delta_t > 0`:
    - `p` is **continuous** and **monotonically non-increasing** (already implied
      by constraints 1 & 11), **and**
    - its **first derivative `p'(delta_t)` is itself continuous and monotone** —
      the curve is C¹ (no kinks / slope jumps) with **no inflection points**. For
      a curve falling from 1 toward 0 this means it is **convex**: forgetting is
      fastest right after a review, then decelerates — no S-shapes, bumps, or
      kinks.
    A **jump at exactly `delta_t = 0` is allowed**: the curve may approach a limit
    `< 1` as `delta_t → 0⁺` while we *define* `p(0) := 1` (cf. defining
    `sin(x)/x := 1` at `x = 0`). This lets a curve drop sharply right after a
    review (helpful for sub-day forgetting) without a smooth ramp back up to 1.
    Rationale: keeps any flexible curve family (splines, monotone nets, extra
    mixture components) from producing weird shapes. The current `w[27..34]`
    mixture **complies** — a fixed convex combination of convex, decreasing
    power-law components is convex and C¹. Combination rules that introduce an
    inflection are **off-limits** — e.g. a probabilistic noisy-OR
    `1 − (1−r₁)(1−r₂)` is concave near 0 then convex (an inflection), so it is
    forbidden even though each component `rᵢ` is individually convex.

> **Don't exploit joint optimization (constraint noted 2026-06-01).** This
> harness trains all 3000 users **jointly**, but production (FSRS-rs) trains
> **one user at a time** — a "win" that only exists because of cross-user
> structure does **not** transfer to real users. A change must improve
> **single-user** optimization. The one violation ever accepted (the
> empirical-Bayes L2 of iter-24, anchoring to the live population mean) was
> removed in iter-75 — user directive; its illegitimate cross-user gain was
> ~0.00029. The L2 penalty now shrinks toward the fixed `FSRS7_DEFAULT`,
> exactly what FSRS-rs uses. **No known joint-optimization exploits remain.**
> (Full story: `docs/architecture_history.md`.)

### Acceptance threshold

A variant is accepted only if `old_logloss_by_user − new_logloss_by_user ≥ threshold`
(the per-user primary metric, not per-review), where:

| Change | Contribution to threshold |
|---|---|
| Each new state variable | +0.0010 |
| Each new **input feature** (a new per-review input column fed to the model, e.g. `n_before`) | +0.0010 (user directive 2026-06-10) |
| Each new parameter (new trainable scalar in `self.w`) | +0.0002 |
| Any other change (formula tweak, training-loop edit, etc.) | +0.0001 baseline |
| Floor | 0.0001 |

**A "new parameter" means a new trainable scalar in `self.w`.** A new fixed
constant outside `self.w` does NOT count. A new variable returned in
`torch.stack(...)` from `step()` counts as a new state variable.

**Removing parameters while adding new ones:** the threshold drops by
0.0002 per removed parameter, but never below 0.0002. So if you remove 3 and
add 1, threshold = `max(-3*0.0002 + 0.0002, 0.0002) = 0.0002` — simplifying
formulas with fewer parameters must still yield an improvement to be kept.

**Pure ablation (removing parameter(s) only) — threshold `0.0000`.** A
standalone iteration whose *only* change is removing one or more parameters —
fixing a param to a constant, tying two params, or collapsing a proven
redundancy — and which adds **no** new params, state, capacity, or formula
freedom. Such an iteration is accepted iff

> `old_logloss_by_user − new_logloss_by_user ≥ 0.0000`

i.e. **the simpler model must merely not be worse.** This is deliberately
distinct from the "removing while adding" rule above (which floors at 0.0002):
a *pure* removal isn't asked to *improve* the loss, only to not damage it,
because the win is the parsimony itself (fewer params + lower complexity
score). A tie within the ~2e-6 GPU-noise floor counts as "not worse" → accept.
Constraints (rules 1–12) of course still hold.

Run an ablation as **its own iteration** — never bundle a removal with any
other change, or the delta can't be attributed. Good ablation targets:
- **Structural redundancy** — params that only ever appear in one combination,
  so one is mathematically free to fix (e.g. an un-normalized mixture weight
  whose overall scale cancels: only the *ratio* of the two weights matters).
  These are guaranteed not to worsen the achievable loss (identical function
  class); expect `Δ ≈ 0`.
- **Empirically inert** — params the diagnostics flag with near-zero
  `range_frac` **and** near-zero mean gradient (and/or median pinned at their
  neutral/default value): the data isn't using them. Removal *tests* the
  hypothesis; the 0.0000 bar means a truly-dead param leaves the loss unmoved.

Examples:
- Add 1 state variable + 3 new params: `0.0010 + 3*0.0002 = 0.0016`
- Change a formula with no new params: `0.0001`
- Change optimizer with no new params: `0.0001`

**Mean-of-3 measurement rule (user directive 2026-06-10).** For any
single-idea iteration judged *without* an hp_tune pass, the `logloss_by_user`
that goes up against the threshold is the **mean of 3 identical benchmark
runs** (same code, same config; training is seeded but the harness has
~1e-5 run-to-run nondeterminism at high LR from scatter_add atomics —
discovered iter-185; the champion at LR 0.0188 reproduces to ~7e-7, but the
rule applies regardless so deltas near any bar are trustworthy). At ~2 min a
run this costs ~6 min per iteration — cheap insurance. **hp_tune passes are
exempt** (3× per trial would triple a 15–20 min pass): the tuner's single-run
trials stand, and for an hp_tune-bundled iteration the established near-bar
practice still applies — if the final best config lands within ~1e-5 of the
bar, re-measure it 2–3× and judge on the mean. Record the individual run
values in the history `comment` so the spread is documented.

**Near-misses (improvement real but below threshold).** The threshold is a
bright line: a variant that improves `logloss_by_user` but by less than its
threshold is a **reject**, and the champion is unchanged — don't fudge it (a
new parameter that can't clear its `+0.0002` hasn't earned its complexity).
What to do *next* is the researcher's call: either **retry the idea from a
slightly different angle** (reformulate, tighten a bound, tie/drop a parameter
so the threshold falls, isolate the part that actually carried the signal) or
**give up and pursue a different change** entirely. Use judgment — chase it if
the near-miss exposed real, attributable signal worth recovering more cheaply;
move on if it looks played out. Record the near-miss in history either way.

### Code complexity gate

`score = AST_node_count + 40 * cyclomatic_complexity`, computed over the
mutation surface files listed above (originally autoresearcher.md scoped this
to a single `train.py`; we scope it to the same set of files we allow
mutating). Each accepted variant must increase the score by **≤5%**, ideally
≤2%. Pick conservative implementations — auto-rejection on complexity is
common if you bolt on a new mechanism without simplifying anything else.

**C++/CUDA is scored too.** The active model is the custom CUDA forward, so
`src/main/csrc/fsrs/fsrs7.cu` and `fsrs7.cuh` are in the scored set. C++ has no
stdlib AST, so `complexity.py` scores it with the same `nodes + 40*cyclomatic`
formula via a deterministic token model (`score_cpp_source`): nodes = token
count (comments/strings/preprocessor stripped); cyclomatic = 1 + if/for/while/
case/catch + `&&`/`||` + ternary. So formula complexity that lives in `.cu` is
**no longer free** — moving a mechanism into CUDA counts against the gate just
like Python. (The off-limits infra C++ — `fsrs_extension.*`, `fsrs_kernel/` —
is *not* scored: it can't be mutated, so it would only add a fixed offset.)

The scored set is wired in `src/main/run.py` (`mutation_files`, passed to
`score_paths`) and must stay in sync with the mutation-surface list above. It
includes the `src/main/fsrs/` helpers, optimizer, scheduler, and the two CUDA
model files so a mutation can't dodge the gate by living in an unscored file.
**Current champion complexity baseline: 17,631** (iter-165 model; re-baselined
2026-06-10: the committed off-by-default tuning tooling — `FSRS_N_USERS` subset,
`FSRS_PARAM_FILE`, `FSRS_VRAM_GB`, cache-RO plumbing in `run.py`/`config.py` —
added +693 over iter-165's 16,938; tooling re-baselines rather than counting
against a research iteration, per the `compute_seconds` precedent). The +5%
gate is measured against this current baseline — update this number in
lock-step with any accepted variant. The full lineage of how the score moved
from 16,766 (when C++ scoring was added) to here is archived in
`docs/architecture_history.md`.

### Pre-submission checklist (verify silently before writing the patch)

- [ ] Forgetting curve monotonic in `delta_t` across the full clamp range?
- [ ] If a clamp lower bound was moved into ≤ 0 territory (i.e. the param
      is now allowed to be zero or negative), trace every formula that
      consumes `w[i]` and confirm it still does the right thing for those
      values. `torch.pow(base, -w[i])` blows up when `w[i]` flips sign,
      `s ** -w[33]` reverses curvature, `exp(w[7] - 1.5)` is fine but
      `log(w[i])` would die, etc. Don't hand-wave this — read each call site.
- [ ] If a bound change lets a trainable base/exponent reach a region where a
      `powf`/`exp` can **overflow float** (e.g. `base^(1/decay)` with small base
      *and* small decay), clamping the *result* (`fminf` after the `powf`) is NOT
      enough under Enzyme: it differentiates the pre-clamp expression whose
      derivative also overflows, and `inf*(clamp'=0)=NaN` poisons the **backward**
      pass — surfacing as a CUDA `input_val>=zero && input_val<=one` assert in the
      BCE eval, not a forward NaN. Compute it in **log-space and clamp the exponent
      before `exp()`** so value AND gradient stay finite (iter-50: base1 floor
      0.5→0.2 needed `exp(min((1/decay1)·ln base1, 60)) - 1`).
- [ ] `w[0..3]` still ordered after any init/clamp changes?
- [ ] `stability_after_review` monotonic in rating?
- [ ] Higher `D` still slows `S` growth?
- [ ] Forgetting curve still pinned at `p(0)=1` and `p(∞)→0` (NO learned
      floor/ceiling), monotonic in `delta_t`, and `p ∈ [0,1]`?
- [ ] `N_EPOCHS` untouched?
- [ ] Review filtering / splits / preprocessing unchanged?
- [ ] All new stochastic ops seeded?
- [ ] No eval→train leakage?
- [ ] Threshold computed correctly?
- [ ] All new trainable params: initialized in `init_w`, clamped in
      `FSRS7ParameterClipper`, given an L2 sigma in `batch_process`?
- [ ] If params were added / removed / reordered: did you update
      `FSRS7_BOUNDS_STATIC` in `src/autoresearch/diagnostics.py` AND the
      param table + role description in this CLAUDE.md? Both stay in sync
      with `FSRS7ParameterClipper` or the diagnostic report goes wrong.
- [ ] Complexity score within budget?

### Diagnostics report (shown to Claude each iteration)

After every training run the loop emits a Markdown report with this shape
(produced by `src/autoresearch/diagnostics.py::format_markdown_report` and
the JSON twin from `build_diagnostics_dict`):

```
Diagnostics:
    Code complexity score (AST_node_count + 40 * cyclomatic_complexity_score): __
    Log loss, per-user (primary):  __
    Log loss, per-review (sanity): __
    Log loss on Again (rating=1) only: __
    Log loss on Hard  (rating=2) only: __
    Log loss on Good  (rating=3) only: __
    Log loss on Easy  (rating=4) only: __
    Log loss on delta_t <  1 day reviews: __
    Log loss on delta_t >= 1 day reviews: __

    Per-parameter diagnostics
    w[i]:
        bounds: (lower, upper)              # effective per-user bound; lower
                                            # may be "w[j]" for chained clamps
                                            # (w[1..3] >= w[i-1], w[28]>=w[27],
                                            # w[30]>=w[29])
        p01 / median / p99:  __ / __ / __   # over all (user, split) rows
        Hit lower bound on __% of users
        Hit upper bound on __% of users
        (p99-p01)/(upper-lower): __          # range utilization, see below
        Mean gradient across all epochs: __
        Mean gradient at the last epoch: __
```

**Range utilization `(p99 − p01) / (upper_bound − lower_bound)`** (the
`range_frac` field; column `(p99-p01)/rng` in the Markdown table). The
numerator is the spread of the central 98% of the population; the denominator
is the param's static clamp width (the per-user effective bounds — uppers are
constant per column, chained lowers like `w[1..3]`, `w[28]`, `w[30]` resolve to
their realized population floor). It answers "how much of the allowed range
does the population actually use?":
- **`frac → 0`** — the population is pinned into a sliver of its range. Either
  the **bound is far wider than needed**, or (more often here) the **L2 prior
  is pinning the param** to the population mean / its default. A near-zero
  `frac` *and* a near-zero gradient ⇒ the param is **inert** — a prime
  candidate for an ablation iteration (see "Removing parameters", below).
- **`frac → 1`** (or hit-bound % high) — the population **fills or strains the
  clamp**; the bound may be **too tight** and worth widening.
- **mid `frac`** — the param is doing genuine per-user work across a healthy
  span; leave it.
Read it alongside the hit-bound % and gradient columns, not alone (e.g. a wide
`frac` with ~0 median just means the param is used in both directions).

Last two gradient rows require the `GradTracker` patch into `train_iter`
described in `diagnostics.py::GradTracker` (returning `flat_grad` from
`train_iter` and calling `tracker.observe(...)` in the outer `train`
loop). Until that patch is wired in, those rows render as `—`.

The Markdown report is what the next loop iteration feeds back into
Claude's context. The corresponding JSON is what the loop driver parses
to evaluate the accept/reject threshold and complexity gate.

### Don't lump unrelated ideas

**Each proposal must contain ONE idea — or at most TWO ideas that are VERY
closely related (user directive 2026-06-10).** "Change the difficulty formula"
and "change the forgetting curve" are two proposals, not one. "Replace Adam
with NAdam" and "add cosine annealing" are two proposals. In particular,
**bundling several individually-mapped sub-threshold changes into one
"composite" iteration so they clear the bar together is NOT allowed**, even
when each component's solo delta is already attributed in history — tried
once (2026-06-10: a 4-shaving composite measured +1.6e-4, enough to clear
the bar) and still vetoed on methodology, unrecorded per the user's call.
The hp_tune multi-knob
pass stays sanctioned: it tunes numeric hyperparameters only, never bundles
formula changes. Each iteration tests **one** change against the threshold
+ complexity gate so we can attribute the delta cleanly.

### Automated hyperparameter tuning

> **Optimizer/training-loop changes must always be accompanied by an hp_tune
> pass (user directive 2026-06-10).** Any iteration that changes the optimizer
> or training loop (update rule, LR schedule, batching, gradient weighting,
> loss shaping, …) is evaluated **with** hyperparameters re-tuned for it: run
> `hp_tune.py --no-commit` on top of the uncommitted change and judge the
> **combined** best config vs the champion against the iteration's threshold.
> Rationale: such changes move the loss landscape, and HPs tuned for the old
> regime can mask a genuine win (iter-183: batch decorrelation alone was
> +7.0e-5 — sub-threshold at champion HPs tuned for correlated-chunk noise).
> This bundling counts as **one** idea, not a lumping violation: tuning numeric
> HPs for a new training regime is part of the change. On accept, the pass
> also counts as the cadence hp_tune (update `result/.last_hptune_iter`).

Tuning numeric *training* hyperparameters (LR, Adam betas, L2 strength, recency
weighting) by hand is mechanical, so it's automated in
`src/autoresearch/hp_tune.py`. It runs a greedy **coordinate-descent** search —
steps each of the 6 *fine* knobs (LR, PENALTY_W_L2, BETA1, BETA2, RECENCY_C0,
RECENCY_EXP) up/down (multiplicative for LR/L2, on the `1−β` scale for betas),
keeps improvements, freezes knobs that don't help — and is **fully autonomous**:
when the best config beats the champion by ≥ 0.0001 it commits the new champion
(constants + diagnostics + history) and tags `iter-N-hp-tune`; otherwise it
restores the champion and records a rejected pass. Editing numeric literals in
place never changes the AST node count, so these tweaks are complexity-neutral —
only the 0.0001 floor applies, no complexity gate. Training is deterministic
(`seed` fixed) so every delta is real.

**The compute operating point (`n_epoch`, `batch_size`) is NOT in that pass** —
it's a speed/log-loss trade-off, not a pure-loss knob (a loss-only search always
drives batch *down*, ignoring the speed cost), so `batch_size` was pulled out of
the coordinate descent and both live in a separate **epoch×batch Pareto grid**
(`--epoch-batch-grid`, added 2026-06-03). The grid sweeps the 20 combos of
`n_epoch ∈ {5,8,12,16,30}` × `batch_size ∈ {128,256,512,1024}`, measuring each
with an Adam **√ LR-batch-scaled** learning rate (fairness — the winner is
fine-tuned precisely afterward), and re-anchors to any cell that **Pareto-
dominates the gold standard `(8,512)`**: no worse on log loss OR on `compute_seconds`
(the train+eval wall time `run.py` now emits into diagnostics.json), strictly
better on ≥1. Pick rule: lowest log loss among cells **not slower** than gold
(tiebreak: fastest). It leaves the winner's `(batch, n_epoch, scaled-LR)` on disk
**uncommitted** for review and writes `result/epoch_batch_grid.json` — re-anchoring
the champion is a deliberate step you finalise (run the regular pass to fine-tune
LR/betas/L2 at the new point, then record the combined re-anchor as one iteration).
This is an **outer / rare** recalibration — run it once to set the operating point,
then again only after a major architectural change; it is **not** part of the
every-5-iter cadence. Because n_epoch is driven per-cell via the `FSRS_N_EPOCHS`
env var, this is the sanctioned exception to hard-constraint #5.

Run it from the **host** (it shells out to `docker compose` per trial and to
`git`); a fine pass is ~10–16 ~70 s runs (~15–20 min), the grid is ~20 runs
(~30–50 min, the high-epoch cells are slow), so launch it in the background:

```pwsh
python -m src.autoresearch.hp_tune                    # full auto pass (6 fine knobs)
python -m src.autoresearch.hp_tune --epoch-batch-grid # re-anchor n_epoch × batch (no commit)
python src/autoresearch/hp_tune.py --dry-run          # validate edits only, no GPU/git
python src/autoresearch/hp_tune.py --no-commit        # search only, leave best on disk
```

**Cadence:** run it every ~5 iterations. It self-maintains
`result/.last_hptune_iter`; trigger when `latest_iter − last_hptune_iter ≥ 5`.
Re-tuning matters most right after a structural change shifts the loss landscape
and the old optima drift. It requires a clean git tree (refuses to auto-commit
otherwise) and owns one iteration number per pass. The multi-knob pass is one
logical unit, not a "lumping" violation — per-knob deltas are in the trial log
(`result/hp_tune_last.json`).

**Default-parameter meta-opt (`central_diff_init_w.py`) — separate from hp_tune, and
SLOW.** This re-optimizes the user-facing defaults `FSRS7_DEFAULT_35_VALUES` (which are
also the per-user init **and** the L2 anchor) for the 0-epoch `default` loss via ~50
Adam + central-difference steps (**~hours** on the host; see iter-140). Re-run it after a
*sequence* of structural changes shifts the optimal defaults — iter-140 recovered +1.62e-4
this way once the defaults had drifted since iter-67 — but **never more often than every 5
iterations** (user directive 2026-06-06; it's expensive). It's an **outer/rare**
recalibration like the epoch×batch grid, NOT part of the per-iteration loop. Run:
`python -m src.autoresearch.central_diff_init_w` (host; writes `best_params` to
`result/init_w_metaopt/`, restores the champion `constants.py` on exit — wiring the tuned
defaults in is a deliberate, human-reviewed step).

### Compaction cadence

`/compact` is human-typed (it is **not** a tool Claude can call), so the loop
uses **smart-but-manual** compaction: when it comes due, surface a dense,
ready-to-paste `/compact <focus>` block (see the `fsrs-autoresearch-compact-check`
skill) and keep working — don't block on the paste.

**Compaction is due every 8 iterations** (changed from 5 on 2026-06-04): when
`latest_iter − last_compact_iter ≥ 8`. It is keyed to a raw iteration count, **not**
slaved to the auto-tuner. Markers (both gitignored, machine-local):
`result/.last_compact_iter` and `result/.last_hptune_iter`. After a compaction,
set `echo <latest_iter> > result/.last_compact_iter`.

**Auto-tune is a conditional pre-step, not a gate.** hp_tune keeps its own ~5-iter
cadence (above). When a compaction comes due, if a tune is *also* due
(`latest_iter − last_hptune_iter ≥ 5`) run it **first** so the compacted context
captures the freshly-tuned champion; otherwise compact directly. The two cadences
(compaction every 8, tuning every ~5) drift in and out of phase — that's expected.

### Iteration cost

One full training run = **~102 seconds wall** on the RTX 4070 with 3000
users and 152M reviews (training ~51 s, the rest is build-ext check,
tensor-cache load, and eval). That's ~35 iterations / hour, so we can
afford a fairly hungry exploration policy.

### Baseline (champion to beat)

| Metric | Value |
|---|---|
| `logloss_by_user` (primary) | **0.32498** |
| `logloss_by_review` (sanity) | 0.33844 |
| Complexity score | **18,563** (original python-only scoring) |

This is the unmodified FSRS-7 with default `init_w` on anki-revlogs-3k,
`--short --secs --recency`, 8 epochs. Every accepted variant must beat
this on `logloss_by_user` by at least the threshold defined above, and
keep the complexity within +5% of the *current champion's* score (not
the original baseline). **Note:** the 18,563 here is the *original*
python-only figure; complexity scoring now also counts the CUDA model
files, so the current champion baseline is **16,766** on the new scale
(see "Code complexity gate"). The `logloss` figures are unaffected.

### Per-run artifacts

Each `bash src/main/run.sh` writes:

* `result/diagnostics.json` — machine-readable: log loss, per-rating /
  per-delta_t breakdowns, per-parameter distribution + bound-hit %,
  complexity score, gradient stats (when `GradTracker` is wired in).
  Overwritten every run.
* `result/diagnostics.md` — same content as Markdown for human / Claude
  reading.

### Iteration history

* `result/history.jsonl` — append-only, one JSON object per iteration
  (proposal evaluated against the prior champion). Source of truth.
* `result/history.md` — regenerated from the JSONL each append. **Human
  view shown to Claude on the next iteration.** Never edit by hand.
* `result/history_plot.png` — log-loss-vs-iteration plot (champions + rejected
  variants), regenerated from the JSONL by `src/autoresearch/plot_history.py`
  after **every** iteration and committed (the README shows it via a relative
  path, so it auto-updates on GitHub). `hp_tune.py` refreshes it automatically
  on its commits; when recording a *manual* iteration, do the same — run
  `python src/autoresearch/plot_history.py`, then `git add result/history_plot.png`.
  (`plot_history.py` is the human's — **run** it, never edit it.)

The loop driver calls
`src.autoresearch.history.append_iteration(record)` after each variant
runs; record schema is in the `history.py` docstring. The baseline is
iteration 0 with `status="champion"`.

**Pre-register the `summary`, comment after.** Write the record's `summary` —
**just the change you made** (what was modified), **in ≤20 words**, with nothing
about expected effect — **BEFORE running the benchmark**, as a pre-registration;
never rewrite it afterward to fit the result. Everything else — *why* you expected it to help,
and *how it actually went* (cleared or missed the threshold, why, what to try
next) — goes in the `comment` field, written **AFTER** you see the result. So:
`summary` = the factual change you committed to up front; `comment` = the
rationale and the retrospective. This keeps the history honest — a pre-registered
change-log, not post-hoc storytelling.

## Things to be careful about

- **The user (Andrew) edits repo files directly — and those edits must be
  preserved and committed, never reverted (user directive 2026-06-04).** Andrew
  sometimes hand-edits files himself (e.g. `README.md`, `CLAUDE.md`,
  `src/autoresearch/plot_history.py`). When such an edit shows up as an
  uncommitted change in `git status`, it is **intentional**: do **not** revert,
  `git checkout`, stash, or overwrite it on your own initiative, and do **not**
  leave it dangling/excluded indefinitely — **commit it** (a dedicated commit is
  fine, or fold it into related work) so the tree stays clean for `hp_tune`.
  When reverting a *rejected iteration*, still list your own mutation files
  explicitly (never a blanket `git checkout -- .`) so you never clobber one of
  his edits. The one editing restriction that remains: don't *author* changes to
  `plot_history.py` yourself unless he explicitly asks (see its "Hands off"
  note) — but his edits to it still get committed, not reverted.
- You can modify src/main/csrc/fsrs, just not other files in src/main/csrc. Don't worry about wasting a few more minutes if it means big log loss wins
- `WRITE_RESULT = False` in `src/main/config.py:20` by default — flip to
  True only when you actually want per-user output (slow).
- `compose.yaml:13` mounts `../anki-revlogs-3k` read-only — never write
  there.
- The Enzyme extension is built in-place by `setup.py` on each `run.sh`;
  the timestamp gate decides whether to rebuild.
- `pyproject.toml` requires Python 3.12. Don't downgrade.
