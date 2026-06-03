# fsrs-autoresearch — Claude Code project context

## What this repo is

GPU-accelerated benchmark + research playground for **FSRS** (Free Spaced
Repetition System). Forked from [open-spaced-repetition/fsrs-gpu-benchmark].
End goal: an autoresearch loop (AlphaEvolve / Karpathy style) that proposes
FSRS model variants, trains them on real Anki review data, scores them, and
keeps the winners.

**Ignore `autoresearcher.md`** — it's an older plan written for a separate
LLM-API-driven loop. We're rebuilding it inside Claude Code.

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
  features/                 # feature engineers (DASH, FSRS-eng, etc.)
  models/                   # ★ where new variants go
    base.py
    trainable.py
    fsrs.py                 # FSRS shared base
    fsrs_v1.py … fsrs_v7.py
    fsrs_v7_interval_penalty.py
    model_factory.py        # MODEL_REGISTRY — register new variants here
  main/
    config.py               # DEVICE, BATCH_SIZE=512, N_EPOCHS=8 (FSRS_N_EPOCHS env override), WRITE_RESULT
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
      fsrs_kernel/
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

35-parameter PyTorch model in `src/models/fsrs_v7.py:74`. **Snapshot of
the current baseline — not the source of truth.** The clipper in
`FSRS7ParameterClipper` (same file) is authoritative for both shape and
bounds; `FSRS7_BOUNDS_STATIC` in `src/autoresearch/diagnostics.py` is the
mirror the diagnostics report reads. Any patch that adds, removes, or
reorders parameters must update **all three** in lock-step (model class,
clipper, diagnostics table, and this section).

| Param range | Role |
|---|---|
| `w[0..3]` | Initial stability per rating (Again/Hard/Good/Easy) |
| `w[4..6]` | Difficulty |
| `w[7..15]` | Long-term stability update — drives the **slow** trace `s` |
| `w[16..24]` | Short-term stability update — drives the **fast** trace `s_fast` |
| `w[25..32]` | 8-param forgetting curve (fast component keyed to `s_fast`, slow to `s`) |
| `w[33..35]` | Difficulty/stability modulation of the curve (d_weight, d_decay, s_decay1) |

**Champion = iter-43 (dual-trace memory), 36 params.** iter-40 added a 2nd
stability state `s_fast` (fast trace) alongside `s` (slow trace); each drives
its own forgetting-curve component and updates via the short-/long-term
stability dynamics respectively. This **removed the old long/short transition
blend** (its 2 params, formerly `w[25..26]`, were deleted in iter-43 — that's
why the curve params shifted down by 2). The fast trace's initial stability is
**hardcoded** at `0.8 * initial_stability` (a per-user multiplier didn't earn
its param). State variables: **3** (`s`, `d`, `s_fast`).

Training: 8 epochs, batch 1024, Adam, lr 3e-2, betas (0.55, 0.85).
Loss = log-loss(per-review) + sched_penalty_1 + sched_penalty_2 + L2-to-defaults.

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

1. Copy `src/models/fsrs_v7.py` → `src/models/fsrs_vX.py`. Rename the class.
2. Modify the parts you want to mutate (forgetting curve, stability update,
   transition function, clipper bounds). Keep the `init_w` shape unless you
   also update the clipper.
3. Register in `src/models/model_factory.py:8`:
   ```python
   MODEL_REGISTRY: dict[ModelName, Any] = {
       ...
       "FSRS-7": FSRS7,
       "FSRS-vX": FSRSvX,  # new
   }
   ```
4. Add the new name to `ModelName` in `src/prepare/prepare_config.py:11`.
5. Re-run training: `docker compose run --rm srs-benchmark bash src/main/run.sh`
   (no Docker rebuild needed — Python-only changes don't trigger Enzyme rebuild;
   only changes to files under `src/main/csrc/` do).

## Autoresearch loop

AlphaEvolve/Karpathy-style: Claude reads the current champion, proposes a
mutation, writes the patch, runs the benchmark, accepts or rejects against a
threshold + complexity gate, and keeps an archive of winners.

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
- `src/models/fsrs_v7.py` (or a new `fsrs_vX.py`) — model forward / clipper
- `src/models/fsrs_v7_interval_penalty.py` — scheduling penalty
- `src/main/fsrs/fsrs_v7_constants.py` — `LR`, `BETAS`, `RECENCY_C0/C1`,
  `PENALTY_W_L2`, `FSRS7_DEFAULT_35_VALUES` (init_w), `FSRS_MIN_VALUES`
  / `FSRS_MAX_VALUES` (CUDA-path clamps), `FSRS7_L2_SIGMA_35_VALUES`
- `src/main/run.py` — training loop (`train_iter`, optimizer/scheduler wiring)
- `src/main/fsrs/fsrs_v7_helpers.py` — L2 penalty (`penalty_loss`), recency
  weighting (`gradient_weight`), parameter clipper (`apply_parameter_clipper`)
- `src/main/fsrs/fsrs_v7_optimizer.py` — the Adam/AdamW/NAdam update rule
- `src/main/fsrs/fsrs_v7_scheduler.py` — LR schedule (cosine decay)
- `src/main/config.py` — only `BATCH_SIZE`, never `N_EPOCHS`
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
`FSRS_MAX_VALUES` / `FSRS7_DEFAULT_35_VALUES` are used by the CUDA training
path; `fsrs_v7.py`'s `FSRS7ParameterClipper` and `init_w` are used by the
Python pretrain. Any variant that changes one must change the other (and
update `FSRS7_BOUNDS_STATIC` in `src/autoresearch/diagnostics.py`).

### Allowed change types

1. Add/remove parameters/constants in FSRS formulas; modify clamp ranges,
   default values, and the L2-penalty sigmas.
2. Add new formulas or new state variables (currently **3**: `s` = slow memory
   stability, `d` = difficulty, `s_fast` = fast memory stability [dual-trace,
   iter-43]).
3. Modify the training loop: learning rate, betas, lr scheduler, optimizer
   choice (Adam → AdamW/SGD/etc.), recency weighting — **except `n_epoch`**.

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
   stability rise either. In the failure formula `new_s_fail ∝ d^(-fail_d_exp)`,
   this means `fail_d_exp >= 0` is a hard structural requirement
   (`w[11]`, `w[20]`).
5. **Do not change `N_EPOCHS`** (`src/main/config.py:27`). Don't rename it
   either. We're chasing architectural wins, not brute-force epochs.
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

> **A constraint I should have stated but didn't — my oversight (noted 2026-06-01).**
> *Don't do things that exploit the way the CUDA code jointly optimizes all 3000
> users at once.* This harness trains every user **jointly**, but the production
> optimizer (FSRS-rs) trains **one user at a time**, so a "win" that only exists
> because of the joint / cross-user structure does **not** transfer to real users —
> a change should improve **single-user** optimization, not just the joint
> 3000-user fit. The one accepted change that violates this is the **empirical-Bayes
> L2 (iter 24)**: it shrinks each `(user, split)` row toward the *live population
> mean* (`anchor_p = flat_fsrs_params.mean(0)` in `run.py`), which for a single user
> is a no-op (the mean is the user itself) and degrades to the pre-iter-24
> fixed-default L2. Its effect is only ~0.0001 log loss and it reduces harmlessly to
> per-user L2-to-default, so the FSRS-rs port simply keeps that fixed-default L2.

### Acceptance threshold

A variant is accepted only if `old_logloss_by_user − new_logloss_by_user ≥ threshold`
(the per-user primary metric, not per-review), where:

| Change | Contribution to threshold |
|---|---|
| Each new state variable | +0.0010 |
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
**Current champion complexity baseline: 16,766.** (Re-baselined when C++ scoring
was added: the 36-param dual-trace champion was 15,322 python-only; the two CUDA
files add 1,436 — `fsrs7.cu` 1,230, `fsrs7.cuh` 206 — and wiring them into
`mutation_files` added 8 to `run.py`, for 16,766. A measurement change, not a
model change; log loss is unaffected. The +5% gate is measured against this
baseline.)

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

"Change the difficulty formula" and "change the forgetting curve" are two
proposals, not one. "Replace Adam with NAdam" and "add cosine annealing" are
two proposals. Each iteration tests **one** change against the threshold
+ complexity gate so we can attribute the delta cleanly.

### Automated hyperparameter tuning

Tuning numeric *training* hyperparameters (LR, Adam betas, L2 strength) by hand
is mechanical, so it's automated in `src/autoresearch/hp_tune.py`. It runs a
greedy **coordinate-descent** search — steps each knob up/down (multiplicative
for LR/L2, on the `1−β` scale for betas), keeps improvements, freezes knobs that
don't help — and is **fully autonomous**: when the best config beats the champion
by ≥ 0.0001 it commits the new champion (constants + diagnostics + history) and
tags `iter-N-hp-tune`; otherwise it restores the champion and records a rejected
pass. Editing numeric literals in place never changes the AST node count, so
these tweaks are complexity-neutral — only the 0.0001 floor applies, no
complexity gate. Training is deterministic (`seed` fixed) so every delta is real.

Run it from the **host** (it shells out to `docker compose` per trial and to
`git`); a pass is ~10–16 ~70 s runs (~15–20 min), so launch it in the background:

```pwsh
python -m src.autoresearch.hp_tune             # full auto pass (search + commit)
python src/autoresearch/hp_tune.py --dry-run   # validate edits only, no GPU/git
python src/autoresearch/hp_tune.py --no-commit # search only, leave best on disk
```

**Cadence:** run it every ~5 iterations. It self-maintains
`result/.last_hptune_iter`; trigger when `latest_iter − last_hptune_iter ≥ 5`.
Re-tuning matters most right after a structural change shifts the loss landscape
and the old optima drift. It requires a clean git tree (refuses to auto-commit
otherwise) and owns one iteration number per pass. The multi-knob pass is one
logical unit, not a "lumping" violation — per-knob deltas are in the trial log
(`result/hp_tune_last.json`).

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
**just the change you made** (what was modified), with nothing about expected
effect — **BEFORE running the benchmark**, as a pre-registration; never rewrite
it afterward to fit the result. Everything else — *why* you expected it to help,
and *how it actually went* (cleared or missed the threshold, why, what to try
next) — goes in the `comment` field, written **AFTER** you see the result. So:
`summary` = the factual change you committed to up front; `comment` = the
rationale and the retrospective. This keeps the history honest — a pre-registered
change-log, not post-hoc storytelling.

## Things to be careful about

- You can modify src/main/csrc/fsrs, just not other files in src/main/csrc. Don't worry about wasting a few more minutes if it means big log loss wins
- `WRITE_RESULT = False` in `src/main/config.py:20` by default — flip to
  True only when you actually want per-user output (slow).
- `compose.yaml:13` mounts `../anki-revlogs-3k` read-only — never write
  there.
- The Enzyme extension is built in-place by `setup.py` on each `run.sh`;
  the timestamp gate decides whether to rebuild.
- `pyproject.toml` requires Python 3.12. Don't downgrade.
