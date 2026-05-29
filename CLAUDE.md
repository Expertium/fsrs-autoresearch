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
    config.py               # DEVICE, BATCH_SIZE=1024, N_EPOCHS=8, WRITE_RESULT
    run.py                  # training + eval entry point
    run.sh                  # builds Enzyme ext then runs run.py
    fsrs/                   # CUDA-accelerated kernel
    csrc/                   # C++/.cu sources (Enzyme extension)
      fsrs_extension.cpp
      fsrs_extension.cu
      fsrs/, fsrs_kernel/
    srs_ops.py              # Python wrapper around the extension
    tensor_cache.py         # LMDB tensor cache
    tensor_lmdb.py
    tensors.py
    result_metrics.py
  __init__.py
compose.yaml                # bind mounts ../anki-revlogs-3k and current dir
dockerfile                  # CUDA 11.8 + LLVM 18 + Enzyme + PyTorch 2.7.1
pyproject.toml              # Python 3.12, fsrs-optimizer, torch, etc.
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
| `w[7..15]` | Long-term stability update |
| `w[16..24]` | Short-term stability update |
| `w[25..26]` | Long/short transition function |
| `w[27..34]` | 8-param forgetting curve |

Training: 8 epochs, batch 1024, Adam, lr 2e-2, betas (0.8, 0.85).
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
- `src/main/run.py` — training loop (`train_iter`, optimizer setup)
- `src/main/config.py` — only `BATCH_SIZE`, never `N_EPOCHS`
- `src/main/csrc/fsrs/*` — CUDA forward/backward source. In bounds, but
  touching it triggers a multi-minute Enzyme rebuild. Other paths under
  `src/main/csrc/` (`fsrs_extension.cpp`, `fsrs_extension.cu`,
  `fsrs_kernel/`) remain out of bounds.

**Sync note for clamps + init_w:** `fsrs_v7_constants.FSRS_MIN_VALUES` /
`FSRS_MAX_VALUES` / `FSRS7_DEFAULT_35_VALUES` are used by the CUDA training
path; `fsrs_v7.py`'s `FSRS7ParameterClipper` and `init_w` are used by the
Python pretrain. Any variant that changes one must change the other (and
update `FSRS7_BOUNDS_STATIC` in `src/autoresearch/diagnostics.py`).

### Allowed change types

1. Add/remove parameters/constants in FSRS formulas; modify clamp ranges,
   default values, and the L2-penalty sigmas.
2. Add new formulas or new state variables (currently 2: `S` = memory
   stability, `D` = difficulty).
3. Modify the training loop: learning rate, betas, lr scheduler, optimizer
   choice (Adam → AdamW/SGD/etc.), recency weighting — **except `n_epoch`**.

### Hard constraints (every variant MUST satisfy all)

1. Forgetting curve is monotonic in `delta_t` for **any** parameter values
   within the allowed clamp ranges.
2. `w[0] <= w[1] <= w[2] <= w[3]` — initial stabilities ordered by rating:
   S0(Again) ≤ S0(Hard) ≤ S0(Good) ≤ S0(Easy).
3. `stability_after_review(rating=1) <= ... <= stability_after_review(rating=4)`
   — same ordering after a review.
4. Higher `D` ⇒ slower growth of `S`. Difficulty must not let memory grow
   faster.
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

Examples:
- Add 1 state variable + 3 new params: `0.0010 + 3*0.0002 = 0.0016`
- Change a formula with no new params: `0.0001`
- Change optimizer with no new params: `0.0001`

### Code complexity gate

`score = AST_node_count + 40 * cyclomatic_complexity`, computed over the
mutation surface files listed above (originally autoresearcher.md scoped this
to a single `train.py`; we scope it to the same set of files we allow
mutating). Each accepted variant must increase the score by **≤5%**, ideally
≤2%. Pick conservative implementations — auto-rejection on complexity is
common if you bolt on a new mechanism without simplifying anything else.

### Pre-submission checklist (verify silently before writing the patch)

- [ ] Forgetting curve monotonic in `delta_t` across the full clamp range?
- [ ] If a clamp lower bound was moved into ≤ 0 territory (i.e. the param
      is now allowed to be zero or negative), trace every formula that
      consumes `w[i]` and confirm it still does the right thing for those
      values. `torch.pow(base, -w[i])` blows up when `w[i]` flips sign,
      `s ** -w[33]` reverses curvature, `exp(w[7] - 1.5)` is fine but
      `log(w[i])` would die, etc. Don't hand-wave this — read each call site.
- [ ] `w[0..3]` still ordered after any init/clamp changes?
- [ ] `stability_after_review` monotonic in rating?
- [ ] Higher `D` still slows `S` growth?
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
        Mean gradient across all epochs: __
        Mean gradient at the last epoch: __
```

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
| Complexity score | **18,563** |

This is the unmodified FSRS-7 with default `init_w` on anki-revlogs-3k,
`--short --secs --recency`, 8 epochs. Every accepted variant must beat
this on `logloss_by_user` by at least the threshold defined above, and
keep the complexity within +5% of the *current champion's* score (not
the original baseline).

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

The loop driver calls
`src.autoresearch.history.append_iteration(record)` after each variant
runs; record schema is in the `history.py` docstring. The baseline is
iteration 0 with `status="champion"`.

## Things to be careful about

- You can modify src/main/csrc/fsrs, just not other files in src/main/csrc. Don't worry about wasting a few more minutes if it means big log loss wins
- `WRITE_RESULT = False` in `src/main/config.py:20` by default — flip to
  True only when you actually want per-user output (slow).
- `compose.yaml:13` mounts `../anki-revlogs-3k` read-only — never write
  there.
- The Enzyme extension is built in-place by `setup.py` on each `run.sh`;
  the timestamp gate decides whether to rebuild.
- `pyproject.toml` requires Python 3.12. Don't downgrade.
