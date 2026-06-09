# FSRS-7 architecture history (archival)

This file holds the long-form, per-iteration narrative that used to live in
`CLAUDE.md` (moved 2026-06-09 to keep the per-session context lean). Nothing
here is operational: the live operational summary (current champion, param
table, constraints, gates, workflow) stays in `CLAUDE.md`, and
`result/history.jsonl` remains the per-iteration source of truth. This is the
*story* — the mechanisms, the deltas, the rationale, and the reverts — kept so
the reasoning behind the current architecture isn't lost.

---

## Architecture evolution (iter-40 → iter-165)

**Dual-trace memory (introduced iter-43), 34 params; current champion iter-165.**

**(iter-165, 2026-06-07 — current champion):** the LONG forgetting component's
difficulty modulation moved from the decay EXPONENT to the curve's horizontal
TIME-SCALE. `r2` now sees `t * exp((d_decay-0.3)*(d-5))` with `decay2` itself
un-modulated — i.e. `decay2_mag = clamp(decay2, …)` and the d-factor multiplies
`t/s_long` inside `r2 = (1 + factor2 * (t/s_long) * exp((d_decay-0.3)*(d-5)))^(-decay2_mag)`
(was `decay2_mag = clamp(decay2 * exp((d_decay-0.3)*(d-5)), …)` modulating the
exponent). Same param `d_decay`, same (degenerate) pivot 5; the difference is that
hard cards now get a faster effective clock — a *level* shift — rather than a
steeper asymptotic power-law *slope*. by_user 0.31991476 → 0.31980351 (+1.11e-4 ≥
0.0001), by_review +1.82e-4; Easy −1.0e-3 / Good −3.5e-4 / long-term bucket −2.9e-4
better, Hard +4.9e-4 the cost. 0 new params (formula change), complexity 16932→16938.
The decoupling frees both params: `decay2` range_frac 0.64→0.99 (sets the slope),
`d_decay` grad 0.34→0.86 & upper-bound-hit 2.5%→5.6% (sets the D time-shift) — so a
`d_decay` bound-widen / hp_tune / default re-tune may extract more. (A prior probe
iter-164 that only changed the exponent form's curvature exp→linear was flat — the
win is the *channel*, not the curvature.)

**(iter-140, 2026-06-06):** re-tuned the user-facing default parameters
(`FSRS7_DEFAULT_35_VALUES`) via the `central_diff_init_w` 50-step 0-epoch meta-opt for
the current model — they had drifted out of sync since iter-67. Trained `logloss_by_user`
**0.32007685 → 0.31991476 (+1.62e-4)**, re-crossing the 0.31995 finishing line cleanly
(0 new params, 0 complexity change; the gain is the better fixed L2 anchor helping the
L2-dominated light users). Architecture unchanged from iter-105.

**(iter-40/43 — the dual trace):** iter-40 added a 2nd stability state `s_short`
(short trace) alongside `s_long` (long trace); each drives its own
forgetting-curve component and updates via the short-/long-term stability
dynamics respectively. This **removed the old long/short transition blend**
(its 2 params, formerly `w[25..26]`, were deleted in iter-43 — that's why the
curve params shifted down by 2). The short trace's initial stability is
**hardcoded** at `0.8 * initial_stability` (a per-user multiplier didn't earn
its param: iter-69 trainable near-miss, iter-70 confirmed 0.8 is the by_user
optimum).

**iter-71 (trace-specific short learning):** the short trace now updates from
the **short component's own recall `r1`** (the `fsrs7_short_component_recall`
helper), not the mixed-curve retention; the long trace still updates from the
mixed retention. State variables: **3** (`s_long`, `d`, `s_short`).

**(2026-06-05 cosmetic rename):** the two stability traces were `s` / `s_fast`
before; now `s_long` / `s_short`. Confined to `fsrs7.cu`/`fsrs7.cuh`; a pure
no-op.

**iter-85 (ablation):** removed the failure-path difficulty exponent
`fail_d_exp` — a single field in the *shared* stability sub-struct, so it was the
long `w[11]` **and** short `w[20]` at once. Post-lapse stability (`new_s_fail`) is
now **difficulty-independent** (the exponent was the most inert param in the model,
median ~0.005, a quarter of users pinned at its `>=0` floor). 36→34 params; the two
stability-update blocks are now **8 params each**.

**iter-97 (post-lapse relearning):** on a lapse (`rating==1`) the fast trace is
capped at `0.8 * post-lapse-slow-stability` (the same 0.8 init fraction) in
`fsrs7_step` — modeling relearning-from-scratch and restoring the
`fast <= 0.8*slow` invariant that `fsrs7_init` sets but that the short-term
`fail` dynamics had let drift above. 0 new params; D-independent (constraint 4)
and the `min` only lowers the lapse value (constraint 3). Improved by_user
+1.70e-4 (short_term bucket −0.00048, Again −0.00062, all classes better): a
systematic, population-wide post-lapse over-retention fix.

**iter-101 + iter-105 (surprise-weighted lapse difficulty):** on a lapse
(`rating==1`) the difficulty increment `delta_d` is scaled by `(1 + 1.0*(retention − 0.9))`
in `fsrs7_next_d` — a lapse on a card the model expected to recall (high `R`) raises `D`
more; an overdue lapse (low `R`, failure expected) raises it less. (iter-101 introduced this
at coefficient 0.5, +1.79e-4; iter-105 found 0.5 under-shot and raised it to **1.0**, +1.18e-4
more, a clean across-the-board win — surprise-weighting is *selective*, so more spread sharpens
D-discrimination without over-sticking D on recoverable cards.) 0 new trainable params
(the coefficient is hardcoded; factor ∈ [0.55, 1.05] always positive ⇒ lapse still raises `D`,
ordering preserved). Only reshapes the **D-update** — the D→S map is untouched (constr 4),
the current review's stability used the old `D` (constr 2/3), the curve is untouched
(1/11/12). Improved by_user **+1.79e-4** (Again −0.00067 dominant, Easy −0.00027, both
Δt buckets better; Hard +0.00006 tiny offset). **This breaks the surprise→D ceiling that
trainable-`d_surprise` iters 36/37/47/58 hit at ~1.5e-5: the trainable param was walled
(light users pin it at 0 → by_review only); fixing the coefficient forces the coupling on
ALL users → ~10× the by_user gain. Validates a retry pattern: revisit walled trainable-param
mechanisms as fixed-formula changes.**

---

## iter-135 / iter-138 — the sub-day forgetting drop (accepted, then user-reverted)

**iter-138 (continuous sub-day forgetting drop) — ACCEPTED, then REVERTED by the user
2026-06-05 (purely aesthetic: "it's ugly, that's the reason, lol"). The win was real and
constraint-compliant, but the user preferred the cleaner code; champion reverted to iter-105
(by_user 0.32007685), `fsrs7_forgetting_curve` restored byte-identical to iter-105, and the
<0.31995 finishing line was deliberately given up (later re-crossed by iter-140/165).
Mechanism kept below for the record.**

Originally crossed the 0.31995 finishing line (by_user 0.31991): in `fsrs7_forgetting_curve` the FAST component `r1` is
multiplied — **in the curve only**; the fast-trace *update* in `fsrs7_step` keeps the
un-discounted `r1`, so the dynamics are unchanged — by a continuous ramp
`g(delta_t) = 0.85 + 0.15·exp(−delta_t / 0.003)`. `g(0)=1` **exactly**, so the curve still
starts at `p(0)=1` and is continuous (constraints 11/12); over a ~4-min timescale `g` falls to
a 0.85 floor, so sub-day reviews beyond ~15 min get a ~15% discount on the fast component while
1-minute reviews get only a small discount (ramping from 1 — fresher memory). `g` is convex
decreasing, so `g·r1` is convex decreasing ((g·r1)″ = g″r1 + 2g′r1′ + g·r1″ > 0) and the
mixture stays convex/monotone with both endpoints intact. This is the **constraint-compliant
successor** to a discontinuous "jump" (`r1 *= 0.93` for all `t>0`, iter-135) that was tried and
**reverted by the user** because it dipped `p` below 1 as `delta_t→0⁺`; the continuous ramp is
both compliant AND **better** (+1.639e-4 vs the jump's +1.073e-4 — the flat jump over-penalised
the freshest reviews). 0 new trainable params (floor/τ hardcoded). by_user-concentrated gain
(light users are L2-pinned to the default and cannot fit the sub-day over-confidence away).
floor line-searched (0.93→+1.355e-4, 0.91→+1.476e-4, **0.85→+1.629e-4 peak**, 0.78→overshoot);
τ 0.003 beat 0.006. **Validates a general lever: the model is systematically over-confident on
sub-day reviews at the default parameterization, and a fixed, population-wide curve-shape
correction is a real by_user lever.**

---

## Empirical-Bayes L2 (iter-24, removed iter-75) — the joint-optimization exploit

*Don't do things that exploit the way the CUDA code jointly optimizes all 3000
users at once.* This harness trains every user **jointly**, but the production
optimizer (FSRS-rs) trains **one user at a time**, so a "win" that only exists
because of the joint / cross-user structure does **not** transfer to real users —
a change should improve **single-user** optimization, not just the joint
3000-user fit. The one accepted change that ever violated this — the
**empirical-Bayes L2 (iter 24)**, which shrank each `(user, split)` row toward
the *live population mean* (`anchor_p = flat_fsrs_params.mean(0)` in `run.py`),
a no-op for a single user — was **removed in iter-75** (user directive
2026-06-03). Its real cost turned out to be ~0.00029 log loss (≈3× the ~0.0001
originally estimated), all of it an illegitimate cross-user gain; the L2
penalty now shrinks toward the fixed `FSRS7_DEFAULT`, exactly what the FSRS-rs
port uses. **There are now no known joint-optimization exploits in the model.**

---

## Complexity-score lineage

Current champion complexity baseline: **16,938** (iter-165 — moved the
long-curve D-modulation from the decay exponent to the time-scale, +6 over
iter-140's 16,932. The 16,932 = iter-105 + the 2026-06-05 cosmetic refactor:
+2 over 16,930 for the three subtraction literals that shift
d_weight/d_decay/s_decay1 to non-negative bounds in
`fsrs7_forgetting_curve`/`fsrs7_short_component_recall`; the
`s`→`s_long`/`s_fast`→`s_short` rename is token-count-neutral. iter-138's
continuous sub-day ramp briefly pushed this to 16,964 but was REVERTED by the
user 2026-06-05 for being "ugly").

Full history: C++ scoring was added at 16,766 — the 36-param dual-trace
champion was 15,322 python-only, the two CUDA files add 1,436 (`fsrs7.cu`
1,230, `fsrs7.cuh` 206), and wiring them into `mutation_files` added 8 to
`run.py`. Numeric-literal hp-tunes / parsimony edits since then drifted it to
16,756; iter-71's `fsrs7_fast_component_recall` helper added +39 → 16,795;
iter-75 removed the empirical-Bayes anchor plumbing (+3 net → 16,798); the
2026-06-03 `compute_seconds` train+eval timer in `run.py::main()` — the speed
axis for the epoch×batch grid — added +29 → 16,798 → 16,827; iter-85 ablated
the two `fail_d_exp` params (removed the `d^(-fail_d_exp)` factor + struct
field + 8 constant-tuple literals) → **16,804**; iter-97 added the post-lapse
fast-trace reset in `fsrs7_step` (a `rating==1` ternary +1 cyclomatic·40 +
`fminf` tokens) → **16,865**; iter-101 added the surprise-weighted
lapse-difficulty branch in `fsrs7_next_d` (a `rating==1` `if` +1 cyclomatic·40
+ the `retention` arg/expr tokens) → **16,930**; iter-105 raised that branch's
coefficient 0.5→1.0 (literal swap, no token-count change) → **16,930**
(iter-105). iter-138 briefly added the continuous sub-day ramp (`subday_floor`
+ `subday_tau` constants + the `subday` ramp expression and the `* subday *`
factor in `fsrs7_forgetting_curve`) → 16,964, but the user REVERTED it
2026-06-05 (aesthetic — "it's ugly"), so the live baseline returned to
iter-105's 16,930, then the 2026-06-05 cosmetic non-negative-param refactor
nudged it to **16,932** (+2); iter-140 (2026-06-06) re-tuned the numeric
default literals with NO complexity change — **16,932**; iter-165 (2026-06-07)
moved the long-curve D-modulation from the decay exponent to the time-scale
(decay2 left un-modulated; the `exp((d_decay-0.3)*(d-5))` factor now
multiplies `t/s_long` inside r2) → **16,938** (+6), the current champion.
(iter-135's discontinuous jump also reached 16,938 and was reverted —
unrelated.)
