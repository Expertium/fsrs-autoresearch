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

## The post-194 closure campaign (iters 195–220, 2026-06-11/12)

After iter-194 (per-epoch batch reshuffle, the last accepted change) the loop
ran **26 consecutive rejections** — but this stretch is the campaign's most
information-dense product: it converted "we believe these axes are done" into
**closure by direct experiment** across the entire mutation surface.

**Training loop (195–214, exhaustive):** schedules (195), optimizers, gradient
clipping, loss shaping, and the flat-minima family — SWA (+6e-5, the best
training-loop delta ever, still sub-bar), EMA, Lookahead, and finally **SAM
(214, −1.2e-5)**, which closed the family and showed its doubled scatter_add
even reintroduces the ~1e-5 atomics noise. The **gated defaults meta-opt
(211, +1.9e-5)** optimized the cheap `--default` proxy by central differences
while selecting on `--recency`: the two metrics diverge after ~5 steps, so the
proxy is unusable — but the `--gated-recency` tooling is kept in
`central_diff_init_w.py` for re-use after the next structural change. The
**strain map (213)**: widening the one remaining strained bound (w[29] floor)
let pinned users take the freedom and *worsened* test loss — strained bounds
are load-bearing regularization, closing bound-widens as a family. **Dual
difficulty (212, 4th state d_short)** came out inert (w[34] unused), the third
4th-state failure. The split instrumentation (added 2026-06-11) confirmed the
**early-split scarcity** residual (209): split 0 carries a ~0.029
generalization gap vs ~0.017 for later splits — irreducible without more
early-history data.

**The S0-evidence family (215–218) — the one real signal found.** Per-user S0
estimates from each row's own `seq_len==2` card-level (first-rating,
interval ≥ 0.5 d, outcome) triples, via binned sufficient statistics + a
64-point log-S0 grid. The ladder of injection strengths tells the story:
raw MLE init **−4.6e-4** (215: ~20% of estimates are edge-censored and
*persist through SGD* — w[0..3] are unanchored and recency weighting never
revisits early history; this also explains the historical "S0 initialization
made FSRS-7 worse"); log-space shrinkage init (W=32) **+2.2e-5** (216);
per-row L2 anchor (σ = rel·anchor) **+7.9e-5** (217); joint W×rel grid
**+8.35e-5 at the interior optimum (W=16, rel=2.0)** (218, mean-of-3
bit-identical). Lessons: *continuous anchor pressure extracts ~3.5× more than
a one-shot init*, and *design quality swings the outcome by half a millipoint*
— but the mechanism's ceiling (~8e-5) sits below the 1e-4 training-pipeline
bar. The candidate is preserved (C:\Temp\run_iter217.py + helpers_iter217.py,
W=16/rel=2.0); accepting it under-bar is the user's call, and it would also
need a ~170-point complexity shave (candidate 21,065 vs gate).

**Formula phase (219–220) — the success gain closed end-to-end.** With the
2026-06-10 directive's new-params menu: a trainable retrieval-strength gate
`retention^(w[34]−0.5)` on the success gain (219) came out **dead flat
(−2.2e-6)** with the diagnostic signature *spread-without-signal* — median
pinned at neutral, healthy per-user spread (range_frac 0.41) and gradient,
zero net held-out gain = pure per-user noise-fitting. The shifted
S-saturation `(s_long + w[34])^−σ` (220, motivated by 20% of users pinning
w[8] at its 0 floor) was **refuted directly (−1.35e-5)**: the same fifth of
users pinned the *new* param at *its* 0 floor too — they want no
S-saturation at all, not a different shape. Together with iter-38 (trainable
(11−d) ceiling), 79 (concave R-response, −3.7e-4), 124 (easy_bonus floor) and
181 (w[22] ablation), every factor axis of
`s_inc = 1 + exp(base)·(11−d)·s^−σ·(exp((1−R)c)−1)·hp·eb` — base, D, S, R,
rating — has now been probed and rejected against its bar. A 2026-06-12 sweep
confirmed every remaining hardcoded constant in the model traces to an
experimentally closed family.

**Methodology lessons banked:** (1) grep history by *formula keyword*, not
param name — iter-210's mixture ablation was an unknowing third repeat of
46/62; (2) free reparameterizations are not free under L2+Adam (210); (3) the
*spread-without-signal* signature (median at neutral + real range_frac + zero
net) identifies noise-fitting dials in a single run; (4) strained bounds can
be load-bearing regularization (213); (5) a real, reproducible mechanism can
still be correctly rejected — the bars exist to price complexity, and the
~8e-5 S0 anchor is the cleanest example yet.

---

## Complexity-score lineage

Current champion complexity baseline: **19,904** (2026-06-12 — the dead
`fmaxf(pls, s*s_inc)` guard removed from `fsrs7.cu`'s success path, −5,
verified loss-identical; see the lineage tail below for the steps from
17,631 up through 19,909).

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
unrelated.) 2026-06-10 re-baseline: the committed off-by-default tuning
tooling from the joint default+sigma campaign (`FSRS_N_USERS` subset knob,
`FSRS_PARAM_FILE` per-eval override, `FSRS_VRAM_GB` cap, cache-RO plumbing in
`run.py`/`config.py`; commits 8cae55a/cc43307/f7b5511) added +693 →
**17,631**. Tooling re-baselines rather than
counting against a research iteration (the `compute_seconds` precedent).
iter-194 (2026-06-11, accepted) added the per-epoch batch-composition
reshuffle to `run.py` (+811 / +4.6%) → **18,442**; the 2026-06-11
user-requested train/test-gap + training-curve instrumentation in `run.py`
added +1,467 (re-baselined per the tooling precedent; the probe
builder/forward live in unscored `diagnostics.py`) → **19,909**; the
2026-06-12 dead-`fmaxf` chore in `fsrs7.cu` (verified mean-of-3 +1.6e-6 =
noise) removed −5 → **19,904**, the current champion baseline.
