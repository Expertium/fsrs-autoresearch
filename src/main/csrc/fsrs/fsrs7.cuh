#pragma once

#include <stdint.h>

struct fsrs_state_t {
    // DUAL-TRACE MEMORY (iter-40). Two persistent stability states instead of
    // one. `s_long` is the SLOW / consolidated trace (long-term memory);
    // `s_short` is the FAST trace (recent, short-term memory). The forgetting
    // curve mixes a recall component driven by each trace; after each review the
    // long trace updates via the long-term (consolidation) dynamics and the short
    // trace via the short-term dynamics. This replaces the single-S +
    // transition-blend approximation with an explicit two-store memory model,
    // attacking the Markovian (S,D) limitation that the residual short-term /
    // Again loss lives in. Adding a field to fsrs_state_t (NOT fsrs_params_t)
    // carries through the kernel + Enzyme autodiff generically and needs no
    // Python-side glue; param count stays 38. Scratch: peak 5.9M slots => 12
    // B/state = 71 MB of the 500 MB buffer (14%), measured iter-40, headroom ample.
    // (s_long / s_short were named s / s_fast before the 2026-06-05 cosmetic rename.)
    float s_long;
    float d;
    float s_short;
};

struct fsrs_stability_after_review_params_t {
    float sinc_base;
    float sinc_s_exp;
    float sinc_r_mult;
    float fail_mult;
    // iter-85 ABLATION: removed fail_d_exp (the d^(-fail_d_exp) factor on the
    // post-lapse stability). Diagnostics flagged it as the most inert param in
    // the model (long range_frac 0.14 / short 0.067, median ~0.005, 24%/20% of
    // users pinned at the >=0 floor that constraint 4 enforces). The factor
    // changed new_s_fail by <=1.2% even for the hardest cards. Post-lapse
    // stability is now D-INDEPENDENT (constant in D => still non-increasing, so
    // constraint 4 holds trivially). This drops one field from BOTH the long and
    // short sub-struct instances: 36 -> 34 params.
    float fail_s_exp;
    float fail_r_mult;
    float hard_penalty;
    float easy_bonus;
};

struct fsrs_params_t {
    // 0..3: Initial stability by first rating.
    float s0_again;
    float s0_hard;
    float s0_good;
    float s0_easy;

    // 4..6: Difficulty.
    float init_d0;
    float init_d1;
    float next_d_mult;

    // 7..14: Long-term stability after review.
    fsrs_stability_after_review_params_t long_stability;

    // 15..22: Short-term stability after review.
    fsrs_stability_after_review_params_t short_stability;

    // NOTE (iter-43): the iter-40 dual-trace removed the elapsed-time transition
    // blend, orphaning its two params (transition_decay, transition_scale). iter-41
    // repurposed transition_scale into a learnable fast-init multiplier, but that
    // per-user param did not earn its keep (the fast init is now hardcoded at a
    // fixed fraction of the slow init in fsrs7_init). Both orphaned params are
    // REMOVED here, simplifying the layout 38 -> 36 (this is the rule-compliant
    // accounting for the dual-trace's +1 state variable: +1 state, -2 params).

    // 23..30: Forgetting curve — 2-component mixture. (31..33 below are also
    // forgetting-curve params: the curve's D/S modulation. 11 curve params total.)
    float decay1;
    float decay2;
    float base1;
    float base2;
    float base_weight1;
    float base_weight2;
    float s_weight_power1;
    float s_weight_power2;

    // 31: Difficulty modulation of the forgetting-curve mixture. Scales the
    // long-forgetting component's weight by exp((d_weight - 0.5) * (D - 5)). The
    // param is clamped to [0, 1]; the curve subtracts 0.5 so the effective
    // coefficient is in [-0.5, 0.5] (2026-06-05 cosmetic shift to keep the param
    // non-negative). Neutral value 0.5 = no D-dependence (mixture weights stay
    // > 0, so both curve endpoints p(0)=1 and p(inf)=0 are preserved).
    float d_weight;

    // 32: Difficulty modulation of the long-forgetting component's DECAY
    // (curve *shape*, distinct from d_weight's mixture-weight reweighting).
    // decay2_mag = clamp(decay2 * exp((d_decay - 0.3)*(D-5)), 0.01, 0.95). The
    // param is clamped to [0, 0.6]; the curve subtracts 0.3 so the effective
    // coefficient is in [-0.3, 0.3] (2026-06-05 cosmetic shift). Neutral value
    // 0.3 = no D-dependence. The clamp keeps |decay| in the float-safe range so
    // base^(1/decay) never overflows; endpoints p(0)=1, p(inf)=0 still hold.
    float d_decay;

    // 33: Stability modulation of the SHORT-forgetting component's DECAY.
    // decay1_mag = clamp(decay1 * powf(s_short, s_decay1 - 0.3), 0.01, 0.95). The
    // param is clamped to [0, 0.6]; the curve subtracts 0.3 so the effective
    // exponent is in [-0.3, 0.3] (2026-06-05 cosmetic shift). Neutral value 0.3 =
    // no S-dependence (recovers decay1 exactly; decay1 stays in [0.01,0.25]). The
    // short component dominates the mixture only at small s_short, so this lever
    // reshapes the short-term (sub-day) curve while staying near-invisible to
    // high-S recall predictions. Clamp keeps base1^(1/decay1) float-safe and
    // endpoints p(0)=1, p(inf)=0 hold (base1<1 => factor1>0, monotone).
    float s_decay1;
};

__device__
fsrs_state_t fsrs7_init(
    const fsrs_params_t &fsrs_params,
    const int8_t first_rating
);

__device__
fsrs_state_t fsrs7_step(
    const fsrs_params_t &fsrs_params,
    const fsrs_state_t fsrs_state,
    const float elapsed_time,
    const int8_t rating
);

__device__
float fsrs7_forgetting_curve(
    const fsrs_params_t &fsrs_params,
    const float elapsed_time,
    const fsrs_state_t &state
);
