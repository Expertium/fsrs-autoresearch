#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "fsrs7.cuh"

__device__ __forceinline__
float fsrs7_clamp(const float x, const float lo, const float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__
fsrs_state_t fsrs7_clamp_state(
    const float stability,
    const float difficulty,
    const float stability_fast
) {
    return fsrs_state_t{
        fsrs7_clamp(stability, 1e-4f, 36500.0f),
        fsrs7_clamp(difficulty, 1.0f, 10.0f),
        fsrs7_clamp(stability_fast, 1e-4f, 36500.0f),
    };
}

__device__ __forceinline__
float fsrs7_initial_difficulty(
    const fsrs_params_t &fsrs_params,
    const float rating
) {
    return fsrs_params.init_d0 - expf(fsrs_params.init_d1 * (rating - 1.0f)) + 1.0f;
}

__device__ __forceinline__
float fsrs7_linear_damping(const float delta_d, const float old_d) {
    return delta_d * (10.0f - old_d) / 9.0f;
}

__device__ __forceinline__
float fsrs7_mean_reversion(const float init, const float current) {
    return 0.01f * init + 0.99f * current;
}

__device__ __forceinline__
float fsrs7_next_d(
    const fsrs_params_t &fsrs_params,
    const fsrs_state_t fsrs_state,
    const int8_t rating
) {
    const float delta_d = -fsrs_params.next_d_mult * (static_cast<float>(rating) - 3.0f);
    const float new_d = fsrs_state.d + fsrs7_linear_damping(delta_d, fsrs_state.d);
    return fsrs7_mean_reversion(fsrs7_initial_difficulty(fsrs_params, 4.0f), new_d);
}

__device__ __forceinline__
float fsrs7_forgetting_curve(
    const fsrs_params_t &fsrs_params,
    const float elapsed_time,
    const fsrs_state_t &state
) {
    // DUAL-TRACE MEMORY (iter-40): the FAST recall component (r1) is driven by
    // the fast trace s_fast; the SLOW component (r2) by the slow trace s. Each
    // component "reads" its own memory store, so the curve is a true two-store
    // mixture rather than two analytic components sharing a single stability.
    const float t_over_s_fast = elapsed_time / state.s_fast;
    const float t_over_s_slow = elapsed_time / state.s;

    // FAST component. Its decay can still be S-modulated (s_decay1), now keyed
    // to the fast trace it belongs to. s_decay1=0 recovers -decay1 exactly
    // (decay1 in [0.01,0.25]). Clamp keeps base1^(1/decay1) float-safe; base1<1
    // => factor1>0, so r1 stays monotone with r1(0)=1, r1(inf)=0.
    const float decay1_mag = fsrs7_clamp(
        fsrs_params.decay1 * powf(state.s_fast, fsrs_params.s_decay1),
        0.01f, 0.95f);
    const float decay1 = -decay1_mag;
    const float factor1 = powf(fsrs_params.base1, 1.0f / decay1) - 1.0f;
    const float r1 = powf(1.0f + factor1 * t_over_s_fast, decay1);

    // SLOW component. Difficulty modulation of its DECAY (curve shape). d_decay>0
    // => hard cards (D>5) get a steeper slow tail. Clamp to [0.01, 0.95] keeps
    // |decay| safe: factor = base^(1/decay) overflows float once |decay| < ~0.008.
    // d_decay=0 reduces to -decay2 exactly (decay2 already in [0.01,0.95]).
    const float decay2_mag = fsrs7_clamp(
        fsrs_params.decay2 * expf(fsrs_params.d_decay * (state.d - 5.0f)),
        0.01f, 0.95f);
    const float decay2 = -decay2_mag;
    const float factor2 = powf(fsrs_params.base2, 1.0f / decay2) - 1.0f;
    const float r2 = powf(1.0f + factor2 * t_over_s_slow, decay2);

    // Mixture weights, each keyed to the trace its component reads: as the fast
    // trace grows weight1 shrinks (S^-power1) and as the slow trace grows
    // weight2 grows (S^+power2), shifting the mix from fast-dominated (freshly
    // reviewed) to slow-dominated (well consolidated). exp(d_weight*(D-5))>0
    // keeps both weights positive, so the mixture preserves p(0)=1, p(inf)=0.
    const float weight1 = fsrs_params.base_weight1 * powf(state.s_fast, -fsrs_params.s_weight_power1);
    const float weight2 = fsrs_params.base_weight2 * powf(state.s, fsrs_params.s_weight_power2)
        * expf(fsrs_params.d_weight * (state.d - 5.0f));
    const float retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2);

    return 1e-5f + (1.0f - 2e-5f) * retention;
}

__device__ __forceinline__
float fsrs7_stability_after_review_one_term(
    const fsrs_state_t fsrs_state,
    const float retention,
    const int8_t rating,
    const fsrs_stability_after_review_params_t &params
) {
    const float hard_penalty = rating == 2 ? params.hard_penalty : 1.0f;
    const float easy_bonus = rating == 4 ? params.easy_bonus : 1.0f;

    const float new_s_fail =
        params.fail_mult
        * powf(fsrs_state.d, -params.fail_d_exp)
        * (powf(fsrs_state.s + 1.0f, params.fail_s_exp) - 1.0f)
        * expf((1.0f - retention) * params.fail_r_mult);
    const float pls = fminf(fsrs_state.s, new_s_fail);

    const float s_inc =
        1.0f
        + expf(params.sinc_base - 1.5f)
        * (11.0f - fsrs_state.d)
        * powf(fsrs_state.s, -params.sinc_s_exp)
        * (expf((1.0f - retention) * params.sinc_r_mult) - 1.0f)
        * hard_penalty
        * easy_bonus;
    const float new_s_success = fmaxf(pls, fsrs_state.s * s_inc);

    return rating > 1 ? new_s_success : pls;
}

__device__
fsrs_state_t fsrs7_init(
    const fsrs_params_t &fsrs_params,
    const int8_t first_rating
) {
    float initial_stability;
    switch (first_rating) {
        case 2:
            initial_stability = fsrs_params.s0_hard;
            break;
        case 3:
            initial_stability = fsrs_params.s0_good;
            break;
        case 4:
            initial_stability = fsrs_params.s0_easy;
            break;
        case 1:
        default:
            initial_stability = fsrs_params.s0_again;
            break;
    }

    const float initial_difficulty = fsrs7_initial_difficulty(
        fsrs_params,
        static_cast<float>(first_rating)
    );

    // The fast trace starts at a fixed fraction of the slow init (iter-43; the
    // iter-41 per-user multiplier was saturated and didn't earn its param, so it
    // is hardcoded) — smaller than the slow trace for sharper sub-day forgetting.
    // Their dynamics diverge from the first review onward (slow long, fast short).
    constexpr float fast_init_frac = 0.8f;
    return fsrs7_clamp_state(
        initial_stability,
        initial_difficulty,
        fast_init_frac * initial_stability);
}

__device__
fsrs_state_t fsrs7_step(
    const fsrs_params_t &fsrs_params,
    const fsrs_state_t fsrs_state,
    const float elapsed_time,
    const int8_t rating
) {
    const float retention = fsrs7_forgetting_curve(
        fsrs_params,
        elapsed_time,
        fsrs_state
    );

    // DUAL-TRACE update. The SLOW trace evolves by the long-term (consolidation)
    // dynamics reading its own value (fsrs_state.s); the FAST trace evolves by
    // the short-term dynamics reading its own value (fsrs_state.s_fast). This
    // replaces the elapsed-time transition blend that collapsed long/short into
    // a single S — the two persistent traces now carry that short-vs-long
    // structure across reviews. transition_scale / transition_decay (w[25..26])
    // are unused by this formulation; they stay in the struct so the param
    // layout and the 38-wide param tensor are unchanged.
    const float new_s_slow = fsrs7_stability_after_review_one_term(
        fsrs_state,
        retention,
        rating,
        fsrs_params.long_stability
    );

    // Read the fast trace as the stability input (difficulty is shared).
    const fsrs_state_t fast_state{fsrs_state.s_fast, fsrs_state.d, fsrs_state.s_fast};
    const float new_s_fast = fsrs7_stability_after_review_one_term(
        fast_state,
        retention,
        rating,
        fsrs_params.short_stability
    );

    const float new_d = fsrs7_next_d(fsrs_params, fsrs_state, rating);

    return fsrs7_clamp_state(new_s_slow, new_d, new_s_fast);
}
