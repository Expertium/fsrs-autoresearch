# FSRS-7 autoresearch — iteration history

_16 record(s). Generated from `history.jsonl` — do not edit by hand._

| # | Time (UTC) | Thresh. | LL before | LL after | Δ LL | Cx before | Cx after | Δ Cx % | Status | Summary |
|--:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 2026-05-29T15:04:27 | — | — | 0.32498 | — | — | 18,563 | — | champion | baseline FSRS-7 (default init_w, --short --secs --recency, 8 epochs, Adam betas=(0.8, 0.85), lr=2e-2 cosine to 0) |
| 1 | 2026-05-29T15:09:34 | 0.0001 | 0.32498 | 0.32530 | -0.00032 | 18,563 | 18,563 | +0.00% | rejected | Adam betas (0.8, 0.85) -> (0.9, 0.95); higher beta1 for smoother momentum, slightly higher beta2, gap preserved (~0.05) |
| 2 | 2026-05-29T15:22:33 | 0.0001 | 0.32498 | 0.32497 | +0.00001 | 18,563 | 18,563 | +0.00% | rejected | w[25] clamp floor 2.5 -> 1.0 (40.2% of users hit floor at baseline); default 2.5 and L2 sigma 0.4053 unchanged |
| 3 | 2026-05-29T15:32:10 | 0.0001 | 0.32498 | 0.32510 | -0.00012 | 18,563 | 18,563 | +0.00% | rejected | PENALTY_W_L2 0.5 -> 0.25 (relax L2 prior; iter 2 evidence suggested L2 anchored bound-saturated params) |
| 4 | 2026-05-29T15:36:19 | 0.0001 | 0.32498 | 0.32458 | +0.00040 | 18,563 | 18,563 | +0.00% | accepted | RECENCY_C0 0.25->0.10, RECENCY_C1 0.75->0.90 (downweight old reviews further; preserve newest weight=1.0) |
| 5 | 2026-05-29T15:38:27 | 0.0001 | 0.32458 | 0.32456 | +0.00001 | 18,563 | 18,563 | +0.00% | rejected | RECENCY_C0 0.10->0.05, C1 0.90->0.95 (push recency further along axis that worked in iter 4) |
| 6 | 2026-05-29T15:45:06 | 0.0001 | 0.32458 | 0.32514 | -0.00056 | 18,563 | 18,563 | +0.00% | rejected | LR 2e-2 -> 1e-2 (halve initial LR; testing whether iter 4 recency shift changed LR optimum) |
| 7 | 2026-05-29T15:59:27 | 0.0001 | 0.32458 | 0.32452 | +0.00005 | 18,689 | 18,691 | +0.01% | rejected | cosine LR scheduler eta_min 0.0 -> 0.05 (keep LR non-zero at end of decay; tests whether cosine->0 leaves loss on the table) |
| 8 | 2026-05-29T19:57:23 | 0.0001 | 0.32458 | 0.32453 | +0.00005 | 18,689 | 18,885 | +1.05% | rejected | log-transform w[0..3]: store u=log(w) as the optimizer-visible param; CUDA still sees w via exp at boundary. Chain-rule dL/dw -> dL/du = dL/dw*w in train_iter and predict_test_set. Clipper bounds for first 4 -> log space. |
| 9 | 2026-05-29T20:03:33 | 0.0001 | 0.32458 | 0.32457 | +0.00001 | 18,689 | 18,696 | +0.04% | rejected | AdamW weight_decay 0 -> 1e-4 (engage the decoupled L2-to-zero knob that has never been used) |
| 10 | 2026-05-29T20:07:33 | 0.0001 | 0.32458 | 0.33344 | -0.00886 | 18,689 | 18,703 | +0.07% | rejected | 500-step linear warmup multiplied on top of cosine LR schedule (motivated by w[0] \|grad\|~70, w[27] \|grad\|~40 -- Adam moments cold at step 0) |
| 11 | 2026-05-29T20:09:57 | 0.0001 | 0.32458 | 0.32457 | +0.00001 | 18,689 | 18,689 | +0.00% | rejected | PENALTY_W_L2 0.5 -> 1.0 (double L2-to-default prior strength; iter 3 weakening was worse, so test the other direction) |
| 12 | 2026-05-29T20:17:44 | 0.0001 | 0.32458 | 0.32458 | -0.00000 | 18,689 | 18,734 | +0.24% | rejected | Lower clip floor for w[11] (LT fail_d_exp) and w[20] (ST fail_d_exp) from 0.001 to 0.0; targets 36.8%/36.3% floor saturation. d^(-0)=1.0 safe. |
| 13 | 2026-05-29T20:30:55 | 0.0001 | 0.32458 | 0.32459 | -0.00001 | 18,689 | 18,734 | +0.24% | rejected | Anchor fail_d_exp at 0: lower floor (0.001->0) AND L2 default (0.0049->0 for w[11], 0.0107->0 for w[20]). Make 0 the natural resting value. |
| 14 | 2026-05-29T20:34:45 | 0.0001 | 0.32458 | 0.32431 | +0.00027 | 18,689 | 18,734 | +0.24% | accepted | Sharpen recency curve: weight = C0 + C1 * (review_ord/N)^5 (was ^3). Same axis as iter 4 win; tests whether more concentration on most-recent training reviews further improves fit to test (most-recent time chunk). |
| 15 | 2026-05-29T20:38:12 | 0.0001 | 0.32431 | 0.32422 | +0.00009 | 18,734 | 18,734 | +0.00% | rejected | Push recency exponent further: 5 -> 7. Tests whether the axis still has room past iter 14 champion. |
