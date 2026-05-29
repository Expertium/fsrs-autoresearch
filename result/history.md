# FSRS-7 autoresearch — iteration history

_8 record(s). Generated from `history.jsonl` — do not edit by hand._

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
