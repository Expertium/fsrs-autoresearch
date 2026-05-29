# FSRS-7 autoresearch — iteration history

_4 record(s). Generated from `history.jsonl` — do not edit by hand._

| # | Time (UTC) | Summary | Thresh. | LL before | LL after | Δ LL | Cx before | Cx after | Δ Cx % | Status | Reason |
|--:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 2026-05-29T15:04:27 | baseline FSRS-7 (default init_w, --short --secs --recency, 8 epochs, Adam betas=(0.8, 0.85), lr=2e-2 cosine to 0) | — | — | 0.32498 | — | — | 18,563 | — | champion | initial |
| 1 | 2026-05-29T15:09:34 | Adam betas (0.8, 0.85) -> (0.9, 0.95); higher beta1 for smoother momentum, slightly higher beta2, gap preserved (~0.05) | 0.0001 | 0.32498 | 0.32530 | -0.00032 | 18,563 | 18,563 | +0.00% | rejected | improvement -0.00032 < threshold 0.0001 (variant LL is worse than champion); (0.8, 0.85) preserved as champion |
| 2 | 2026-05-29T15:22:33 | w[25] clamp floor 2.5 -> 1.0 (40.2% of users hit floor at baseline); default 2.5 and L2 sigma 0.4053 unchanged | 0.0001 | 0.32498 | 0.32497 | +0.00001 | 18,563 | 18,563 | +0.00% | rejected | improvement +0.00001 < threshold 0.0001. Freeing the bound let w[25] p01 drop 2.5->1.17 and hit_lo collapse 40.2%->0.4%, so the bound WAS active — but L2 prior toward default 2.5 (sigma=0.4053) is the real anchor, so net LL barely moved. Champion floor (2.5) restored. |
| 3 | 2026-05-29T15:32:10 | PENALTY_W_L2 0.5 -> 0.25 (relax L2 prior; iter 2 evidence suggested L2 anchored bound-saturated params) | 0.0001 | 0.32498 | 0.32510 | -0.00012 | 18,563 | 18,563 | +0.00% | rejected | improvement -0.00012 < threshold 0.0001 (variant LL is WORSE). Relaxing L2 broadened p99 modestly (w[25] 2.81->2.95, w[26] p01 0.62->0.55) but did not reduce bound saturation (w[25] hit_lo still 40.1%, ~unchanged). Net LL got worse — the original L2 strength of 0.5 was actually preventing overfitting. PENALTY_W_L2 restored to 0.5. |
