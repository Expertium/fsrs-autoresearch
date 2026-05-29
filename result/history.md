# FSRS-7 autoresearch — iteration history

_2 record(s). Generated from `history.jsonl` — do not edit by hand._

| # | Time (UTC) | Summary | Thresh. | LL before | LL after | Δ LL | Cx before | Cx after | Δ Cx % | Status | Reason |
|--:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 2026-05-29T15:04:27 | baseline FSRS-7 (default init_w, --short --secs --recency, 8 epochs, Adam betas=(0.8, 0.85), lr=2e-2 cosine to 0) | — | — | 0.32498 | — | — | 18,563 | — | champion | initial |
| 1 | 2026-05-29T15:09:34 | Adam betas (0.8, 0.85) -> (0.9, 0.95); higher beta1 for smoother momentum, slightly higher beta2, gap preserved (~0.05) | 0.0001 | 0.32498 | 0.32530 | -0.00032 | 18,563 | 18,563 | +0.00% | rejected | improvement -0.00032 < threshold 0.0001 (variant LL is worse than champion); (0.8, 0.85) preserved as champion |
