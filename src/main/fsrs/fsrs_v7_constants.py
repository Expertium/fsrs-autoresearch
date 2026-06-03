LR: float = 0.0133  # iter 20: was 2e-2; recency sharpening (iter 14, exp=5) cut gradient mass, raise LR to compensate
BETAS: tuple = (0.55, 0.9913)  # this is for Adam, default is (0.9, 0.999)

RECENCY_C0 = 0.0667  # iter 4: was 0.25; shift relative grad weight toward most-recent reviews (test split is time-most-recent chunk). Rounded to 4 dp (was 0.0666667) to match the hp_tune <=4-dp resolution.
RECENCY_EXP = 11.25  # iter-65: recency-ramp exponent, promoted from a hardcoded 5 in gradient_weight so hp_tune can search it. weight = C0 + (1-C0)*ord_frac^EXP (C1 dropped — newest-review weight is pinned at 1 by construction).

PENALTY_W_L2 = 0.3333

# iter-65: the iter-52 per-group LR multipliers (LR_GROUP_MULT / LR_GROUP_PER_PARAM)
# were removed — a single global LR is used again. Their gain was too small to
# justify the per-group complexity; hp_tune now tunes recency weighting
# (RECENCY_C0 / RECENCY_EXP) instead.

FSRS7_DEFAULT_35_VALUES = (
    0.107,
    2.2526,
    3.8514,
    11.9504,
    6.176,
    0.6711,
    3.5464,
    2.0207,
    0.0384,
    1.4093,
    0.6931,
    0.001,
    0.5703,
    0.8688,
    0.5846,
    1.0003,
    1.3878,
    0.6956,
    3.9311,
    0.3819,
    0.001,
    0.0866,
    1.8235,
    0.73,
    1.0,
    0.176,
    0.0815,
    0.2447,
    0.9497,
    0.14,
    0.7113,
    0.0,
    0.5989,
    -0.0519,
    0.2048,
    0.0498,
)

FSRS_MIN_VALUES = (
    0.0001,
    0.0001,
    0.0001,
    0.0001,
    1.0,
    0.001,
    0.1,
    0.0,
    0.0,
    0.3,
    0.01,
    0.001,
    0.1,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.5,
    0.001,
    0.001,
    0.001,
    0.0,
    0.0,
    1.0,
    0.01,
    0.01,
    0.2,  # 27 base1 min: lowered 0.5->0.2 (iter-50) for steeper sub-day fast-curve forgetting; factor1 clamped float-safe in fsrs7.cu
    0.5,
    0.01,
    0.1,
    0.0,
    0.1,
    -0.5,  # 35 d_weight min
    -0.3,  # 36 d_decay min
    -0.3,  # 37 s_decay1 min
)

FSRS_MAX_VALUES = (
    50.0,
    100.0,
    100.0,
    100.0,
    10.0,
    4.0,
    4.0,
    4.0,
    1.2,
    3.0,
    1.5,
    0.9,
    1.0,
    3.5,
    1.0,
    7.0,
    4.0,
    2.0,
    6.0,
    1.5,
    2.0,
    1.0,
    5.0,
    1.0,
    7.0,
    0.25,
    0.95,
    0.85,
    0.99,
    1.0,
    1.0,
    0.9,
    1.1,
    0.5,  # 35 d_weight max
    0.3,  # 36 d_decay max
    0.3,  # 37 s_decay1 max
)

FSRS7_L2_SIGMA_35_VALUES = (
    9999.0,
    9999.0,
    9999.0,
    9999.0,
    0.523,
    0.2528,
    0.4329,
    0.2966,
    0.2139,
    0.2889,
    0.1862,
    0.0829,
    0.175,
    0.3812,
    0.3013,
    0.9104,
    0.3234,
    0.2448,
    0.3273,
    0.1842,
    0.1542,
    0.1735,
    0.4608,
    0.311,
    0.864,
    0.0418,
    0.2596,
    0.0798,
    0.0682,
    0.1282,
    0.1397,
    0.1407,
    0.1489,
    0.2,  # 35 d_weight sigma (weak L2 prior toward 0 so data can move it)
    0.15,  # 36 d_decay sigma (weak prior toward the fixed default)
    0.15,  # 37 s_decay1 sigma (weak prior; mirrors d_decay)
)
