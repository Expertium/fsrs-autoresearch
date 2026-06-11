LR: float = 0.0282  # iter 20: was 2e-2; recency sharpening (iter 14, exp=5) cut gradient mass, raise LR to compensate
BETAS: tuple = (0.55, 0.9942)  # this is for Adam, default is (0.9, 0.999)

RECENCY_C0 = 0.0667  # iter 4: was 0.25; shift relative grad weight toward most-recent reviews (test split is time-most-recent chunk). Rounded to 4 dp (was 0.0666667) to match the hp_tune <=4-dp resolution.
RECENCY_EXP = 11.25  # iter-65: recency-ramp exponent, promoted from a hardcoded 5 in gradient_weight so hp_tune can search it. weight = C0 + (1-C0)*ord_frac^EXP (C1 dropped — newest-review weight is pinned at 1 by construction).

PENALTY_W_L2 = 0.5

# iter-65: the iter-52 per-group LR multipliers (LR_GROUP_MULT / LR_GROUP_PER_PARAM)
# were removed — a single global LR is used again. Their gain was too small to
# justify the per-group complexity; hp_tune now tunes recency weighting
# (RECENCY_C0 / RECENCY_EXP) instead.

FSRS7_DEFAULT_35_VALUES = (
    0.11039221453545266,
    2.2394934874691153,
    3.9220859596436135,
    11.784057540400967,
    6.168552230911006,
    0.6457255826197822,
    3.680704175188553,
    1.9795479315653393,
    0.0,
    1.3826473134308452,
    0.7023970826603881,
    0.5998956533912253,
    0.8145563344431239,
    0.6397912762333635,
    1.0,
    1.3206843176869136,
    0.6707051260828588,
    3.8667972762906326,
    0.44161358242645404,
    0.0933587998441009,
    1.8630702388426579,
    0.6162036848970999,
    1.0868843534046273,
    0.1566963957224671,
    0.08012072623492521,
    0.24206093576886362,
    0.9463799075560653,
    0.14328762877880125,
    0.714473829362035,
    0.0,
    0.5667322886723738,
    0.3734077003798003,  # 31 d_weight (shifted +0.5; curve subtracts 0.5)
    0.5333456073318997,  # 32 d_decay  (shifted +0.3; curve subtracts 0.3)
    0.3047711964822912,  # 33 s_decay1 (shifted +0.3; curve subtracts 0.3)
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
    0.1,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.5,
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
    0.0,  # 31 d_weight min  (was -0.5; +0.5 shift to make param >=0, curve formula subtracts 0.5)
    0.0,  # 32 d_decay min   (was -0.3; +0.3 shift, curve formula subtracts 0.3)
    0.0,  # 33 s_decay1 min  (was -0.3; +0.3 shift, curve formula subtracts 0.3)
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
    1.0,
    3.5,
    1.0,
    7.0,
    4.0,
    2.0,
    6.0,
    1.5,
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
    1.0,  # 31 d_weight max  (was 0.5; +0.5 shift of the [-0.5,0.5] range -> [0,1])
    0.6,  # 32 d_decay max   (was 0.3; +0.3 shift of the [-0.3,0.3] range -> [0,0.6])
    0.6,  # 33 s_decay1 max  (was 0.3; +0.3 shift of the [-0.3,0.3] range -> [0,0.6])
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
    0.175,
    0.3812,
    0.3013,
    0.9104,
    0.3234,
    0.2448,
    0.3273,
    0.1842,
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
