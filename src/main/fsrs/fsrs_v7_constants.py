LR: float = 0.045  # iter 20: was 2e-2; recency sharpening (iter 14, exp=5) cut gradient mass, raise LR to compensate
BETAS: tuple = (0.55, 0.899999)  # this is for Adam, default is (0.9, 0.999)

RECENCY_C0 = 0.10  # iter 4: was 0.25; shift relative grad weight toward most-recent reviews (test split is time-most-recent chunk)
RECENCY_C1 = 0.90  # iter 4: was 0.75; kept so newest review weight = C0 + C1 = 1.0

PENALTY_W_L2 = 0.75

# Per-group learning-rate multipliers (iter-52): a separate effective LR per
# functional parameter group, scaling the global LR. Tuned by coordinate descent
# on logloss_by_user: the forgetting-curve + initial-stability groups OSCILLATE at
# the global LR (largest last-epoch gradients; Adam already per-param normalizes,
# so a big persistent gradient = bouncing around the optimum) and want a LOWER
# rate to settle; the 18-param stability-update block is UNDER-converged and wants
# a HIGHER rate; difficulty is already converged (neutral). A single LR can't
# express this heterogeneity. (1,1,1,1) recovers the single-LR champion exactly.
# Tuned, not back-propagated -> training hyperparameters, no new self.w scalar.
LR_GROUP_MULT = (0.65, 1.0, 1.6, 0.7)  # (init_S w0-3, difficulty w4-6, stability-update w7-24, forgetting-curve w25-35)

# Expanded to one multiplier per parameter index (length 36, matches the param layout).
LR_GROUP_PER_PARAM = (
    (LR_GROUP_MULT[0],) * 4     # w0-3   initial stability
    + (LR_GROUP_MULT[1],) * 3   # w4-6   difficulty
    + (LR_GROUP_MULT[2],) * 18  # w7-24  stability-after-review (long + short)
    + (LR_GROUP_MULT[3],) * 11  # w25-35 forgetting curve (+ d_weight/d_decay/s_decay1)
)

FSRS7_DEFAULT_35_VALUES = (
    0.041,
    2.4175,
    4.1283,
    11.9709,
    5.6385,
    0.4468,
    3.262,
    2.3054,
    0.1688,
    1.3325,
    0.3524,
    0.0049,
    0.7503,
    0.0896,
    0.6625,
    1.3,
    0.882,
    0.3072,
    3.5875,
    0.303,
    0.0107,
    0.2279,
    2.6413,
    0.5594,
    1.3,
    0.0723,
    0.1634,
    0.5,
    0.9555,
    0.2245,
    0.6232,
    0.1362,
    0.3862,
    0.0,  # 35 d_weight (difficulty modulation of forgetting curve; 0 = neutral)
    0.0,  # 36 d_decay (difficulty modulation of slow-component decay/shape; 0 = neutral)
    0.0,  # 37 s_decay1 (stability modulation of fast-component decay; 0 = neutral)
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
    0.15,  # 36 d_decay sigma (weak prior; EB anchor lets population mean drift)
    0.15,  # 37 s_decay1 sigma (weak prior; mirrors d_decay)
)
