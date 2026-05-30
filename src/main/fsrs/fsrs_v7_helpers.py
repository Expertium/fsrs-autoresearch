import torch

from src.main.fsrs import fsrs_v7_constants

def get_initial_params_for_optimization():
    return torch.tensor(fsrs_v7_constants.FSRS7_DEFAULT_35_VALUES, dtype=torch.float32)

@torch.compile(fullgraph=True)
def apply_parameter_clipper(parameters_b):
    lo = torch.tensor(
        fsrs_v7_constants.FSRS_MIN_VALUES,
        device=parameters_b.device,
        dtype=parameters_b.dtype,
    )
    hi = torch.tensor(
        fsrs_v7_constants.FSRS_MAX_VALUES,
        device=parameters_b.device,
        dtype=parameters_b.dtype,
    )

    with torch.no_grad():
        clipped = parameters_b.clamp(min=lo, max=hi).clone()
        clipped[..., 1] = torch.maximum(clipped[..., 1], clipped[..., 0])
        clipped[..., 2] = torch.maximum(clipped[..., 2], clipped[..., 1])
        clipped[..., 3] = torch.maximum(clipped[..., 3], clipped[..., 2])
        # iter-43: curve params shifted down by 2 after removing the two dead
        # transition-era params (old 27/29 -> 25/27, old 28/30 -> 26/28).
        # iter-54: dropped the decay2 >= decay1 chain (a vestigial pre-dual-trace
        # ordering). r1 reads s_fast and r2 reads s -- independent curves -- and
        # 11.6% of users were pinned at decay2 == decay1, wanting the slow trace a
        # heavier tail (decay2 < decay1). decay2 now clamps to its own [0.01, 0.95].
        clipped[..., 28] = torch.maximum(clipped[..., 28], clipped[..., 27])  # base2  >= base1
    return clipped

@torch.compile(fullgraph=True)
def penalty_loss(parameters_kp, batch_size_k, training_set_size_k, anchor_p):
    # iter 24: empirical-Bayes L2. Shrink each (user, split) row toward the
    # live population mean (anchor_p, passed in detached) instead of the fixed
    # FSRS7_DEFAULT. The population centroid is then free to drift to where the
    # data collectively wants it, while individual outliers stay regularized.
    # anchor_p == FSRS7_DEFAULT at init (all rows start there), so this smoothly
    # generalizes the old fixed-anchor penalty.
    sigma = torch.tensor(
        fsrs_v7_constants.FSRS7_L2_SIGMA_35_VALUES,
        device=parameters_kp.device,
        dtype=parameters_kp.dtype,
    )
    l2_k = torch.sum(
        torch.square(parameters_kp - anchor_p.unsqueeze(0))
        / torch.square(sigma.unsqueeze(0)),
        dim=-1,
    )
    penalty_k = (
        fsrs_v7_constants.PENALTY_W_L2
        * batch_size_k / training_set_size_k
        * l2_k
    )
    return penalty_k

def gradient_weight(review_ord_kb, training_set_size_k):
    # more recent reviews get a higher weight
    review_ord_lin = review_ord_kb / training_set_size_k.unsqueeze(-1)
    return fsrs_v7_constants.RECENCY_C0 + fsrs_v7_constants.RECENCY_C1 * torch.pow(review_ord_lin, 5)
