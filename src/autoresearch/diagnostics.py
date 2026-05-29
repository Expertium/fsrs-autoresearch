"""
Post-training diagnostics for the autoresearch loop.

Produces the per-parameter stats and loss breakdowns described in
autoresearcher.md (adapted for the multi-user, custom-CUDA setup):

    Code complexity score (see complexity.py)
    Average log loss
    Log loss on Again / Hard / Good / Easy reviews only
    Log loss on delta_t < 1 day reviews
    Log loss on delta_t >= 1 day reviews
    Per-parameter w[i]:
        bounds (effective per-user)
        p01 / median / p99 across all (user, split) rows
        Hit lower bound on X% of users
        Hit upper bound on X% of users
        Mean gradient across all epochs        ← requires gradient capture
        Mean gradient at the last epoch        ← requires gradient capture

Provides:

* :func:`fsrs7_effective_bounds` — clamp ranges that mirror
  ``FSRS7ParameterClipper`` exactly, including chained lower bounds
  (``w[1] >= w[0]`` etc.).
* :func:`param_distribution_stats` — p01/median/p99 + hit-bound percentages
  over the flat ``[user * N_SPLITS, 35]`` weight tensor produced by
  ``flatten_fsrs_params_for_summary``.
* :func:`loss_breakdown_by_rating` and :func:`loss_breakdown_by_delta_t`
  — per-bucket log loss, computed at eval time from the same
  ``(p, label, rating, delta_t)`` tensors that ``evaluate_on_test_set``
  already has access to.
* :class:`GradTracker` — accumulates per-parameter gradient magnitudes
  during training. Needs a small wiring change in ``src/main/run.py``
  (see the class docstring) because gradients come out of a custom CUDA
  op, not standard PyTorch autograd.
* :func:`build_diagnostics_dict` and :func:`format_markdown_report`
  — produce the JSON+Markdown payload the loop driver will feed back
  into Claude on the next iteration.

All tensor ops use only what's already in the project's dependency set
(torch). No new third-party deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


# ── FSRS-7 clamp bounds ──────────────────────────────────────────────────────
#
# Source of truth: src/models/fsrs_v7.py, class FSRS7ParameterClipper.
# These tuples mirror the clipper's literal `.clamp(lower, upper)` calls.
# Numeric lower means "constant"; a string lower like "w[0]" means
# "chained to the current value of that parameter for this user" — i.e. a
# dynamic lower bound that the diagnostics report keeps symbolic for
# readability.
#
# Values reflect the --secs=True config the loop always runs with:
# s_min = 0.0001, init_s_max = 100.0 (substituted into idx 0/1/2/3 here).
# If you change the clipper, change `fsrs7_effective_bounds` to match —
# the *displayed* bound and the *resolved* per-user bound used for the
# hit-bound percentage must agree.

FSRS7_BOUNDS_STATIC: list[tuple[float | str, float | str]] = [
    # idx, role                          (lower,    upper)
    (0.0001,         50.0),            # 0  Initial S, rating 1   (s_min, init_s_max/2)
    ("w[0]",         100.0),           # 1  Initial S, rating 2   (chained, init_s_max)
    ("w[1]",         100.0),           # 2  Initial S, rating 3
    ("w[2]",         100.0),           # 3  Initial S, rating 4
    (1.0,            10.0),            # 4  Difficulty
    (0.001,          4.0),             # 5  Difficulty
    (0.1,            4.0),             # 6  Difficulty
    (0.0,            4.0),             # 7  Long-term stability
    (0.0,            1.2),             # 8
    (0.3,            3.0),             # 9
    (0.01,           1.5),             # 10
    (0.001,          0.9),             # 11
    (0.1,            1.0),             # 12
    (0.0,            3.5),             # 13
    (0.0,            1.0),             # 14
    (1.0,            7.0),             # 15
    (0.0,            4.0),             # 16 Short-term stability
    (0.0,            2.0),             # 17
    (0.5,            6.0),             # 18
    (0.001,          1.5),             # 19
    (0.001,          2.0),             # 20
    (0.001,          1.0),             # 21
    (0.0,            5.0),             # 22
    (0.0,            1.0),             # 23
    (1.0,            7.0),             # 24
    (2.5,            15.0),            # 25 Long-short transition
    (0.0,            1.0),             # 26
    (0.01,           0.25),            # 27 Forgetting curve, decay 1
    ("w[27]",        0.95),            # 28 decay 2 (lower chained to w[27])
    (0.5,            0.85),            # 29 base 1
    ("w[29]",        0.99),            # 30 base 2 (lower chained to w[29])
    (0.01,           1.0),             # 31 weight 1
    (0.1,            1.0),             # 32 weight 2
    (0.0,            0.9),             # 33 S weight power 1
    (0.1,            1.1),             # 34 S weight power 2
    (-0.5,           0.5),             # 35 d_weight (difficulty modulation)
    (-0.3,           0.3),             # 36 d_decay (difficulty modulation of slow decay)
]

# Source of truth for the parameter count — derived from the bounds table so
# adding/removing a param only requires editing FSRS7_BOUNDS_STATIC above.
N_PARAMS: int = len(FSRS7_BOUNDS_STATIC)


def fsrs7_effective_bounds(
    rows: torch.Tensor,
    s_min: float = 0.0001,
    init_s_max: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-row (per-user) effective clamp bounds, resolving the chained
    lower bounds w[1]>=w[0], w[2]>=w[1], w[3]>=w[2], w[28]>=w[27], w[30]>=w[29].

    Args:
        rows: weights tensor of shape ``[N, 35]`` (one row per user/split).
        s_min: floor for ``w[0]``. Default 0.0001 matches FSRS-7 with --secs;
            pass 0.001 for the no-secs variant.
        init_s_max: ceiling for w[0..3].

    Returns:
        (lower, upper) each of shape ``[N, 35]``.
    """
    if rows.ndim != 2 or rows.size(1) != N_PARAMS:
        raise ValueError(f"expected rows of shape [N, {N_PARAMS}], got {tuple(rows.shape)}")
    n = rows.size(0)
    device = rows.device
    dtype = rows.dtype

    lower = torch.empty(n, N_PARAMS, dtype=dtype, device=device)
    upper = torch.empty(n, N_PARAMS, dtype=dtype, device=device)

    # idx 0: (s_min, init_s_max/2)
    lower[:, 0] = s_min
    upper[:, 0] = init_s_max / 2.0
    # idx 1..3: chained lower
    lower[:, 1] = rows[:, 0]
    lower[:, 2] = rows[:, 1]
    lower[:, 3] = rows[:, 2]
    upper[:, 1] = init_s_max
    upper[:, 2] = init_s_max
    upper[:, 3] = init_s_max

    static_pairs = {
        4: (1.0, 10.0),
        5: (0.001, 4.0),
        6: (0.1, 4.0),
        7: (0.0, 4.0), 8: (0.0, 1.2), 9: (0.3, 3.0),
        10: (0.01, 1.5), 11: (0.001, 0.9), 12: (0.1, 1.0),
        13: (0.0, 3.5), 14: (0.0, 1.0), 15: (1.0, 7.0),
        16: (0.0, 4.0), 17: (0.0, 2.0), 18: (0.5, 6.0),
        19: (0.001, 1.5), 20: (0.001, 2.0), 21: (0.001, 1.0),
        22: (0.0, 5.0), 23: (0.0, 1.0), 24: (1.0, 7.0),
        25: (2.5, 15.0), 26: (0.0, 1.0),
        27: (0.01, 0.25),
        29: (0.5, 0.85),
        31: (0.01, 1.0), 32: (0.1, 1.0),
        33: (0.0, 0.9), 34: (0.1, 1.1),
        35: (-0.5, 0.5),
        36: (-0.3, 0.3),
    }
    for idx, (lo, hi) in static_pairs.items():
        lower[:, idx] = lo
        upper[:, idx] = hi

    # idx 28: lower chained to w[27]; upper 0.95
    lower[:, 28] = rows[:, 27]
    upper[:, 28] = 0.95
    # idx 30: lower chained to w[29]; upper 0.99
    lower[:, 30] = rows[:, 29]
    upper[:, 30] = 0.99

    return lower, upper


# ── Per-parameter distribution stats ─────────────────────────────────────────


@dataclass(frozen=True)
class ParamStats:
    index: int
    bound_lower: str  # human-readable, may be "w[0]" etc.
    bound_upper: str
    p01: float
    median: float
    p99: float
    hit_lower_pct: float
    hit_upper_pct: float


def param_distribution_stats(
    rows: torch.Tensor,
    s_min: float = 0.0001,
    init_s_max: float = 100.0,
    *,
    hit_eps: float = 1e-4,
) -> list[ParamStats]:
    """
    Compute p01/median/p99 and hit-lower/upper percentages for each of the 35
    FSRS-7 parameters across all rows.

    "Hit lower" means the parameter's value is within ``hit_eps`` of its
    effective lower bound for that row (i.e. the clipper most likely clamped
    it). Same for upper.

    Args:
        rows: tensor of shape ``[N, 35]`` (e.g. result of
            ``flatten_fsrs_params_for_summary`` from result_metrics.py).
        s_min, init_s_max: passed through to :func:`fsrs7_effective_bounds`.
        hit_eps: tolerance for the "is value at the bound" check.

    Returns:
        A list of 35 :class:`ParamStats`, one per parameter index.
    """
    rows_f32 = rows.detach().to(dtype=torch.float32).cpu()
    if rows_f32.size(0) == 0:
        raise ValueError("rows is empty — no per-user weights to summarise.")

    p01 = torch.quantile(rows_f32, 0.01, dim=0)
    med = torch.quantile(rows_f32, 0.50, dim=0)
    p99 = torch.quantile(rows_f32, 0.99, dim=0)

    lower, upper = fsrs7_effective_bounds(
        rows_f32, s_min=s_min, init_s_max=init_s_max,
    )
    hit_lower = (rows_f32 <= lower + hit_eps).float().mean(dim=0) * 100.0
    hit_upper = (rows_f32 >= upper - hit_eps).float().mean(dim=0) * 100.0

    def _fmt(v: float | str) -> str:
        return v if isinstance(v, str) else f"{v:g}"

    out: list[ParamStats] = []
    for i in range(N_PARAMS):
        lo_desc, hi_desc = FSRS7_BOUNDS_STATIC[i]
        out.append(
            ParamStats(
                index=i,
                bound_lower=_fmt(lo_desc),
                bound_upper=_fmt(hi_desc),
                p01=float(p01[i]),
                median=float(med[i]),
                p99=float(p99[i]),
                hit_lower_pct=float(hit_lower[i]),
                hit_upper_pct=float(hit_upper[i]),
            )
        )
    return out


# ── Loss breakdowns ──────────────────────────────────────────────────────────


def _bce_mean(p: torch.Tensor, label: torch.Tensor, eps: float = 1e-7) -> float:
    if p.numel() == 0:
        return float("nan")
    p_clamped = p.clamp(min=eps, max=1.0 - eps)
    return float(
        torch.nn.functional.binary_cross_entropy(p_clamped, label, reduction="mean").item()
    )


def loss_breakdown_by_rating(
    p: torch.Tensor,
    rating: torch.Tensor,
) -> dict[str, float]:
    """
    Compute log loss restricted to reviews of each rating.

    Note: with ``label = (rating > 1).float()``, rating=1 reviews always
    have label=0 (so their loss is ``-log(1-p)``), while ratings 2/3/4 have
    label=1 (loss is ``-log(p)``). The breakdown is still computed against
    the same label definition the training/eval pipeline uses.
    """
    if p.shape != rating.shape:
        raise ValueError(f"p and rating must match shape, got {p.shape} vs {rating.shape}")
    label = (rating > 1).to(dtype=p.dtype)
    out: dict[str, float] = {}
    for r, name in [(1, "again"), (2, "hard"), (3, "good"), (4, "easy")]:
        mask = rating == r
        out[name] = _bce_mean(p[mask], label[mask])
        out[f"{name}_count"] = int(mask.sum().item())
    return out


def loss_breakdown_by_delta_t(
    p: torch.Tensor,
    rating: torch.Tensor,
    delta_t_days: torch.Tensor,
    threshold_days: float = 1.0,
) -> dict[str, float]:
    """
    Split log loss into sub-day vs >= 1 day intervals.
    """
    if not (p.shape == rating.shape == delta_t_days.shape):
        raise ValueError(
            f"shape mismatch: p={p.shape} rating={rating.shape} delta_t={delta_t_days.shape}"
        )
    label = (rating > 1).to(dtype=p.dtype)
    short_mask = delta_t_days < threshold_days
    long_mask = ~short_mask
    return {
        "short_term": _bce_mean(p[short_mask], label[short_mask]),
        "short_term_count": int(short_mask.sum().item()),
        "long_term": _bce_mean(p[long_mask], label[long_mask]),
        "long_term_count": int(long_mask.sum().item()),
        "threshold_days": threshold_days,
    }


# ── Gradient tracking ────────────────────────────────────────────────────────


@dataclass
class GradTracker:
    """
    Accumulates per-parameter gradient magnitude statistics during training.

    Why this is non-trivial here
    ----------------------------
    The FSRS-7 training loop uses a custom CUDA op (see
    ``src/main/run.py::run_cpp_train_pass`` and ``train_iter``) that
    computes ``per_example_grad`` directly, not through standard PyTorch
    autograd. Standard ``register_hook`` won't fire. To capture gradients we
    need to modify ``train_iter`` to return ``flat_grad`` as an additional
    output, and have the outer ``train`` loop feed it into the tracker.

    Suggested wiring (apply as a patch in run.py)
    ---------------------------------------------
    1. Add ``flat_grad`` to the return tuple of ``train_iter``::

           return new_flat_fsrs_params, new_optim_state, new_step_i_cat, flat_grad

       and adjust the ``@torch.compile(fullgraph=True, dynamic=False)``
       wrapper accordingly (it's fine; ``flat_grad`` is already a tensor in
       scope).
    2. In ``train``, unpack the new output and call::

           tracker.observe(flat_grad, step_i_cat, num_training_steps_per_epoch_cat)

       once per outer iteration.
    3. After ``train`` returns, call ``tracker.summarise()`` to get the
       mean-over-all-iters and mean-over-last-epoch tensors of shape ``[35]``.

    Behaviour
    ---------
    For each parameter index ``i``, we track:
      * ``abs_grad_sum_all`` and ``count_all`` — running totals over every
        active (user, split) gradient observed across all training steps.
      * ``abs_grad_sum_last`` and ``count_last`` — same, but only over
        gradients observed during the **last epoch** (we use the per-user
        ``num_training_steps_per_epoch_cat`` to detect when ``step_i_cat``
        has entered each user's final epoch).

    Mean is computed as ``sum / count`` at summarise time, returning NaN
    where ``count == 0``.

    The tracker is RNG-free and adds no autograd graph, so it can be called
    inside the training loop without affecting the optimisation.
    """

    n_params: int = N_PARAMS

    # Initialised lazily on first observation so we adopt the model's dtype/device.
    abs_grad_sum_all: Optional[torch.Tensor] = None
    count_all: int = 0
    abs_grad_sum_last: Optional[torch.Tensor] = None
    count_last: int = 0

    # Filled by observe(): per-user epoch index = step_i_cat // steps_per_epoch.
    # We track the maximum number of epochs seen so far and treat the highest
    # so-far value as "the last epoch" — this matches the loop's behaviour of
    # all users finishing at the same epoch count (N_EPOCHS).
    max_epoch_seen: int = 0

    def observe(
        self,
        flat_grad: torch.Tensor,
        step_i_cat: torch.Tensor,
        num_training_steps_per_epoch_cat: torch.Tensor,
    ) -> None:
        """
        Update accumulators with one iteration's gradient tensor.

        Args:
            flat_grad: shape ``[U*N_SPLITS, 35]``, gradient of the loss w.r.t.
                each (user, split) parameter row at this iteration. Rows that
                were inactive at this step should have grad = 0 (which is how
                ``run.py`` already zeros them via the ``legal`` mask).
            step_i_cat: shape ``[U*N_SPLITS]``, the step counter per
                (user, split) at the **end** of this iteration.
            num_training_steps_per_epoch_cat: shape ``[U*N_SPLITS]``, the
                number of training steps each user runs per epoch.
        """
        if flat_grad.ndim != 2 or flat_grad.size(1) != self.n_params:
            raise ValueError(
                f"flat_grad should have shape [N, {self.n_params}], got {tuple(flat_grad.shape)}"
            )
        g = flat_grad.detach().abs()

        # Active = any nonzero gradient on the row (so we don't count
        # already-finished users with their zero grads).
        active = (g.sum(dim=-1) > 0)
        active_g = g[active]

        if self.abs_grad_sum_all is None:
            self.abs_grad_sum_all = torch.zeros(self.n_params, dtype=g.dtype, device=g.device)
            self.abs_grad_sum_last = torch.zeros(self.n_params, dtype=g.dtype, device=g.device)

        self.abs_grad_sum_all.add_(active_g.sum(dim=0))
        self.count_all += int(active.sum().item())

        # Per-user epoch index AFTER this step. Rows whose step_i is in the
        # final epoch contribute to abs_grad_sum_last.
        epoch_idx = (step_i_cat.to(torch.int64) - 1).clamp_min(0) // num_training_steps_per_epoch_cat.to(torch.int64)
        self.max_epoch_seen = max(self.max_epoch_seen, int(epoch_idx.max().item()))
        last_mask = (epoch_idx == self.max_epoch_seen) & active
        last_g = g[last_mask]
        self.abs_grad_sum_last.add_(last_g.sum(dim=0))
        self.count_last += int(last_mask.sum().item())

    def summarise(self) -> dict[str, list[float]]:
        """Return per-parameter mean |grad| over all iters / last epoch only."""
        def _mean(total: Optional[torch.Tensor], count: int) -> list[float]:
            if total is None or count == 0:
                return [float("nan")] * self.n_params
            return (total / count).cpu().tolist()

        return {
            "mean_abs_grad_all_epochs": _mean(self.abs_grad_sum_all, self.count_all),
            "mean_abs_grad_last_epoch": _mean(self.abs_grad_sum_last, self.count_last),
            "active_observations_all": self.count_all,
            "active_observations_last_epoch": self.count_last,
        }


# ── Top-level assembly + formatting ──────────────────────────────────────────


def build_diagnostics_dict(
    *,
    avg_log_loss: float,
    logloss_by_user: float,
    rows: torch.Tensor,
    loss_by_rating: dict,
    loss_by_delta_t: dict,
    grad_summary: Optional[dict[str, list[float]]] = None,
    complexity_score: Optional[dict] = None,
    s_min: float = 0.0001,
    init_s_max: float = 100.0,
) -> dict:
    """
    Bundle every diagnostic into one JSON-serialisable dict.

    Args:
        avg_log_loss: ``logloss_by_review`` from ``evaluate_on_test_set``.
        logloss_by_user: ditto, ``logloss_by_user``.
        rows: per-user trained weights, shape ``[N, 35]``.
        loss_by_rating: pre-aggregated dict in the flat shape that
            :func:`loss_breakdown_by_rating` returns — e.g.
            ``{"again": 0.42, "again_count": 12345, ...}``. The training
            loop computes this incrementally across splits to avoid
            concatenating per-review tensors.
        loss_by_delta_t: same idea — pre-aggregated short_term/long_term
            log loss + counts plus ``threshold_days``.
        grad_summary: result of ``GradTracker.summarise()``, or None if
            gradient capture isn't wired up yet.
        complexity_score: result of ``complexity.score_paths(...)``, or None.
    """
    per_param = param_distribution_stats(rows, s_min=s_min, init_s_max=init_s_max)
    return {
        "logloss": {
            "by_review": avg_log_loss,
            "by_user": logloss_by_user,
        },
        "loss_by_rating": loss_by_rating,
        "loss_by_delta_t": loss_by_delta_t,
        "per_param": [
            {
                "index": ps.index,
                "bound_lower": ps.bound_lower,
                "bound_upper": ps.bound_upper,
                "p01": ps.p01,
                "median": ps.median,
                "p99": ps.p99,
                "hit_lower_pct": ps.hit_lower_pct,
                "hit_upper_pct": ps.hit_upper_pct,
                "mean_abs_grad_all_epochs": (
                    grad_summary["mean_abs_grad_all_epochs"][ps.index]
                    if grad_summary else None
                ),
                "mean_abs_grad_last_epoch": (
                    grad_summary["mean_abs_grad_last_epoch"][ps.index]
                    if grad_summary else None
                ),
            }
            for ps in per_param
        ],
        "complexity": complexity_score,
        "config": {
            "s_min": s_min,
            "init_s_max": init_s_max,
            "n_rows": int(rows.size(0)),
            "grad_capture_wired": grad_summary is not None,
        },
    }


def format_markdown_report(diag: dict) -> str:
    """Render a diagnostics dict as a Markdown report for human/Claude reading."""
    lines: list[str] = []
    lines.append("# FSRS-7 training diagnostics\n")

    ll = diag["logloss"]
    lines.append(f"**Log loss, per-user (primary):** `{ll['by_user']:.5f}`")
    lines.append(f"**Log loss, per-review (sanity):** `{ll['by_review']:.5f}`\n")

    if diag.get("complexity"):
        c = diag["complexity"]
        lines.append(
            f"**Complexity score:** `{c['total_score']}` "
            f"(AST nodes: `{c['total_ast_nodes']}`, "
            f"cyclomatic: `{c['total_cyclomatic']}`)\n"
        )

    lbr = diag["loss_by_rating"]
    lines.append("## Loss by rating\n")
    lines.append("| Rating | Count | Log loss |")
    lines.append("|---|---:|---:|")
    for name in ("again", "hard", "good", "easy"):
        lines.append(
            f"| {name.capitalize()} | {lbr[f'{name}_count']} | "
            f"{lbr[name]:.5f} |"
        )
    lines.append("")

    lbd = diag["loss_by_delta_t"]
    lines.append("## Loss by interval\n")
    lines.append("| Bucket | Count | Log loss |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Short-term (< {lbd['threshold_days']}d) | "
        f"{lbd['short_term_count']} | {lbd['short_term']:.5f} |"
    )
    lines.append(
        f"| Long-term (>= {lbd['threshold_days']}d) | "
        f"{lbd['long_term_count']} | {lbd['long_term']:.5f} |"
    )
    lines.append("")

    lines.append("## Per-parameter stats\n")
    lines.append(
        "| i | bound_lo | bound_hi | p01 | median | p99 | hit_lo% | hit_hi% | mean|g| all | mean|g| last |"
    )
    lines.append("|--:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for pp in diag["per_param"]:
        g_all = pp.get("mean_abs_grad_all_epochs")
        g_last = pp.get("mean_abs_grad_last_epoch")
        g_all_s = f"{g_all:.2e}" if isinstance(g_all, (int, float)) else "—"
        g_last_s = f"{g_last:.2e}" if isinstance(g_last, (int, float)) else "—"
        lines.append(
            f"| {pp['index']} | {pp['bound_lower']} | {pp['bound_upper']} | "
            f"{pp['p01']:.4g} | {pp['median']:.4g} | {pp['p99']:.4g} | "
            f"{pp['hit_lower_pct']:.1f} | {pp['hit_upper_pct']:.1f} | "
            f"{g_all_s} | {g_last_s} |"
        )
    if not diag["config"]["grad_capture_wired"]:
        lines.append(
            "\n> _Gradient columns are blank because the GradTracker isn't wired into "
            "`train_iter` yet — see `src/autoresearch/diagnostics.py::GradTracker` "
            "docstring for the patch._"
        )
    return "\n".join(lines) + "\n"
