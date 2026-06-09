#!/usr/bin/env python3
"""
batch_eval_default.py — load-once / eval-many batch evaluator for the
central_diff default-parameter meta-opt (the FSRS_N_EPOCHS=0 phase).

WHY: each central-difference meta-step needs 2*N+1 evals, and the current
``evaluate()`` runs each as a SEPARATE ``docker compose run`` — every eval pays
container spin-up + CUDA init + LMDB tensor-cache load + per-split GPU load (the
idle "troughs" in VRAM usage) only to do one brief forward pass (the "spike").
That overhead dominates (~60% of an ~8 s eval), so 69 evals/step is ~6x more
work than the compute alone.

WHAT: evaluate a LIST of candidate FSRS7_DEFAULT vectors in ONE process — load
the env + tensor cache + each user-split's GPU tensors ONCE, then loop all
candidates against the loaded split data. Loads drop from 2*N+1 per step to ~1.
Measured ~2-2.5x faster default tuning.

CORRECTNESS: produces ``logloss_by_user`` BIT-IDENTICAL (modulo the ~2e-6 GPU
atomics noise) to a full ``python -m src.main.run`` per candidate — same splits,
same ``evaluate_on_test_set``, same aggregation; the only change is loop order.
Each candidate is injected by monkeypatching ``fsrs_v7_constants
.FSRS7_DEFAULT_35_VALUES`` (which ``get_initial_params_for_optimization`` reads
fresh per call and is NOT torch.compile'd, so the patch takes). Verify with
``--self-check`` (re-evaluates the live champion default and prints by_user).

SCOPE: FSRS_N_EPOCHS=0 only (user-facing default phase). Training is skipped, so
each candidate is just init -> evaluate. Sigma tuning (N_EPOCHS=8) would need
training per candidate AND handling the torch.compile'd penalty_loss reading the
sigma tuple — a deliberate later addition, not this file.

IO (paths under result/init_w_metaopt/, the repo dir is bind-mounted):
  in : batch_candidates.json = {"candidates": [[...34 floats...], ...]}
  out: batch_results.json     = {"by_user": [f, ...]}  one per candidate, in order

Run inside docker (build-gate the ext first, like run.sh):
  docker compose --progress quiet run --rm -e FSRS_N_EPOCHS=0 srs-benchmark \
      bash -c "python setup.py -q build_ext --inplace && \
               python -m src.autoresearch.batch_eval_default"
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lmdb
import torch

from src.main import run as R
from src.main.config import (
    DEVICE,
    EXPECTED_N_USERS,
    LMDB_PATH,
    LMDB_SIZE,
    N_EPOCHS,
    N_USERS,
    SUBSET_SEED,
    USER_END,
    USER_MAX_TRAIN_SPLIT_LENGTHS_KEY,
    USER_START,
)
from src.main.fsrs import fsrs_v7_constants
from src.main.tensor_cache import (
    load_cached_review_data,
    load_cached_test_only,
    load_or_rebuild_tensor_cache,
)

OUTPUT_DIR = Path("result/init_w_metaopt")
CAND_PATH = OUTPUT_DIR / "batch_candidates.json"
OUT_PATH = OUTPUT_DIR / "batch_results.json"


def _setup_splits_and_cache():
    """Replicate src.main.run.main()'s split + tensor-cache setup EXACTLY so the
    cache key matches (no rebuild) and the eval is identical."""
    env = lmdb.open(str(LMDB_PATH), map_size=LMDB_SIZE, readonly=True, lock=False)
    users = list(range(USER_START, USER_END + 1))
    # Mirror run.py's fast-tuning subset so the 0-epoch default phase evaluates the
    # SAME seeded 2k proxy as the 8-epoch sigma/gate evals (consistent metric).
    if N_USERS and N_USERS < len(users):
        import random as _random
        users = sorted(_random.Random(SUBSET_SEED).sample(users, N_USERS))
    with env.begin(write=False) as txn:
        user_max_train_split_lengths = R.load_metadata_tensor(
            txn, USER_MAX_TRAIN_SPLIT_LENGTHS_KEY
        )
    rel_user_sum = user_max_train_split_lengths[torch.tensor(users) - 1]
    split_factor_k = R.get_split_factor_k(rel_user_sum.sum())
    user_splits = R.split_users_by_train_length(
        users, user_max_train_split_lengths, split_factor_k
    )
    user_splits.reverse()
    for l in user_splits:
        l.sort()
    cache_env = load_or_rebuild_tensor_cache(env, user_splits)
    return env, cache_env, user_splits


def evaluate_candidates(candidates: list[list[float]]) -> list[float]:
    """Return logloss_by_user for each candidate FSRS7_DEFAULT vector, loading
    each split's data ONCE and evaluating all candidates against it."""
    assert DEVICE == "cuda", "Only cuda is supported."
    assert N_EPOCHS == 0, (
        f"batch_eval_default is the FSRS_N_EPOCHS=0 (default) phase only; got "
        f"N_EPOCHS={N_EPOCHS}. Set -e FSRS_N_EPOCHS=0."
    )
    K = len(candidates)
    aggs = [R.EvaluationAggregate() for _ in range(K)]

    env, cache_env, user_splits = _setup_splits_and_cache()
    try:
        for split_i, user_subset in enumerate(user_splits):
            torch.cuda.empty_cache()
            review_data = load_cached_review_data(cache_env, split_i, DEVICE)
            # 0-epoch: no training, so we never load train_data — just test_data once.
            test_data = load_cached_test_only(
                cache_env, split_i, DEVICE, review_data, load_rmse_bins=False
            )
            for ci, cand in enumerate(candidates):
                # Inject this candidate as the user-facing default; 0-epoch means
                # the init IS the evaluated param vector (no per-user SGD).
                fsrs_v7_constants.FSRS7_DEFAULT_35_VALUES = tuple(float(v) for v in cand)
                fsrs_params = R.make_initial_fsrs_params(len(user_subset))
                with torch.no_grad():
                    result = R.evaluate_on_test_set(fsrs_params, user_subset, test_data)
                aggs[ci].add(result)
            del review_data, test_data
            torch.cuda.empty_cache()
    finally:
        cache_env.close()
        env.close()

    # Preprocessing guard (mirror run.py): every candidate must see all users
    # (or the full seeded subset when FSRS_N_USERS>0).
    expected_users = N_USERS if N_USERS else EXPECTED_N_USERS
    for ci, a in enumerate(aggs):
        assert a.user_count == expected_users, (
            f"candidate {ci}: user_count={a.user_count} != expected={expected_users} "
            f"(N_USERS={N_USERS}) — split/cache setup diverged from run.py."
        )
    return [a.logloss_by_user for a in aggs]


def main() -> None:
    if "--self-check" in sys.argv:
        # Re-evaluate the live champion default twice: result must equal a normal
        # run's by_user and be stable across the two list positions.
        champ = [float(v) for v in fsrs_v7_constants.FSRS7_DEFAULT_35_VALUES]
        t0 = time.perf_counter()
        out = evaluate_candidates([champ, champ])
        print(f"[self-check] champion default by_user (x2): {out}")
        print(f"[self-check] {time.perf_counter() - t0:.0f}s for 2 evals")
        return

    candidates = json.loads(CAND_PATH.read_text())["candidates"]
    t0 = time.perf_counter()
    by_user = evaluate_candidates(candidates)
    dt = time.perf_counter() - t0
    OUT_PATH.write_text(json.dumps({"by_user": by_user}))
    print(
        f"batch_eval_default: {len(candidates)} candidates in {dt:.0f}s "
        f"({dt / max(len(candidates), 1):.1f}s/cand) -> {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
