from __future__ import annotations

import os
from pathlib import Path

DEVICE = "cuda"  # not optional

# These paths point to Docker volumes
LMDB_PATH = Path("/data/parallel_db")
TENSOR_CACHE_PATH = Path("/data/tensor_cache_db")
LMDB_SIZE = 32_569_171_968
TENSOR_CACHE_SIZE = 26_319_083_520

TENSOR_CACHE_VERSION = 6
USER_MAX_TRAIN_SPLIT_LENGTHS_KEY = "metadata_user_max_train_split_lengths"
TEST_BATCH_SIZE_MAX = 10_000_000

BATCH_PERM_SEED = 1234

# Writes to the result file if set to True, but incurs a large time cost.
WRITE_RESULT = False
WRITE_RESULT_FILE = "result/FSRS-7-dev.jsonl"

# Only print the result metrics at the end
HIDE_PROGRESS = True

BATCH_SIZE = 256  # Should be a multiple of 128, the block sized used by the cuda kernel
N_EPOCHS = int(os.environ.get("FSRS_N_EPOCHS", "8"))  # nonnegative integer, set to 0 for no optimization. Default 8 (unchanged for the autoresearch loop); the FSRS_N_EPOCHS env override lets the init_w meta-optimizer evaluate the user-facing default with no per-user SGD (FSRS_N_EPOCHS=0).

# Invalidates the cache (slow)
USER_START = 1
USER_END = 3000

# ── Preprocessing invariants ────────────────────────────────────────────────
# Asserted at the end of run.py::main(). These guard against accepting a
# variant that silently changes user filtering or review preprocessing —
# one of the autoresearch hard constraints in CLAUDE.md.
# After the first baseline run on anki-revlogs-3k completes, update
# EXPECTED_N_REVIEWS with the value printed as "n reviews:" by run.py.
EXPECTED_N_USERS: int = 3000
EXPECTED_N_REVIEWS: int | None = 152_354_175  # locked from 2026-05-29 baseline

# Fast-tuning subset knob (PROXY metric). If FSRS_N_USERS > 0, run.py trains+evals
# a SEEDED-RANDOM subset of this many users instead of all 3000 — used ONLY by the
# offline meta-tuners to speed up exploration; the winner is always re-verified on
# the full 3000-user metric before acceptance. Default 0 = full dataset, so
# normal/champion runs and the preprocessing-guard asserts are UNCHANGED. The guard
# against a *model variant* secretly dropping users still holds: a variant is code
# and cannot set this env var.
N_USERS = int(os.environ.get("FSRS_N_USERS", "0"))
SUBSET_SEED = 42  # deterministic subset (stable across evals => stable cache key + metric)

# Requires a prepare.py run to change, and untested
N_SPLITS = 5
