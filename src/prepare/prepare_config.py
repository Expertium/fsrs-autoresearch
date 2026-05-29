from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal, get_args

import torch


ModelName = Literal[
    "FSRS-7",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", default=8, type=int)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--max-user-id", type=int, default=None)
    parser.add_argument("--partitions", default="none", choices=["none", "deck", "preset"])
    # FSRS-7 is always run with --short --secs --recency, so these default to True.
    # Use --no-short / --no-secs / --no-recency to disable.
    parser.add_argument("--recency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--secs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--short", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default", action="store_true")
    parser.add_argument("--S0", action="store_true")
    parser.add_argument("--sched_penalties", default=False, action="store_true")
    parser.add_argument("--two_buttons", action="store_true")
    parser.add_argument("--data", default="../anki-revlogs-3k")
    parser.add_argument("--duration", action="store_true")
    parser.add_argument("--no_test_same_day", action="store_true")
    parser.add_argument("--no_train_same_day", action="store_true")
    parser.add_argument("--equalize_test_with_non_secs", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--file", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--algo", default="FSRS-7")
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--train_equals_test", action="store_true")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--torch_num_threads", type=int, default=1)
    return parser


def _parse_cuda_devices(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    value = raw.strip()
    if value.lower() in {"all", "*"}:
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    return [int(part) for part in re.split(r"[,\s]+", value) if part]


class Config:
    def __init__(self, args: argparse.Namespace):
        self.raw_args = args
        self.dev_mode: bool = args.dev
        self.default_params: bool = args.default
        self.model_name: ModelName = args.algo
        self.max_user_id: int | None = args.max_user_id
        self.use_secs_intervals: bool = args.secs
        self.lstm_use_duration: bool = args.duration
        self.no_test_same_day: bool = args.no_test_same_day
        self.no_train_same_day: bool = args.no_train_same_day
        self.equalize_test_with_non_secs: bool = args.equalize_test_with_non_secs
        self.two_buttons: bool = args.two_buttons
        self.only_S0: bool = args.S0
        self.sched_penalties: bool = args.sched_penalties
        self.save_evaluation_file: bool = args.file
        self.generate_plots: bool = args.plot
        self.save_weights: bool = args.weights
        self.partitions: str = args.partitions
        self.save_raw_output: bool = args.raw
        self.num_processes: int = args.processes
        self.data_path: Path = Path(args.data)
        self.use_recency_weighting: bool = args.recency
        self.train_equals_test: bool = args.train_equals_test
        self.cuda_device_ids: list[int] | None = _parse_cuda_devices(args.gpus)
        self.n_splits: int = args.n_splits
        self.batch_size: int = args.batch_size
        self.max_seq_len: int = args.max_seq_len
        self.include_short_term: bool = args.short
        self.torch_num_threads: int = args.torch_num_threads
        torch.set_num_threads(self.torch_num_threads)

        if self.model_name not in get_args(ModelName):
            raise ValueError(
                f"Model name '{self.model_name}' must be one of {get_args(ModelName)}"
            )

        self.fsrs_optimizer_module_path: str = "../fsrs-optimizer/src/fsrs_optimizer/"
        self.device: torch.device = torch.device("cpu")
        self.verbose_logging: bool = False
        self.verbose_inadequate_data: bool = False
        # FSRS-7 with --secs (the default) uses s_min = 0.0001; without --secs, 0.01.
        self.s_min: float = 0.0001 if self.use_secs_intervals else 0.01
        self.init_s_max: float = 100.0
        self.s_max: float = 36500.0
        self.seed: int = 42
