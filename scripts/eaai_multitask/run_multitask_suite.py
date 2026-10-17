#!/usr/bin/env python3
"""Run an explicitly selected set of EAAI multi-task jobs sequentially.

This launcher delegates every job to :mod:`train_multitask`. It never retries
or overwrites a run directory. Restrict ``--config-ids``, ``--folds``, or
``--seeds`` for a smoke test; the frozen full grid is four configurations,
five folds, and three seeds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-ids", nargs="+", choices=("B1", "B2", "B3", "B5"), required=True)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 52, 62), required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--split-root", default="data/images3/grouped_outer_cv_20260730")
    parser.add_argument("--before-dir", default="data/images3/before")
    parser.add_argument("--after-dir", default="data/images3/after")
    parser.add_argument("--observability-root", default="outputs/grouped_outer_cv_20260730")
    parser.add_argument("--output-root", default="outputs/eaai_multitask")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = [
        "--project-root", args.project_root,
        "--split-root", args.split_root,
        "--before-dir", args.before_dir,
        "--after-dir", args.after_dir,
        "--observability-root", args.observability_root,
        "--output-root", args.output_root,
        "--device", args.device,
    ]
    for config_id in args.config_ids:
        for fold in args.folds:
            for seed in args.seeds:
                command = [
                    sys.executable,
                    "-m",
                    "scripts.eaai_multitask.train_multitask",
                    "--config-id", config_id,
                    "--fold", str(fold),
                    "--seed", str(seed),
                    *common,
                ]
                print("Running {} fold={} seed={}".format(config_id, fold, seed), flush=True)
                subprocess.run(command, check=True, cwd=args.project_root)


if __name__ == "__main__":
    main()
