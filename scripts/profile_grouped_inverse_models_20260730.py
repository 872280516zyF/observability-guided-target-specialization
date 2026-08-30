#!/usr/bin/env python3
"""Record trainable parameters and profiler-counted FLOPs for Fig. 3/SI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dpi_branch_ablation import (  # noqa: E402
    Shared4BNNet,
    TargetedIntegratedExpertBNNet,
    count_trainable_parameters,
)
from scripts.run_plain_cnn_inverse_baseline import PlainCNNInverseNet  # noqa: E402


def counted_flops(model: torch.nn.Module, image_size: int) -> int:
    model = model.cpu().eval()
    batch = {
        "effect": torch.zeros(1, 3, image_size, image_size),
        "before": torch.zeros(1, 3, image_size, image_size),
    }
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True,
        ) as profile:
            model(batch)
    return int(sum(int(event.flops or 0) for event in profile.key_averages()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    models = {
        "plain_cnn": PlainCNNInverseNet(width=32),
        "shared_head_resnet": Shared4BNNet(
            "resnet18", False, 256, 0.3
        ),
        "texture_expert_any_target": TargetedIntegratedExpertBNNet(
            "resnet18", False, 256, 0.3, 64, "dpi", True
        ),
        "nonguided_equal_capacity": TargetedIntegratedExpertBNNet(
            "resnet18", False, 256, 0.3, 64, "dpi", False
        ),
    }
    rows = []
    for model_id, model in models.items():
        rows.append(
            {
                "model_id": model_id,
                "image_size": args.image_size,
                "trainable_parameters": count_trainable_parameters(model),
                "profiler_counted_flops_batch1": counted_flops(
                    model, args.image_size
                ),
                "flop_scope": (
                    "PyTorch CPU profiler-counted convolution and linear "
                    "operations; use for within-study comparison only"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        output / "model_complexity.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": args.image_size,
        "batch_size": 1,
        "note": (
            "All four specialist-target placements share the "
            "texture_expert_any_target complexity. The non-guided control is "
            "constructed to have identical trainable parameter count."
        ),
        "rows": frame.to_dict(orient="records"),
    }
    (output / "model_complexity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
