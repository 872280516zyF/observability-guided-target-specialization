#!/usr/bin/env python3
"""Train one smallNORB image-to-factor model on frozen object-instance splits."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import prepare_smallnorb_observability_20260807 as prepared  # noqa: E402
from scripts import train_observability_gradient_protection_20260806 as gradient  # noqa: E402


FACTORS = ("elevation", "azimuth", "lighting")
CLASS_COUNTS = {"elevation": 9, "azimuth": 18, "lighting": 6}
MODES = ("shared", "selected_spatial", "all_spatial")
OPTIMIZATIONS = ("standard", "symmetric_pcgrad")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SmallNORBSplit(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.frame = pd.read_csv(csv_path)
        required = {"source_split", "row_index", "group_id", *FACTORS}
        if not required.issubset(self.frame.columns):
            raise ValueError("Missing smallNORB columns {}".format(sorted(required - set(self.frame.columns))))
        self.maps = {}
        for split in self.frame["source_split"].unique():
            values, shape, _ = prepared.matrix_memmap(prepared.matrix_path(str(split), "dat"))
            if shape != (24300, 2, 96, 96):
                raise ValueError("Unexpected smallNORB image shape")
            self.maps[str(split)] = values

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = np.asarray(
            self.maps[str(row["source_split"])][int(row["row_index"]), 0], dtype=np.float32
        ).copy()
        tensor = torch.from_numpy(image).unsqueeze(0) / 255.0
        targets = torch.as_tensor([int(row[factor]) for factor in FACTORS], dtype=torch.long)
        return tensor, targets, str(row["group_id"]), int(row["row_index"])


class SmallNORBFactorModel(nn.Module):
    def __init__(self, mode: str, specialist_targets: Sequence[str], dropout: float) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        if mode == "shared" and specialist_targets:
            raise ValueError("Shared model cannot receive specialist targets")
        if mode == "all_spatial" and set(specialist_targets) != set(FACTORS):
            raise ValueError("all_spatial must specialize all factors")
        self.mode = mode
        self.specialist_targets = tuple(specialist_targets)
        channels = [1, 32, 64, 128, 256]
        blocks: List[nn.Module] = []
        for incoming, outgoing in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing),
                    nn.SiLU(inplace=True),
                ]
            )
        self.encoder = nn.Sequential(*blocks)
        self.shared = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(inplace=True), nn.Dropout(dropout)
        )
        self.base_heads = nn.ModuleDict(
            {factor: nn.Linear(256, CLASS_COUNTS[factor]) for factor in FACTORS}
        )
        self.attention = nn.ModuleDict()
        self.specialist_heads = nn.ModuleDict()
        for factor in self.specialist_targets:
            self.attention[factor] = nn.Sequential(
                nn.Conv2d(256, 64, 1), nn.SiLU(inplace=True), nn.Conv2d(64, 1, 1)
            )
            head = nn.Sequential(
                nn.Linear(512, 256), nn.SiLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(256, CLASS_COUNTS[factor]),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.specialist_heads[factor] = head

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        maps = self.encoder(images)
        global_feature = self.shared(F.adaptive_avg_pool2d(maps, 1).flatten(1))
        output = {factor: self.base_heads[factor](global_feature) for factor in FACTORS}
        for factor in self.specialist_targets:
            scores = self.attention[factor](maps).flatten(2)
            weights = torch.softmax(scores, dim=2)
            attended = (maps.flatten(2) * weights).sum(2)
            output[factor] = output[factor] + self.specialist_heads[factor](
                torch.cat([global_feature, attended], dim=1)
            )
        return output


def loader(path: Path, batch_size: int, workers: int, train: bool, seed: int) -> DataLoader:
    dataset = SmallNORBSplit(path)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        generator=generator if train else None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def error_values(factor: str, truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    difference = np.abs(truth.astype(float) - prediction.astype(float))
    if factor == "azimuth":
        difference = np.minimum(difference, CLASS_COUNTS[factor] - difference)
        return difference / (CLASS_COUNTS[factor] / 2.0)
    return difference / float(CLASS_COUNTS[factor] - 1)


def predictions(
    model: nn.Module, current_loader: DataLoader, device: torch.device
) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for images, targets, group_ids, row_indices in current_loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = {factor: torch.softmax(logits[factor], dim=1).cpu().numpy() for factor in FACTORS}
            target_values = targets.numpy()
            for row_index in range(len(group_ids)):
                row: Dict[str, object] = {
                    "condition_id": "{}__{}".format(group_ids[row_index], int(row_indices[row_index])),
                    "group_id": group_ids[row_index],
                    "row_index": int(row_indices[row_index]),
                }
                for factor_index, factor in enumerate(FACTORS):
                    row["true_{}".format(factor)] = int(target_values[row_index, factor_index])
                    current = probabilities[factor][row_index]
                    row["pred_{}".format(factor)] = int(np.argmax(current))
                    for class_index, value in enumerate(current):
                        row["prob_{}_{}".format(factor, class_index)] = float(value)
                rows.append(row)
    frame = pd.DataFrame(rows)
    for factor in FACTORS:
        frame["nmae_{}".format(factor)] = error_values(
            factor, frame["true_{}".format(factor)].to_numpy(), frame["pred_{}".format(factor)].to_numpy()
        )
        frame["correct_{}".format(factor)] = (
            frame["true_{}".format(factor)] == frame["pred_{}".format(factor)]
        ).astype(float)
    return frame


def metrics(frame: pd.DataFrame, high_targets: Sequence[str]) -> Dict[str, object]:
    grouped = frame.groupby("group_id", sort=False)
    nmae = {factor: float(grouped["nmae_{}".format(factor)].mean().mean()) for factor in FACTORS}
    accuracy = {factor: float(grouped["correct_{}".format(factor)].mean().mean()) for factor in FACTORS}
    low = [factor for factor in FACTORS if factor not in set(high_targets)]
    return {
        "nmae": nmae,
        "accuracy": accuracy,
        "high_nmae": float(np.mean([nmae[factor] for factor in high_targets])),
        "mean_nmae": float(np.mean(list(nmae.values()))),
        "worst_low_nmae": float(max([nmae[factor] for factor in low], default=0.0)),
        "groups": int(frame["group_id"].nunique()),
        "images": int(len(frame)),
    }


def selection_score(values: Dict[str, object]) -> float:
    return float(values["high_nmae"] + 0.25 * values["mean_nmae"] + 0.10 * values["worst_low_nmae"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--optimization", choices=OPTIMIZATIONS, default="standard")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", default="")
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--high-targets", nargs="+", required=True)
    parser.add_argument("--specialist-targets", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "pilot" and (not args.val_csv or args.test_csv):
        raise ValueError("Pilot requires validation and forbids official test")
    if args.phase == "confirm" and (args.val_csv or not args.test_csv):
        raise ValueError("Confirm uses fixed epochs on official train and evaluates official test only")
    if not set(args.high_targets).issubset(FACTORS):
        raise ValueError("Unknown high target")
    if not set(args.specialist_targets).issubset(FACTORS):
        raise ValueError("Unknown specialist target")
    if args.mode == "selected_spatial" and not args.specialist_targets:
        raise ValueError("selected_spatial requires specialist targets")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / ("validation_summary.json" if args.phase == "pilot" else "test_metrics.json")
    if summary_path.is_file():
        print("[SKIP] {}".format(output), flush=True)
        return

    seed_everything(args.seed)
    train_loader = loader(Path(args.train_csv), args.batch_size, args.num_workers, True, args.seed)
    evaluation_loader = loader(
        Path(args.val_csv if args.phase == "pilot" else args.test_csv),
        args.batch_size, args.num_workers, False, args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNORBFactorModel(args.mode, args.specialist_targets, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    pcgrad_generator = torch.Generator(device="cpu").manual_seed(args.seed + 20260807)
    best_score, best_epoch, stale = math.inf, -1, 0
    checkpoint = output / "best_model.pth"
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_losses, conflicts, cosines = [], [], []
        for images, targets, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            per_task = [F.cross_entropy(logits[factor], targets[:, index]) for index, factor in enumerate(FACTORS)]
            optimizer.zero_grad(set_to_none=True)
            if args.optimization == "symmetric_pcgrad":
                diagnostic = gradient.apply_gradient_rule(
                    model, per_task, "symmetric_pcgrad", [], 1.0, pcgrad_generator
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
                optimizer.step()
                conflicts.append(float(diagnostic["conflict_fraction"]))
                cosines.append(float(diagnostic["mean_preprojection_cosine"]))
                total = torch.stack(per_task).mean().detach()
            else:
                total = torch.stack(per_task).mean()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
                optimizer.step()
            epoch_losses.append(float(total.cpu()))
        scheduler.step()

        if args.phase == "pilot":
            frame = predictions(model, evaluation_loader, device)
            current_metrics = metrics(frame, args.high_targets)
            score = selection_score(current_metrics)
        else:
            current_metrics, score = {}, float("nan")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses)),
                "val_selection_score": score,
                "conflict_fraction": float(np.mean(conflicts)) if conflicts else np.nan,
                "mean_preprojection_cosine": float(np.mean(cosines)) if cosines else np.nan,
            }
        )
        print(
            "[{} {} {} seed{}] epoch={:03d} loss={:.6f} score={}".format(
                args.phase, args.mode, args.optimization, args.seed, epoch + 1,
                float(np.mean(epoch_losses)), "{:.6f}".format(score) if np.isfinite(score) else "fixed"
            ),
            flush=True,
        )
        if args.phase == "pilot":
            if score < best_score - 1e-8:
                best_score, best_epoch, stale = score, epoch + 1, 0
                torch.save({"model": model.state_dict()}, checkpoint)
            else:
                stale += 1
                if stale >= args.patience:
                    break

    if args.phase == "pilot":
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    else:
        best_epoch = args.epochs
        torch.save({"model": model.state_dict()}, checkpoint)
    frame = predictions(model, evaluation_loader, device)
    frame.to_csv(
        output / ("validation_predictions.csv" if args.phase == "pilot" else "test_predictions.csv"),
        index=False,
    )
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    final_metrics = metrics(frame, args.high_targets)
    payload = {
        "phase": args.phase,
        "mode": args.mode,
        "optimization": args.optimization,
        "seed": args.seed,
        "high_targets": list(args.high_targets),
        "specialist_targets": list(args.specialist_targets),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "selection_score": selection_score(final_metrics),
        "metrics": final_metrics,
        "input": "left 96x96 grayscale image only",
        "augmentation": "none; geometric and photometric transforms would alter labeled factors",
        "official_test_was_supplied": args.phase == "confirm",
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
