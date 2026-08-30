#!/usr/bin/env python3
"""Evaluate a frozen forward checkpoint on the corrected locked test partition.

This script never trains or selects a checkpoint. It verifies partition
independence, computes forward-image fidelity, runs one-parameter
counterfactuals with equal physical-range steps, and records reproducibility
metadata for the manuscript evidence package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.laser_dataset_v2 import LaserParamDatasetV2, adaptive_collate_fn
from experiments.metadata import load_table
from models.pix2pixhd import Pix2PixHD
from train_forward_model import create_param_map, denormalize_image


PARAMETER_KEYS = ["frequency", "pulse_width", "speed", "dpi"]
PARAMETER_LABELS = {
    "frequency": "Parameter 2",
    "pulse_width": "Parameter 1",
    "speed": "Parameter 3",
    "dpi": "P*",
}
PHYSICAL_RANGES = np.asarray(
    [
        [20.0, 95.0],
        [25.0, 100.0],
        [30000.0, 50000.0],
        [25.0, 175.0],
    ],
    dtype=np.float32,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (project_root / path).resolve()


def clean_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def read_ids(path: Path, column: str = "sample_id") -> set[str]:
    table = load_table(path)
    if column not in table.columns:
        if column == "sample_id":
            column = table.columns[0]
        else:
            raise ValueError(f"Missing partition column {column!r}: {path}")
    return set(table[column].map(clean_id))


def verify_partitions(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    expected_sizes: tuple[int, int, int] | None = (871, 185, 185),
    group_column: str | None = None,
) -> dict[str, object]:
    train_ids = read_ids(train_path)
    val_ids = read_ids(val_path)
    test_ids = read_ids(test_path)
    result = {
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids),
        "overlap_train_val": len(train_ids & val_ids),
        "overlap_train_test": len(train_ids & test_ids),
        "overlap_val_test": len(val_ids & test_ids),
    }
    if expected_sizes is not None and (
        result["train_samples"],
        result["val_samples"],
        result["test_samples"],
    ) != expected_sizes:
        raise RuntimeError(f"Unexpected partition sizes: {result}")
    if result["overlap_train_val"] or result["overlap_train_test"] or result["overlap_val_test"]:
        raise RuntimeError(f"Partition overlap detected: {result}")
    if group_column is not None:
        train_groups = read_ids(train_path, group_column)
        val_groups = read_ids(val_path, group_column)
        test_groups = read_ids(test_path, group_column)
        result.update(
            {
                "group_column": group_column,
                "train_groups": len(train_groups),
                "val_groups": len(val_groups),
                "test_groups": len(test_groups),
                "group_overlap_train_val": len(
                    train_groups & val_groups
                ),
                "group_overlap_train_test": len(
                    train_groups & test_groups
                ),
                "group_overlap_val_test": len(val_groups & test_groups),
            }
        )
        if any(
            result[key]
            for key in [
                "group_overlap_train_val",
                "group_overlap_train_test",
                "group_overlap_val_test",
            ]
        ):
            raise RuntimeError(
                f"Grouped partition overlap detected: {result}"
            )
    return result


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_state(checkpoint) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state", "model_state_dict", "state_dict", "generator_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def build_model(config: dict, num_params: int, device: torch.device) -> Pix2PixHD:
    model_config = config.get("model", {})
    if str(model_config.get("type", "")).lower() != "pix2pixhd":
        raise ValueError("Only the manuscript Pix2PixHD forward interpreter is supported.")
    return Pix2PixHD(
        num_params=num_params,
        use_local_enhancer=model_config.get("use_local_enhancer", True),
        base_channels=model_config.get("base_channels", 64),
        local_channels=model_config.get("local_channels", 32),
        output_mode=model_config.get("output_mode", "direct"),
        use_pattern_mask_channel=model_config.get("use_pattern_mask_channel", False),
    ).to(device)


def ssim_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(pred * pred, 11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 11, stride=1, padding=5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1)
        * (sigma_x + sigma_y + c2)
        + 1e-8
    )
    return ssim_map.flatten(1).mean(dim=1)


def to_numpy_rgb(batch: torch.Tensor) -> np.ndarray:
    return (
        batch.detach()
        .cpu()
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .numpy()
        .astype(np.float32)
    )


def build_action_mask(pattern01: np.ndarray, threshold: float) -> np.ndarray:
    gray = pattern01.mean(axis=2).astype(np.float32)
    denominator = max(float(gray.max() - gray.min()), 1e-6)
    normalized = (gray - float(gray.min())) / denominator
    image_u8 = (normalized * 255.0).round().astype(np.uint8)
    _, otsu = cv2.threshold(image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = otsu > 0
    if mask.mean() < 0.02 or mask.mean() > 0.90:
        mask = normalized > threshold
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask.astype(bool)


def edge_map(image01: np.ndarray) -> np.ndarray:
    gray = (image01.mean(axis=2) * 255.0).round().astype(np.uint8)
    return (cv2.Canny(gray, 50, 150) > 0).astype(np.float32)


def region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else float("nan")


def batch_values(batch: dict, key: str, size: int) -> list[object]:
    value = batch.get(key, [""] * size)
    if isinstance(value, torch.Tensor):
        return [item.item() for item in value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * size


def equal_step_bounds(base: np.ndarray, parameter_index: int, step_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = PHYSICAL_RANGES[parameter_index]
    step = float(upper - lower) * step_fraction
    low = base.copy()
    high = base.copy()
    center = base[:, parameter_index]
    low_value = center - step / 2.0
    high_value = center + step / 2.0
    below = low_value < lower
    high_value[below] += lower - low_value[below]
    low_value[below] = lower
    above = high_value > upper
    low_value[above] -= high_value[above] - upper
    high_value[above] = upper
    low[:, parameter_index] = np.clip(low_value, lower, upper)
    high[:, parameter_index] = np.clip(high_value, lower, upper)
    actual = high[:, parameter_index] - low[:, parameter_index]
    if not np.allclose(actual, step, atol=1e-4):
        raise RuntimeError(f"Unable to construct equal counterfactual step for index {parameter_index}")
    return low, high


def physical_to_normalized(
    values: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    normalized = (values - means[None, :]) / stds[None, :]
    return torch.as_tensor(normalized, dtype=torch.float32, device=device)


def summarize_fidelity(frame: pd.DataFrame, inference_ms_per_sample: float) -> dict[str, float]:
    return {
        "n_test": int(len(frame)),
        "mean_image_mae": float(frame["image_mae"].mean()),
        "median_image_mae": float(frame["image_mae"].median()),
        "p90_image_mae": float(frame["image_mae"].quantile(0.90)),
        "mean_ssim": float(frame["ssim"].mean()),
        "median_ssim": float(frame["ssim"].median()),
        "mean_psnr_db": float(frame["psnr_db"].mean()),
        "mean_edge_mae": float(frame["edge_mae"].mean()),
        "mean_roi_mae": float(frame["roi_mae"].mean()),
        "mean_background_mae": float(frame["background_mae"].mean()),
        "inference_ms_per_sample": float(inference_ms_per_sample),
    }


def summarize_counterfactual(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["rgb_mae", "one_minus_ssim", "edge_mae", "roi_mae"]
    summary = frame.groupby(["parameter", "parameter_label"], as_index=False).agg(
        samples=("sample_id", "nunique"),
        rgb_mae=("rgb_mae", "mean"),
        one_minus_ssim=("one_minus_ssim", "mean"),
        edge_mae=("edge_mae", "mean"),
        roi_mae=("roi_mae", "mean"),
        background_mae=("background_mae", "mean"),
        roi_background_mae_ratio=("roi_background_mae_ratio", "mean"),
    )
    normalized_columns: list[str] = []
    for metric in metrics:
        values = summary[metric].astype(float)
        denominator = float(values.max() - values.min())
        normalized = (values - values.min()) / denominator if denominator > 0 else values * 0
        column = f"{metric}_normalized"
        summary[column] = normalized
        normalized_columns.append(column)
    summary["composite_response_score"] = summary[normalized_columns].mean(axis=1)
    summary["rank"] = summary["composite_response_score"].rank(method="min", ascending=False).astype(int)
    return summary.sort_values(["rank", "parameter"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-manifest", default=None, type=Path)
    parser.add_argument("--allow-unverified-checkpoint", action="store_true")
    parser.add_argument("--test-csv", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--counterfactual-step", type=float, default=0.10)
    parser.add_argument("--mask-threshold", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    if not 0 < args.counterfactual_step < 1:
        raise ValueError("--counterfactual-step must lie in (0, 1).")
    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, project_root)
    checkpoint_path = resolve(args.checkpoint, project_root)
    output_dir = resolve(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_config = config["data"]
    train_path = resolve(data_config["train_manifest"], project_root)
    val_path = resolve(data_config["val_manifest"], project_root)
    config_test = data_config.get("test_manifest", "")
    test_path = (
        resolve(args.test_csv, project_root)
        if args.test_csv is not None
        else resolve(config_test, project_root)
    )
    if not test_path.exists():
        raise FileNotFoundError(f"Locked test manifest missing: {test_path}")
    grouped_protocol = config.get("grouped_outer_protocol", {})
    is_grouped_outer = (
        grouped_protocol.get("policy") == "before_image_grouped_outer_cv"
    )
    partition_audit = verify_partitions(
        train_path,
        val_path,
        test_path,
        expected_sizes=None if is_grouped_outer else (871, 185, 185),
        group_column=(
            grouped_protocol.get("group_column", "before_id")
            if is_grouped_outer
            else None
        ),
    )
    training_manifest_path = (
        resolve(args.training_manifest, project_root)
        if args.training_manifest is not None
        else checkpoint_path.parent / "locked_training_manifest.json"
    )
    training_manifest = None
    if training_manifest_path.exists():
        training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
        if training_manifest.get("status") != "checkpoint_frozen_before_test_evaluation":
            raise RuntimeError("Training manifest does not record a frozen pre-test checkpoint.")
        if training_manifest.get("config_sha256") != sha256(config_path):
            raise RuntimeError("Training manifest config hash does not match the evaluation config.")
        if training_manifest.get("checkpoint_sha256") != sha256(checkpoint_path):
            raise RuntimeError("Training manifest checkpoint hash does not match the evaluated checkpoint.")
        if training_manifest.get("test_used_during_training") is not False:
            raise RuntimeError("Training manifest does not certify test isolation.")
    elif not args.allow_unverified_checkpoint:
        raise FileNotFoundError(
            "A locked_training_manifest.json is required before formal test evaluation. "
            "Run scripts/finalize_locked_forward_training.py after training, or use "
            "--allow-unverified-checkpoint only for a clearly labelled smoke test."
        )

    transforms_config = dict(config.get("transforms") or {})
    transforms_config.setdefault("augment", False)
    dataset = LaserParamDatasetV2(
        annotation_path=str(test_path),
        before_dir=str(resolve(data_config["before_dir"], project_root)),
        after_dir=str(resolve(data_config["after_dir"], project_root)),
        pattern_dir=str(resolve(data_config["pattern_dir"], project_root)),
        pattern_manifest=str(resolve(data_config["pattern_manifest"], project_root))
        if data_config.get("pattern_manifest")
        else None,
        label_stats_path=str(resolve(data_config["label_stats"], project_root)),
        has_after=True,
        transforms_cfg=transforms_config,
    )
    if args.max_samples > 0:
        dataset.df = dataset.df.iloc[: args.max_samples].reset_index(drop=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=adaptive_collate_fn,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = build_model(config, len(dataset.param_cols), device)
    checkpoint = safe_torch_load(checkpoint_path, device)
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/config mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    means = dataset.normalizer.means.detach().cpu().numpy().astype(np.float32)
    stds = dataset.normalizer.stds.detach().cpu().numpy().astype(np.float32)
    if len(means) != 4:
        raise RuntimeError(f"Expected four parameter statistics, found {len(means)}")

    first_batch = next(iter(loader))
    with torch.no_grad():
        before = first_batch["before_img"].to(device)
        pattern = first_batch["pattern_img"].to(device)
        params = first_batch["targets"].to(device)
        model(before, pattern, create_param_map(params, before.shape[-2], before.shape[-1]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    fidelity_rows: list[dict[str, object]] = []
    counterfactual_rows: list[dict[str, object]] = []
    prediction_dir = output_dir / "predictions"
    total_inference_seconds = 0.0
    total_inference_samples = 0

    for batch in tqdm(loader, desc="locked forward evidence"):
        before = batch["before_img"].to(device, non_blocking=True)
        pattern = batch["pattern_img"].to(device, non_blocking=True)
        params = batch["targets"].to(device, non_blocking=True)
        target = denormalize_image(batch["after_img"].to(device, non_blocking=True), device=device).clamp(0, 1)
        param_map = create_param_map(params, before.shape[-2], before.shape[-1])

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            prediction, _ = model(before, pattern, param_map)
            prediction = prediction.clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_inference_seconds += time.perf_counter() - start
        total_inference_samples += prediction.shape[0]

        prediction_np = to_numpy_rgb(prediction)
        target_np = to_numpy_rgb(target)
        pattern_np = to_numpy_rgb(denormalize_image(pattern, device=device).clamp(0, 1))
        absolute = np.abs(prediction_np - target_np).mean(axis=3)
        mse = ((prediction - target) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
        psnr = (20.0 * torch.log10(torch.ones_like(mse) / torch.sqrt(mse))).detach().cpu().numpy()
        ssim = ssim_score(prediction, target).detach().cpu().numpy()
        sample_ids = [clean_id(value) for value in batch_values(batch, "sample_id", prediction.shape[0])]
        before_ids = [clean_id(value) for value in batch_values(batch, "before_id", prediction.shape[0])]
        pattern_ids = [clean_id(value) for value in batch_values(batch, "pattern_id", prediction.shape[0])]

        for index, sample_id in enumerate(sample_ids):
            action = build_action_mask(pattern_np[index], args.mask_threshold)
            background = np.logical_not(action)
            edge_error = np.abs(edge_map(prediction_np[index]) - edge_map(target_np[index]))
            row = {
                "sample_id": sample_id,
                "before_id": before_ids[index],
                "pattern_id": pattern_ids[index],
                "image_mae": float(absolute[index].mean()),
                "ssim": float(ssim[index]),
                "psnr_db": float(psnr[index]),
                "edge_mae": float(edge_error.mean()),
                "roi_mae": region_mean(absolute[index], action),
                "background_mae": region_mean(absolute[index], background),
                "roi_area_ratio": float(action.mean()),
            }
            fidelity_rows.append(row)
            if args.save_predictions:
                prediction_dir.mkdir(parents=True, exist_ok=True)
                save_image(prediction[index].detach().cpu(), prediction_dir / f"{sample_id}_pred.png")

        physical = params.detach().cpu().numpy() * stds[None, :] + means[None, :]
        for parameter_index, parameter in enumerate(PARAMETER_KEYS):
            low_physical, high_physical = equal_step_bounds(
                physical,
                parameter_index,
                args.counterfactual_step,
            )
            low_params = physical_to_normalized(low_physical, means, stds, device)
            high_params = physical_to_normalized(high_physical, means, stds, device)
            with torch.no_grad():
                low_prediction, _ = model(
                    before,
                    pattern,
                    create_param_map(low_params, before.shape[-2], before.shape[-1]),
                )
                high_prediction, _ = model(
                    before,
                    pattern,
                    create_param_map(high_params, before.shape[-2], before.shape[-1]),
                )
            low_prediction = low_prediction.clamp(0, 1)
            high_prediction = high_prediction.clamp(0, 1)
            low_np = to_numpy_rgb(low_prediction)
            high_np = to_numpy_rgb(high_prediction)
            difference = np.abs(high_np - low_np).mean(axis=3)
            one_minus_ssim = 1.0 - ssim_score(high_prediction, low_prediction).detach().cpu().numpy()
            for index, sample_id in enumerate(sample_ids):
                action = build_action_mask(pattern_np[index], args.mask_threshold)
                background = np.logical_not(action)
                edge_error = np.abs(edge_map(high_np[index]) - edge_map(low_np[index]))
                roi_mae = region_mean(difference[index], action)
                background_mae = region_mean(difference[index], background)
                counterfactual_rows.append(
                    {
                        "sample_id": sample_id,
                        "parameter": parameter,
                        "parameter_label": PARAMETER_LABELS[parameter],
                        "step_fraction": args.counterfactual_step,
                        "low_physical": float(low_physical[index, parameter_index]),
                        "high_physical": float(high_physical[index, parameter_index]),
                        "rgb_mae": float(difference[index].mean()),
                        "one_minus_ssim": float(one_minus_ssim[index]),
                        "edge_mae": float(edge_error.mean()),
                        "roi_mae": roi_mae,
                        "background_mae": background_mae,
                        "roi_background_mae_ratio": float(roi_mae / max(background_mae, 1e-8)),
                    }
                )

    fidelity = pd.DataFrame(fidelity_rows)
    counterfactual = pd.DataFrame(counterfactual_rows)
    expected_test = len(dataset)
    if len(fidelity) != expected_test:
        raise RuntimeError(f"Expected {expected_test} fidelity rows, found {len(fidelity)}")
    if len(counterfactual) != expected_test * 4:
        raise RuntimeError(
            f"Expected {expected_test * 4} counterfactual rows, found {len(counterfactual)}"
        )

    inference_ms = total_inference_seconds / max(total_inference_samples, 1) * 1000.0
    fidelity_summary = summarize_fidelity(fidelity, inference_ms)
    counterfactual_summary = summarize_counterfactual(counterfactual)
    fidelity.to_csv(output_dir / "forward_fidelity_per_sample.csv", index=False, encoding="utf-8-sig")
    counterfactual.to_csv(
        output_dir / "forward_counterfactual_per_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    counterfactual_summary.to_csv(
        output_dir / "forward_counterfactual_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "forward_fidelity_summary.json").write_text(
        json.dumps(fidelity_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "frozen_validation_selected_checkpoint_test_once",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_manifest": str(training_manifest_path) if training_manifest else "",
        "training_manifest_sha256": sha256(training_manifest_path) if training_manifest else "",
        "checkpoint_provenance_verified": training_manifest is not None,
        "train_csv": str(train_path),
        "val_csv": str(val_path),
        "test_csv": str(test_path),
        "label_stats": str(resolve(data_config["label_stats"], project_root)),
        "partition_audit": partition_audit,
        "evaluated_samples": len(dataset),
        "counterfactual_step_fraction": args.counterfactual_step,
        "parameter_order": PARAMETER_KEYS,
        "physical_ranges": PHYSICAL_RANGES.tolist(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch_version": torch.__version__,
        "fidelity_summary": fidelity_summary,
        "counterfactual_rank": counterfactual_summary[
            ["parameter", "parameter_label", "composite_response_score", "rank"]
        ].to_dict("records"),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
