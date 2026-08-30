#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.metadata import infer_metadata_columns, load_table  # noqa: E402


PARAM_SPECS = [
    ("frequency", "frequency", 20.0, 95.0),
    ("pulse_width", "pulse_width", 25.0, 100.0),
    ("speed", "speed", 30000.0, 50000.0),
    ("dpi", "dpi", 25.0, 175.0),
]
PARAM_ALIASES = {
    "frequency": ["frequency", "freq", "频率", "棰戠巼"],
    "pulse_width": ["pulse_width", "pulse", "脉宽", "鑴夊"],
    "speed": ["speed", "速度", "閫熷害"],
    "dpi": ["dpi", "DPI"],
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_split(path: str | Path, split_name: str) -> pd.DataFrame:
    df = infer_metadata_columns(load_table(path))
    df["split"] = split_name
    for _, name, _, _ in PARAM_SPECS:
        column = resolve_param_column(df, name)
        df[name] = pd.to_numeric(df[column], errors="coerce")
    return df.reset_index(drop=True)


def resolve_param_column(df: pd.DataFrame, param_name: str) -> str:
    for column in PARAM_ALIASES[param_name]:
        if column in df.columns:
            return column
    raise ValueError(f"Missing parameter column for {param_name}; tried {PARAM_ALIASES[param_name]}")


def image_path(row: pd.Series, before_dir: Path, after_dir: Path, image_role: str) -> Path:
    sample_id = str(row["sample_id"])
    before_id = str(row["before_id"]) if "before_id" in row and pd.notna(row["before_id"]) else sample_id
    if image_role == "before":
        return before_dir / f"{before_id}.jpg"
    return after_dir / f"{sample_id}.jpg"


def open_rgb(path: Path, size: int | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def normalize_targets(values: np.ndarray) -> np.ndarray:
    out = []
    for idx, (_, _name, lower, upper) in enumerate(PARAM_SPECS):
        out.append(np.clip((values[:, idx] - lower) / (upper - lower), 0.0, 1.0))
    return np.stack(out, axis=1).astype(np.float32)


def denormalize_targets(values: np.ndarray) -> np.ndarray:
    out = []
    for idx, (_, _name, lower, upper) in enumerate(PARAM_SPECS):
        out.append(values[:, idx] * (upper - lower) + lower)
    return np.stack(out, axis=1).astype(np.float32)


def target_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[name for _, name, _, _ in PARAM_SPECS]].to_numpy(dtype=np.float32)


def color_hist(channel: np.ndarray, bins: int = 16) -> np.ndarray:
    hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else hist


def simple_lbp_hist(gray: np.ndarray) -> np.ndarray:
    # 8-neighbor local binary pattern histogram on the interior pixels.
    center = gray[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    for bit, nb in enumerate(neighbors):
        code |= ((nb >= center).astype(np.uint8) << bit)
    hist, _ = np.histogram(code, bins=32, range=(0, 256), density=False)
    hist = hist.astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else hist


def texture_features_one(path: Path, size: int) -> np.ndarray:
    img = open_rgb(path, size=size)
    gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)
    feats: list[np.ndarray] = []

    for c in range(3):
        channel = img[:, :, c]
        stats = np.array(
            [
                channel.mean(),
                channel.std(),
                np.percentile(channel, 10),
                np.percentile(channel, 50),
                np.percentile(channel, 90),
            ],
            dtype=np.float32,
        )
        feats.append(stats)
        feats.append(color_hist(channel, bins=16))

    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    edge_density = np.array(
        [
            gray.mean(),
            gray.std(),
            np.percentile(gray, 10),
            np.percentile(gray, 50),
            np.percentile(gray, 90),
            grad.mean(),
            grad.std(),
            float((grad > 0.04).mean()),
            float((grad > 0.08).mean()),
            float((gray > 0.20).mean()),
            float((gray > 0.50).mean()),
        ],
        dtype=np.float32,
    )
    feats.append(edge_density)
    feats.append(color_hist(gray, bins=24))
    feats.append(color_hist(np.clip(grad / (grad.max() + 1e-6), 0.0, 1.0), bins=16))
    feats.append(simple_lbp_hist(gray))
    return np.concatenate(feats).astype(np.float32)


def build_texture_features(df: pd.DataFrame, before_dir: Path, after_dir: Path, image_role: str, size: int) -> np.ndarray:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"texture-{image_role}"):
        rows.append(texture_features_one(image_path(row, before_dir, after_dir, image_role), size=size))
    return np.stack(rows, axis=0)


def build_resnet_features(
    df: pd.DataFrame,
    before_dir: Path,
    after_dir: Path,
    image_role: str,
    size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    class ImageFeatureDataset(Dataset):
        def __init__(self, table: pd.DataFrame):
            self.table = table.reset_index(drop=True)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        def __len__(self) -> int:
            return len(self.table)

        def __getitem__(self, idx: int):
            row = self.table.iloc[idx]
            path = image_path(row, before_dir, after_dir, image_role)
            img = Image.open(path).convert("RGB")
            return self.transform(img)

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    loader = DataLoader(ImageFeatureDataset(df), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"resnet-{image_role}"):
            out = model(batch.to(dev)).detach().cpu().numpy()
            feats.append(out.astype(np.float32))
    return np.concatenate(feats, axis=0)


def cache_path(cache_dir: Path, feature_kind: str, split_name: str, image_role: str, size: int) -> Path:
    return cache_dir / f"{feature_kind}_{image_role}_{split_name}_{size}.npy"


def get_features(
    df: pd.DataFrame,
    split_name: str,
    feature_kind: str,
    before_dir: Path,
    after_dir: Path,
    image_role: str,
    size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    cache_dir: Path,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, feature_kind, split_name, image_role, size)
    if path.exists():
        return np.load(path)
    if feature_kind == "texture":
        x = build_texture_features(df, before_dir, after_dir, image_role, size)
    elif feature_kind == "resnet18":
        x = build_resnet_features(df, before_dir, after_dir, image_role, size, batch_size, num_workers, device)
    else:
        raise ValueError(f"Unsupported feature kind: {feature_kind}")
    np.save(path, x)
    return x


def build_estimator(variant: str, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    if variant in {"texture_rf", "resnet_rf"}:
        return RandomForestRegressor(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if variant == "resnet_extra_trees":
        return ExtraTreesRegressor(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if variant == "resnet_svr":
        return make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=10.0, epsilon=0.02, gamma="scale")))
    if variant == "resnet_xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError(f"xgboost is not available: {exc}") from exc
        base = XGBRegressor(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
            reg_lambda=1.0,
        )
        return MultiOutputRegressor(base)
    raise ValueError(f"Unsupported variant: {variant}")


def variant_feature_kind(variant: str) -> str:
    if variant.startswith("texture_"):
        return "texture"
    return "resnet18"


def variant_label(variant: str) -> str:
    return {
        "texture_rf": "Texture-feature RF",
        "resnet_rf": "ResNet-feature RF",
        "resnet_extra_trees": "ResNet-feature ExtraTrees",
        "resnet_svr": "ResNet-feature SVR",
        "resnet_xgboost": "ResNet-feature XGBoost",
    }.get(variant, variant)


def export_predictions(
    df: pd.DataFrame,
    pred_norm: np.ndarray,
    output_csv: Path,
    run_name: str,
    split_name: str,
) -> pd.DataFrame:
    pred_norm = np.clip(pred_norm, 0.0, 1.0)
    pred_raw = denormalize_targets(pred_norm)
    true_raw = target_matrix(df)
    ape = np.abs(pred_raw - true_raw) / np.maximum(np.abs(true_raw), 1e-6) * 100.0
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        item = {
            "run_name": run_name,
            "split": split_name,
            "sample_id": str(row["sample_id"]),
            "before_id": str(row["before_id"]) if "before_id" in row and pd.notna(row["before_id"]) else str(row["sample_id"]),
            "pattern_id": str(row["pattern_id"]) if "pattern_id" in row and pd.notna(row["pattern_id"]) else "",
            "batch_id": str(row["batch_id"]) if "batch_id" in row and pd.notna(row["batch_id"]) else "",
        }
        for param_idx, (_, name, _, _) in enumerate(PARAM_SPECS):
            item[f"true_{name}"] = float(true_raw[idx, param_idx])
            item[f"pred_{name}"] = float(pred_raw[idx, param_idx])
            item[f"abs_err_{name}"] = abs(item[f"pred_{name}"] - item[f"true_{name}"])
            item[f"ape_{name}"] = float(ape[idx, param_idx])
        item["mean_ape"] = float(np.mean(ape[idx]))
        rows.append(item)
    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return out


def summarize_predictions(pred: pd.DataFrame) -> dict:
    return {
        "mape_physical": float(pred["mean_ape"].mean()),
        "param_mape_physical": {name: float(pred[f"ape_{name}"].mean()) for _, name, _, _ in PARAM_SPECS},
        "param_abs_err_physical": {name: float(pred[f"abs_err_{name}"].mean()) for _, name, _, _ in PARAM_SPECS},
        "n_samples": int(len(pred)),
    }


def run_variant(args: argparse.Namespace, variant: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    output_dir = Path(args.output_dir) / variant
    run_name = f"seed{args.seed}" if args.grouped_layout else f"{variant}_seed{args.seed}"
    run_dir = output_dir / run_name
    predictions_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    feature_kind = variant_feature_kind(variant)
    before_dir = Path(args.before_dir)
    after_dir = Path(args.after_dir)
    cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else Path(args.output_dir) / "_feature_cache"

    try:
        x_train = get_features(
            train_df,
            "train",
            feature_kind,
            before_dir,
            after_dir,
            args.image_role,
            args.img_size,
            args.batch_size,
            args.num_workers,
            args.device,
            cache_dir,
        )
        x_val = get_features(
            val_df,
            "val",
            feature_kind,
            before_dir,
            after_dir,
            args.image_role,
            args.img_size,
            args.batch_size,
            args.num_workers,
            args.device,
            cache_dir,
        )
        x_test = get_features(
            test_df,
            "test",
            feature_kind,
            before_dir,
            after_dir,
            args.image_role,
            args.img_size,
            args.batch_size,
            args.num_workers,
            args.device,
            cache_dir,
        )
        y_train = normalize_targets(target_matrix(train_df))
        estimator = build_estimator(variant, args.seed)
        started = time.perf_counter()
        estimator.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started
        val_pred = export_predictions(val_df, estimator.predict(x_val), predictions_dir / "val_predictions.csv", run_name, "val")
        test_started = time.perf_counter()
        test_pred_norm = estimator.predict(x_test)
        test_seconds = time.perf_counter() - test_started
        test_pred = export_predictions(test_df, test_pred_norm, predictions_dir / "test_predictions.csv", run_name, "test")
        summary = {
            "run_name": run_name,
            "variant": variant,
            "model_label": variant_label(variant),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "ok",
            "script": str(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "seed": args.seed,
            "feature_kind": feature_kind,
            "image_role": args.image_role,
            "img_size": args.img_size,
            "estimator": estimator.__class__.__name__,
            "parameter_count": np.nan,
            "selection_metric": "fixed_traditional_baseline_no_test_selection",
            "loss_type": "not_applicable_sklearn",
            "best_selection_score": np.nan,
            "num_train_samples": int(len(train_df)),
            "num_val_samples": int(len(val_df)),
            "num_test_samples": int(len(test_df)),
            "train_seconds": train_seconds,
            "test_inference_ms_per_sample": float(test_seconds / max(len(test_df), 1) * 1000.0),
            "mean_val_prediction_mape": float(val_pred["mean_ape"].mean()),
            "mean_test_prediction_mape": float(test_pred["mean_ape"].mean()),
            "test_metrics": summarize_predictions(test_pred),
        }
        print(f"[OK] {variant}: test mean APE={summary['mean_test_prediction_mape']:.4f}")
    except Exception as exc:
        summary = {
            "run_name": run_name,
            "variant": variant,
            "model_label": variant_label(variant),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "skipped_or_failed",
            "error": repr(exc),
            "seed": args.seed,
            "feature_kind": feature_kind,
            "image_role": args.image_role,
        }
        print(f"[WARN] {variant} skipped/failed: {exc}")
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "config.json", vars(args) | {"variant": variant, "model_label": variant_label(variant)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traditional inverse-regression baselines on the locked split.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", default="texture_rf,resnet_rf,resnet_extra_trees,resnet_svr,resnet_xgboost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-role", choices=["after", "before"], default="after")
    parser.add_argument("--feature-cache-dir", default="")
    parser.add_argument(
        "--grouped-layout",
        action="store_true",
        help="Write <output>/<variant>/seed<seed> for grouped outer-CV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = read_split(args.train_csv, "train")
    val_df = read_split(args.val_csv, "val")
    test_df = read_split(args.test_csv, "test")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    for variant in variants:
        run_variant(args, variant, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
