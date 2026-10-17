#!/usr/bin/env python3
"""Train one EAAI multi-task configuration for one fold and seed.

Run from the repository root, for example::

    python -m scripts.eaai_multitask.train_multitask \
        --config-id B2 --fold 0 --seed 42 \
        --split-root data/images3/grouped_outer_cv_20260730 \
        --before-dir data/images3/before --after-dir data/images3/after \
        --observability-root outputs/grouped_outer_cv_20260730

The four frozen configurations are B1 (GradNorm), B2 (four-task PCGrad),
B3 (homoscedastic uncertainty weighting), and B5 (guided architecture with
PCGrad restricted to the three-task general branch). A best checkpoint is
selected only by inner-validation physical MAPE. The outer-test loader is not
constructed until that checkpoint has been frozen and reloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .configs import get_config
from .guided_pcgrad import (
    EXPECTED_MODEL_PARAMETERS,
    PARAMETER_NAMES,
    build_model_and_topology,
    trainable_parameter_count,
)
from .methods import (
    GradNormController,
    HomoscedasticUncertainty,
    NUM_TASKS,
    apply_pcgrad_to_shared_groups,
    assert_finite_tensor,
    per_task_smooth_l1,
)


ALLOWED_SEEDS = (42, 52, 62)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True, choices=("B1", "B2", "B3", "B5"))
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--seed", required=True, type=int, choices=ALLOWED_SEEDS)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root; defaults to the repository containing this module.",
    )
    parser.add_argument(
        "--split-root",
        default="data/images3/grouped_outer_cv_20260730",
        help="Project-relative or absolute directory containing fold_0 through fold_4.",
    )
    parser.add_argument("--before-dir", default="data/images3/before")
    parser.add_argument("--after-dir", default="data/images3/after")
    parser.add_argument(
        "--observability-root",
        default="outputs/grouped_outer_cv_20260730",
        help="Frozen per-fold observability result root.",
    )
    parser.add_argument("--output-root", default="outputs/eaai_multitask")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Generic PyTorch device; CPU is intended only for small debugging runs.",
    )
    return parser.parse_args()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def assert_group_disjoint(
    train_csv: Path,
    validation_csv: Path,
    outer_test_csv: Path,
    group_column: str = "before_id",
) -> Dict[str, int]:
    """Fail if any before-image group occurs in more than one split."""

    frames = {
        "train": pd.read_csv(train_csv),
        "validation": pd.read_csv(validation_csv),
        "outer_test": pd.read_csv(outer_test_csv),
    }
    for name, frame in frames.items():
        if group_column not in frame.columns:
            raise ValueError("{} is missing group column {}".format(name, group_column))
    groups = {
        name: set(frame[group_column].astype(str)) for name, frame in frames.items()
    }
    pairs = (("train", "validation"), ("train", "outer_test"), ("validation", "outer_test"))
    for left, right in pairs:
        overlap = groups[left].intersection(groups[right])
        if overlap:
            raise ValueError(
                "Group leakage between {} and {}: {}".format(
                    left, right, sorted(overlap)
                )
            )
    return {name: len(value) for name, value in groups.items()}


def checkpoint_improves(inner_validation_mape: float, best_score: float) -> bool:
    """Return the frozen checkpoint decision based only on inner validation."""

    if not math.isfinite(inner_validation_mape):
        raise FloatingPointError("Inner-validation MAPE is not finite")
    return inner_validation_mape < best_score


@torch.no_grad()
def evaluate(model_api, model, loader, device: torch.device) -> Dict[str, object]:
    model.eval()
    loss_sums = np.zeros(NUM_TASKS, dtype=np.float64)
    count = 0
    prediction_rows = []
    truth_norm_rows = []
    truth_raw_rows = []
    for batch in loader:
        batch = model_api.move_tensor_batch(batch, device)
        output = model(batch)
        prediction, _ = model_api.unpack_model_output(output)
        losses = per_task_smooth_l1(prediction, batch["params"])
        batch_size = int(prediction.shape[0])
        loss_sums += np.asarray([float(value.cpu()) for value in losses]) * batch_size
        count += batch_size
        prediction_rows.append(prediction.detach().cpu().numpy())
        truth_norm_rows.append(batch["params"].detach().cpu().numpy())
        truth_raw_rows.append(batch["params_raw"].detach().cpu().numpy())
    prediction_norm = np.concatenate(prediction_rows, axis=0)
    truth_norm = np.concatenate(truth_norm_rows, axis=0)
    truth_raw = np.concatenate(truth_raw_rows, axis=0)
    _, ape = model_api.compute_physical_errors(prediction_norm, truth_raw)
    result = {
        "loss": float(loss_sums.mean() / max(count, 1)),
        "task_losses": {
            name: float(loss_sums[index] / max(count, 1))
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "mae_norm": float(np.mean(np.abs(prediction_norm - truth_norm))),
        "mape_physical": float(np.mean(ape)),
        "param_mape_physical": {
            name: float(np.mean(ape[:, index]))
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "n_samples": int(count),
    }
    numeric = [
        result["loss"],
        result["mae_norm"],
        result["mape_physical"],
        *result["task_losses"].values(),
        *result["param_mape_physical"].values(),
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise FloatingPointError("Evaluation produced NaN or Inf")
    return result


def _mean_vector(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values = [row[key] for row in rows if key in row]
    if not values:
        return []
    return np.asarray(values, dtype=np.float64).mean(axis=0).tolist()


def _load_public_model_api(project_root: Path):
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts import run_dpi_branch_ablation as model_api
    from utils.seed import set_seed

    return model_api, set_seed


def run(args: argparse.Namespace) -> Dict[str, object]:
    project_root = Path(args.project_root).resolve()
    config = get_config(args.config_id)
    config_hash = canonical_json_sha256(config)
    protocol = config["protocol"]

    split_root = _resolve(project_root, args.split_root)
    before_dir = _resolve(project_root, args.before_dir)
    after_dir = _resolve(project_root, args.after_dir)
    observability_root = _resolve(project_root, args.observability_root)
    output_root = _resolve(project_root, args.output_root)
    run_dir = output_root / args.config_id / "fold_{}".format(args.fold) / "seed{}".format(args.seed)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty run directory: {}".format(run_dir))
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    fold_dir = split_root / "fold_{}".format(args.fold)
    train_csv = fold_dir / "label_train.csv"
    validation_csv = fold_dir / "label_val.csv"
    outer_test_csv = fold_dir / "label_test.csv"
    selected_target_path = (
        observability_root
        / "fold_{}".format(args.fold)
        / "observability"
        / "grouped"
        / "selected_observability_target.json"
    )
    required = [
        train_csv,
        validation_csv,
        outer_test_csv,
        before_dir,
        after_dir,
        selected_target_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required frozen inputs: {}".format(missing))
    split_group_counts = assert_group_disjoint(train_csv, validation_csv, outer_test_csv)
    selected_payload = json.loads(selected_target_path.read_text(encoding="utf-8"))
    selected_target = str(selected_payload["selected_target"])
    if selected_target not in PARAMETER_NAMES:
        raise ValueError("Invalid frozen selected target {}".format(selected_target))
    split_hashes = {
        "train": sha256(train_csv),
        "validation": sha256(validation_csv),
        "outer_test": sha256(outer_test_csv),
        "selected_target": sha256(selected_target_path),
    }

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model_api, set_seed = _load_public_model_api(project_root)
    set_seed(args.seed)
    model, pcgrad_groups, gradnorm_reference, topology = build_model_and_topology(
        model_api, config, selected_target, device
    )
    observed_parameters = trainable_parameter_count(model)
    expected_parameters = EXPECTED_MODEL_PARAMETERS[str(config["architecture"])]
    if observed_parameters != expected_parameters:
        raise RuntimeError(
            "Model capacity mismatch: observed {}, expected {}".format(
                observed_parameters, expected_parameters
            )
        )
    write_json(run_dir / "gradient_topology.json", topology)

    loader_args = {
        "before_dir": str(before_dir),
        "after_dir": str(after_dir),
        "img_size": int(protocol["image_size"]),
        "batch_size": int(protocol["batch_size"]),
        "num_workers": int(protocol["num_workers"]),
        "resize_mode": str(protocol["resize_mode"]),
        "group_balanced_sampler": False,
    }
    train_loader = model_api.build_loader(
        csv_path=str(train_csv),
        is_train=True,
        augmentation_mode=str(protocol["augmentation_mode"]),
        **loader_args,
    )
    validation_loader = model_api.build_loader(
        csv_path=str(validation_csv),
        is_train=False,
        augmentation_mode="none",
        **loader_args,
    )

    method = str(config["method"])
    controller = None
    controller_optimizer = None
    model_lr = float(protocol["learning_rate"])
    weight_decay = float(protocol["weight_decay"])
    if method == "gradnorm":
        controller = GradNormController(alpha=float(config["gradnorm"]["alpha"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_lr, weight_decay=weight_decay)
        controller_optimizer = torch.optim.Adam(
            controller.parameters(),
            lr=float(config["gradnorm"]["task_weight_learning_rate"]),
        )
    elif method == "homoscedastic_uncertainty":
        initial_values = config["uncertainty"]["initial_log_variances"]
        if len(set(float(value) for value in initial_values)) != 1:
            raise ValueError("A common initial log variance is required")
        controller = HomoscedasticUncertainty(float(initial_values[0])).to(device)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "weight_decay": weight_decay},
                {"params": controller.parameters(), "weight_decay": 0.0},
            ],
            lr=model_lr,
        )
    elif method == "pcgrad":
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported method {}".format(method))

    epochs = int(protocol["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=model_lr * float(protocol["eta_min_fraction"]),
    )
    pcgrad_generator = torch.Generator(device="cpu")
    pcgrad_seed = args.seed + int(config.get("pcgrad", {}).get("generator_seed_offset", 0))
    pcgrad_generator.manual_seed(pcgrad_seed)

    write_json(
        run_dir / "config_resolved.json",
        {
            "config": config,
            "config_sha256": config_hash,
            "fold": args.fold,
            "seed": args.seed,
            "selected_target": selected_target,
            "split_hashes_sha256": split_hashes,
            "split_group_counts": split_group_counts,
            "model_parameter_count": observed_parameters,
            "method_parameter_count": (
                trainable_parameter_count(controller) if controller is not None else 0
            ),
            "pcgrad_generator_seed": pcgrad_seed if method == "pcgrad" else None,
            "prediction_calibration": "raw",
            "outer_test_used_for_training_or_checkpoint_selection": False,
        },
    )

    history_rows: List[Dict[str, object]] = []
    best_score = float("inf")
    best_epoch = -1
    checkpoint_path = run_dir / "checkpoints" / "best_model.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    max_gradient_norm = float(protocol["max_gradient_norm"])

    for epoch in range(1, epochs + 1):
        model.train()
        batch_rows: List[Dict[str, object]] = []
        for batch_index, batch in enumerate(train_loader):
            batch = model_api.move_tensor_batch(batch, device)
            output = model(batch)
            prediction, _ = model_api.unpack_model_output(output)
            task_losses = per_task_smooth_l1(prediction, batch["params"])
            unweighted_mean = torch.stack(task_losses).mean()

            optimizer.zero_grad(set_to_none=True)
            if controller_optimizer is not None:
                controller_optimizer.zero_grad(set_to_none=True)
            if method == "pcgrad":
                diagnostics = apply_pcgrad_to_shared_groups(
                    task_losses, pcgrad_groups, pcgrad_generator
                )
                optimized_loss = unweighted_mean.detach()
            elif method == "gradnorm":
                model_loss, weight_loss, diagnostics = controller.losses(
                    task_losses, gradnorm_reference
                )
                model_loss.backward()
                weight_loss.backward()
                optimized_loss = model_loss.detach()
            else:
                optimized, diagnostics = controller(task_losses)
                optimized.backward()
                optimized_loss = optimized.detach()

            assert_finite_tensor("optimized_loss", optimized_loss)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            assert_finite_tensor("model_gradient_norm", gradient_norm)
            optimizer.step()
            if method == "gradnorm":
                controller_optimizer.step()
                controller.renormalize_()
            if controller is not None:
                for name, value in controller.named_parameters():
                    assert_finite_tensor("controller_{}".format(name), value)

            row: Dict[str, object] = {
                "batch": batch_index,
                "unweighted_mean_loss": float(unweighted_mean.detach().cpu()),
                "optimized_loss": float(optimized_loss.cpu()),
                "model_gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "task_losses": [float(value.detach().cpu()) for value in task_losses],
            }
            if method == "pcgrad":
                groups = diagnostics["groups"]
                row["pcgrad_conflict_fraction"] = float(
                    np.mean([value["conflict_fraction"] for value in groups])
                )
                row["pcgrad_mean_preprojection_cosine"] = float(
                    np.mean([value["mean_preprojection_cosine"] for value in groups])
                )
            else:
                row.update(diagnostics)
            batch_rows.append(row)

        scheduler.step()
        validation_metrics = evaluate(model_api, model, validation_loader, device)
        selection_score = float(validation_metrics["mape_physical"])
        history_row: Dict[str, object] = {
            "epoch": epoch,
            "train_unweighted_mean_loss": float(
                np.mean([value["unweighted_mean_loss"] for value in batch_rows])
            ),
            "train_optimized_loss": float(
                np.mean([value["optimized_loss"] for value in batch_rows])
            ),
            "validation_loss": validation_metrics["loss"],
            "validation_mape": validation_metrics["mape_physical"],
            "selection_score": selection_score,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        for index, name in enumerate(PARAMETER_NAMES):
            history_row["train_{}_loss".format(name)] = float(
                np.mean([value["task_losses"][index] for value in batch_rows])
            )
            history_row["validation_{}_mape".format(name)] = validation_metrics[
                "param_mape_physical"
            ][name]
        if method == "pcgrad":
            history_row["pcgrad_conflict_fraction"] = float(
                np.mean([value["pcgrad_conflict_fraction"] for value in batch_rows])
            )
            history_row["pcgrad_mean_preprojection_cosine"] = float(
                np.mean([value["pcgrad_mean_preprojection_cosine"] for value in batch_rows])
            )
        elif method == "gradnorm":
            for key in (
                "task_weights",
                "base_gradient_norms",
                "weighted_gradient_norms",
                "target_gradient_norms",
                "relative_training_rates",
            ):
                history_row[key] = json.dumps(_mean_vector(batch_rows, key))
        else:
            history_row["log_variances"] = json.dumps(_mean_vector(batch_rows, "log_variances"))
            history_row["effective_weights"] = json.dumps(_mean_vector(batch_rows, "effective_weights"))
        history_rows.append(history_row)
        pd.DataFrame(history_rows).to_csv(run_dir / "logs" / "history.csv", index=False)
        print(
            "epoch {}/{} train={:.6f} validation_mape={:.6f}".format(
                epoch, epochs, history_row["train_unweighted_mean_loss"], selection_score
            )
        )

        if checkpoint_improves(selection_score, best_score):
            best_score = selection_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "method_state_dict": controller.state_dict() if controller is not None else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "controller_optimizer_state_dict": (
                        controller_optimizer.state_dict()
                        if controller_optimizer is not None
                        else None
                    ),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_inner_validation_mean_mape": best_score,
                    "config_sha256": config_hash,
                    "split_hashes_sha256": split_hashes,
                    "outer_test_used_for_selection": False,
                },
                checkpoint_path,
            )

    if best_epoch < 0 or not checkpoint_path.exists():
        raise RuntimeError("Training completed without a valid checkpoint")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint["config_sha256"] != config_hash:
        raise RuntimeError("Checkpoint/config hash mismatch")
    if checkpoint["split_hashes_sha256"] != split_hashes:
        raise RuntimeError("Checkpoint/split hash mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    if controller is not None:
        controller.load_state_dict(checkpoint["method_state_dict"])

    # Dataset construction reads the label table, so the outer-test loader is
    # deliberately constructed only after checkpoint selection and reload.
    outer_test_loader = model_api.build_loader(
        csv_path=str(outer_test_csv),
        is_train=False,
        augmentation_mode="none",
        **loader_args,
    )
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    validation_metrics = evaluate(model_api, model, validation_loader, device)
    model_api.export_predictions(
        model,
        validation_loader,
        device,
        prediction_dir / "validation_predictions.csv",
        "{}_fold{}_seed{}".format(args.config_id, args.fold, args.seed),
        "validation",
    )
    outer_test_metrics = evaluate(model_api, model, outer_test_loader, device)
    model_api.export_predictions(
        model,
        outer_test_loader,
        device,
        prediction_dir / "outer_test_predictions.csv",
        "{}_fold{}_seed{}".format(args.config_id, args.fold, args.seed),
        "outer_test",
    )

    summary = {
        "status": "COMPLETE",
        "config_id": args.config_id,
        "architecture": config["architecture"],
        "method": method,
        "fold": args.fold,
        "seed": args.seed,
        "selected_target": selected_target,
        "best_epoch": best_epoch,
        "best_inner_validation_mean_mape": best_score,
        "validation_metrics": validation_metrics,
        "outer_test_metrics": outer_test_metrics,
        "outer_test_used_for_training_or_checkpoint_selection": False,
        "prediction_calibration": "raw",
        "checkpoint_sha256": sha256(checkpoint_path),
        "outer_test_predictions_sha256": sha256(
            prediction_dir / "outer_test_predictions.csv"
        ),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
