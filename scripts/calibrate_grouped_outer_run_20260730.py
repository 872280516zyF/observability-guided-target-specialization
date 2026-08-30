#!/usr/bin/env python3
"""Apply the identical inner-validation DPI ridge calibration to one run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PARAMS = ["frequency", "pulse_width", "speed", "dpi"]
DPI_MIN = 25.0
DPI_MAX = 175.0


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    prediction_columns = [
        "pred_dpi",
        "pred_frequency",
        "pred_pulse_width",
        "pred_speed",
    ]
    predictions = frame[prediction_columns].astype(float).to_numpy()
    dpi = predictions[:, :1]
    return np.concatenate(
        [
            np.ones((len(frame), 1), dtype=float),
            predictions,
            dpi * dpi / 100.0,
            (predictions[:, 1:2] - 75.0) / 25.0,
            (predictions[:, 2:3] - 40000.0) / 10000.0,
        ],
        axis=1,
    )


def fit_ridge(validation: pd.DataFrame, alpha: float) -> np.ndarray:
    design = feature_matrix(validation)
    target = validation["true_dpi"].astype(float).to_numpy()
    penalty = np.eye(design.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )


def apply(frame: pd.DataFrame, coefficients: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["pred_dpi"] = np.clip(
        feature_matrix(output) @ coefficients, DPI_MIN, DPI_MAX
    )
    output["abs_err_dpi"] = np.abs(
        output["pred_dpi"].astype(float)
        - output["true_dpi"].astype(float)
    )
    output["ape_dpi"] = (
        output["abs_err_dpi"]
        / np.maximum(np.abs(output["true_dpi"].astype(float)), 1e-6)
        * 100.0
    )
    output["mean_ape"] = output[
        [f"ape_{parameter}" for parameter in PARAMS]
    ].astype(float).mean(axis=1)
    return output


def metrics(frame: pd.DataFrame) -> dict:
    return {
        "n_samples": int(frame["sample_id"].nunique()),
        "mean_ape": float(frame["mean_ape"].mean()),
        "parameter_mape": {
            parameter: float(frame[f"ape_{parameter}"].mean())
            for parameter in PARAMS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    validation_path = run_dir / "predictions" / "val_predictions.csv"
    test_path = run_dir / "predictions" / "test_predictions.csv"
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing raw predictions in {run_dir / 'predictions'}"
        )
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    coefficients = fit_ridge(validation, args.alpha)
    calibrated_validation = apply(validation, coefficients)
    calibrated_test = apply(test, coefficients)
    output = run_dir / "calibrated"
    output.mkdir(parents=True, exist_ok=True)
    calibrated_validation.to_csv(
        output / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibrated_test.to_csv(
        output / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "fit_partition": "inner_validation_only",
        "test_partition_used_for_fit": False,
        "alpha": args.alpha,
        "feature_map": [
            "intercept",
            "pred_dpi",
            "pred_frequency",
            "pred_pulse_width",
            "pred_speed",
            "pred_dpi_squared_div_100",
            "centered_frequency",
            "centered_speed",
        ],
        "coefficients": [float(value) for value in coefficients],
        "raw_validation": metrics(validation),
        "calibrated_validation": metrics(calibrated_validation),
        "raw_outer_test": metrics(test),
        "calibrated_outer_test": metrics(calibrated_test),
    }
    (output / "calibration_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
