#!/usr/bin/env python3
"""Combine and package the frozen DED, NIST, and smallNORB confirmations."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs" / "three_dataset_observability_20260807"
DOMAIN_OUTPUT = PROJECT / "outputs" / "dataset_specific_refinement_20260807"
NORB_OUTPUT = PROJECT / "outputs" / "smallnorb_observability_20260807"
NORB_DATA = PROJECT / "data" / "smallnorb_20260807"

ACTIVE_SCRIPTS = (
    "train_dataset_specific_refinement_20260807.py",
    "run_dataset_specific_refinement_20260807.py",
    "prepare_smallnorb_observability_20260807.py",
    "train_smallnorb_observability_20260807.py",
    "run_smallnorb_observability_20260807.py",
    "aggregate_three_dataset_observability_20260807.py",
    "run_5090_three_dataset_observability_20260807.sh",
)


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def aggregate() -> None:
    domain = read_json(DOMAIN_OUTPUT / "development_summary.json")
    norb = read_json(NORB_OUTPUT / "development_summary.json")
    domain_models = pd.read_csv(DOMAIN_OUTPUT / "model_oof_summary.csv")
    domain_comparisons = pd.read_csv(DOMAIN_OUTPUT / "paired_group_comparisons.csv")
    norb_models = pd.read_csv(NORB_OUTPUT / "model_summary.csv")
    norb_comparisons = pd.read_csv(NORB_OUTPUT / "paired_group_comparisons.csv")

    domain_models.insert(0, "benchmark", domain_models["dataset"])
    norb_models.insert(0, "benchmark", "smallnorb")
    domain_comparisons.insert(0, "benchmark", domain_comparisons["dataset"])
    norb_comparisons.insert(0, "benchmark", "smallnorb")
    domain_models.to_csv(OUTPUT / "ded_nist_model_summary.csv", index=False)
    norb_models.to_csv(OUTPUT / "smallnorb_model_summary.csv", index=False)
    domain_comparisons.to_csv(OUTPUT / "ded_nist_paired_comparisons.csv", index=False)
    norb_comparisons.to_csv(OUTPUT / "smallnorb_paired_comparisons.csv", index=False)

    verdicts = {
        "ded": bool(domain["datasets"]["ded"]["workflow_support_rule_passed"]),
        "nist": bool(domain["datasets"]["nist"]["workflow_support_rule_passed"]),
        "smallnorb": bool(norb["workflow_support_rule_passed"]),
    }
    summary = {
        "scope": "three independent image-to-factor benchmarks with dataset-tailored models",
        "selection_policy": (
            "observability and hyperparameter selection used training/inner-validation data only; "
            "held-out outer or official test data were evaluated after freezing"
        ),
        "benchmarks": {
            "ded": {
                "input": "DED melt-pool/track image sets",
                "targets": ["laser_power", "print_speed", "powder_feed_rate"],
                "mechanism": "high-target primary loss, weak auxiliary losses, and high-target residual adapter",
                "verdict": domain["datasets"]["ded"],
            },
            "nist": {
                "input": "NIST MDS2 condition-level melt-pool image sets",
                "targets": ["spot_diameter", "laser_power", "scan_speed"],
                "mechanism": "all-task residual adapters with symmetric PCGrad",
                "verdict": domain["datasets"]["nist"],
            },
            "smallnorb": {
                "input": "single left-camera 96 x 96 grayscale image",
                "targets": ["elevation", "azimuth", "lighting"],
                "mechanism": "training-only observability selection and spatial specialist routing",
                "verdict": norb,
            },
        },
        "workflow_support_by_benchmark": verdicts,
        "supported_benchmark_count": int(sum(verdicts.values())),
        "all_three_supported": bool(all(verdicts.values())),
        "interpretation_rule": (
            "A negative result is retained as a boundary condition and is not converted into a claim of support."
        ),
    }
    write_json(OUTPUT / "three_dataset_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def add_tree(archive: tarfile.TarFile, root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.suffix.lower() in (".pth", ".pt"):
            continue
        archive.add(path, arcname=str(path.relative_to(PROJECT)))


def package() -> None:
    summary = OUTPUT / "three_dataset_summary.json"
    if not summary.is_file():
        raise FileNotFoundError("Aggregate before package: {}".format(summary))
    archive_path = PROJECT / "three_dataset_observability_20260807_no_weights.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for root in (OUTPUT, DOMAIN_OUTPUT, NORB_OUTPUT):
            add_tree(archive, root)
        for path in sorted(NORB_DATA.rglob("*")):
            if path.is_dir() or "raw" in path.parts or path.suffix.lower() in (".mat", ".gz"):
                continue
            archive.add(path, arcname=str(path.relative_to(PROJECT)))
        for name in ACTIVE_SCRIPTS:
            path = PROJECT / "scripts" / name
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.add(path, arcname=str(path.relative_to(PROJECT)))
        plan = PROJECT / "THREE_DATASET_OBSERVABILITY_PLAN_20260807.md"
        if plan.is_file():
            archive.add(plan, arcname=plan.name)
    print("[ARCHIVE] {}".format(archive_path), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("aggregate", "package"))
    args = parser.parse_args()
    {"aggregate": aggregate, "package": package}[args.stage]()


if __name__ == "__main__":
    main()
