"""Frozen public configurations for B1, B2, B3, and B5."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict


_PROTOCOL = {
    "epochs": 50,
    "batch_size": 32,
    "num_workers": 2,
    "image_size": 224,
    "resize_mode": "stretch",
    "augmentation_mode": "weak",
    "backbone": "resnet18",
    "pretrained": True,
    "hidden_dim": 256,
    "dropout": 0.3,
    "loss": "four independent SmoothL1 means",
    "model_optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "scheduler": "CosineAnnealingLR",
    "eta_min_fraction": 0.01,
    "max_gradient_norm": 1.0,
    "checkpoint_metric": "inner_validation_mean_physical_MAPE",
}


def _base(config_id: str, architecture: str, method: str, variant: str) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "config_id": config_id,
        "architecture": architecture,
        "method": method,
        "model_variant": variant,
        "protocol": deepcopy(_PROTOCOL),
    }


CONFIGS: Dict[str, Dict[str, object]] = {}

CONFIGS["B1"] = _base("B1", "shared_head", "gradnorm", "target_shared4_bn")
CONFIGS["B1"]["gradnorm"] = {
    "alpha": 1.5,
    "task_weight_optimizer": "Adam",
    "task_weight_learning_rate": 0.025,
    "initial_weights": [1.0, 1.0, 1.0, 1.0],
    "normalization": "positive weights renormalized to sum 4 after every step",
    "reference_parameters": "last shared hidden Linear+BatchNorm block",
}

CONFIGS["B2"] = _base("B2", "shared_head", "pcgrad", "target_shared4_bn")
CONFIGS["B2"]["pcgrad"] = {
    "projection": "standard symmetric pairwise conflict projection",
    "projection_scope": "backbone and shared hidden head parameters only",
    "private_parameter_rule": "ordinary gradient of the four-task mean loss",
    "merge": "sum projected task gradients divided by 4",
    "order": "independent deterministic random permutation per task and minibatch",
    "generator_seed_offset": 20261012,
}

CONFIGS["B3"] = _base(
    "B3", "shared_head", "homoscedastic_uncertainty", "target_shared4_bn"
)
CONFIGS["B3"]["uncertainty"] = {
    "parameterization": "four global trainable s_i = log(sigma_i^2) scalars",
    "initial_log_variances": [0.0, 0.0, 0.0, 0.0],
    "formula": "mean_i(exp(-s_i) * SmoothL1_i + s_i)",
    "weight_decay_on_log_variances": 0.0,
    "per_sample_head": False,
}

CONFIGS["B5"] = _base(
    "B5",
    "observability_guided_specialist",
    "pcgrad",
    "target_integrated_texture_expert_any_bn",
)
CONFIGS["B5"]["selected_target_source"] = (
    "frozen per-fold outer-development observability JSON"
)
CONFIGS["B5"]["protocol"]["texture_dim"] = 64
CONFIGS["B5"]["pcgrad"] = {
    "projection": "standard symmetric pairwise conflict projection",
    "projection_scope": (
        "general backbone, auxiliary encoder, and shared hidden general-head parameters"
    ),
    "active_tasks_in_general_group": "the three non-selected targets",
    "specialist_private_rule": (
        "ordinary gradient of the selected task contribution to the four-task mean"
    ),
    "private_output_rule": "ordinary gradient of the four-task mean loss",
    "merge": "sum projected active-task gradients divided by 4",
    "order": (
        "independent deterministic random permutation per active task and minibatch"
    ),
    "generator_seed_offset": 20261012,
}


def get_config(config_id: str) -> Dict[str, object]:
    """Return an isolated copy of a frozen configuration."""

    if config_id not in CONFIGS:
        raise ValueError("config_id must be one of {}".format(sorted(CONFIGS)))
    return deepcopy(CONFIGS[config_id])
