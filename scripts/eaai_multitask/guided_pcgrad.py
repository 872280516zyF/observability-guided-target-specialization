"""Model topology and gradient-boundary definitions for B2 and B5."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .methods import SharedParameterGroup


PARAMETER_NAMES = ["frequency", "pulse_width", "speed", "dpi"]
EXPECTED_MODEL_PARAMETERS = {
    "shared_head": 11342020,
    "observability_guided_specialist": 23347844,
}


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(value.numel() for value in module.parameters() if value.requires_grad))


def parameters_except(module: nn.Module, excluded: nn.Module) -> List[nn.Parameter]:
    excluded_ids = {id(value) for value in excluded.parameters()}
    return [
        value
        for value in module.parameters()
        if value.requires_grad and id(value) not in excluded_ids
    ]


def _named_parameter_records(
    model: nn.Module, parameters: Sequence[nn.Parameter]
) -> List[Dict[str, object]]:
    name_by_id = {id(value): name for name, value in model.named_parameters()}
    return [
        {
            "name": name_by_id[id(value)],
            "shape": list(value.shape),
            "elements": int(value.numel()),
        }
        for value in parameters
    ]


def build_model_and_topology(
    model_api,
    config: Dict[str, object],
    selected_target: str,
    device: torch.device,
) -> Tuple[nn.Module, List[SharedParameterGroup], List[nn.Parameter], Dict[str, object]]:
    """Build the frozen architecture and declare its shared/private boundary."""

    protocol = config["protocol"]
    architecture = str(config["architecture"])
    if architecture == "shared_head":
        model = model_api.Shared4BNNet(
            str(protocol["backbone"]),
            bool(protocol["pretrained"]),
            int(protocol["hidden_dim"]),
            float(protocol["dropout"]),
        )
        final_output = model.head.head[-2]
        shared_parameters = parameters_except(model, final_output)
        groups = [
            SharedParameterGroup(
                name="shared_backbone_and_hidden_head",
                parameters=shared_parameters,
                task_indices=(0, 1, 2, 3),
            )
        ]
        # The final shared hidden Linear and BatchNorm are the GradNorm reference.
        gradnorm_reference = list(model.head.head[4].parameters()) + list(
            model.head.head[5].parameters()
        )
        private_parameters = list(final_output.parameters())
    elif architecture == "observability_guided_specialist":
        model = model_api.TargetedIntegratedExpertBNNet(
            str(protocol["backbone"]),
            bool(protocol["pretrained"]),
            int(protocol["hidden_dim"]),
            float(protocol["dropout"]),
            texture_dim=int(protocol["texture_dim"]),
            expert_target=selected_target,
            texture_guided=True,
        )
        final_general_output = model.general_head.head[-2]
        general_modules = [
            model.general_backbone,
            model.general_aux_encoder,
            model.general_head,
        ]
        excluded_ids = {id(value) for value in final_general_output.parameters()}
        shared_parameters = []
        for module in general_modules:
            for value in module.parameters():
                if value.requires_grad and id(value) not in excluded_ids:
                    shared_parameters.append(value)
        groups = [
            SharedParameterGroup(
                name="general_branch_shared_by_nonselected_tasks",
                parameters=shared_parameters,
                task_indices=tuple(int(value) for value in model.remaining_indices),
            )
        ]
        gradnorm_reference = []
        shared_ids = {id(value) for value in shared_parameters}
        private_parameters = [
            value
            for value in model.parameters()
            if value.requires_grad and id(value) not in shared_ids
        ]
    else:
        raise ValueError("Unsupported architecture {}".format(architecture))

    model = model.to(device)
    topology = {
        "architecture": architecture,
        "selected_target": selected_target,
        "selected_target_index": PARAMETER_NAMES.index(selected_target),
        "pcgrad_groups": [
            {
                "name": group.name,
                "task_indices": list(group.task_indices),
                "task_names": [PARAMETER_NAMES[index] for index in group.task_indices],
                "parameters": _named_parameter_records(model, group.parameters),
            }
            for group in groups
        ],
        "private_parameters": _named_parameter_records(model, private_parameters),
        "gradnorm_reference_parameters": _named_parameter_records(
            model, gradnorm_reference
        ),
    }
    return model, groups, gradnorm_reference, topology
