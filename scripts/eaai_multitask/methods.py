"""Four-task optimization methods used in the EAAI experiments.

PCGrad is applied only to parameters genuinely shared by at least two tasks.
Parameters outside a declared shared group keep the ordinary gradient of the
four-task mean loss, preventing cross-task gradients from entering private
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_TASKS = 4


def assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError("{} contains NaN or Inf".format(name))


def per_task_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor
) -> List[torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction/target shape mismatch: {} vs {}".format(
                tuple(prediction.shape), tuple(target.shape)
            )
        )
    if prediction.ndim != 2 or prediction.shape[1] != NUM_TASKS:
        raise ValueError("EAAI multi-task training requires exactly four outputs")
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    losses = [elementwise[:, index].mean() for index in range(NUM_TASKS)]
    for index, loss in enumerate(losses):
        assert_finite_tensor("task_loss_{}".format(index), loss)
    return losses


def _tensor_list_dot(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> torch.Tensor:
    if len(left) != len(right) or not left:
        raise ValueError("Gradient lists must be non-empty and aligned")
    result = left[0].new_zeros(())
    for left_item, right_item in zip(left, right):
        result = result + (left_item * right_item).sum()
    return result


def _tensor_list_norm(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(_tensor_list_dot(values, values).clamp_min(0.0))


def _task_gradients(
    losses: Sequence[torch.Tensor],
    parameters: Sequence[nn.Parameter],
) -> List[List[torch.Tensor]]:
    if not parameters:
        raise ValueError("PCGrad parameter group is empty")
    output: List[List[torch.Tensor]] = []
    for loss in losses:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        output.append(
            [
                torch.zeros_like(parameter)
                if gradient is None
                else gradient.detach().clone()
                for parameter, gradient in zip(parameters, gradients)
            ]
        )
    return output


def project_pcgrad(
    task_gradients: Sequence[Sequence[torch.Tensor]],
    generator: torch.Generator,
    normalization_tasks: int = NUM_TASKS,
    eps: float = 1e-12,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    """Apply symmetric PCGrad to one shared parameter block.

    Each task is projected against the other tasks in an independently shuffled
    order. Projected gradients are divided by ``normalization_tasks`` rather
    than by the active-task count, preserving the scale of the original
    four-task mean loss when only a subset of tasks uses the block.
    """

    if len(task_gradients) < 2:
        raise ValueError("PCGrad needs at least two active task gradients")
    if normalization_tasks < len(task_gradients):
        raise ValueError("normalization_tasks cannot be smaller than active tasks")
    projected: List[List[torch.Tensor]] = []
    orders: List[List[int]] = []
    conflicts = 0
    comparisons = 0
    cosines: List[float] = []
    for task_index, original in enumerate(task_gradients):
        current = [item.clone() for item in original]
        order = torch.randperm(len(task_gradients), generator=generator).tolist()
        orders.append([int(value) for value in order if value != task_index])
        for other_index in order:
            if other_index == task_index:
                continue
            reference = task_gradients[other_index]
            dot = _tensor_list_dot(current, reference)
            current_norm = _tensor_list_norm(current)
            reference_norm = _tensor_list_norm(reference)
            cosine = dot / (current_norm * reference_norm).clamp_min(eps)
            cosines.append(float(cosine.detach().cpu()))
            comparisons += 1
            if bool((dot < 0).item()):
                denominator = _tensor_list_dot(reference, reference).clamp_min(eps)
                coefficient = dot / denominator
                current = [
                    value - coefficient * reference_value
                    for value, reference_value in zip(current, reference)
                ]
                conflicts += 1
        projected.append(current)

    combined = []
    for parts in zip(*projected):
        combined.append(sum(parts) / float(normalization_tasks))
    for index, gradient in enumerate(combined):
        assert_finite_tensor("pcgrad_combined_{}".format(index), gradient)
    return combined, {
        "active_tasks": len(task_gradients),
        "normalization_tasks": int(normalization_tasks),
        "conflicts": int(conflicts),
        "comparisons": int(comparisons),
        "conflict_fraction": float(conflicts / max(comparisons, 1)),
        "mean_preprojection_cosine": float(sum(cosines) / max(len(cosines), 1)),
        "projection_orders": orders,
    }


@dataclass
class SharedParameterGroup:
    name: str
    parameters: Sequence[nn.Parameter]
    task_indices: Sequence[int]


def apply_pcgrad_to_shared_groups(
    task_losses: Sequence[torch.Tensor],
    groups: Sequence[SharedParameterGroup],
    generator: torch.Generator,
) -> Dict[str, object]:
    """Populate gradients using mean-loss gradients plus PCGrad overrides."""

    if len(task_losses) != NUM_TASKS:
        raise ValueError("PCGrad requires four independent task losses")
    if not groups:
        raise ValueError("At least one shared parameter group is required")

    # The ordinary backward pass supplies four-task-mean gradients to private
    # parameters. Gradients for explicitly shared groups are overwritten below.
    torch.stack(list(task_losses)).mean().backward(retain_graph=True)
    seen = set()
    group_diagnostics: List[Dict[str, object]] = []
    for group in groups:
        parameters = list(group.parameters)
        task_indices = [int(index) for index in group.task_indices]
        if len(task_indices) < 2:
            raise ValueError("PCGrad group {} has fewer than two tasks".format(group.name))
        if len(set(task_indices)) != len(task_indices):
            raise ValueError("PCGrad group {} repeats a task".format(group.name))
        for parameter in parameters:
            identity = id(parameter)
            if identity in seen:
                raise ValueError("A parameter appears in multiple PCGrad groups")
            seen.add(identity)
        gradients = _task_gradients(
            [task_losses[index] for index in task_indices], parameters
        )
        combined, diagnostics = project_pcgrad(
            gradients,
            generator=generator,
            normalization_tasks=NUM_TASKS,
        )
        for parameter, gradient in zip(parameters, combined):
            parameter.grad = gradient
        diagnostics = dict(diagnostics)
        diagnostics["name"] = group.name
        diagnostics["task_indices"] = task_indices
        diagnostics["parameter_tensors"] = len(parameters)
        diagnostics["parameter_elements"] = int(
            sum(parameter.numel() for parameter in parameters)
        )
        group_diagnostics.append(diagnostics)
    return {"groups": group_diagnostics}


class GradNormController(nn.Module):
    """Four-task GradNorm controller with positive, sum-to-four weights."""

    def __init__(self, alpha: float = 1.5, num_tasks: int = NUM_TASKS):
        super().__init__()
        if num_tasks != NUM_TASKS:
            raise ValueError("Phase B GradNorm is fixed to four tasks")
        self.alpha = float(alpha)
        self.task_weights = nn.Parameter(torch.ones(num_tasks, dtype=torch.float32))
        self.register_buffer(
            "initial_losses", torch.full((num_tasks,), float("nan"))
        )
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def losses(
        self,
        task_losses: Sequence[torch.Tensor],
        reference_parameters: Sequence[nn.Parameter],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
        if len(task_losses) != NUM_TASKS:
            raise ValueError("GradNorm requires four task losses")
        if not reference_parameters:
            raise ValueError("GradNorm reference parameter set is empty")
        loss_vector = torch.stack(list(task_losses))
        if not bool(self.initialized.item()):
            self.initial_losses.copy_(loss_vector.detach().clamp_min(1e-12))
            self.initialized.fill_(True)

        base_norms = []
        for loss in task_losses:
            gradients = torch.autograd.grad(
                loss,
                reference_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            squared = loss.new_zeros(())
            for parameter, gradient in zip(reference_parameters, gradients):
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                squared = squared + gradient.detach().square().sum()
            base_norms.append(torch.sqrt(squared.clamp_min(1e-24)))
        base_norm_vector = torch.stack(base_norms)
        weighted_norms = self.task_weights * base_norm_vector

        relative_losses = (loss_vector.detach() / self.initial_losses).clamp_min(1e-12)
        inverse_training_rates = relative_losses / relative_losses.mean().clamp_min(1e-12)
        target_norms = (
            weighted_norms.detach().mean()
            * inverse_training_rates.pow(self.alpha)
        )
        gradnorm_loss = torch.abs(weighted_norms - target_norms).sum()
        model_loss = (
            self.task_weights.detach() * loss_vector
        ).sum() / float(NUM_TASKS)

        assert_finite_tensor("gradnorm_model_loss", model_loss)
        assert_finite_tensor("gradnorm_weight_loss", gradnorm_loss)
        return model_loss, gradnorm_loss, {
            "task_weights": [float(value) for value in self.task_weights.detach().cpu()],
            "base_gradient_norms": [float(value) for value in base_norm_vector.cpu()],
            "weighted_gradient_norms": [float(value) for value in weighted_norms.detach().cpu()],
            "target_gradient_norms": [float(value) for value in target_norms.cpu()],
            "relative_training_rates": [float(value) for value in inverse_training_rates.cpu()],
        }

    @torch.no_grad()
    def renormalize_(self) -> None:
        self.task_weights.clamp_(min=1e-6)
        self.task_weights.mul_(
            float(NUM_TASKS) / self.task_weights.sum().clamp_min(1e-12)
        )
        assert_finite_tensor("gradnorm_task_weights", self.task_weights)


class HomoscedasticUncertainty(nn.Module):
    """Four global task-level log variances, not a per-sample head."""

    def __init__(self, initial_log_variance: float = 0.0):
        super().__init__()
        self.log_variances = nn.Parameter(
            torch.full((NUM_TASKS,), float(initial_log_variance), dtype=torch.float32)
        )

    def forward(
        self, task_losses: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        if len(task_losses) != NUM_TASKS:
            raise ValueError("Uncertainty weighting requires four task losses")
        loss_vector = torch.stack(list(task_losses))
        precisions = torch.exp(-self.log_variances)
        # This is twice the conventional 0.5*[exp(-s)L+s] expression. The
        # common factor preserves the initial model-gradient scale of the
        # ordinary four-task mean without changing the optimum in s.
        total = (precisions * loss_vector + self.log_variances).mean()
        assert_finite_tensor("uncertainty_total_loss", total)
        return total, {
            "log_variances": [float(value) for value in self.log_variances.detach().cpu()],
            "effective_weights": [float(value) for value in precisions.detach().cpu()],
        }
