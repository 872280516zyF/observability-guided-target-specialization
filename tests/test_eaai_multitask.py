"""Unit and small functional tests for the public EAAI training code."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.eaai_multitask.configs import get_config
from scripts.eaai_multitask.guided_pcgrad import (
    EXPECTED_MODEL_PARAMETERS,
    build_model_and_topology,
    trainable_parameter_count,
)
from scripts.eaai_multitask.methods import (
    GradNormController,
    HomoscedasticUncertainty,
    SharedParameterGroup,
    apply_pcgrad_to_shared_groups,
    project_pcgrad,
)
from scripts.eaai_multitask.train_multitask import (
    assert_group_disjoint,
    checkpoint_improves,
)
from utils.seed import set_seed


class TestPCGrad(unittest.TestCase):
    def test_no_conflict_is_mean_with_four_task_scaling(self):
        gradients = [[torch.tensor([value])] for value in (1.0, 2.0, 3.0, 4.0)]
        generator = torch.Generator().manual_seed(7)
        combined, diagnostics = project_pcgrad(gradients, generator)
        self.assertTrue(torch.allclose(combined[0], torch.tensor([2.5])))
        self.assertEqual(diagnostics["conflicts"], 0)
        self.assertEqual(diagnostics["normalization_tasks"], 4)

    def test_conflicting_gradients_are_projected(self):
        gradients = [
            [torch.tensor([1.0, 0.0])],
            [torch.tensor([-1.0, 1.0])],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([1.0, 1.0])],
        ]
        combined, diagnostics = project_pcgrad(
            gradients, torch.Generator().manual_seed(11)
        )
        self.assertGreater(diagnostics["conflicts"], 0)
        self.assertTrue(torch.isfinite(combined[0]).all())

    def test_b5_private_specialist_keeps_ordinary_mean_gradient(self):
        shared = nn.Parameter(torch.tensor(1.0))
        specialist = nn.Parameter(torch.tensor(3.0))
        losses = [shared.square(), 2.0 * shared.square(), 3.0 * shared.square(), (specialist - 2.0).square()]
        group = SharedParameterGroup("general", [shared], (0, 1, 2))
        apply_pcgrad_to_shared_groups(losses, [group], torch.Generator().manual_seed(3))
        expected_specialist = 2.0 * (specialist.detach() - 2.0) / 4.0
        self.assertTrue(torch.allclose(specialist.grad, expected_specialist))
        self.assertEqual(group.task_indices, (0, 1, 2))


class TestGradNorm(unittest.TestCase):
    def test_dynamic_weights_update_positive_and_sum_to_four(self):
        layer = nn.Linear(2, 4, bias=False)
        controller = GradNormController(alpha=1.5)
        optimizer = torch.optim.Adam(controller.parameters(), lr=0.025)
        x = torch.tensor([[1.0, -2.0], [0.5, 1.0]])
        targets = torch.tensor([[0.0, 1.0, -1.0, 2.0], [1.0, -1.0, 0.5, -2.0]])
        prediction = layer(x)
        losses = [(prediction[:, i] - targets[:, i]).square().mean() for i in range(4)]
        _, weight_loss, diagnostics = controller.losses(losses, list(layer.parameters()))
        optimizer.zero_grad()
        weight_loss.backward()
        before = controller.task_weights.detach().clone()
        optimizer.step()
        controller.renormalize_()
        self.assertFalse(torch.allclose(before, controller.task_weights.detach()))
        self.assertTrue(bool((controller.task_weights > 0).all()))
        self.assertAlmostEqual(float(controller.task_weights.sum()), 4.0, places=6)
        self.assertEqual(len(diagnostics["base_gradient_norms"]), 4)
        self.assertTrue(bool(controller.initialized.item()))


class TestUncertaintyWeighting(unittest.TestCase):
    def test_four_log_variances_receive_gradients_and_update(self):
        controller = HomoscedasticUncertainty(0.0)
        optimizer = torch.optim.AdamW(
            [{"params": controller.parameters(), "weight_decay": 0.0}], lr=0.001
        )
        losses = [torch.tensor(value) for value in (0.2, 0.6, 1.4, 2.2)]
        total, _ = controller(losses)
        optimizer.zero_grad()
        total.backward()
        self.assertEqual(tuple(controller.log_variances.grad.shape), (4,))
        before = controller.log_variances.detach().clone()
        optimizer.step()
        self.assertFalse(torch.allclose(before, controller.log_variances.detach()))


class _TinyGuidedModel(nn.Module):
    def __init__(self, backbone, pretrained, hidden_dim, dropout, texture_dim, expert_target, texture_guided):
        super().__init__()
        selected = ["frequency", "pulse_width", "speed", "dpi"].index(expert_target)
        self.remaining_indices = [index for index in range(4) if index != selected]
        self.general_backbone = nn.Linear(2, 3)
        self.general_aux_encoder = nn.Linear(2, 3)
        self.general_head = nn.Module()
        self.general_head.head = nn.Sequential(
            nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3), nn.Identity()
        )
        self.specialist_backbone = nn.Linear(2, 3)
        self.specialist_head = nn.Linear(3, 1)


class _TinyModelAPI:
    TargetedIntegratedExpertBNNet = _TinyGuidedModel


class TestTopologyAndProtocol(unittest.TestCase):
    def test_b5_pcgrad_group_excludes_specialist_parameters(self):
        config = get_config("B5")
        model, groups, reference, topology = build_model_and_topology(
            _TinyModelAPI, config, "dpi", torch.device("cpu")
        )
        shared_ids = {id(parameter) for parameter in groups[0].parameters}
        specialist_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("specialist_")
        }
        self.assertTrue(specialist_ids)
        self.assertTrue(shared_ids.isdisjoint(specialist_ids))
        self.assertEqual(tuple(groups[0].task_indices), (0, 1, 2))
        self.assertEqual(reference, [])
        self.assertIn("specialist_head.weight", {row["name"] for row in topology["private_parameters"]})

    def test_public_models_match_frozen_capacities_and_gradient_boundaries(self):
        from scripts import run_dpi_branch_ablation as model_api

        shared_config = get_config("B2")
        shared_config["protocol"]["pretrained"] = False
        shared_model, shared_groups, _, _ = build_model_and_topology(
            model_api, shared_config, "dpi", torch.device("cpu")
        )
        self.assertEqual(
            trainable_parameter_count(shared_model),
            EXPECTED_MODEL_PARAMETERS["shared_head"],
        )
        self.assertEqual(tuple(shared_groups[0].task_indices), (0, 1, 2, 3))

        guided_config = get_config("B5")
        guided_config["protocol"]["pretrained"] = False
        guided_model, guided_groups, _, _ = build_model_and_topology(
            model_api, guided_config, "dpi", torch.device("cpu")
        )
        self.assertEqual(
            trainable_parameter_count(guided_model),
            EXPECTED_MODEL_PARAMETERS["observability_guided_specialist"],
        )
        shared_ids = {id(parameter) for parameter in guided_groups[0].parameters}
        specialist_ids = {
            id(parameter)
            for name, parameter in guided_model.named_parameters()
            if name.startswith("specialist_")
        }
        self.assertTrue(shared_ids.isdisjoint(specialist_ids))
        self.assertEqual(tuple(guided_groups[0].task_indices), (0, 1, 2))

    def test_frozen_configuration_values(self):
        b1 = get_config("B1")
        self.assertEqual(b1["protocol"]["epochs"], 50)
        self.assertEqual(b1["gradnorm"]["alpha"], 1.5)
        self.assertEqual(b1["gradnorm"]["task_weight_learning_rate"], 0.025)
        self.assertEqual(get_config("B2")["pcgrad"]["generator_seed_offset"], 20261012)
        self.assertFalse(get_config("B3")["uncertainty"]["per_sample_head"])

    def test_seed_is_reproducible(self):
        set_seed(52)
        first = (np.random.rand(3), torch.rand(3))
        set_seed(52)
        second = (np.random.rand(3), torch.rand(3))
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_group_disjoint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, groups in (("train", ["a", "b"]), ("val", ["c"]), ("test", ["d"])):
                path = root / "{}.csv".format(name)
                pd.DataFrame({"before_id": groups, "value": range(len(groups))}).to_csv(path, index=False)
                paths.append(path)
            counts = assert_group_disjoint(*paths)
            self.assertEqual(counts, {"train": 2, "validation": 1, "outer_test": 1})
            pd.DataFrame({"before_id": ["b"]}).to_csv(paths[2], index=False)
            with self.assertRaises(ValueError):
                assert_group_disjoint(*paths)

    def test_checkpoint_decision_has_no_outer_test_argument(self):
        parameters = tuple(inspect.signature(checkpoint_improves).parameters)
        self.assertEqual(parameters, ("inner_validation_mape", "best_score"))
        self.assertTrue(checkpoint_improves(4.0, 5.0))
        self.assertFalse(checkpoint_improves(6.0, 5.0))

    def test_outer_loader_is_created_after_checkpoint_reload(self):
        from scripts.eaai_multitask import train_multitask

        source = inspect.getsource(train_multitask.run)
        reload_position = source.index('model.load_state_dict(checkpoint["model_state_dict"])')
        outer_loader_position = source.index("outer_test_loader = model_api.build_loader")
        self.assertLess(reload_position, outer_loader_position)


if __name__ == "__main__":
    unittest.main()
