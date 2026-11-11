from copy import deepcopy
from pathlib import Path
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.ctrgcn_mamba import Model as StandardHybrid
from model.ctrgcn_mamba_robust_v2 import (Model, ReliabilityEstimator, local_history, reliable_mamba)
from test_robust_mamba import OfficialShapeTestMamba, cpu_official_operators, reference_scan


class V2StateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_history_uses_past_only_with_no_zero_padding_bias(self):
        values = torch.tensor([[[1., 3., 5., 7., 9.]]])
        torch.testing.assert_close(local_history(values, 3), torch.tensor([[[1., 1., 2., 3., 5.]]]))
        changed = values.clone()
        changed[..., 3:] = 100
        torch.testing.assert_close(local_history(changed, 3)[..., :4], local_history(values, 3)[..., :4])

    def test_fully_observed_gate_is_exactly_one_and_missing_gate_has_gradients(self):
        estimator = ReliabilityEstimator()
        tokens = torch.randn(2, 8, 256)
        full = estimator(tokens, torch.ones(2, 8, 1))
        torch.testing.assert_close(full, torch.ones_like(full), rtol=0, atol=0)
        partial = estimator(tokens, torch.full((2, 8, 1), 0.8))
        self.assertTrue(((partial >= 0.8) & (partial <= 1)).all())
        partial.sum().backward()
        self.assertGreater(estimator.score[-1].weight.grad.abs().max().item(), 0)

    def test_gate_one_matches_standard_outputs_and_parameter_gradients(self):
        mamba = OfficialShapeTestMamba(256)
        standard = deepcopy(mamba)
        first = torch.randn(2, 7, 256, requires_grad=True)
        second = first.detach().clone().requires_grad_()
        expected = standard(first)
        with cpu_official_operators():
            actual = reliable_mamba(mamba, second, torch.ones(2, 7, 1))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-4, atol=1e-6)
        for (_, a), (_, b) in zip(mamba.named_parameters(), standard.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)

    def test_local_substitution_affects_write_input_not_skip_delta_B_or_C(self):
        calls = []
        def record(*args, **kwargs):
            calls.append((args, kwargs))
            return reference_scan(*args, **kwargs)
        mamba = OfficialShapeTestMamba(256)
        tokens = torch.randn(2, 7, 256)
        gate = torch.full((2, 7, 1), 0.6)
        with cpu_official_operators(record):
            full = reliable_mamba(mamba, tokens, torch.ones_like(gate))
            mixed = reliable_mamba(mamba, tokens, gate)
        u = calls[0][0][0]
        torch.testing.assert_close(calls[1][0][0], 0.6 * u + 0.4 * local_history(u))
        for index in (1, 2, 3, 4):
            torch.testing.assert_close(calls[0][0][index], calls[1][0][index])
        self.assertIsNone(calls[1][1]['D'])
        self.assertIsNone(calls[1][1]['z'])
        # With state readout disabled, all gates must retain the same D*u skip.
        def skip_only(u, *args, **kwargs):
            return torch.zeros_like(u)
        with cpu_official_operators(skip_only):
            a = reliable_mamba(mamba, tokens, torch.ones_like(gate))
            b = reliable_mamba(mamba, tokens, gate)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertGreater((full - mixed).abs().max().item(), 0)


class V2ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.options = dict(num_class=60, num_point=25, num_person=2,
                            graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})

    def test_disabled_model_matches_standard_and_initial_clean_model_is_close(self):
        with cpu_official_operators():
            torch.manual_seed(1)
            standard = StandardHybrid(**self.options).eval()
            torch.manual_seed(1)
            disabled = Model(use_reliability=False, use_local_compensation=False, **self.options).eval()
            torch.manual_seed(1)
            improved = Model(**self.options).eval()
            data = torch.randn(1, 3, 64, 25, 2)
            with torch.no_grad():
                expected = standard(data)
                torch.testing.assert_close(disabled(data), expected, rtol=0, atol=0)
                torch.testing.assert_close(improved(data), expected, rtol=2e-4, atol=2e-4)

    def test_independent_switches_budget_and_masked_backward(self):
        for reliability, local, extra in ((False, True, 17921), (True, False, 8321),
                                           (True, True, 26242)):
            with self.subTest(reliability=reliability, local=local), cpu_official_operators():
                model = Model(use_reliability=reliability, use_local_compensation=local, **self.options)
                self.assertEqual(model.extra_parameters, extra)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 1884944 + extra)
                if local:
                    model.temporal_local.scale.data.fill_(0.1)
                data = torch.randn(2, 3, 64, 25, 2)
                data[0, :, :, :5] = 0
                loss = F.cross_entropy(model(data), torch.tensor([0, 1]))
                loss.backward()
                parameters = dict(model.named_parameters())
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in parameters.values()
                                    if p.grad is not None))
                for name in ('temporal_reliability.projection.weight', 'temporal_local.dilated.weight',
                             'temporal_mamba.in_proj.weight'):
                    if name in parameters:
                        self.assertGreater(parameters[name].grad.abs().max().item(), 0)

    def test_exactly_closed_gate_retains_local_fallback(self):
        class ClosedGate(nn.Module):
            def forward(self, tokens, observed):
                return torch.zeros_like(observed)
        with cpu_official_operators():
            model = Model(**self.options).eval()
            model.temporal_reliability = ClosedGate()
            model.temporal_local.scale.data.fill_(0.2)
            data = torch.randn(1, 3, 64, 25, 2)
            with torch.no_grad():
                features = model.forward_features(data)
                tokens = features.mean(-1).permute(0, 1, 3, 2).contiguous().view(2, 16, 256)
                expected = tokens + model.temporal_local(model.temporal_norm(tokens))
                expected = model.fc(expected.view(1, 2, 16, 256).mean(2).mean(1))
                torch.testing.assert_close(model(data), expected, rtol=0, atol=0)

    def test_matched_augmentation_control_differs_only_in_algorithm_and_output(self):
        root = Path(__file__).resolve().parents[1] / 'config/nturgbd-cross-subject'
        original = yaml.safe_load((root / 'ctrgcn_mamba_fast128.yaml').read_text())
        control = yaml.safe_load((root / 'ctrgcn_mamba_maskaug_fast128.yaml').read_text())
        improved = yaml.safe_load((root / 'ctrgcn_mamba_robust_v2_fast128.yaml').read_text())
        original['work_dir'] = control['work_dir']
        original['train_feeder_args'].update(joint_mask_prob=0.25, joint_mask_max=5)
        self.assertEqual(original, control)
        control['work_dir'] = improved['work_dir']
        control['model'] = 'model.ctrgcn_mamba_robust_v2.Model'
        control['model_args'].update(use_reliability=True, use_local_compensation=True, history_window=3)
        self.assertEqual(control, improved)


if __name__ == '__main__':
    unittest.main()
