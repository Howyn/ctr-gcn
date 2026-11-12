from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
import yaml

from model.ctrgcn_mamba import Model as StandardHybrid
from model.ctrgcn_mamba_robust import observed_joint_fraction
from model.ctrgcn_mamba_robust_v2 import local_history, reliable_mamba
from model.ctrgcn_mamba_robust_v3 import (Model, ResidualWriteSelector, VisibilityPooling,
                                        joint_observability)
from test_robust_mamba import OfficialShapeTestMamba, cpu_official_operators, reference_scan
from tools import audit_occlusion as audit
from tools.check_clean_comparison import check_comparison
from tools.evaluate_occlusion import matching_test_settings, matching_training_inputs, read_config


class VisibilityPoolingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_missing_joints_remain_zero_weight_and_empty_people_padding_are_ignored(self):
        data = torch.ones(1, 3, 8, 25, 2)
        data[..., 1] = 0
        data[:, :, 6:] = 0
        missing = [0, 4, 8, 12, 16]
        data[:, :, :, missing, 0] = 0
        weights = joint_observability(data, 2)
        self.assertEqual(tuple(weights.shape), (1, 2, 2, 25))
        self.assertEqual(torch.count_nonzero(weights[0, 0, :, missing]).item(), 0)
        torch.testing.assert_close(weights[0, 1], torch.ones(2, 25), rtol=0, atol=0)
        expected = observed_joint_fraction(data, 2)
        torch.testing.assert_close(weights.mean(-1).reshape(2, 2, 1), expected)

    def test_pool_uses_visible_features_not_missing_position_features(self):
        data = torch.ones(1, 3, 8, 25, 2)
        data[:, :, :, :5] = 0
        features = torch.ones(1, 2, 256, 2, 25, requires_grad=True)
        with torch.no_grad():
            features[..., :5] = 100
        original = features.detach().clone()
        result = VisibilityPooling()(features, data)
        torch.testing.assert_close(result, torch.ones(1, 2, 256, 2), rtol=0, atol=0)
        result.sum().backward()
        self.assertEqual(torch.count_nonzero(features.grad[..., :5]).item(), 0)
        self.assertGreater(features.grad[..., 5:].abs().max().item(), 0)
        torch.testing.assert_close(features.detach(), original, rtol=0, atol=0)

    def test_fully_observed_and_empty_slots_preserve_the_exact_ordinary_mean(self):
        features = torch.randn(2, 2, 256, 4, 25)
        data = torch.ones(2, 3, 16, 25, 2)
        data[1, ..., 1] = 0
        torch.testing.assert_close(VisibilityPooling()(features, data), features.mean(-1), rtol=0, atol=0)


class ResidualSelectorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_zero_scale_is_exact_identity_even_when_observation_is_missing(self):
        selector = ResidualWriteSelector()
        tokens = torch.randn(2, 8, 256)
        observed = torch.full((2, 8, 1), 0.8)
        coefficient = selector(tokens, observed)
        torch.testing.assert_close(coefficient, torch.ones_like(observed), rtol=0, atol=0)
        coefficient.sum().backward()
        self.assertGreater(selector.scale.grad.abs().item(), 0)
        self.assertEqual(selector.score[-1].weight.grad.abs().max().item(), 0)

    def test_clean_gate_learns_after_residual_scale_moves(self):
        selector = ResidualWriteSelector()
        tokens = torch.randn(2, 8, 256)
        observed = torch.ones(2, 8, 1)
        optimizer = torch.optim.SGD(selector.parameters(), lr=0.02)
        for _ in range(2):
            optimizer.zero_grad()
            selector(tokens, observed).square().mean().backward()
            optimizer.step()
        self.assertGreater(selector.score[-1].weight.grad.abs().max().item(), 0)
        self.assertTrue(torch.isfinite(selector.score[-1].weight.grad).all())

    def test_signed_residual_changes_state_write_but_preserves_standard_skip(self):
        calls = []
        def record(*args, **kwargs):
            calls.append((args, kwargs))
            return reference_scan(*args, **kwargs)
        mamba = OfficialShapeTestMamba(256)
        selector = ResidualWriteSelector()
        tokens = torch.randn(2, 8, 256)
        observed = torch.ones(2, 8, 1)
        with cpu_official_operators(record):
            reliable_mamba(mamba, tokens, torch.ones_like(observed))
            for scale in (-0.4, 0.4):
                selector.scale.data.fill_(scale)
                coefficient = selector(tokens, observed)
                self.assertTrue(((coefficient >= 0) & (coefficient <= 2)).all())
                reliable_mamba(mamba, tokens, coefficient)
                u = calls[0][0][0]
                expected = coefficient.transpose(1, 2) * u + (1 - coefficient.transpose(1, 2)) * local_history(u)
                torch.testing.assert_close(calls[-1][0][0], expected)
                for index in (1, 2, 3, 4):
                    torch.testing.assert_close(calls[0][0][index], calls[-1][0][index])
                self.assertIsNone(calls[-1][1]['D'])
                self.assertIsNone(calls[-1][1]['z'])


class V3ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.options = dict(num_class=60, num_point=25, num_person=2,
                            graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})

    def test_initial_clean_model_matches_standard_and_all_disabled_is_exact_standard(self):
        with cpu_official_operators():
            torch.manual_seed(1)
            standard = StandardHybrid(**self.options).eval()
            torch.manual_seed(1)
            improved = Model(**self.options).eval()
            torch.manual_seed(1)
            disabled = Model(use_visibility_pooling=False, use_reliability=False,
                             use_local_compensation=False, use_history_compensation=False, **self.options).eval()
            data = torch.randn(2, 3, 16, 25, 2)
            with torch.no_grad():
                expected = standard(data)
                torch.testing.assert_close(improved(data), expected, rtol=2e-4, atol=2e-4)
                torch.testing.assert_close(disabled(data), expected, rtol=0, atol=0)

    def test_parameter_budget_and_independent_module_switches(self):
        with cpu_official_operators():
            for pool, gate, local, history, extra in ((True, True, True, True, 18051),
                    (False, True, True, True, 18051), (True, False, True, False, 17921),
                    (True, True, False, False, 130), (True, False, False, False, 0)):
                model = Model(use_visibility_pooling=pool, use_reliability=gate,
                              use_local_compensation=local, use_history_compensation=history, **self.options)
                self.assertEqual(model.extra_parameters, extra)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 1884944 + extra)
                self.assertLess(model.extra_parameters, 200000)

    def test_classification_loss_updates_selector_on_fully_observed_training_data(self):
        with cpu_official_operators():
            model = Model(**self.options).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9,
                                        nesterov=True, weight_decay=0.0004)
            data = torch.randn(2, 3, 16, 25, 2)
            for _ in range(3):
                optimizer.zero_grad()
                F.cross_entropy(model(data), torch.tensor([0, 1])).backward()
                optimizer.step()
            gradient = model.temporal_reliability.score[-1].weight.grad
            self.assertGreater(gradient.abs().max().item(), 0)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertNotEqual(model.temporal_reliability.scale.item(), 0)

    def test_audit_checks_the_current_masked_coordinates_and_pooled_temporal_path(self):
        with cpu_official_operators():
            model = Model(**self.options).eval()
            data = torch.randn(2, 3, 16, 25, 2)
            data[:, :, :, :5] = 0
            details = {}
            first = audit.checked_forward(model, data, diagnostics=details)
            audit.checked_forward(model, torch.ones_like(data))
            torch.testing.assert_close(audit.checked_forward(model, data), first, rtol=0, atol=0)
            self.assertIn('residual_write_coefficient', details)
            self.assertEqual(details['signed_residual_scale'], 0)

    def test_config_matches_original_training_and_existing_v2_preprocessing(self):
        root = Path(__file__).resolve().parents[1] / 'config/nturgbd-cross-subject'
        standard = yaml.safe_load((root / 'ctrgcn_mamba_fast128.yaml').read_text())
        v2 = yaml.safe_load((root / 'ctrgcn_mamba_robust_v2_clean_fast128.yaml').read_text())
        v3 = yaml.safe_load((root / 'ctrgcn_mamba_robust_v3_clean_fast128.yaml').read_text())
        check_comparison(standard, v3)
        self.assertTrue(matching_training_inputs(dict(baseline=standard, mamba=v3)))
        matching_test_settings(dict(baseline=v2, mamba=v3))
        expected = dict(v2)
        expected.update(work_dir=v3['work_dir'], model=v3['model'], model_args=v3['model_args'])
        self.assertEqual(expected, v3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.yaml').write_text(yaml.safe_dump(v3))
            self.assertEqual(read_config(path, 'model.ctrgcn_mamba_robust_v3.Model')['model'], v3['model'])


if __name__ == '__main__':
    unittest.main()
