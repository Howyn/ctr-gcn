"""V4 equations and gradient checks using test-only official-shape CPU ops."""

from copy import deepcopy
import unittest

import torch
import torch.nn.functional as F

from model.ctrgcn_mamba import Model as StandardHybrid
from model.ctrgcn_mamba_robust_v2 import Model as V2Model, reliable_mamba as v2_mamba
from model.ctrgcn_mamba_robust_v3 import VisibilityPooling
from model.ctrgcn_mamba_robust_v4 import (
    ALGORITHM_ID, ConsistencyReliability, Model, SoftVisibilityPooling, consistent_mamba,
)
from test_robust_mamba import OfficialShapeTestMamba, cpu_official_operators, reference_scan


class SoftPoolingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_missing_node_features_are_retained_with_finite_nonzero_gradients(self):
        data = torch.ones(1, 3, 8, 25, 2)
        data[:, :, :, :5] = 0
        features = torch.ones(1, 2, 4, 2, 25, requires_grad=True)
        with torch.no_grad():
            features[..., :5] = 100
        original = features.detach().clone()
        soft = SoftVisibilityPooling()(features, data)
        hard = VisibilityPooling()(features, data)
        torch.testing.assert_close(hard, torch.ones_like(hard))
        torch.testing.assert_close(soft, torch.full_like(soft, 16.84))
        soft.sum().backward()
        self.assertGreater(features.grad[..., :5].abs().min().item(), 0)
        self.assertTrue(torch.isfinite(features.grad).all())
        torch.testing.assert_close(features.detach(), original, rtol=0, atol=0)

    def test_full_observation_empty_people_and_padding_preserve_ordinary_mean(self):
        features = torch.randn(2, 2, 16, 4, 25)
        data = torch.ones(2, 3, 16, 25, 2)
        data[1, ..., 1] = 0
        data[0, :, 8:] = 0
        torch.testing.assert_close(SoftVisibilityPooling()(features, data), features.mean(-1),
                                   rtol=0, atol=0)


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_parameter_free_gate_is_bounded_and_full_observation_is_exact_one(self):
        gate = ConsistencyReliability()
        self.assertEqual(sum(p.numel() for p in gate.parameters()), 0)
        values = torch.randn(2, 8, 256, requires_grad=True)
        q = torch.full((2, 8, 1), 0.8)
        result = gate(values, q)
        self.assertTrue((result >= q - 1e-7).all())
        self.assertTrue((result <= (1 + q) / 2 + 1e-7).all())
        torch.testing.assert_close(gate(values, torch.ones_like(q)), torch.ones_like(q), rtol=0, atol=0)
        result.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(values.grad.abs().max().item(), 0)

    def test_gate_reads_no_future_and_is_finite_at_zero(self):
        gate = ConsistencyReliability()
        values = torch.randn(2, 9, 256)
        changed = values.clone()
        changed[:, 5:] += 100
        q = torch.full((2, 9, 1), 0.8)
        torch.testing.assert_close(gate(values, q)[:, :5], gate(changed, q)[:, :5], rtol=0, atol=0)
        torch.testing.assert_close(gate(torch.zeros_like(values), q), torch.full_like(q, 0.9))


class ConsistentStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_unit_gate_matches_standard_outputs_and_parameter_gradients(self):
        model = OfficialShapeTestMamba(256)
        reference = deepcopy(model)
        first = torch.randn(2, 7, 256, requires_grad=True)
        second = first.detach().clone().requires_grad_()
        expected = reference(first)
        with cpu_official_operators():
            actual = consistent_mamba(model, second, torch.ones(2, 7, 1))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-4, atol=1e-6)
        for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)

    def test_write_and_delta_B_C_use_consistent_input_with_an_independent_switch(self):
        calls = []
        def record(*args, **kwargs):
            calls.append((args, kwargs))
            return reference_scan(*args, **kwargs)
        model = OfficialShapeTestMamba(256)
        tokens = torch.randn(2, 7, 256)
        gate = torch.full((2, 7, 1), 0.8)
        with cpu_official_operators(record):
            v2_mamba(model, tokens, gate)
            consistent_mamba(model, tokens, gate)
            consistent_mamba(model, tokens, gate, parameter_correction=False)
        torch.testing.assert_close(calls[0][0][0], calls[1][0][0])
        for index in (1, 3, 4):
            self.assertGreater((calls[0][0][index] - calls[1][0][index]).abs().max().item(), 0)
            torch.testing.assert_close(calls[0][0][index], calls[2][0][index], rtol=0, atol=0)
        torch.testing.assert_close(calls[0][0][2], calls[1][0][2], rtol=0, atol=0)
        self.assertIsNone(calls[1][1]['D'])
        self.assertIsNone(calls[1][1]['z'])

    def test_direct_skip_is_unchanged_by_state_parameter_correction(self):
        def no_state(u, *args, **kwargs):
            return torch.zeros_like(u)
        model = OfficialShapeTestMamba(256)
        tokens = torch.randn(2, 7, 256)
        with cpu_official_operators(no_state):
            full = consistent_mamba(model, tokens, torch.ones(2, 7, 1))
            masked = consistent_mamba(model, tokens, torch.full((2, 7, 1), 0.8))
        torch.testing.assert_close(full, masked, rtol=0, atol=0)

    def test_corrected_scan_reads_no_future_and_masked_backward_is_finite(self):
        model = OfficialShapeTestMamba(256)
        tokens = torch.randn(2, 9, 256, requires_grad=True)
        changed = tokens.detach().clone()
        changed[:, 5:] += 30
        gate = torch.full((2, 9, 1), 0.8)
        with cpu_official_operators():
            actual = consistent_mamba(model, tokens, gate)
            alternative = consistent_mamba(model, changed, gate)
        torch.testing.assert_close(actual[:, :5], alternative[:, :5])
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(tokens.grad.abs().max().item(), 0)


class V4ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.options = dict(num_class=60, num_point=25, num_person=2,
                            graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})

    def test_shared_v2_initialization_and_clean_logits_are_preserved(self):
        with cpu_official_operators():
            torch.manual_seed(1)
            reference = V2Model(**self.options).eval()
            torch.manual_seed(1)
            candidate = Model(**self.options).eval()
            original = reference.state_dict()
            for name, value in candidate.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            self.assertFalse(any(name.startswith('temporal_reliability.') for name in candidate.state_dict()))
            values = torch.randn(2, 3, 16, 25, 2)
            with torch.no_grad():
                torch.testing.assert_close(candidate(values), reference(values), rtol=2e-4, atol=2e-4)

    def test_independent_switches_and_exact_parameter_budget(self):
        with cpu_official_operators():
            for pool, gate, local, history, parameters in (
                    (True, True, True, True, True), (False, True, True, False, False),
                    (True, False, True, False, False), (True, True, False, True, True),
                    (False, False, False, False, False)):
                model = Model(use_visibility_pooling=pool, use_reliability=gate,
                              use_local_compensation=local, use_history_compensation=history,
                              use_parameter_correction=parameters, **self.options)
                extra = 17921 if local else 0
                self.assertEqual(model.extra_parameters, extra)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 1884944 + extra)
                self.assertLess(model.extra_parameters, 200000)
                self.assertEqual(model.algorithm_id, ALGORITHM_ID)

    def test_all_switches_off_match_standard_exactly(self):
        with cpu_official_operators():
            torch.manual_seed(1)
            reference = StandardHybrid(**self.options).eval()
            torch.manual_seed(1)
            candidate = Model(use_visibility_pooling=False, use_reliability=False,
                              use_local_compensation=False, use_history_compensation=False,
                              use_parameter_correction=False, **self.options).eval()
            values = torch.randn(2, 3, 16, 25, 2)
            with torch.no_grad():
                torch.testing.assert_close(candidate(values), reference(values), rtol=0, atol=0)

    def test_clean_local_training_and_masked_backward_are_finite(self):
        with cpu_official_operators():
            model = Model(**self.options).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
            values = torch.randn(2, 3, 16, 25, 2)
            for _ in range(3):
                optimizer.zero_grad()
                F.cross_entropy(model(values), torch.tensor([0, 1])).backward()
                optimizer.step()
            self.assertGreater(model.temporal_local.short.weight.grad.abs().max().item(), 0)
            values[0, :, :, :5] = 0
            optimizer.zero_grad()
            F.cross_entropy(model(values), torch.tensor([0, 1])).backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_wrong_algorithm_identifier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'algorithm_id'):
            Model(algorithm_id='v4_v3_rerun', **self.options)

    def test_exactly_closed_gate_keeps_backbone_and_local_fallback(self):
        class ClosedGate(torch.nn.Module):
            def forward(self, tokens, observed):
                return torch.zeros_like(observed)

        with cpu_official_operators():
            model = Model(**self.options).eval()
            model.temporal_reliability = ClosedGate()
            model.temporal_local.scale.data.fill_(0.2)
            values = torch.randn(1, 3, 16, 25, 2)
            with torch.no_grad():
                features = model.forward_features(values)
                batch, people, channels, frames, _ = features.shape
                tokens = features.mean(-1).permute(0, 1, 3, 2).contiguous()
                tokens = tokens.view(batch * people, frames, channels)
                expected = tokens + model.temporal_local(model.temporal_norm(tokens))
                expected = expected.view(batch, people, frames, channels).mean(2).mean(1)
                expected = model.fc(model.drop_out(expected))
                torch.testing.assert_close(model(values), expected, rtol=0, atol=0)

    def test_three_dimensional_single_person_input(self):
        with cpu_official_operators():
            model = Model(**dict(self.options, num_person=1)).eval()
            with torch.no_grad():
                output = model(torch.randn(2, 16, 75))
            self.assertEqual(tuple(output.shape), (2, 60))
            self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
