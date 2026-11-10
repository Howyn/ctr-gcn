"""CPU algorithm checks using test-only stand-ins for official CUDA kernels.

Real fused/explicit CUDA equivalence and AMP are checked on AutoDL by
tools/check_robust_mamba.py. None of these reference operators is a model fallback.
"""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.ctrgcn_mamba import Model as StandardHybrid
from model.ctrgcn_mamba_robust import (LocalTemporalCompensation, Model,
                                      ReliabilityEstimator, observed_joint_fraction, reliable_mamba)


def reference_scan(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
    dt = delta + delta_bias[None, :, None]
    if delta_softplus:
        dt = F.softplus(dt)
    state = torch.zeros(u.shape[0], u.shape[1], A.shape[1], dtype=u.dtype, device=u.device)
    output = []
    for t in range(u.shape[-1]):
        decay = torch.exp(dt[:, :, t, None] * A[None, :, :])
        write = dt[:, :, t, None] * B[:, None, :, t] * u[:, :, t, None]
        state = decay * state + write
        output.append((state * C[:, None, :, t]).sum(-1) + D[None, :] * u[:, :, t])
    return torch.stack(output, dim=-1) * F.silu(z)


def reference_conv(x, weight, bias, activation):
    assert activation == 'silu'
    result = F.conv1d(x, weight[:, None, :], bias, padding=weight.shape[-1] - 1,
                      groups=x.shape[1])[..., :x.shape[-1]]
    return F.silu(result)


class OfficialShapeTestMamba(nn.Module):
    """Exact parameter shapes, independent plain-PyTorch forward for CPU tests."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.d_inner, self.d_state = expand * d_model, d_state
        self.dt_rank = (d_model + 15) // 16
        self.activation = 'silu'
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, tokens):
        x, z = self.in_proj(tokens).transpose(1, 2).chunk(2, dim=1)
        u = F.silu(self.conv1d(x)[..., :tokens.shape[1]])
        dt, B, C = self.x_proj(u.transpose(1, 2)).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.linear(dt, self.dt_proj.weight).transpose(1, 2)
        y = reference_scan(u, delta, -torch.exp(self.A_log), B.transpose(1, 2),
                           C.transpose(1, 2), self.D, z, self.dt_proj.bias, True)
        return self.out_proj(y.transpose(1, 2))


@contextmanager
def cpu_official_operators(scan=reference_scan):
    mamba = types.ModuleType('mamba_ssm')
    mamba.Mamba = OfficialShapeTestMamba
    conv = types.ModuleType('causal_conv1d')
    conv.causal_conv1d_fn = reference_conv
    interface = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    interface.selective_scan_fn = scan
    with patch.dict(sys.modules, {'mamba_ssm': mamba, 'causal_conv1d': conv,
                                 'mamba_ssm.ops.selective_scan_interface': interface}):
        yield


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_visibility_excludes_empty_person_and_padding_but_detects_fixed_five_joints(self):
        data = torch.ones(1, 3, 8, 25, 2)
        data[..., 1] = 0
        data[:, :, 6:] = 0
        data[:, :, :, [0, 4, 8, 12, 16], 0] = 0
        observed = observed_joint_fraction(data, 2)
        torch.testing.assert_close(observed[0], torch.full((2, 1), 0.8))
        torch.testing.assert_close(observed[1], torch.ones(2, 1))

    def test_estimator_finite_at_zero_and_responsive_to_persistent_missingness(self):
        estimator = ReliabilityEstimator()
        tokens = torch.zeros(2, 16, 256, requires_grad=True)
        full = estimator(tokens, torch.ones(2, 16, 1))
        missing = estimator(tokens, torch.full((2, 16, 1), 0.8))
        self.assertTrue(torch.isfinite(full).all())
        self.assertTrue((missing < full).all())
        self.assertTrue(((missing >= 0) & (missing <= 0.8)).all())
        (full.sum() + missing.sum()).backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_estimator_and_local_branch_do_not_read_future_tokens(self):
        estimator = ReliabilityEstimator()
        local = LocalTemporalCompensation()
        local.scale.data.fill_(1)
        first = torch.randn(2, 9, 256)
        changed = first.clone()
        changed[:, 5:] += 30
        observed = torch.ones(2, 9, 1)
        torch.testing.assert_close(estimator(first, observed)[:, :5],
                                   estimator(changed, observed)[:, :5])
        torch.testing.assert_close(local(first)[:, :5], local(changed)[:, :5])

    def test_local_residual_is_zero_initially_but_scale_can_learn(self):
        local = LocalTemporalCompensation()
        output = local(torch.randn(2, 8, 256))
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        output.sum().backward()
        self.assertGreater(local.scale.grad.abs().item(), 0)


class StateWriteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.mamba = OfficialShapeTestMamba(256)
        self.tokens = torch.randn(2, 7, 256)

    def test_unit_reliability_preserves_standard_outputs_and_gradients(self):
        reference = deepcopy(self.mamba)
        original_input = self.tokens.clone().requires_grad_()
        changed_input = self.tokens.clone().requires_grad_()
        expected = reference(original_input)
        with cpu_official_operators():
            actual = reliable_mamba(self.mamba, changed_input, torch.ones(2, 7, 1))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(changed_input.grad, original_input.grad, rtol=1e-4, atol=1e-6)
        for (_, actual_parameter), (_, expected_parameter) in zip(
                self.mamba.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=1e-4, atol=1e-6)

    def test_gate_changes_only_write_coefficients_and_receives_gradient(self):
        calls = []
        def record(*args, **kwargs):
            calls.append((args, kwargs))
            return reference_scan(*args, **kwargs)
        gate = torch.full((2, 7, 1), 0.3, requires_grad=True)
        with cpu_official_operators(scan=record):
            reliable_mamba(self.mamba, self.tokens, torch.ones_like(gate))
            result = reliable_mamba(self.mamba, self.tokens, gate)
        for index in (0, 1, 2, 4, 5):
            torch.testing.assert_close(calls[0][0][index], calls[1][0][index])
        torch.testing.assert_close(calls[1][0][3], calls[0][0][3] * 0.3)
        result.sum().backward()
        self.assertTrue(torch.isfinite(gate.grad).all())
        self.assertGreater(gate.grad.abs().max().item(), 0)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.options = dict(num_class=60, num_point=25, num_person=2,
                            graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})

    def test_both_switches_off_match_standard_weights_logits_and_initialization(self):
        with cpu_official_operators():
            torch.manual_seed(1)
            standard = StandardHybrid(**self.options).eval()
            torch.manual_seed(1)
            disabled = Model(use_reliability=False, use_local_compensation=False, **self.options).eval()
            self.assertEqual(disabled.extra_parameters, 0)
            self.assertEqual(standard.state_dict().keys(), disabled.state_dict().keys())
            for key, value in standard.state_dict().items():
                torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
            data = torch.randn(1, 3, 64, 25, 2)
            with torch.no_grad():
                torch.testing.assert_close(standard(data), disabled(data), rtol=0, atol=0)

    def test_independent_switches_parameter_budget_and_full_backward(self):
        for reliability, local, extra in ((False, True, 17153), (True, False, 8321),
                                           (True, True, 25474)):
            with self.subTest(reliability=reliability, local=local), cpu_official_operators():
                model = Model(use_reliability=reliability, use_local_compensation=local, **self.options)
                self.assertEqual(model.extra_parameters, extra)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 1884944 + extra)
                if local:
                    model.temporal_local.scale.data.fill_(0.1)
                data = torch.randn(2, 3, 64, 25, 2)
                data[0, :, :, :5] = 0
                logits = model(data)
                self.assertEqual(tuple(logits.shape), (2, 60))
                F.cross_entropy(logits, torch.tensor([0, 1])).backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                for name in ('temporal_mamba.in_proj.weight', 'temporal_reliability.projection.weight',
                             'temporal_local.expand.weight'):
                    parameters = dict(model.named_parameters())
                    if name in parameters:
                        self.assertGreater(parameters[name].grad.abs().max().item(), 0)

    def test_closed_gate_leaves_backbone_and_independent_local_path(self):
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
                local = model.temporal_local(model.temporal_norm(tokens))
                expected = model.fc((tokens + local).view(1, 2, 16, 256).mean(2).mean(1))
                torch.testing.assert_close(model(data), expected, rtol=0, atol=0)

    def test_only_algorithm_and_output_change_in_training_config(self):
        root = Path(__file__).resolve().parents[1] / 'config' / 'nturgbd-cross-subject'
        original = yaml.safe_load((root / 'ctrgcn_mamba_fast128.yaml').read_text())
        improved = yaml.safe_load((root / 'ctrgcn_mamba_robust_fast128.yaml').read_text())
        original['work_dir'] = './work_dir/ntu60/xsub/ctrgcn_mamba_robust_joint_fast128'
        original['model'] = 'model.ctrgcn_mamba_robust.Model'
        original['model_args'].update(use_reliability=True, use_local_compensation=True)
        self.assertEqual(improved, original)


if __name__ == '__main__':
    unittest.main()
