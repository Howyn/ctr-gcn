import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn as nn


spec = importlib.util.spec_from_file_location(
    'check_mamba', Path(__file__).resolve().parents[1] / 'tools' / 'check_mamba.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class TinyHybrid(nn.Module):
    """Exercise the diagnostic flow with real CPU autograd and GradScaler."""
    def __init__(self):
        super().__init__()
        self.temporal_mamba = nn.Module()
        self.temporal_mamba.in_proj = nn.Linear(4, 4, bias=False)
        self.fc = nn.Linear(4, 3)

    def forward(self, data):
        return self.fc(self.temporal_mamba.in_proj(data).tanh())


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, lr=0.02):
        super().__init__(parameters, lr=lr)
        self.actual_steps = 0

    def step(self, *args, **kwargs):
        self.actual_steps += 1
        return super().step(*args, **kwargs)


class PrecisionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.model = TinyHybrid()
        self.data = torch.randn(2, 4) * 0.1
        self.labels = torch.tensor([0, 1])
        self.optimizer = CountingSGD(self.model.parameters())

    def run_check(self, amp=True, attempts=6):
        scaler = torch.amp.GradScaler('cpu', init_scale=8, enabled=amp)
        return checker.run_backward_check(self.model, self.optimizer, scaler,
                                          self.data, self.labels, amp, max_attempts=attempts)

    def test_initial_overflows_are_skipped_and_finite_updates_are_required(self):
        calls = [0]
        def overflow_first_two(gradient):
            calls[0] += 1
            return torch.full_like(gradient, float('inf')) if calls[0] <= 2 else gradient
        self.model.temporal_mamba.in_proj.weight.register_hook(overflow_first_two)
        result = self.run_check()
        self.assertEqual(result['overflow_attempts'], 2)
        self.assertEqual(result['successful_steps'], 2)
        self.assertEqual(result['attempts'], 4)
        self.assertEqual(result['final_scale'], 2)
        self.assertEqual(self.optimizer.actual_steps, 2)

    def test_persistent_overflow_never_passes_or_updates_weights(self):
        original = {name: value.clone() for name, value in self.model.state_dict().items()}
        self.model.temporal_mamba.in_proj.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float('inf')))
        with self.assertRaisesRegex(RuntimeError, 'Only 0 finite updates'):
            self.run_check(attempts=3)
        self.assertEqual(self.optimizer.actual_steps, 0)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_fp32_reports_bad_parameter_name_without_an_update(self):
        self.model.temporal_mamba.in_proj.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float('nan')))
        with self.assertRaisesRegex(RuntimeError, 'FP32.*temporal_mamba.in_proj.weight'):
            self.run_check(amp=False)
        self.assertEqual(self.optimizer.actual_steps, 0)

    def test_finite_fp32_path_passes_after_two_real_updates(self):
        result = self.run_check(amp=False)
        self.assertEqual(result['successful_steps'], 2)
        self.assertEqual(self.optimizer.actual_steps, 2)
        self.assertEqual(result['overflow_attempts'], 0)

    def test_disconnected_mamba_branch_cannot_pass(self):
        self.model.forward = lambda data: self.model.fc(data)
        with self.assertRaisesRegex(RuntimeError, 'Mamba branch'):
            self.run_check(amp=False)
        self.assertEqual(self.optimizer.actual_steps, 0)

    def test_zero_learning_rate_cannot_claim_weights_were_updated(self):
        self.optimizer = CountingSGD(self.model.parameters(), lr=0)
        with self.assertRaisesRegex(RuntimeError, 'weights did not change'):
            self.run_check(amp=False)

    def test_nonfinite_forward_cannot_be_hidden_by_scale_retries(self):
        forward = self.model.forward
        self.model.forward = lambda data: forward(data) * float('nan')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite forward'):
            self.run_check()
        self.assertEqual(self.optimizer.actual_steps, 0)


if __name__ == '__main__':
    unittest.main()
