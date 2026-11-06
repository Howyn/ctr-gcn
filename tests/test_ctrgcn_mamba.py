"""CPU integration tests; the real CUDA kernel is checked by tools/check_mamba.py."""

import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from model.ctrgcn import Model as CTRGCN
from model.ctrgcn_mamba import Model as CTRGCNMamba


class TestMixer(nn.Module):
    """CPU test double only; it never appears in the production model."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.last_shape = None

    def forward(self, tokens):
        self.last_shape = tuple(tokens.shape)
        return self.projection(tokens)


class LegacyCTRGCN(CTRGCN):
    """Reference classifier path before exposing forward_features."""
    def forward(self, x):
        batch, channels, frames, joints, people = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(batch, people * joints * channels, frames)
        x = self.data_bn(x)
        x = x.view(batch, people, joints, channels, frames).permute(
            0, 1, 3, 4, 2).contiguous().view(batch * people, channels, frames, joints)
        for index in range(1, 11):
            x = getattr(self, 'l{}'.format(index))(x)
        x = x.view(batch, people, x.size(1), -1).mean(3).mean(1)
        return self.fc(self.drop_out(x))


class HybridTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.options = dict(num_class=60, num_point=25, num_person=2,
                            graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})

    def hybrid(self):
        module = types.ModuleType('mamba_ssm')
        module.Mamba = TestMixer
        with patch.dict(sys.modules, {'mamba_ssm': module}):
            return CTRGCNMamba(**self.options)

    def test_backbone_refactor_preserves_logits_and_checkpoint_keys(self):
        legacy = LegacyCTRGCN(**self.options).eval()
        updated = CTRGCN(**self.options).eval()
        updated.load_state_dict(legacy.state_dict(), strict=True)
        data = torch.randn(1, 3, 64, 25, 2)
        with torch.no_grad():
            torch.testing.assert_close(updated(data), legacy(data), rtol=0, atol=0)

    def test_mamba_receives_time_sequence_and_backward_reaches_both_modules(self):
        model = self.hybrid().train()
        data = torch.randn(2, 3, 64, 25, 2)
        logits = model(data)
        self.assertEqual(tuple(logits.shape), (2, 60))
        self.assertEqual(model.temporal_mamba.last_shape, (4, 16, 256))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        for parameter in (model.temporal_mamba.projection.weight, model.l1.gcn1.convs[0].conv3.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().max().item(), 0)

    def test_zero_mixer_preserves_person_joint_and_time_pooling(self):
        model = self.hybrid().eval()
        baseline = CTRGCN(**self.options).eval()
        baseline.load_state_dict({name: value for name, value in model.state_dict().items()
                                  if not name.startswith(('temporal_norm.', 'temporal_mamba.'))}, strict=True)
        with torch.no_grad():
            model.temporal_mamba.projection.weight.zero_()
            data = torch.randn(2, 3, 64, 25, 2)
            torch.testing.assert_close(model(data), baseline(data), rtol=1e-5, atol=1e-6)

    def test_missing_official_package_fails_instead_of_using_a_fallback(self):
        with patch.dict(sys.modules, {'mamba_ssm': None}):
            with self.assertRaisesRegex(ImportError, 'Official mamba_ssm'):
                CTRGCNMamba(**self.options)


if __name__ == '__main__':
    unittest.main()
