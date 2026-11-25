"""Matched mean, attention and reliability-write variants for NTU60 Joint."""

import torch
import torch.nn as nn

from model.ctrgcn_mamba import Model as StandardHybrid
from model.ctrgcn_mamba_robust import reliable_mamba
from model.ctrgcn_mamba_robust_v3 import joint_observability


def write_reliability(attention, observability):
    """Retain the attention mass assigned to observed joints at each token."""
    with torch.autocast(device_type=attention.device.type, enabled=False):
        missing_mass = (attention.float() * (1 - observability.float())).sum(dim=-1, keepdim=True)
        return (1 - missing_mass).clamp(0, 1)


class Model(StandardHybrid):
    def __init__(self, use_joint_attention=False, use_write_gate=False, **kwargs):
        if use_write_gate and not use_joint_attention:
            raise ValueError('The attention-weighted gate requires joint attention.')
        super().__init__(**kwargs)
        self.use_write_gate = use_write_gate
        self.joint_score = None
        if use_joint_attention:
            # Uniform initial attention and the same RNG state for loader seeding.
            with torch.random.fork_rng(devices=[]):
                self.joint_score = nn.Conv2d(self.fc.in_features, 1, kernel_size=1, bias=False)
            nn.init.zeros_(self.joint_score.weight)

    def forward(self, x, *, disable_write_gate=False):
        if x.ndim == 3:
            batch, frames, _ = x.shape
            x = x.reshape(batch, frames, self.num_point, -1).permute(0, 3, 1, 2).unsqueeze(-1)
        features = self.forward_features(x)
        batch, people, channels, frames, joints = features.shape
        features = features.reshape(batch * people, channels, frames, joints)
        reliability = features.new_ones((batch * people, frames, 1))
        if self.joint_score is None:
            pooled = features.mean(dim=-1)
        else:
            attention = self.joint_score(features).squeeze(1).softmax(dim=-1, dtype=torch.float32)
            # Scale before averaging so uniform attention starts at mean pooling.
            pooled = (features * (joints * attention).to(features.dtype).unsqueeze(1)).mean(dim=-1)
            if self.use_write_gate and not disable_write_gate:
                observed = joint_observability(x, frames).reshape(batch * people, frames, joints)
                reliability = write_reliability(attention, observed)

        tokens = pooled.transpose(1, 2).contiguous()
        # Match the explicit Mamba-1 CUDA path in all three arms. A unit
        # coefficient leaves the native recurrence and direct skip intact.
        tokens = tokens + reliable_mamba(self.temporal_mamba, self.temporal_norm(tokens), reliability)
        pooled = tokens.reshape(batch, people, frames, channels).mean(dim=2).mean(dim=1)
        return self.fc(self.drop_out(pooled))
