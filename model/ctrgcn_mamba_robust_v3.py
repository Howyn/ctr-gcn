"""Clean-trained visibility pooling and residual correction of Mamba state writes.

No training masks, extra loss, teacher, extra Mamba layer or external mask input.
V1/V2 implementations are preserved. The visibility map is a zero-coordinate
heuristic derived solely from the current input, not ground-truth confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ctrgcn_mamba import Model as StandardHybrid
from model.ctrgcn_mamba_robust import observed_joint_fraction
from model.ctrgcn_mamba_robust_v2 import LocalTemporalCompensation, reliable_mamba


def joint_observability(data, frames, eps=1e-6):
    """Return N,M,T',V observed fractions, ignoring padding and empty people.

An entirely empty person/frame uses ones to retain the original empty-slot
behavior; it is not evidence that an entirely missing person can be recovered.
"""
    if data.ndim != 5 or frames < 1:
        raise ValueError('Expected N,C,T,V,M coordinates and a positive frame count.')
    visible = (data.detach().float().abs().amax(dim=1) > eps).permute(0, 3, 2, 1).float()
    batch, people, joints, length = visible.shape
    visible = visible.reshape(batch * people, joints, length)
    valid = visible.amax(dim=1, keepdim=True)
    numerator = F.adaptive_avg_pool1d(visible, frames)
    denominator = F.adaptive_avg_pool1d(valid, frames)
    fractions = torch.where(denominator > 0, numerator / denominator.clamp_min(eps),
                            torch.ones_like(numerator))
    return fractions.view(batch, people, joints, frames).transpose(2, 3).contiguous()


class VisibilityPooling(nn.Module):
    """Normalize over observed joints; preserve the exact ordinary mean if full."""

    def forward(self, features, coordinates):
        if features.ndim != 5 or features.shape[:2] != (coordinates.shape[0], coordinates.shape[-1]):
            raise ValueError('Expected features N,M,C,T,V aligned with coordinates N,C,T,V,M.')
        if features.shape[-1] != coordinates.shape[3]:
            raise ValueError('Features and coordinates have different joint counts.')
        ordinary = features.mean(dim=-1)
        with torch.autocast(device_type=features.device.type, enabled=False):
            weights = joint_observability(coordinates, features.shape[-2])[:, :, None]
            pooled = (features.float() * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-6)
        full = (weights == 1).all(dim=-1)
        return torch.where(full, ordinary, pooled.to(dtype=features.dtype))


class ResidualWriteSelector(nn.Module):
    """Always-active feature selector with a zero-initialized signed residual.

g = sigmoid(score(stats) + log(q)); rho = tanh(scale)
w = u + rho * (1-g) * (history-u)

Returns r = 1-rho*(1-g), so the existing official-scan adapter constructs w.
r is a write coefficient in [0,2], NOT a confidence probability. Negative rho
permits a bounded extrapolating correction rather than a convex interpolation.
At rho=0 the write is exactly standard. The selector receives classification
gradients after rho starts moving; q=1 no longer permanently disables it.
"""

    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.normal_(self.score[-1].weight, std=1e-3)
        nn.init.zeros_(self.score[-1].bias)
        self.scale = nn.Parameter(torch.zeros(()))
        self.coefficient_kind = 'residual_write_coefficient'

    def forward(self, tokens, observed):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            values = tokens.float()
            previous = torch.cat((values[:, :1], values[:, :-1]), dim=1)
            difference = values - previous
            previous_difference = torch.cat((torch.zeros_like(difference[:, :1]), difference[:, :-1]), dim=1)
            magnitude = values.norm(dim=-1, keepdim=True)
            previous_magnitude = previous.norm(dim=-1, keepdim=True)
            total = magnitude + previous_magnitude
            cosine = F.cosine_similarity(values, previous, dim=-1, eps=1e-6).unsqueeze(-1)
            cosine_distance = torch.where(total > 1e-6, (1 - cosine).clamp(0, 2) / 2,
                                          torch.zeros_like(total))
            first = difference.norm(dim=-1, keepdim=True) / total.clamp_min(1e-6)
            innovation = (difference - previous_difference).norm(dim=-1, keepdim=True)
            innovation = innovation / (difference.norm(dim=-1, keepdim=True)
                                       + previous_difference.norm(dim=-1, keepdim=True)).clamp_min(1e-6)
            magnitude_change = (magnitude - previous_magnitude).abs() / total.clamp_min(1e-6)
            observed = observed.float().clamp(0, 1)
            previous_observed = torch.cat((observed[:, :1], observed[:, :-1]), dim=1)
            statistics = torch.cat((observed, cosine_distance, first, innovation,
                                    magnitude_change, (observed - previous_observed).abs()), dim=-1)
            gate = torch.sigmoid(self.score(statistics) + observed.clamp_min(1e-6).log())
            return 1 - self.scale.tanh() * (1 - gate)


class Model(StandardHybrid):
    def __init__(self, use_visibility_pooling=True, use_reliability=True,
                 use_local_compensation=True, use_history_compensation=True,
                 history_window=3, **kwargs):
        super().__init__(**kwargs)
        if history_window < 1:
            raise ValueError('History window must be positive.')
        standard_parameters = sum(p.numel() for p in self.parameters())
        self.use_visibility_pooling = use_visibility_pooling
        self.use_reliability = use_reliability
        self.use_local_compensation = use_local_compensation
        self.use_history_compensation = use_history_compensation
        self.history_window = history_window
        if use_visibility_pooling:
            self.temporal_pool = VisibilityPooling()
        if use_reliability:
            self.temporal_reliability = ResidualWriteSelector()
        if use_local_compensation:
            self.temporal_local = LocalTemporalCompensation(self.fc.in_features)
        self.extra_parameters = sum(p.numel() for p in self.parameters()) - standard_parameters
        if self.extra_parameters > 200_000:
            raise ValueError('New modules exceed the 0.2M parameter budget.')

    def forward(self, x):
        if x.ndim == 3:
            x = x.view(x.shape[0], x.shape[1], self.num_point, -1).permute(
                0, 3, 1, 2).contiguous().unsqueeze(-1)
        features = self.forward_features(x)
        batch, people, channels, frames, _ = features.shape
        pooled = self.temporal_pool(features, x) if self.use_visibility_pooling else features.mean(dim=-1)
        tokens = pooled.permute(0, 1, 3, 2).contiguous().view(batch * people, frames, channels)
        normalized = self.temporal_norm(tokens)
        if self.use_reliability:
            coefficient = self.temporal_reliability(tokens, observed_joint_fraction(x, frames))
            long_range = reliable_mamba(self.temporal_mamba, normalized, coefficient,
                                        self.use_history_compensation, self.history_window)
            long_range = torch.where(coefficient > 0, long_range, torch.zeros_like(long_range))
        else:
            long_range = self.temporal_mamba(normalized)
        output = tokens + long_range
        if self.use_local_compensation:
            output = output + self.temporal_local(normalized)
        pooled = output.view(batch, people, frames, channels).mean(dim=2).mean(dim=1)
        return self.fc(self.drop_out(pooled))
