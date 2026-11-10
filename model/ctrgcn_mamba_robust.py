"""Reliability-controlled Mamba state writes with a local temporal residual.

The backbone, Mamba parameters and classifier keep the standard hybrid's names.
The CUDA scan is still the official mamba_ssm kernel; no Python scan is used in
production. Missingness is inferred from input coordinates, never evaluator IDs.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ctrgcn_mamba import Model as StandardHybrid


def observed_joint_fraction(data, frames, eps=1e-6):
    """Return NM,T',1 observability, excluding empty people and padded frames.

All-zero xyz is a missingness heuristic, not ground-truth confidence. A joint
at the coordinate origin can also be zero. An entirely empty person/frame is
ignored rather than called 100% occluded; no person pooling rule is changed.
"""
    if data.ndim != 5:
        raise ValueError('Expected skeleton input N,C,T,V,M.')
    visible = data.detach().float().abs().amax(dim=1) > eps  # N,T,V,M
    valid = visible.any(dim=2).permute(0, 2, 1).float()  # N,M,T
    fraction = visible.float().mean(dim=2).permute(0, 2, 1)
    batch, people, _ = fraction.shape
    valid = valid.reshape(batch * people, 1, -1)
    fraction = fraction.reshape(batch * people, 1, -1)
    numerator = F.adaptive_avg_pool1d(fraction * valid, frames)
    denominator = F.adaptive_avg_pool1d(valid, frames)
    fraction = torch.where(denominator > 0, numerator / denominator.clamp_min(eps),
                           torch.ones_like(numerator))
    return fraction.transpose(1, 2).contiguous()


class ReliabilityEstimator(nn.Module):
    """Small causal consistency estimator; natural fast motion is not hard-masked."""

    def __init__(self, channels=256, projection=32):
        super().__init__()
        self.projection = nn.Linear(channels, projection, bias=False)
        self.score = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 1))
        # Start near the standard hybrid on observable clean input. Not a
        # guarantee of unchanged classification accuracy after optimization.
        nn.init.normal_(self.score[-1].weight, std=1e-3)
        nn.init.constant_(self.score[-1].bias, math.log(99.0))

    def forward(self, tokens, observed):
        # Ratios/cosines must remain stable under the training AMP context.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            projected = self.projection(tokens.float())
            previous = torch.cat((projected[:, :1], projected[:, :-1]), dim=1)
            difference = projected - previous
            previous_difference = torch.cat((torch.zeros_like(difference[:, :1]),
                                             difference[:, :-1]), dim=1)
            magnitude = projected.norm(dim=-1, keepdim=True)
            previous_magnitude = previous.norm(dim=-1, keepdim=True)
            total = magnitude + previous_magnitude
            cosine = F.cosine_similarity(projected, previous, dim=-1, eps=1e-6).unsqueeze(-1)
            cosine_distance = torch.where(total > 1e-6, (1 - cosine).clamp(0, 2) / 2,
                                          torch.zeros_like(total))
            first_difference = difference.norm(dim=-1, keepdim=True) / total.clamp_min(1e-6)
            innovation = (difference - previous_difference).norm(dim=-1, keepdim=True)
            innovation = innovation / (difference.norm(dim=-1, keepdim=True)
                                       + previous_difference.norm(dim=-1, keepdim=True)).clamp_min(1e-6)
            magnitude_change = (magnitude - previous_magnitude).abs() / total.clamp_min(1e-6)
            observed = observed.float().clamp(0, 1)
            previous_observed = torch.cat((observed[:, :1], observed[:, :-1]), dim=1)
            statistics = torch.cat((observed, cosine_distance, first_difference, innovation,
                                    magnitude_change, (observed - previous_observed).abs()), dim=-1)
            # The explicit observed factor also responds to persistent missing
            # joints: temporal differences alone cannot detect a fixed mask.
            return observed * torch.sigmoid(self.score(statistics))


class LocalTemporalCompensation(nn.Module):
    """Independent short-range path, zero-strength residual at initialization."""

    def __init__(self, channels=256, bottleneck=32):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 3, groups=channels, bias=False)
        self.reduce = nn.Conv1d(channels, bottleneck, 1, bias=False)
        self.expand = nn.Conv1d(bottleneck, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, normalized_tokens):
        sequence = normalized_tokens.transpose(1, 2)
        local = self.depthwise(F.pad(sequence, (2, 0)))
        local = self.expand(F.silu(self.reduce(local)))
        return self.scale * local.transpose(1, 2)


def reliable_mamba(mamba, tokens, reliability):
    """Mamba-1's explicit CUDA path, with B_t <- r_t B_t only.

Delta, A, C, D and the internal output gate z retain their standard meanings.
Using explicit causal_conv1d + selective_scan loses the whole-block fused path,
but retains the official parallel CUDA scan and its backward implementation.
"""
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    batch, frames, _ = tokens.shape
    xz = mamba.in_proj(tokens).transpose(1, 2).contiguous()
    x, z = xz.chunk(2, dim=1)
    u = causal_conv1d_fn(x=x.contiguous(), weight=mamba.conv1d.weight[:, 0, :],
                        bias=mamba.conv1d.bias, activation=mamba.activation)
    parameters = mamba.x_proj(u.transpose(1, 2).reshape(-1, mamba.d_inner))
    dt, B, C = torch.split(parameters, [mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1)
    # Bias is passed separately: selective_scan adds it before softplus.
    delta = F.linear(dt, mamba.dt_proj.weight).view(batch, frames, mamba.d_inner)
    delta = delta.transpose(1, 2).contiguous()
    B = B.view(batch, frames, mamba.d_state).transpose(1, 2).contiguous()
    C = C.view(batch, frames, mamba.d_state).transpose(1, 2).contiguous()
    B = (B * reliability.transpose(1, 2).to(dtype=B.dtype)).contiguous()
    result = selective_scan_fn(u, delta, -torch.exp(mamba.A_log.float()), B, C,
                               mamba.D.float(), z=z.contiguous(),
                               delta_bias=mamba.dt_proj.bias.float(), delta_softplus=True)
    return mamba.out_proj(result.transpose(1, 2))


class Model(StandardHybrid):
    def __init__(self, use_reliability=True, use_local_compensation=True, **kwargs):
        super().__init__(**kwargs)
        standard_parameters = sum(parameter.numel() for parameter in self.parameters())
        self.use_reliability = use_reliability
        self.use_local_compensation = use_local_compensation
        channels = self.fc.in_features
        if use_reliability:
            self.temporal_reliability = ReliabilityEstimator(channels)
        if use_local_compensation:
            self.temporal_local = LocalTemporalCompensation(channels)
        self.extra_parameters = sum(parameter.numel() for parameter in self.parameters()) - standard_parameters
        if self.extra_parameters > 200_000:
            raise ValueError('New modules exceed the 0.2M parameter budget.')

    def forward(self, x):
        if x.ndim == 3:
            # Match the backbone's alternative N,T,V*C input convention.
            x = x.view(x.shape[0], x.shape[1], self.graph.num_node, -1).permute(
                0, 3, 1, 2).contiguous().unsqueeze(-1)
        features = self.forward_features(x)
        batch, people, channels, frames, _ = features.shape
        tokens = features.mean(dim=-1).permute(0, 1, 3, 2).contiguous()
        tokens = tokens.view(batch * people, frames, channels)
        normalized = self.temporal_norm(tokens)
        if self.use_reliability:
            observed = observed_joint_fraction(x, frames)
            reliability = self.temporal_reliability(tokens, observed)
            long_range = reliable_mamba(self.temporal_mamba, normalized, reliability)
            # Suppress potentially unreliable historical readout as well as new
            # state writes. r=0 leaves the backbone + local convolution path.
            long_range = long_range * reliability.to(dtype=long_range.dtype)
        else:
            long_range = self.temporal_mamba(normalized)
        output = tokens + long_range
        if self.use_local_compensation:
            output = output + self.temporal_local(normalized)
        pooled = output.view(batch, people, frames, channels).mean(dim=2).mean(dim=1)
        return self.fc(self.drop_out(pooled))
