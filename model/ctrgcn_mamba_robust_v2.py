"""Missingness-conditioned state writes with a local-history alternative.

V1 is preserved in ctrgcn_mamba_robust.py. V2 removes ordinary output scaling,
keeps the Mamba D skip untouched, and blends a local history into the write input.
No missing coordinates or true clean states are available at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ctrgcn_mamba import Model as StandardHybrid
from model.ctrgcn_mamba_robust import observed_joint_fraction


class ReliabilityEstimator(nn.Module):
    """Complete observation gives r=1; missingness activates learned regulation."""

    def __init__(self, channels=256, projection=32):
        super().__init__()
        self.projection = nn.Linear(channels, projection, bias=False)
        self.score = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.normal_(self.score[-1].weight, std=1e-3)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, tokens, observed):
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
            # No unconditional sigmoid attenuation on complete clean input.
            # With q=.8 this lies between .8 and 1, instead of being capped at .8.
            return 1 - (1 - observed) * torch.sigmoid(self.score(statistics))


class LocalTemporalCompensation(nn.Module):
    def __init__(self, channels=256, bottleneck=32):
        super().__init__()
        self.short = nn.Conv1d(channels, channels, 3, groups=channels, bias=False)
        self.dilated = nn.Conv1d(channels, channels, 3, dilation=2, groups=channels, bias=False)
        self.reduce = nn.Conv1d(channels, bottleneck, 1, bias=False)
        self.expand = nn.Conv1d(bottleneck, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, tokens):
        sequence = tokens.transpose(1, 2)
        local = (self.short(F.pad(sequence, (2, 0)))
                 + self.dilated(F.pad(sequence, (4, 0)))) * 0.5
        local = self.expand(F.silu(self.reduce(local)))
        return self.scale.tanh() * local.transpose(1, 2)


def local_history(sequence, window=3):
    """Mean of at most W preceding tokens; no future or current token leakage.

    At the first token no history exists, so use the current token itself.
    The history is an alternative feature estimate, not a recovered clean pose.
    """
    if window < 1:
        raise ValueError('History window must be positive.')
    with torch.autocast(device_type=sequence.device.type, enabled=False):
        values = sequence.float()
        prefix = F.pad(values.cumsum(dim=-1), (1, 0))[..., :-1]
        before_window = F.pad(prefix, (window, 0))[..., :sequence.shape[-1]]
        counts = torch.arange(sequence.shape[-1], device=sequence.device).clamp(max=window)
        average = (prefix - before_window) / counts.clamp_min(1)
        average = torch.where(counts[None, None, :] > 0, average, values)
    return average.to(dtype=sequence.dtype)


def reliable_mamba(mamba, tokens, reliability, local_compensation=True, history_window=3):
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    batch, frames, _ = tokens.shape
    x, z = mamba.in_proj(tokens).transpose(1, 2).contiguous().chunk(2, dim=1)
    u = causal_conv1d_fn(x=x.contiguous(), weight=mamba.conv1d.weight[:, 0, :],
                        bias=mamba.conv1d.bias, activation=mamba.activation)
    parameters = mamba.x_proj(u.transpose(1, 2).reshape(-1, mamba.d_inner))
    dt, B, C = parameters.split([mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1)
    delta = F.linear(dt, mamba.dt_proj.weight).view(batch, frames, mamba.d_inner)
    delta = delta.transpose(1, 2).contiguous()
    B = B.view(batch, frames, mamba.d_state).transpose(1, 2).contiguous()
    C = C.view(batch, frames, mamba.d_state).transpose(1, 2).contiguous()
    r = reliability.transpose(1, 2).to(dtype=u.dtype)
    write_input = r * u
    if local_compensation:
        write_input = write_input + (1 - r) * local_history(u, history_window)
    # The scan's u affects both its write and D skip. Separate the skip so that
    # local substitution changes state writes only, never the direct D*u path.
    state_output = selective_scan_fn(write_input.contiguous(), delta, -torch.exp(mamba.A_log.float()),
                                    B, C, D=None, z=None,
                                    delta_bias=mamba.dt_proj.bias.float(), delta_softplus=True)
    with torch.autocast(device_type=u.device.type, enabled=False):
        result = (state_output.float() + mamba.D.float()[None, :, None] * u.float()) * F.silu(z.float())
    result = result.to(dtype=u.dtype)
    return mamba.out_proj(result.transpose(1, 2))


class Model(StandardHybrid):
    def __init__(self, use_reliability=True, use_local_compensation=True,
                 history_window=3, **kwargs):
        super().__init__(**kwargs)
        if history_window < 1:
            raise ValueError('History window must be positive.')
        standard_parameters = sum(p.numel() for p in self.parameters())
        self.use_reliability = use_reliability
        self.use_local_compensation = use_local_compensation
        self.history_window = history_window
        channels = self.fc.in_features
        if use_reliability:
            self.temporal_reliability = ReliabilityEstimator(channels)
        if use_local_compensation:
            self.temporal_local = LocalTemporalCompensation(channels)
        self.extra_parameters = sum(p.numel() for p in self.parameters()) - standard_parameters
        if self.extra_parameters > 200_000:
            raise ValueError('New modules exceed the 0.2M parameter budget.')

    def forward(self, x):
        if x.ndim == 3:
            x = x.view(x.shape[0], x.shape[1], self.graph.num_node, -1).permute(
                0, 3, 1, 2).contiguous().unsqueeze(-1)
        features = self.forward_features(x)
        batch, people, channels, frames, _ = features.shape
        tokens = features.mean(dim=-1).permute(0, 1, 3, 2).contiguous()
        tokens = tokens.view(batch * people, frames, channels)
        normalized = self.temporal_norm(tokens)
        if self.use_reliability:
            reliability = self.temporal_reliability(tokens, observed_joint_fraction(x, frames))
            long_range = reliable_mamba(self.temporal_mamba, normalized, reliability,
                                        self.use_local_compensation, self.history_window)
            # Only an exactly closed gate invokes the hard fallback. Ordinary
            # reliability never multiplies the whole output a second time.
            long_range = torch.where(reliability > 0, long_range, torch.zeros_like(long_range))
        else:
            long_range = self.temporal_mamba(normalized)
        output = tokens + long_range
        if self.use_local_compensation:
            output = output + self.temporal_local(normalized)
        pooled = output.view(batch, people, frames, channels).mean(dim=2).mean(dim=1)
        return self.fc(self.drop_out(pooled))
