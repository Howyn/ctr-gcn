"""Soft visibility pooling and consistent selective-state parameter correction.

The fully observed path retains V2's ordinary mean and unit write coefficient.
The innovation rule is parameter-free: no learned scorer is disabled by q=1.
Only current observed coordinates and their backbone features are available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ctrgcn_mamba_robust import observed_joint_fraction
from model.ctrgcn_mamba_robust_v2 import Model as V2Model, local_history
from model.ctrgcn_mamba_robust_v3 import joint_observability


ALGORITHM_ID = 'v4_soft_pool_consistent_selective_scan'


class SoftVisibilityPooling(nn.Module):
    """Blend ordinary and visible means, retaining missing-node inferred features.

At visible fraction q, output = ordinary + (1-q)*(visible-ordinary).
Zero coordinates remain a missingness heuristic, not ground-truth confidence.
"""

    def forward(self, features, coordinates):
        if (features.ndim != 5 or coordinates.ndim != 5
                or features.shape[:2] != (coordinates.shape[0], coordinates.shape[-1])
                or features.shape[-1] != coordinates.shape[3]):
            raise ValueError('Expected aligned N,M,C,T,V features and N,C,T,V,M coordinates.')
        ordinary = features.mean(dim=-1)
        with torch.autocast(device_type=features.device.type, enabled=False):
            weights = joint_observability(coordinates, features.shape[-2])[:, :, None]
            visible = (features.float() * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-6)
            observed = weights.mean(dim=-1)
            pooled = ordinary.float() + (1 - observed) * (visible - ordinary.float())
        full = (weights == 1).all(dim=-1)
        return torch.where(full, ordinary, pooled.to(dtype=features.dtype))


class ConsistencyReliability(nn.Module):
    """Past-only innovation coefficient with exact identity at full observation.

e = ||x-history|| / (||x|| + ||history|| + eps)
r = 1-(1-q)*(0.5+0.5*e), hence q <= r <= (1+q)/2.
History is also missing the same persistent joints; it is not clean recovery.
"""

    coefficient_kind = 'reliability'

    def __init__(self, history_window=3, eps=1e-6):
        super().__init__()
        if history_window < 1 or eps <= 0:
            raise ValueError('History window and numerical epsilon must be positive.')
        self.history_window = history_window
        self.eps = eps

    def forward(self, tokens, observed):
        if tokens.ndim != 3 or observed.shape != (*tokens.shape[:2], 1):
            raise ValueError('Expected tokens N,T,C and observed fractions N,T,1.')
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            values = tokens.float()
            history = local_history(values.transpose(1, 2), self.history_window).transpose(1, 2)
            difference = (values - history).norm(dim=-1, keepdim=True)
            denominator = values.norm(dim=-1, keepdim=True) + history.norm(dim=-1, keepdim=True)
            innovation = (difference / (denominator + self.eps)).clamp(0, 1)
            q = observed.float().clamp(0, 1)
            return 1 - (1 - q) * (0.5 + 0.5 * innovation)


def consistent_mamba(mamba, tokens, reliability, local_compensation=True,
                     history_window=3, parameter_correction=True):
    """Condition writes and delta/B/C on one corrected convolution feature.

Official CUDA scan is retained. D*u and standard z use original observed u;
reliability=1 recovers standard Mamba under either correction switch.
"""
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    batch, frames, _ = tokens.shape
    if reliability.shape != (batch, frames, 1):
        raise ValueError('Expected one reliability coefficient per temporal token.')
    x, z = mamba.in_proj(tokens).transpose(1, 2).contiguous().chunk(2, dim=1)
    u = causal_conv1d_fn(x=x.contiguous(), weight=mamba.conv1d.weight[:, 0, :],
                        bias=mamba.conv1d.bias, activation=mamba.activation)
    with torch.autocast(device_type=u.device.type, enabled=False):
        r = reliability.float().transpose(1, 2)
        write = r * u.float()
        if local_compensation:
            write = write + (1 - r) * local_history(u.float(), history_window)
    write = write.to(dtype=u.dtype)
    # V2/V3 still generated parameters from u after replacing writes. V4 uses
    # the actual write feature for input-dependent discretization/read/write.
    parameter_input = write if parameter_correction else u
    parameters = mamba.x_proj(parameter_input.transpose(1, 2).reshape(-1, mamba.d_inner))
    dt, B, C = parameters.split([mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1)
    delta = F.linear(dt, mamba.dt_proj.weight).view(batch, frames, mamba.d_inner)
    delta = delta.transpose(1, 2).contiguous()
    B = B.view(batch, frames, mamba.d_state).transpose(1, 2).contiguous()
    C = C.view(batch, frames, mamba.d_state).transpose(1, 2).contiguous()
    state = selective_scan_fn(write.contiguous(), delta, -torch.exp(mamba.A_log.float()),
                              B, C, D=None, z=None,
                              delta_bias=mamba.dt_proj.bias.float(), delta_softplus=True)
    with torch.autocast(device_type=u.device.type, enabled=False):
        result = (state.float() + mamba.D.float()[None, :, None] * u.float()) * F.silu(z.float())
    return mamba.out_proj(result.to(dtype=u.dtype).transpose(1, 2))


class Model(V2Model):
    def __init__(self, use_visibility_pooling=True, use_reliability=True,
                 use_local_compensation=True, use_history_compensation=True,
                 use_parameter_correction=True, history_window=3,
                 algorithm_id=ALGORITHM_ID, **kwargs):
        if algorithm_id != ALGORITHM_ID:
            raise ValueError('V4 algorithm_id does not match this implementation.')
        # Construct V2 in its original order before replacing its scorer. This
        # preserves shared backbone/Mamba/local initialization at a given seed.
        super().__init__(use_reliability=use_reliability,
                         use_local_compensation=use_local_compensation,
                         history_window=history_window, **kwargs)
        standard_parameters = sum(p.numel() for p in self.parameters()) - self.extra_parameters
        self.algorithm_id = algorithm_id
        self.use_visibility_pooling = use_visibility_pooling
        self.use_history_compensation = use_history_compensation
        self.use_parameter_correction = use_parameter_correction
        if use_visibility_pooling:
            self.temporal_pool = SoftVisibilityPooling()
        if use_reliability:
            # Old scoring parameters are removed from both state_dict/optimizer.
            self.temporal_reliability = ConsistencyReliability(history_window)
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
            reliability = self.temporal_reliability(tokens, observed_joint_fraction(x, frames))
            long_range = consistent_mamba(self.temporal_mamba, normalized, reliability,
                                          self.use_history_compensation, self.history_window,
                                          self.use_parameter_correction)
            long_range = torch.where(reliability > 0, long_range, torch.zeros_like(long_range))
        else:
            long_range = self.temporal_mamba(normalized)
        output = tokens + long_range
        if self.use_local_compensation:
            output = output + self.temporal_local(normalized)
        pooled = output.view(batch, people, frames, channels).mean(dim=2).mean(dim=1)
        return self.fc(self.drop_out(pooled))
