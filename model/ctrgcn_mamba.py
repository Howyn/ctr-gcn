"""CTR-GCN backbone followed by an official, unidirectional Mamba-1 block."""

import torch.nn as nn

from model.ctrgcn import Model as CTRGCN


class Model(CTRGCN):
    def __init__(self, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, **kwargs):
        # Keep the backbone and classifier initialization identical to CTR-GCN.
        super().__init__(**kwargs)
        if min(mamba_d_state, mamba_d_conv, mamba_expand) < 1:
            raise ValueError('Mamba state size, convolution width and expansion must be positive.')
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as error:
            raise ImportError(
                'Official mamba_ssm and its CUDA extensions are required. '
                'On the AutoDL instance run: python tools/install_mamba.py'
            ) from error
        channels = self.fc.in_features
        self.temporal_norm = nn.LayerNorm(channels)
        self.temporal_mamba = Mamba(
            d_model=channels,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            use_fast_path=True,
        )
        # Do not run the backbone's generic initializer over the Mamba module:
        # the official block has specialized dt/A/D initialization.

    def forward(self, x):
        features = self.forward_features(x)  # N, M, 256, T', 25
        batch, people, channels, frames, _ = features.shape
        # Pool joints only: keep time and each person's sequence separate.
        tokens = features.mean(dim=-1).permute(0, 1, 3, 2).contiguous()
        tokens = tokens.view(batch * people, frames, channels)
        tokens = tokens + self.temporal_mamba(self.temporal_norm(tokens))
        pooled = tokens.view(batch, people, frames, channels).mean(dim=2).mean(dim=1)
        return self.fc(self.drop_out(pooled))
