"""Independently named rerun of the existing clean-trained V3 architecture.

V4 changes the experiment namespace only. It does not introduce a new model,
forward computation, initialization, parameter or training recipe.
"""

from model.ctrgcn_mamba_robust_v3 import Model as V3Model


class Model(V3Model):
    """Exact V3 architecture under a separate V4 model/config/output name."""
