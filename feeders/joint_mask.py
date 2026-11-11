"""Training-only joint dropout; evaluation masks are generated independently."""

import numpy as np
import torch


def mask_training_joints(data, probability=0.0, max_joints=5):
    """Occasionally zero 1..K joint trajectories, identically for all people.

    Accepts a NumPy array or the Tensor returned by random_rot. Preserve input
    representation, dtype/device and storage ownership. Probability zero
    consumes no RNG values, preserving the original augmentation sequence.
    """
    if not 0 <= probability <= 1 or not 1 <= max_joints <= data.shape[2]:
        raise ValueError('Invalid joint masking probability or maximum joint count.')
    if probability == 0 or np.random.random() >= probability:
        return data
    count = np.random.randint(1, max_joints + 1)
    joints = np.random.choice(data.shape[2], count, replace=False).tolist()
    masked = data.clone() if torch.is_tensor(data) else data.copy()
    masked[:, :, joints, :] = 0
    return masked
