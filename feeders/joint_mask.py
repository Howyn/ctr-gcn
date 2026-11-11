"""Training-only joint dropout; evaluation masks are generated independently."""

import numpy as np


def mask_training_joints(data, probability=0.0, max_joints=5):
    """Occasionally zero 1..K joint trajectories, identically for all people.

    Returns a copy only when masking. Probability zero consumes no RNG values,
    preserving the original feeder's augmentation sequence by default.
    """
    if not 0 <= probability <= 1 or not 1 <= max_joints <= data.shape[2]:
        raise ValueError('Invalid joint masking probability or maximum joint count.')
    if probability == 0 or np.random.random() >= probability:
        return data
    count = np.random.randint(1, max_joints + 1)
    joints = np.random.choice(data.shape[2], count, replace=False)
    masked = data.copy()
    masked[:, :, joints, :] = 0
    return masked
