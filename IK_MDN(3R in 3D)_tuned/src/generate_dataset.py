import numpy as np
from robot_params import RobotParams3R3D
from helper_fk import forward_kinematics_numpy


def generate_dataset(
    n_samples: int,
    params: RobotParams3R3D,
    seed: int = 42,
):
    """
    Generate dataset using joint-space sampling.

    This preserves multiple IK branches because many different joint
    configurations can produce nearby or similar Cartesian positions.

    Args:
        n_samples:
            Number of samples.

        params:
            Robot parameters.

        seed:
            Random seed.

    Returns:
        features:
            Shape (N, 3)
            [x, y, z]

        targets:
            Shape (N, 3)
            [theta1, theta2, theta3]
    """

    rng = np.random.default_rng(seed)

    theta1 = rng.uniform(
        params.theta1_lim[0],
        params.theta1_lim[1],
        size=n_samples,
    )

    theta2 = rng.uniform(
        params.theta2_lim[0],
        params.theta2_lim[1],
        size=n_samples,
    )

    theta3 = rng.uniform(
        params.theta3_lim[0],
        params.theta3_lim[1],
        size=n_samples,
    )

    targets = np.stack([theta1, theta2, theta3], axis=1).astype(np.float32)

    features = forward_kinematics_numpy(targets, params).astype(np.float32)

    return features, targets


def split_and_normalize(
    features,
    targets,
    train_fraction=0.8,
    val_fraction=0.1,
    test_fraction=0.1,
    seed=0,
):
    """
    Split and normalize dataset.

    Returns a dictionary containing raw data, normalized data, indices,
    and normalization statistics.
    """

    assert abs(train_fraction + val_fraction + test_fraction - 1.0) < 1e-6

    n_samples = len(features)

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(train_fraction * n_samples)
    n_val = int(val_fraction * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    feature_mean = features[train_idx].mean(axis=0, keepdims=True)
    feature_std = features[train_idx].std(axis=0, keepdims=True) + 1e-8

    theta_mean = targets[train_idx].mean(axis=0, keepdims=True)
    theta_std = targets[train_idx].std(axis=0, keepdims=True) + 1e-8

    features_norm = (features - feature_mean) / feature_std
    targets_norm = (targets - theta_mean) / theta_std

    return {
        "features_original": features.astype(np.float32),
        "targets_original": targets.astype(np.float32),
        "features_normalized": features_norm.astype(np.float32),
        "targets_normalized": targets_norm.astype(np.float32),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "theta_mean": theta_mean.astype(np.float32),
        "theta_std": theta_std.astype(np.float32),
    }