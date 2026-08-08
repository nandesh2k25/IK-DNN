import numpy as np
import torch
from robot_params import RobotParams3R3D


def forward_kinematics_numpy(theta: np.ndarray, params: RobotParams3R3D) -> np.ndarray:
    """
    NumPy FK for 3R robot in 3D.

    Args:
        theta:
            Shape (N, 3)
            Columns: theta1, theta2, theta3

    Returns:
        pose:
            Shape (N, 3)
            Columns: x, y, z
    """

    theta = np.asarray(theta, dtype=np.float32)

    t1 = theta[:, 0]
    t2 = theta[:, 1]
    t3 = theta[:, 2]

    r = params.L1 + params.L2 * np.cos(t2) + params.L3 * np.cos(t2 + t3)

    x = r * np.cos(t1)
    y = r * np.sin(t1)
    z = params.L2 * np.sin(t2) + params.L3 * np.sin(t2 + t3)

    return np.stack([x, y, z], axis=1).astype(np.float32)


def forward_kinematics_torch(
    theta: torch.Tensor,
    params: RobotParams3R3D,
) -> torch.Tensor:
    """
    Torch FK for 3R robot in 3D.

    Args:
        theta:
            Shape can be:
                (batch, 3)
                (batch, components, 3)

    Returns:
        pose:
            Shape:
                (batch, 1, 3), if input was (batch, 3)
                (batch, components, 3), if input was (batch, components, 3)
    """

    if theta.dim() == 2:
        theta = theta.unsqueeze(1)

    t1 = theta[:, :, 0]
    t2 = theta[:, :, 1]
    t3 = theta[:, :, 2]

    r = params.L1 + params.L2 * torch.cos(t2) + params.L3 * torch.cos(t2 + t3)

    x = r * torch.cos(t1)
    y = r * torch.sin(t1)
    z = params.L2 * torch.sin(t2) + params.L3 * torch.sin(t2 + t3)

    return torch.stack([x, y, z], dim=2)


def joint_limit_penalty(theta: torch.Tensor, params: RobotParams3R3D) -> torch.Tensor:
    """
    Soft penalty for predictions outside joint limits.

    Args:
        theta:
            Shape (batch, components, 3)

    Returns:
        penalty:
            Shape (batch, components)
    """

    limits = torch.tensor(
        [
            params.theta1_lim,
            params.theta2_lim,
            params.theta3_lim,
        ],
        dtype=theta.dtype,
        device=theta.device,
    )

    lower = limits[:, 0].view(1, 1, 3)
    upper = limits[:, 1].view(1, 1, 3)

    below = torch.relu(lower - theta)
    above = torch.relu(theta - upper)

    penalty = torch.mean(below**2 + above**2, dim=2)

    return penalty