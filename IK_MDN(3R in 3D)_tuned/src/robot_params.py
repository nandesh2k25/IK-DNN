from dataclasses import dataclass, asdict
import math
from typing import Tuple, Dict, Any


@dataclass
class RobotParams3R3D:
    """
    3R robot in 3D.

    Kinematic structure:
        theta1 = base yaw around z-axis
        theta2 = shoulder pitch
        theta3 = elbow pitch

    FK model:
        r = L1 + L2*cos(theta2) + L3*cos(theta2 + theta3)

        x = r*cos(theta1)
        y = r*sin(theta1)
        z = L2*sin(theta2) + L3*sin(theta2 + theta3)

    This is a simple 3D position-control robot.
    It predicts x, y, z only, not full 6D pose.
    """

    L1: float = 1.0
    L2: float = 1.0
    L3: float = 0.5

    theta1_lim: Tuple[float, float] = (-math.pi, math.pi)
    theta2_lim: Tuple[float, float] = (-math.pi / 2, math.pi / 2)
    theta3_lim: Tuple[float, float] = (-math.pi / 2, math.pi / 2)

    @property
    def n_dof(self) -> int:
        return 3

    @property
    def input_dim(self) -> int:
        return 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)