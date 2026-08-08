import math
import torch
from torch import nn
from helper_fk import forward_kinematics_torch, joint_limit_penalty
from robot_params import RobotParams3R3D


class MDN3R3D(nn.Module):
    """
    Mixture Density Network for 3R inverse kinematics in 3D.

    Input:
        [x, y, z]

    Output:
        Mixture distribution over:
            [theta1, theta2, theta3]

    For every pose, the model predicts K possible joint configurations.
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 3,
        hidden_size: int = 256,
        n_components: int = 8,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

        final_dim = hidden_size // 2

        self.pi_head = nn.Linear(final_dim, n_components)
        self.mu_head = nn.Linear(final_dim, n_components * output_dim)
        self.log_sigma_head = nn.Linear(final_dim, n_components * output_dim)

    def forward(self, x):
        """
        Args:
            x:
                Shape (batch, input_dim)

        Returns:
            pi:
                Shape (batch, components)

            mu:
                Shape (batch, components, output_dim)

            sigma:
                Shape (batch, components, output_dim)
        """

        h = self.backbone(x)

        pi = torch.softmax(self.pi_head(h), dim=1)

        mu = self.mu_head(h).view(
            -1,
            self.n_components,
            self.output_dim,
        )

        log_sigma = self.log_sigma_head(h).view(
            -1,
            self.n_components,
            self.output_dim,
        )

        log_sigma = torch.clamp(log_sigma, min=-5.0, max=2.0)

        sigma = torch.exp(log_sigma)

        return pi, mu, sigma

    def mdn_loss(self, pi, mu, sigma, target):
        """
        Negative log likelihood loss for MDN.

        target is normalized joint angle:
            Shape (batch, 3)
        """

        target = target.unsqueeze(1)

        var = sigma**2

        log_prob = -0.5 * torch.sum(((target - mu) ** 2) / var, dim=2)

        log_prob += -0.5 * self.output_dim * math.log(2.0 * math.pi)

        log_prob += -torch.sum(torch.log(sigma + 1e-8), dim=2)

        weighted_log_prob = log_prob + torch.log(pi + 1e-8)

        log_likelihood = torch.logsumexp(weighted_log_prob, dim=1)

        return -torch.mean(log_likelihood)

    def predict_best_fk(
        self,
        x_norm,
        x_original,
        theta_mean,
        theta_std,
        params: RobotParams3R3D,
        probability_weight: float = 0.01,
        limit_weight: float = 0.05,
    ):
        """
        Predict best joint configuration using FK consistency.

        The model predicts multiple candidate joint solutions.
        We run FK on every candidate and select the one with the lowest
        physical pose error.

        Args:
            x_norm:
                Normalized input.

            x_original:
                Original pose [x, y, z].

            theta_mean:
                Shape (1, 1, 3)

            theta_std:
                Shape (1, 1, 3)

            params:
                Robot parameters.

        Returns:
            best_theta:
                Shape (batch, 3)

            info:
                Dictionary containing all candidates and selected component.
        """

        self.eval()

        with torch.no_grad():
            pi, mu_norm, sigma = self.forward(x_norm)

            mu_real = mu_norm * theta_std + theta_mean

            fk_candidates = forward_kinematics_torch(mu_real, params)

            target_pose = x_original.unsqueeze(1)

            fk_error = torch.mean(
                (fk_candidates - target_pose) ** 2,
                dim=2,
            )

            limit_penalty = joint_limit_penalty(mu_real, params)

            score = (
                fk_error
                + limit_weight * limit_penalty
                - probability_weight * torch.log(pi + 1e-8)
            )

            best_component = torch.argmin(score, dim=1)

            batch_indices = torch.arange(
                x_norm.shape[0],
                device=x_norm.device,
            )

            best_theta = mu_real[batch_indices, best_component]

            info = {
                "pi": pi,
                "mu_real": mu_real,
                "fk_candidates": fk_candidates,
                "fk_error": fk_error,
                "best_component": best_component,
            }

            return best_theta, info