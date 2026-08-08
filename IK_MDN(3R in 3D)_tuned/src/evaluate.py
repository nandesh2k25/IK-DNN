import math
import numpy as np
import torch
from helper_fk import forward_kinematics_torch


def evaluate_best_of_k(
    model,
    loader,
    device,
    params,
    theta_mean,
    theta_std,
):
    """
    Best-of-K evaluation.

    For each input, the MDN predicts K candidate joint configurations.
    We compute:
        best joint-angle error
        best FK pose error
        best component index
    """

    model.eval()

    theta_mean_t = torch.tensor(
        theta_mean,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, -1)

    theta_std_t = torch.tensor(
        theta_std,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, -1)

    all_x_original = []
    all_y_true_real = []
    all_mu_real = []
    all_pi = []
    all_fk_candidates = []
    all_best_component = []
    all_best_fk_pose_mse = []
    all_best_angle_mse = []
    all_best_pred = []

    with torch.no_grad():
        for x_norm, x_original, y_norm in loader:
            x_norm = x_norm.to(device)
            x_original = x_original.to(device)
            y_norm = y_norm.to(device)

            pi, mu_norm, sigma = model(x_norm)

            mu_real = mu_norm * theta_std_t + theta_mean_t

            y_true_real = (
                y_norm * theta_std_t.squeeze(0).squeeze(0)
                + theta_mean_t.squeeze(0).squeeze(0)
            )

            fk_candidates = forward_kinematics_torch(mu_real, params)

            pose_target = x_original.unsqueeze(1)

            fk_error = torch.mean(
                (fk_candidates - pose_target) ** 2,
                dim=2,
            )

            y_expanded = y_true_real.unsqueeze(1)

            angle_error = torch.mean(
                (mu_real - y_expanded) ** 2,
                dim=2,
            )

            score = fk_error - 0.01 * torch.log(pi + 1e-8)

            best_component = torch.argmin(score, dim=1)

            batch_indices = torch.arange(
                x_norm.shape[0],
                device=device,
            )

            best_pred = mu_real[batch_indices, best_component]

            best_fk_pose_mse = fk_error[batch_indices, best_component]
            best_angle_mse = angle_error[batch_indices, best_component]

            all_x_original.append(x_original.cpu().numpy())
            all_y_true_real.append(y_true_real.cpu().numpy())
            all_mu_real.append(mu_real.cpu().numpy())
            all_pi.append(pi.cpu().numpy())
            all_fk_candidates.append(fk_candidates.cpu().numpy())
            all_best_component.append(best_component.cpu().numpy())
            all_best_fk_pose_mse.append(best_fk_pose_mse.cpu().numpy())
            all_best_angle_mse.append(best_angle_mse.cpu().numpy())
            all_best_pred.append(best_pred.cpu().numpy())

    x_original = np.concatenate(all_x_original, axis=0)
    y_true_real = np.concatenate(all_y_true_real, axis=0)
    predicted_all_components_real = np.concatenate(all_mu_real, axis=0)
    mixture_probabilities = np.concatenate(all_pi, axis=0)
    fk_pose_candidates = np.concatenate(all_fk_candidates, axis=0)
    best_component_indices = np.concatenate(all_best_component, axis=0)
    best_fk_pose_mse = np.concatenate(all_best_fk_pose_mse, axis=0)
    best_angle_mse = np.concatenate(all_best_angle_mse, axis=0)
    predicted_best_fk_real = np.concatenate(all_best_pred, axis=0)

    metrics = {
        "best_of_k_angle_mse": float(np.mean(best_angle_mse)),
        "best_of_k_angle_rmse": float(math.sqrt(np.mean(best_angle_mse))),
        "best_of_k_pose_mse": float(np.mean(best_fk_pose_mse)),
        "best_of_k_pose_rmse": float(math.sqrt(np.mean(best_fk_pose_mse))),
        "theta1_mae": float(np.mean(np.abs(predicted_best_fk_real[:, 0] - y_true_real[:, 0]))),
        "theta2_mae": float(np.mean(np.abs(predicted_best_fk_real[:, 1] - y_true_real[:, 1]))),
        "theta3_mae": float(np.mean(np.abs(predicted_best_fk_real[:, 2] - y_true_real[:, 2]))),
    }

    predictions = {
        "x_original": x_original,
        "y_true_real": y_true_real,
        "predicted_all_components_real": predicted_all_components_real,
        "predicted_best_fk_real": predicted_best_fk_real,
        "mixture_probabilities": mixture_probabilities,
        "fk_pose_candidates": fk_pose_candidates,
        "best_component_indices": best_component_indices,
        "best_fk_pose_mse": best_fk_pose_mse,
        "best_angle_mse": best_angle_mse,
    }

    return metrics, predictions