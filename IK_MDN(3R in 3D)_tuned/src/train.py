from pathlib import Path
import torch
from torch.utils.data import Dataset
from helper_fk import forward_kinematics_torch, joint_limit_penalty
from robot_params import RobotParams3R3D


class IKDataset3D(Dataset):
    """
    Dataset returns:
        x_norm
        x_original
        y_norm
    """

    def __init__(self, x_norm, x_original, y_norm):
        self.x_norm = torch.tensor(x_norm, dtype=torch.float32)
        self.x_original = torch.tensor(x_original, dtype=torch.float32)
        self.y_norm = torch.tensor(y_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.x_norm)

    def __getitem__(self, idx):
        return self.x_norm[idx], self.x_original[idx], self.y_norm[idx]


def fk_consistency_loss(
    mu_real,
    x_original,
    params: RobotParams3R3D,
    limit_weight: float = 0.05,
):
    """
    FK consistency loss.

    Uses the best FK candidate among all mixture components.
    """

    fk_candidates = forward_kinematics_torch(mu_real, params)

    target_pose = x_original.unsqueeze(1)

    fk_error = torch.mean(
        (fk_candidates - target_pose) ** 2,
        dim=2,
    )

    limit_penalty = joint_limit_penalty(mu_real, params)

    total_error = fk_error + limit_weight * limit_penalty

    best_error = torch.min(total_error, dim=1)[0]

    return best_error.mean()


def evaluate_loss(
    model,
    loader,
    device,
    theta_mean_t,
    theta_std_t,
    params,
    lambda_fk,
):
    model.eval()

    total_loss = 0.0
    total_nll = 0.0
    total_fk = 0.0
    total_count = 0

    with torch.no_grad():
        for x_norm, x_original, y_norm in loader:
            x_norm = x_norm.to(device)
            x_original = x_original.to(device)
            y_norm = y_norm.to(device)

            pi, mu_norm, sigma = model(x_norm)

            nll_loss = model.mdn_loss(pi, mu_norm, sigma, y_norm)

            mu_real = mu_norm * theta_std_t + theta_mean_t

            fk_loss = fk_consistency_loss(mu_real, x_original, params)

            loss = nll_loss + lambda_fk * fk_loss

            batch_size = x_norm.shape[0]

            total_loss += loss.item() * batch_size
            total_nll += nll_loss.item() * batch_size
            total_fk += fk_loss.item() * batch_size
            total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "nll_loss": total_nll / total_count,
        "fk_loss": total_fk / total_count,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    feature_mean,
    feature_std,
    theta_mean,
    theta_std,
    params,
    config,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "robot_params": params.to_dict(),
        "training_config": config,
    }

    torch.save(checkpoint, path)


def train_mdn(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    params,
    theta_mean,
    theta_std,
    feature_mean,
    feature_std,
    config,
    experiment_dirs,
    logger,
):
    """
    Train MDN with:
        MDN NLL loss
        FK consistency loss
        early stopping
        best checkpoint saving
    """

    max_epochs = config["max_epochs"]
    patience = config["patience"]
    min_delta = config["min_delta"]
    lambda_fk = config["lambda_fk"]

    model.to(device)

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

    best_val_loss = float("inf")
    best_epoch = 0
    no_improvement_count = 0
    early_stopped = False

    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()

        total_loss = 0.0
        total_nll = 0.0
        total_fk = 0.0
        total_count = 0

        for x_norm, x_original, y_norm in train_loader:
            x_norm = x_norm.to(device)
            x_original = x_original.to(device)
            y_norm = y_norm.to(device)

            optimizer.zero_grad()

            pi, mu_norm, sigma = model(x_norm)

            nll_loss = model.mdn_loss(pi, mu_norm, sigma, y_norm)

            mu_real = mu_norm * theta_std_t + theta_mean_t

            fk_loss = fk_consistency_loss(mu_real, x_original, params)

            loss = nll_loss + lambda_fk * fk_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            batch_size = x_norm.shape[0]

            total_loss += loss.item() * batch_size
            total_nll += nll_loss.item() * batch_size
            total_fk += fk_loss.item() * batch_size
            total_count += batch_size

        train_loss = total_loss / total_count
        train_nll = total_nll / total_count
        train_fk = total_fk / total_count

        val_stats = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
            theta_mean_t=theta_mean_t,
            theta_std_t=theta_std_t,
            params=params,
            lambda_fk=lambda_fk,
        )

        val_loss = val_stats["loss"]
        val_nll = val_stats["nll_loss"]
        val_fk = val_stats["fk_loss"]

        learning_rate = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_nll_loss": train_nll,
            "train_fk_loss": train_fk,
            "val_nll_loss": val_nll,
            "val_fk_loss": val_fk,
            "learning_rate": learning_rate,
        }

        history.append(row)

        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}/{max_epochs}, "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )

        improved = val_loss < best_val_loss - min_delta

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement_count = 0

            best_model_path = experiment_dirs["models"] / "best_model.pt"
            best_checkpoint_path = experiment_dirs["models"] / "checkpoint_best.pt"

            torch.save(model.state_dict(), best_model_path)

            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                feature_mean=feature_mean,
                feature_std=feature_std,
                theta_mean=theta_mean,
                theta_std=theta_std,
                params=params,
                config=config,
            )

            logger.info(f"Validation improved to {best_val_loss:.6f} at epoch {epoch}.")
            logger.info(f"Saved best model to {best_model_path}.")
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            early_stopped = True
            logger.info(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    final_model_path = experiment_dirs["models"] / "final_model.pt"

    torch.save(model.state_dict(), final_model_path)

    logger.info(f"Saved final model to {final_model_path}.")

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "early_stopped": early_stopped,
    }