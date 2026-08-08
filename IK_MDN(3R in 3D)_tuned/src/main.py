from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from robot_params import RobotParams3R3D
from generate_dataset import generate_dataset, split_and_normalize
from train import IKDataset3D, train_mdn
from evaluate import evaluate_best_of_k
from ik_mdn import MDN3R3D
from plots_tables import (
    create_experiment_dirs,
    setup_logger,
    save_json,
    save_npz,
    save_train_history,
    save_final_metrics,
    save_prediction_tables,
    save_training_plots,
    save_dataset_plots,
    save_prediction_plots,
)


def make_loader(processed, indices, batch_size, shuffle=False):
    return DataLoader(
        IKDataset3D(
            processed["features_normalized"][indices],
            processed["features_original"][indices],
            processed["targets_normalized"][indices],
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_unseen_experiment(
    model,
    params,
    processed,
    config,
    experiment_dirs,
    device,
    logger,
):
    """
    Generate 1000 unseen samples and evaluate model on them.
    """

    logger.info("Running unseen-data experiment with 1000 samples.")

    unseen_features, unseen_targets = generate_dataset(
        n_samples=1000,
        params=params,
        seed=config["unseen_seed"],
    )

    feature_mean = processed["feature_mean"]
    feature_std = processed["feature_std"]
    theta_mean = processed["theta_mean"]
    theta_std = processed["theta_std"]

    unseen_features_norm = (unseen_features - feature_mean) / feature_std
    unseen_targets_norm = (unseen_targets - theta_mean) / theta_std

    unseen_dataset = {
        "features_original": unseen_features.astype(np.float32),
        "targets_original": unseen_targets.astype(np.float32),
        "features_normalized": unseen_features_norm.astype(np.float32),
        "targets_normalized": unseen_targets_norm.astype(np.float32),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
    }

    save_npz(
        unseen_dataset,
        experiment_dirs["unseen_dataset"] / "unseen_dataset.npz",
    )

    unseen_loader = DataLoader(
        IKDataset3D(
            unseen_features_norm,
            unseen_features,
            unseen_targets_norm,
        ),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    unseen_metrics, unseen_predictions = evaluate_best_of_k(
        model=model,
        loader=unseen_loader,
        device=device,
        params=params,
        theta_mean=theta_mean,
        theta_std=theta_std,
    )

    save_npz(
        unseen_predictions,
        experiment_dirs["unseen_predictions"] / "unseen_predictions.npz",
    )

    save_json(
        unseen_metrics,
        experiment_dirs["unseen_metrics"] / "unseen_metrics.json",
    )

    import pandas as pd

    pd.DataFrame([unseen_metrics]).to_csv(
        experiment_dirs["unseen_metrics"] / "unseen_metrics.csv",
        index=False,
    )

    save_prediction_tables(
        unseen_predictions,
        experiment_dirs["unseen_tables"],
        config["n_components"],
        prefix="unseen",
    )

    save_prediction_plots(
        unseen_predictions,
        experiment_dirs["unseen_plots"],
        config["n_components"],
        prefix="unseen",
    )

    logger.info("Saved unseen-data dataset, predictions, metrics, tables, and plots.")

    return unseen_metrics


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    config = {
        "project_name": "ik_mdn_3r_3d",
        "n_samples": 15000,
        "batch_size": 256,
        "max_epochs": 200,
        "patience": 20,
        "min_delta": 1e-5,
        "learning_rate": 5e-4,
        "lambda_fk": 0.1,
        "n_components": 8,
        "hidden_size": 256,
        "train_fraction": 0.8,
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "dataset_seed": 42,
        "split_seed": 0,
        "unseen_seed": 999,
    }

    experiment_dirs = create_experiment_dirs()

    logger = setup_logger(
        experiment_dirs["logs"] / "training.log"
    )

    logger.info(f"Experiment directory: {experiment_dirs['root']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    params = RobotParams3R3D()

    config["robot_params"] = params.to_dict()

    save_json(
        config,
        experiment_dirs["configs"] / "config.json",
    )

    logger.info("Saved config.")

    features, targets = generate_dataset(
        n_samples=config["n_samples"],
        params=params,
        seed=config["dataset_seed"],
    )

    save_npz(
        {
            "features": features,
            "targets": targets,
        },
        experiment_dirs["dataset"] / "raw_dataset.npz",
    )

    logger.info("Saved raw dataset.")

    processed = split_and_normalize(
        features=features,
        targets=targets,
        train_fraction=config["train_fraction"],
        val_fraction=config["val_fraction"],
        test_fraction=config["test_fraction"],
        seed=config["split_seed"],
    )

    save_npz(
        processed,
        experiment_dirs["processed_data"] / "processed_dataset.npz",
    )

    logger.info("Saved processed dataset.")

    train_loader = make_loader(
        processed,
        processed["train_indices"],
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = make_loader(
        processed,
        processed["val_indices"],
        batch_size=config["batch_size"],
        shuffle=False,
    )

    test_loader = make_loader(
        processed,
        processed["test_indices"],
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = MDN3R3D(
        input_dim=params.input_dim,
        output_dim=params.n_dof,
        hidden_size=config["hidden_size"],
        n_components=config["n_components"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    train_result = train_mdn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        params=params,
        theta_mean=processed["theta_mean"],
        theta_std=processed["theta_std"],
        feature_mean=processed["feature_mean"],
        feature_std=processed["feature_std"],
        config=config,
        experiment_dirs=experiment_dirs,
        logger=logger,
    )

    history = train_result["history"]

    save_train_history(
        history,
        experiment_dirs["metrics"],
    )

    logger.info("Saved training history JSON and CSV.")

    best_model_path = experiment_dirs["models"] / "best_model.pt"

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
        )
    )

    model.to(device)

    logger.info("Loaded best model for final evaluation.")

    test_metrics, test_predictions = evaluate_best_of_k(
        model=model,
        loader=test_loader,
        device=device,
        params=params,
        theta_mean=processed["theta_mean"],
        theta_std=processed["theta_std"],
    )

    final_metrics = {
        **test_metrics,
        "best_val_loss": train_result["best_val_loss"],
        "best_epoch": train_result["best_epoch"],
        "total_epochs_trained": train_result["total_epochs_trained"],
        "early_stopped": train_result["early_stopped"],
        "timestamp": datetime.now().isoformat(),
    }

    save_final_metrics(
        final_metrics,
        experiment_dirs["metrics"],
    )

    logger.info(f"Test best-of-K angle MSE: {test_metrics['best_of_k_angle_mse']:.6f}")
    logger.info(f"Test best-of-K pose MSE: {test_metrics['best_of_k_pose_mse']:.6f}")

    save_npz(
        test_predictions,
        experiment_dirs["predictions"] / "test_predictions.npz",
    )

    save_prediction_tables(
        test_predictions,
        experiment_dirs["predictions"],
        config["n_components"],
        prefix="test",
    )

    logger.info("Saved test predictions and CSV tables.")

    save_training_plots(
        history,
        experiment_dirs["plots"],
    )

    save_dataset_plots(
        processed["features_original"],
        experiment_dirs["plots"],
    )

    save_prediction_plots(
        test_predictions,
        experiment_dirs["plots"],
        config["n_components"],
        prefix="test",
    )

    logger.info("Saved thesis verification plots.")

    unseen_metrics = run_unseen_experiment(
        model=model,
        params=params,
        processed=processed,
        config=config,
        experiment_dirs=experiment_dirs,
        device=device,
        logger=logger,
    )

    logger.info(f"Unseen best-of-K angle MSE: {unseen_metrics['best_of_k_angle_mse']:.6f}")
    logger.info(f"Unseen best-of-K pose MSE: {unseen_metrics['best_of_k_pose_mse']:.6f}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()