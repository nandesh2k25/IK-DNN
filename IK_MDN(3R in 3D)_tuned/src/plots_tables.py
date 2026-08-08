import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_experiment_dirs(base_dir="outputs/experiments", run_name=None):
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    root = Path(base_dir) / run_name

    dirs = {
        "root": root,
        "models": root / "models",
        "dataset": root / "dataset",
        "processed_data": root / "processed_data",
        "metrics": root / "metrics",
        "plots": root / "plots",
        "configs": root / "configs",
        "logs": root / "logs",
        "predictions": root / "predictions",
        "unseen_root": root / "unseen_data",
        "unseen_dataset": root / "unseen_data" / "dataset",
        "unseen_predictions": root / "unseen_data" / "predictions",
        "unseen_metrics": root / "unseen_data" / "metrics",
        "unseen_tables": root / "unseen_data" / "tables",
        "unseen_plots": root / "unseen_data" / "plots",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def setup_logger(log_path: Path):
    logger = logging.getLogger("ik_mdn_3d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_json(data, path):
    path = Path(path)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=convert)


def save_npz(data_dict, path):
    np.savez_compressed(path, **data_dict)


def save_train_history(history, metrics_dir):
    df = pd.DataFrame(history)

    json_path = metrics_dir / "train_history.json"
    csv_path = metrics_dir / "train_history.csv"

    df.to_json(json_path, orient="records", indent=4)
    df.to_csv(csv_path, index=False)


def save_final_metrics(metrics, metrics_dir):
    json_path = metrics_dir / "final_metrics.json"
    csv_path = metrics_dir / "final_metrics.csv"

    save_json(metrics, json_path)

    pd.DataFrame([metrics]).to_csv(csv_path, index=False)


def prediction_summary_dataframe(predictions):
    x = predictions["x_original"]
    y_true = predictions["y_true_real"]
    y_pred = predictions["predicted_best_fk_real"]
    pi = predictions["mixture_probabilities"]
    best_component = predictions["best_component_indices"]

    rows = []

    for i in range(len(x)):
        selected = int(best_component[i])

        rows.append(
            {
                "sample_id": i,
                "x": x[i, 0],
                "y": x[i, 1],
                "z": x[i, 2],
                "theta1_true": y_true[i, 0],
                "theta2_true": y_true[i, 1],
                "theta3_true": y_true[i, 2],
                "theta1_pred": y_pred[i, 0],
                "theta2_pred": y_pred[i, 1],
                "theta3_pred": y_pred[i, 2],
                "best_component": selected,
                "best_fk_pose_mse": predictions["best_fk_pose_mse"][i],
                "best_angle_mse": predictions["best_angle_mse"][i],
                "max_mixture_probability": np.max(pi[i]),
                "selected_component_probability": pi[i, selected],
            }
        )

    return pd.DataFrame(rows)


def component_usage_dataframe(predictions, n_components):
    best_component = predictions["best_component_indices"]

    counts = np.bincount(best_component, minlength=n_components)

    rows = []

    total = len(best_component)

    for component_id, count in enumerate(counts):
        rows.append(
            {
                "component_id": component_id,
                "selected_count": int(count),
                "selected_percentage": float(100.0 * count / total),
            }
        )

    return pd.DataFrame(rows)


def error_summary_dataframe(predictions):
    y_true = predictions["y_true_real"]
    y_pred = predictions["predicted_best_fk_real"]

    values = {
        "best_fk_pose_mse": predictions["best_fk_pose_mse"],
        "best_angle_mse": predictions["best_angle_mse"],
        "theta1_abs_error": np.abs(y_pred[:, 0] - y_true[:, 0]),
        "theta2_abs_error": np.abs(y_pred[:, 1] - y_true[:, 1]),
        "theta3_abs_error": np.abs(y_pred[:, 2] - y_true[:, 2]),
    }

    rows = []

    for name, arr in values.items():
        rows.append(
            {
                "metric_name": name,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "median": float(np.median(arr)),
                "max": float(np.max(arr)),
            }
        )

    return pd.DataFrame(rows)


def save_prediction_tables(predictions, output_dir, n_components, prefix="test"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = prediction_summary_dataframe(predictions)
    component_usage = component_usage_dataframe(predictions, n_components)
    error_summary = error_summary_dataframe(predictions)

    summary.to_csv(output_dir / f"{prefix}_prediction_summary.csv", index=False)
    component_usage.to_csv(output_dir / f"{prefix}_component_usage.csv", index=False)
    error_summary.to_csv(output_dir / f"{prefix}_error_summary.csv", index=False)


def save_training_plots(history, plots_dir):
    df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")

    best_idx = df["val_loss"].idxmin()
    best_epoch = df.loc[best_idx, "epoch"]
    best_val = df.loc[best_idx, "val_loss"]

    plt.scatter([best_epoch], [best_val], marker="o", label="best_val_epoch")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "training_loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_nll_loss"], label="train_nll")
    plt.plot(df["epoch"], df["val_nll_loss"], label="val_nll")
    plt.plot(df["epoch"], df["train_fk_loss"], label="train_fk")
    plt.plot(df["epoch"], df["val_fk_loss"], label="val_fk")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NLL and FK Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "nll_fk_loss_curve.png", dpi=150)
    plt.close()


def save_dataset_plots(features_original, plots_dir):
    x = features_original[:, 0]
    y = features_original[:, 1]
    z = features_original[:, 2]

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=2, alpha=0.35)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Workspace Distribution: x-y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plots_dir / "workspace_xy_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(x, z, s=2, alpha=0.35)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Workspace Distribution: x-z")
    plt.tight_layout()
    plt.savefig(plots_dir / "workspace_xz_distribution.png", dpi=150)
    plt.close()


def save_prediction_plots(predictions, plots_dir, n_components, prefix="test"):
    y_true = predictions["y_true_real"]
    y_pred = predictions["predicted_best_fk_real"]
    best_fk = predictions["best_fk_pose_mse"]
    best_angle = predictions["best_angle_mse"]
    best_component = predictions["best_component_indices"]
    x_original = predictions["x_original"]

    plt.figure(figsize=(7, 5))
    plt.hist(best_fk, bins=50)
    plt.xlabel("Best-FK pose MSE")
    plt.ylabel("Count")
    plt.title(f"{prefix}: FK Pose Error Histogram")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{prefix}_fk_error_histogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(best_angle, bins=50)
    plt.xlabel("Best-of-K angle MSE")
    plt.ylabel("Count")
    plt.title(f"{prefix}: Angle Error Histogram")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{prefix}_angle_error_histogram.png", dpi=150)
    plt.close()

    sample_count = min(1000, len(y_true))
    idx = np.arange(sample_count)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    names = ["theta1", "theta2", "theta3"]

    for j, ax in enumerate(axes):
        ax.plot(idx, y_true[:sample_count, j], label="true", linewidth=1)
        ax.plot(idx, y_pred[:sample_count, j], label="pred", linewidth=1)
        ax.set_ylabel(names[j])
        ax.legend()

    axes[-1].set_xlabel("Sample index")
    fig.suptitle(f"{prefix}: Predicted vs True Joint Angles")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_predicted_vs_true_angles.png", dpi=150)
    plt.close(fig)

    counts = np.bincount(best_component, minlength=n_components)

    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(n_components), counts)
    plt.xlabel("MDN component")
    plt.ylabel("Selected count")
    plt.title(f"{prefix}: Mixture Component Usage")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{prefix}_component_usage.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        x_original[:, 0],
        x_original[:, 1],
        c=best_fk,
        s=5,
        alpha=0.7,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{prefix}: Workspace Colored by FK Error")
    plt.colorbar(label="FK pose MSE")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{prefix}_workspace_error_xy.png", dpi=150)
    plt.close()