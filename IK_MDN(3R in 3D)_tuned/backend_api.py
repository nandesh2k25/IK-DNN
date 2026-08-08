import os
import sys
import json
import logging
import threading
from pathlib import Path

# Add src to sys.path so that internal imports in src files work correctly
sys.path.append(str(Path(__file__).parent / "src"))

from typing import Optional, List, Dict, Any

import numpy as np
import torch
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.robot_params import RobotParams3R3D
from src.generate_dataset import generate_dataset, split_and_normalize
from src.train import IKDataset3D, train_mdn
from src.evaluate import evaluate_best_of_k
from src.ik_mdn import MDN3R3D
from src.plots_tables import (
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
from torch.utils.data import DataLoader

app = FastAPI(title="MDN-Based Inverse Kinematics System API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
OUTPUTS_DIR = Path("outputs/experiments")
SRC_OUTPUTS_DIR = Path("src/outputs/experiments")

# Global state for training
training_state = {
    "is_training": False,
    "current_run": None,
    "stop_requested": False
}

class DatasetRequest(BaseModel):
    n_samples: int = 15000
    seed: int = 42

class TrainRequest(BaseModel):
    n_samples: int = 15000
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    learning_rate: float = 5e-4
    lambda_fk: float = 0.1
    n_components: int = 8
    hidden_size: int = 256
    dataset_seed: int = 42

class PredictRequest(BaseModel):
    run_name: str
    x: float
    y: float
    z: float

def get_all_runs():
    runs = []
    # Check both potential output directories
    for d in [OUTPUTS_DIR, SRC_OUTPUTS_DIR]:
        if d.exists():
            for run_dir in d.iterdir():
                if run_dir.is_dir():
                    runs.append(run_dir)
    # Return unique run names, sorted by name (which is timestamp-based)
    run_dict = {r.name: r for r in runs}
    return sorted(run_dict.values(), key=lambda x: x.name, reverse=True)

@app.get("/experiments")
async def list_experiments():
    runs = get_all_runs()
    return [{"name": r.name, "path": str(r)} for r in runs]

@app.get("/experiments/{run_name}/config")
async def get_config(run_name: str):
    runs = get_all_runs()
    run_dir = next((r for r in runs if r.name == run_name), None)
    if not run_dir:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    config_path = run_dir / "configs" / "config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    
    with open(config_path, "r") as f:
        return json.load(f)

@app.get("/experiments/{run_name}/metrics")
async def get_metrics(run_name: str):
    runs = get_all_runs()
    run_dir = next((r for r in runs if r.name == run_name), None)
    if not run_dir:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    metrics_dir = run_dir / "metrics"
    final_metrics_path = metrics_dir / "final_metrics.json"
    history_path = metrics_dir / "train_history.json"
    
    result = {}
    if final_metrics_path.exists():
        with open(final_metrics_path, "r") as f:
            result["final"] = json.load(f)
    
    if history_path.exists():
        with open(history_path, "r") as f:
            result["history"] = json.load(f)
            
    return result

@app.get("/experiments/{run_name}/logs")
async def get_logs(run_name: str):
    runs = get_all_runs()
    run_dir = next((r for r in runs if r.name == run_name), None)
    if not run_dir:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    log_path = run_dir / "logs" / "training.log"
    if not log_path.exists():
        return {"logs": ""}
    
    with open(log_path, "r") as f:
        return {"logs": f.read()}

@app.post("/dataset/generate")
async def generate_dataset_endpoint(req: DatasetRequest):
    params = RobotParams3R3D()
    features, targets = generate_dataset(
        n_samples=req.n_samples,
        params=params,
        seed=req.seed,
    )
    
    # Return some statistics
    return {
        "n_samples": req.n_samples,
        "feature_shape": list(features.shape),
        "target_shape": list(targets.shape),
        "x_range": [float(features[:, 0].min()), float(features[:, 0].max())],
        "y_range": [float(features[:, 1].min()), float(features[:, 1].max())],
        "z_range": [float(features[:, 2].min()), float(features[:, 2].max())],
    }

def run_training_task(req: TrainRequest):
    global training_state
    training_state["is_training"] = True
    training_state["stop_requested"] = False
    
    try:
        torch.manual_seed(0)
        np.random.seed(0)

        config = {
            "project_name": "ik_mdn_3r_3d",
            "n_samples": req.n_samples,
            "batch_size": req.batch_size,
            "max_epochs": req.max_epochs,
            "patience": req.patience,
            "min_delta": 1e-5,
            "learning_rate": req.learning_rate,
            "lambda_fk": req.lambda_fk,
            "n_components": req.n_components,
            "hidden_size": req.hidden_size,
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "dataset_seed": req.dataset_seed,
            "split_seed": 0,
            "unseen_seed": 999,
        }

        experiment_dirs = create_experiment_dirs()
        training_state["current_run"] = experiment_dirs["root"].name
        
        logger = setup_logger(experiment_dirs["logs"] / "training.log")
        logger.info(f"Starting training from API. Experiment: {experiment_dirs['root'].name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = RobotParams3R3D()
        config["robot_params"] = params.to_dict()

        save_json(config, experiment_dirs["configs"] / "config.json")

        features, targets = generate_dataset(
            n_samples=config["n_samples"],
            params=params,
            seed=config["dataset_seed"],
        )

        save_npz({"features": features, "targets": targets}, experiment_dirs["dataset"] / "raw_dataset.npz")

        processed = split_and_normalize(
            features=features,
            targets=targets,
            train_fraction=config["train_fraction"],
            val_fraction=config["val_fraction"],
            test_fraction=config["test_fraction"],
            seed=config["split_seed"],
        )

        save_npz(processed, experiment_dirs["processed_data"] / "processed_dataset.npz")

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

        train_loader = make_loader(processed, processed["train_indices"], config["batch_size"], True)
        val_loader = make_loader(processed, processed["val_indices"], config["batch_size"], False)

        model = MDN3R3D(
            input_dim=params.input_dim,
            output_dim=params.n_dof,
            hidden_size=config["hidden_size"],
            n_components=config["n_components"],
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        # Note: train_mdn doesn't support stopping internally without modification.
        # For now, we'll just run it. If stopping is critical, we'd need to modify train_mdn.
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

        save_train_history(train_result["history"], experiment_dirs["metrics"])
        logger.info("Training task complete.")

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Training failed: {str(e)}")
        print(f"Training error: {e}")
    finally:
        training_state["is_training"] = False

@app.post("/train")
async def train_endpoint(req: TrainRequest, background_tasks: BackgroundTasks):
    if training_state["is_training"]:
        return {"status": "error", "message": "Training already in progress"}
    
    background_tasks.add_task(run_training_task, req)
    return {"status": "success", "message": "Training started"}

@app.get("/train/status")
async def train_status():
    return training_state

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    runs = get_all_runs()
    run_dir = next((r for r in runs if r.name == req.run_name), None)
    if not run_dir:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Load config and normalization stats
    config_path = run_dir / "configs" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    processed_path = run_dir / "processed_data" / "processed_dataset.npz"
    if not processed_path.exists():
        raise HTTPException(status_code=404, detail="Processed dataset not found for this run")
    
    processed = np.load(processed_path)
    feature_mean = processed["feature_mean"]
    feature_std = processed["feature_std"]
    theta_mean = processed["theta_mean"]
    theta_std = processed["theta_std"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = RobotParams3R3D()
    
    model = MDN3R3D(
        input_dim=params.input_dim,
        output_dim=params.n_dof,
        hidden_size=config["hidden_size"],
        n_components=config["n_components"],
    )
    
    model_path = run_dir / "models" / "best_model.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare input
    x_orig = np.array([req.x, req.y, req.z], dtype=np.float32)
    x_norm = (x_orig - feature_mean) / feature_std
    
    x_norm_t = torch.tensor(x_norm, device=device).unsqueeze(0)
    x_orig_t = torch.tensor(x_orig, device=device).unsqueeze(0)
    theta_mean_t = torch.tensor(theta_mean, device=device).view(1, 1, -1)
    theta_std_t = torch.tensor(theta_std, device=device).view(1, 1, -1)
    
    best_theta, info = model.predict_best_fk(
        x_norm=x_norm_t,
        x_original=x_orig_t,
        theta_mean=theta_mean_t,
        theta_std=theta_std_t,
        params=params
    )
    
    # Extract results
    pi = info["pi"][0].cpu().numpy().tolist()
    mu_real = info["mu_real"][0].cpu().numpy()
    fk_candidates = info["fk_candidates"][0].cpu().numpy()
    fk_error = info["fk_error"][0].cpu().numpy().tolist()
    best_idx = int(info["best_component"][0].cpu().item())
    
    all_components = []
    for i in range(config["n_components"]):
        comp = {
            "component_id": i,
            "mixture_probability": pi[i],
            "theta1_deg": float(np.degrees(mu_real[i, 0])),
            "theta2_deg": float(np.degrees(mu_real[i, 1])),
            "theta3_deg": float(np.degrees(mu_real[i, 2])),
            "theta1_rad": float(mu_real[i, 0]),
            "theta2_rad": float(mu_real[i, 1]),
            "theta3_rad": float(mu_real[i, 2]),
            "fk_x": float(fk_candidates[i, 0]),
            "fk_y": float(fk_candidates[i, 1]),
            "fk_z": float(fk_candidates[i, 2]),
            "fk_error": float(fk_error[i]),
            "selected_best": i == best_idx
        }
        all_components.append(comp)
        
    return {
        "target_pose": [req.x, req.y, req.z],
        "all_components": all_components,
        "best_component": all_components[best_idx],
        "predicted_best_fk_pose": [
            float(fk_candidates[best_idx, 0]),
            float(fk_candidates[best_idx, 1]),
            float(fk_candidates[best_idx, 2])
        ],
        "error": float(fk_error[best_idx])
    }

# Serve plots and datasets statically if needed
# Note: In a real app, we might want to serve these files
@app.get("/experiments/{run_name}/plots/{plot_name}")
async def get_plot(run_name: str, plot_name: str):
    runs = get_all_runs()
    run_dir = next((r for r in runs if r.name == run_name), None)
    if not run_dir:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Try multiple paths for plots
    plot_paths = [
        run_dir / "plots" / plot_name,
        run_dir / "unseen_data" / "plots" / plot_name
    ]
    
    for p in plot_paths:
        if p.exists():
            from fastapi.responses import FileResponse
            return FileResponse(p)
            
    raise HTTPException(status_code=404, detail="Plot not found")

@app.get("/experiments/{run_name}/tables/{table_name}")
async def get_table(run_name: str, table_name: str):
    runs = get_all_runs()
    run_dir = next((r for r in runs if r.name == run_name), None)
    if not run_dir:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    table_paths = [
        run_dir / "metrics" / table_name,
        run_dir / "predictions" / table_name,
        run_dir / "unseen_data" / "tables" / table_name,
        run_dir / "unseen_data" / "metrics" / table_name
    ]
    
    for p in table_paths:
        if p.exists():
            df = pd.read_csv(p)
            return df.to_dict(orient="records")
            
    raise HTTPException(status_code=404, detail="Table not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
