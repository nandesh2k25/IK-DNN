# MDN-Based Inverse Kinematics System UI

A modern, full-stack interface for the 3R Robot Inverse Kinematics system using Mixture Density Networks (MDN).

## Features

- **Experiment Management:** Browse and select from previous training runs.
- **Dataset Generation:** Create new datasets directly from the UI.
- **Interactive Training:** Start model training with custom hyperparameters and monitor logs in real-time.
- **Multi-Solution Prediction:** Visualize all candidate solutions predicted by the MDN.
- **3D Visualization:** Dynamic 3D robot arm visualization with target vs. predicted end-effector comparison.
- **Analytics:** View training plots and metric tables within the app.

## Project Structure

- `backend_api.py`: FastAPI backend that wraps the existing ML logic in `src/`.
- `frontend/`: React + Vite + Tailwind CSS frontend application.

## Setup Instructions

### 1. Backend Dependencies

Ensure you have the core ML dependencies installed:
```bash
pip install numpy torch matplotlib pandas
```

Install FastAPI and Uvicorn for the API:
```bash
pip install fastapi uvicorn python-multipart
```

### 2. Frontend Dependencies

Navigate to the `frontend` directory and install npm packages:
```bash
cd frontend
npm install
```

## Running the Application

### 1. Start the Backend API

From the root directory:
```bash
python backend_api.py
```
The API will be available at `http://localhost:8000`.

### 2. Start the Frontend Development Server

From the `frontend` directory:
```bash
npm run dev
```
The UI will be available at `http://localhost:5173`.

## Usage Guide

1. **Dashboard:** Select an experiment from the sidebar to see its performance metrics and configuration.
2. **Dataset:** Use this panel to generate new training data.
3. **Train:** Configure hyperparameters and start a new training run. You can monitor the progress in the **Logs** tab.
4. **Predict:** Enter target `(x, y, z)` coordinates to see how the model predicts multiple joint configurations. The system will automatically select and visualize the one with the best Forward Kinematics consistency.
5. **Plots/Tables:** Review the visual and tabular outputs of your experiments.

## Technical Details

- **Model:** Mixture Density Network (MDN) predicting Gaussian mixture parameters.
- **Selection:** FK-based consistency selection among $K$ mixture components.
- **Frontend Stack:** React, Tailwind CSS, Lucide Icons, Plotly.js for 3D rendering.
- **Backend Stack:** FastAPI, PyTorch integration.
