import React, { useState } from 'react';
import { Target, Play, Loader2, CheckCircle2, AlertCircle, BarChart2 } from 'lucide-react';
import { getExperimentMetrics } from '../utils/api';

const EvaluationPanel = ({ selectedRun }) => {
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  const handleEvaluate = async () => {
    if (!selectedRun) return;
    setLoading(true);
    setError(null);
    try {
      // In this system, evaluation usually happens at the end of training
      // and results are saved in metrics. We can just refetch them.
      const res = await getExperimentMetrics(selectedRun);
      setMetrics(res.data.final);
      if (!res.data.final) {
        setError("Final metrics not found. Has training completed?");
      }
    } catch (err) {
      setError("Failed to fetch evaluation metrics");
    } finally {
      setLoading(false);
    }
  };

  if (!selectedRun) return (
    <div className="flex flex-col items-center justify-center h-64 text-slate-400">
      <BarChart2 className="h-12 w-12 mb-4 opacity-20" />
      <p>Select an experiment to view evaluation results</p>
    </div>
  );

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <div className="flex items-center justify-between mb-8">
          <h3 className="text-lg font-semibold text-slate-900 flex items-center">
            <BarChart2 className="h-5 w-5 mr-2 text-blue-600" />
            Model Evaluation Results
          </h3>
          <button
            onClick={handleEvaluate}
            disabled={loading}
            className="flex items-center px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold rounded-lg transition-all"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            Refresh Metrics
          </button>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-amber-50 border border-amber-100 rounded-lg flex items-center gap-3 text-amber-700 text-sm">
            <AlertCircle className="h-5 w-5 shrink-0" />
            {error}
          </div>
        )}

        {metrics ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <MetricCard label="Angle MSE" value={metrics.best_of_k_angle_mse} sub="Best-of-K" />
            <MetricCard label="Angle RMSE" value={metrics.best_of_k_angle_rmse} sub="Best-of-K" />
            <MetricCard label="Pose MSE" value={metrics.best_of_k_pose_mse} sub="Best-of-K" />
            <MetricCard label="Pose RMSE" value={metrics.best_of_k_pose_rmse} sub="Best-of-K" />
            <MetricCard label="Theta 1 MAE" value={metrics.theta1_mae} sub="Mean Absolute Error" />
            <MetricCard label="Theta 2 MAE" value={metrics.theta2_mae} sub="Mean Absolute Error" />
            <MetricCard label="Theta 3 MAE" value={metrics.theta3_mae} sub="Mean Absolute Error" />
            <MetricCard label="Epochs" value={metrics.total_epochs_trained} sub="Total Trained" />
            <MetricCard label="Status" value={metrics.early_stopped ? "Early Stopped" : "Completed"} sub="Termination" />
          </div>
        ) : (
          <div className="text-center py-12 text-slate-400">
            <p>Click "Refresh Metrics" to load evaluation results for this run.</p>
          </div>
        )}
      </div>

      <div className="bg-blue-50 border border-blue-100 rounded-xl p-6">
        <h4 className="text-sm font-bold text-blue-900 mb-2 flex items-center">
          <Info className="h-4 w-4 mr-2" />
          Best-of-K Evaluation Strategy
        </h4>
        <p className="text-sm text-blue-800 leading-relaxed">
          For each target pose in the test set, the MDN predicts multiple candidates. The system selects the candidate that minimizes the Forward Kinematics (FK) error. These metrics reflect the performance of that selected candidate.
        </p>
      </div>
    </div>
  );
};

const MetricCard = ({ label, value, sub }) => (
  <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
    <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold mb-1">{label}</p>
    <p className="text-lg font-mono font-bold text-slate-900">
      {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(6)) : value}
    </p>
    <p className="text-[10px] text-slate-500 mt-1">{sub}</p>
  </div>
);

const Info = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
);

export default EvaluationPanel;
