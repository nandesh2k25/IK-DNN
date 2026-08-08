import React, { useState } from 'react';
import { Activity, Play, Square, Loader2, AlertCircle } from 'lucide-react';
import { trainModel } from '../utils/api';

const TrainingPanel = ({ trainStatus }) => {
  const [params, setParams] = useState({
    n_samples: 15000,
    batch_size: 256,
    max_epochs: 200,
    patience: 20,
    learning_rate: 0.0005,
    lambda_fk: 0.1,
    n_components: 8,
    hidden_size: 256,
    dataset_seed: 42
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    try {
      await trainModel(params);
    } catch (err) {
      setError(err.response?.data?.message || "Failed to start training");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setParams(prev => ({
      ...prev,
      [name]: name === 'learning_rate' || name === 'lambda_fk' ? parseFloat(value) : parseInt(value)
    }));
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h3 className="text-lg font-semibold text-slate-900 mb-6 flex items-center">
          <Activity className="h-5 w-5 mr-2 text-indigo-600" />
          Train New MDN Model
        </h3>

        {trainStatus.is_training ? (
          <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-8 flex flex-col items-center justify-center text-center">
            <Loader2 className="h-12 w-12 text-indigo-600 animate-spin mb-4" />
            <h4 className="text-lg font-bold text-indigo-900 mb-2">Training in Progress</h4>
            <p className="text-indigo-700 max-w-md">
              The model is currently training. You can check the logs in the "Logs" tab for real-time progress.
            </p>
            <div className="mt-6 p-4 bg-white rounded-lg shadow-sm border border-indigo-200 w-full max-w-sm">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-500">Run Name</span>
                <span className="font-mono font-medium text-indigo-700">{trainStatus.current_run}</span>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <InputGroup label="Samples" name="n_samples" value={params.n_samples} onChange={handleChange} />
              <InputGroup label="Batch Size" name="batch_size" value={params.batch_size} onChange={handleChange} />
              <InputGroup label="Max Epochs" name="max_epochs" value={params.max_epochs} onChange={handleChange} />
              <InputGroup label="Patience" name="patience" value={params.patience} onChange={handleChange} />
              <InputGroup label="Learning Rate" name="learning_rate" value={params.learning_rate} onChange={handleChange} step="0.0001" />
              <InputGroup label="Lambda FK" name="lambda_fk" value={params.lambda_fk} onChange={handleChange} step="0.01" />
              <InputGroup label="Components" name="n_components" value={params.n_components} onChange={handleChange} />
              <InputGroup label="Hidden Size" name="hidden_size" value={params.hidden_size} onChange={handleChange} />
              <InputGroup label="Dataset Seed" name="dataset_seed" value={params.dataset_seed} onChange={handleChange} />
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-100 rounded-lg flex items-center gap-3 text-red-700 text-sm">
                <AlertCircle className="h-5 w-5 shrink-0" />
                {error}
              </div>
            )}

            <button
              onClick={handleTrain}
              disabled={loading}
              className="flex items-center justify-center px-8 py-3 bg-indigo-600 text-white text-sm font-bold rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-all shadow-lg shadow-indigo-500/20"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2 fill-current" />
                  Start Training
                </>
              )}
            </button>
          </>
        )}
      </div>
      
      <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
        <h4 className="text-sm font-bold text-slate-900 mb-4 uppercase tracking-wider">Training Tips</h4>
        <ul className="space-y-3 text-sm text-slate-600">
          <li className="flex gap-2">
            <span className="text-indigo-600 font-bold">•</span>
            <span><b>Lambda FK:</b> Higher values force the model to prioritize physical pose accuracy over joint distribution.</span>
          </li>
          <li className="flex gap-2">
            <span className="text-indigo-600 font-bold">•</span>
            <span><b>Components:</b> For a 3R robot, 8-16 components are usually sufficient to capture multiple solutions.</span>
          </li>
          <li className="flex gap-2">
            <span className="text-indigo-600 font-bold">•</span>
            <span><b>Patience:</b> Higher patience helps avoid early stopping in noisy loss landscapes.</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

const InputGroup = ({ label, name, value, onChange, step = "1" }) => (
  <div>
    <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5">
      {label}
    </label>
    <input
      type="number"
      name={name}
      value={value}
      onChange={onChange}
      step={step}
      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all"
    />
  </div>
);

export default TrainingPanel;
