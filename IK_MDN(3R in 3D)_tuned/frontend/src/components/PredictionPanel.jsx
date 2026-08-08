import React, { useState } from 'react';
import { Target, Play, Loader2, CheckCircle2, AlertCircle, Info } from 'lucide-react';
import { predictJoints } from '../utils/api';
import Robot3DView from './Robot3DView';

const PredictionPanel = ({ selectedRun, config }) => {
  const [pose, setPose] = useState({ x: 0.2, y: 0.1, z: 0.3 });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (!selectedRun) {
      setError("Please select an active experiment first");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await predictJoints({ run_name: selectedRun, ...pose });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setPose(prev => ({ ...prev, [e.target.name]: parseFloat(e.target.value) }));
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900 mb-6 flex items-center">
              <Target className="h-5 w-5 mr-2 text-blue-600" />
              Target End-Effector
            </h3>

            <div className="space-y-4 mb-6">
              <InputGroup label="X (m)" name="x" value={pose.x} onChange={handleChange} />
              <InputGroup label="Y (m)" name="y" value={pose.y} onChange={handleChange} />
              <InputGroup label="Z (m)" name="z" value={pose.z} onChange={handleChange} />
            </div>

            <button
              onClick={handlePredict}
              disabled={loading || !selectedRun}
              className="w-full flex items-center justify-center px-6 py-3 bg-blue-600 text-white text-sm font-bold rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-all shadow-lg shadow-blue-500/20"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Prediction
                </>
              )}
            </button>

            {!selectedRun && (
              <p className="mt-3 text-xs text-red-500 text-center flex items-center justify-center gap-1">
                <AlertCircle className="h-3 w-3" /> Select a run first
              </p>
            )}
          </div>

          {result && (
            <div className="bg-white p-6 rounded-xl border border-blue-100 shadow-sm animate-in zoom-in-95 duration-300">
              <h4 className="text-sm font-bold text-slate-900 mb-4 uppercase tracking-wider flex items-center justify-between">
                Selected Solution
                <span className="text-[10px] bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                  Component #{result.best_component.component_id}
                </span>
              </h4>
              
              <div className="space-y-3">
                <AngleDisplay label="Theta 1" deg={result.best_component.theta1_deg} rad={result.best_component.theta1_rad} />
                <AngleDisplay label="Theta 2" deg={result.best_component.theta2_deg} rad={result.best_component.theta2_rad} />
                <AngleDisplay label="Theta 3" deg={result.best_component.theta3_deg} rad={result.best_component.theta3_rad} />
                
                <div className="pt-3 border-t border-slate-100 mt-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">Pose Error (m)</span>
                    <span className="font-mono font-bold text-blue-600">{result.error.toFixed(8)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Probability</span>
                    <span className="font-mono font-bold text-slate-700">{(result.best_component.mixture_probability * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm h-[500px]">
            <Robot3DView 
              prediction={result} 
              target={pose}
              robotParams={config?.robot_params}
            />
          </div>

          {result && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-200 bg-slate-50">
                <h4 className="text-sm font-bold text-slate-900 flex items-center">
                  <Info className="h-4 w-4 mr-2 text-slate-400" />
                  All MDN Mixture Components
                </h4>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-white text-slate-500 uppercase tracking-wider">
                    <tr>
                      <th className="px-4 py-3 font-bold">ID</th>
                      <th className="px-4 py-3 font-bold">Prob</th>
                      <th className="px-4 py-3 font-bold">Theta 1,2,3 (deg)</th>
                      <th className="px-4 py-3 font-bold">FK Position (x,y,z)</th>
                      <th className="px-4 py-3 font-bold">FK Error</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {result.all_components.map((comp) => (
                      <tr 
                        key={comp.component_id}
                        className={`transition-colors ${comp.selected_best ? 'bg-green-50' : 'hover:bg-slate-50'}`}
                      >
                        <td className="px-4 py-3 font-mono">
                          {comp.component_id}
                          {comp.selected_best && <CheckCircle2 className="h-3 w-3 inline ml-1.5 text-green-600" />}
                        </td>
                        <td className="px-4 py-3">{(comp.mixture_probability * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 font-mono">
                          {comp.theta1_deg.toFixed(1)}, {comp.theta2_deg.toFixed(1)}, {comp.theta3_deg.toFixed(1)}
                        </td>
                        <td className="px-4 py-3 font-mono">
                          {comp.fk_x.toFixed(3)}, {comp.fk_y.toFixed(3)}, {comp.fk_z.toFixed(3)}
                        </td>
                        <td className={`px-4 py-3 font-mono font-bold ${
                          comp.fk_error < 0.001 ? 'text-green-600' : comp.fk_error < 0.01 ? 'text-amber-600' : 'text-slate-400'
                        }`}>
                          {comp.fk_error.toFixed(6)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const InputGroup = ({ label, name, value, onChange }) => (
  <div>
    <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5">
      {label}
    </label>
    <input
      type="number"
      name={name}
      value={value}
      onChange={onChange}
      step="0.01"
      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
    />
  </div>
);

const AngleDisplay = ({ label, deg, rad }) => (
  <div className="flex justify-between items-end">
    <div>
      <p className="text-[10px] text-slate-400 uppercase tracking-wider font-bold mb-0.5">{label}</p>
      <p className="text-sm font-bold text-slate-900">{deg.toFixed(2)}°</p>
    </div>
    <p className="text-[10px] text-slate-400 font-mono">{rad.toFixed(4)} rad</p>
  </div>
);

export default PredictionPanel;
