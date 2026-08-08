import React, { useState } from 'react';
import { Database, Save, Loader2, CheckCircle2 } from 'lucide-react';
import { generateDataset as apiGenerateDataset } from '../utils/api';

const DatasetPanel = ({ config }) => {
  const [nSamples, setNSamples] = useState(config?.n_samples || 15000);
  const [seed, setSeed] = useState(config?.dataset_seed || 42);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await apiGenerateDataset({ n_samples: nSamples, seed });
      setResult(res.data);
    } catch (err) {
      console.error("Failed to generate dataset", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h3 className="text-lg font-semibold text-slate-900 mb-6 flex items-center">
          <Database className="h-5 w-5 mr-2 text-blue-600" />
          Generate New Dataset
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">
              Number of Samples
            </label>
            <input
              type="number"
              value={nSamples}
              onChange={(e) => setNSamples(parseInt(e.target.value))}
              className="w-full bg-white border border-slate-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
            />
            <p className="mt-1.5 text-xs text-slate-500">
              Total number of (x,y,z) to theta1,2,3 pairs.
            </p>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">
              Random Seed
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value))}
              className="w-full bg-white border border-slate-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
            />
            <p className="mt-1.5 text-xs text-slate-500">
              For reproducibility of dataset generation.
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="flex items-center justify-center px-6 py-2.5 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md shadow-blue-500/20"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Database className="h-4 w-4 mr-2" />
                Generate Dataset
              </>
            )}
          </button>
        </div>
      </div>
      
      {result && (
        <div className="bg-white p-6 rounded-xl border border-green-200 shadow-sm animate-in zoom-in-95 duration-300">
          <div className="flex items-center gap-2 text-green-700 mb-4">
            <CheckCircle2 className="h-5 w-5" />
            <h4 className="font-semibold">Dataset Generated Successfully</h4>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <ResultItem label="Samples" value={result.n_samples} />
            <ResultItem label="X Range" value={`${result.x_range[0].toFixed(2)} to ${result.x_range[1].toFixed(2)}`} />
            <ResultItem label="Y Range" value={`${result.y_range[0].toFixed(2)} to ${result.y_range[1].toFixed(2)}`} />
            <ResultItem label="Z Range" value={`${result.z_range[0].toFixed(2)} to ${result.z_range[1].toFixed(2)}`} />
          </div>
        </div>
      )}
    </div>
  );
};

const ResultItem = ({ label, value }) => (
  <div>
    <p className="text-xs text-slate-400 mb-0.5 uppercase tracking-wider">{label}</p>
    <p className="text-sm font-bold text-slate-900">{value}</p>
  </div>
);

export default DatasetPanel;
