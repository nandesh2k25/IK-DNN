import React from 'react';
import { Search, ChevronRight } from 'lucide-react';

const ExperimentSelector = ({ experiments, selectedRun, onSelect }) => {
  return (
    <div className="space-y-2">
      <div className="relative mb-4">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
        <input
          type="text"
          placeholder="Search runs..."
          className="w-full bg-slate-50 border border-slate-200 rounded-md py-1.5 pl-9 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
        />
      </div>
      
      <div className="space-y-1 max-h-64 overflow-y-auto pr-1 custom-scrollbar">
        {experiments.length === 0 && (
          <p className="text-xs text-slate-400 text-center py-4 italic">No runs found</p>
        )}
        {experiments.map((run) => (
          <button
            key={run.name}
            onClick={() => onSelect(run.name)}
            className={`w-full flex items-center justify-between px-3 py-2 text-xs font-medium rounded-md transition-all ${
              selectedRun === run.name
                ? 'bg-white border border-blue-200 shadow-sm text-blue-700 ring-1 ring-blue-500/10'
                : 'text-slate-600 hover:bg-slate-50 border border-transparent'
            }`}
          >
            <span className="truncate mr-2">{run.name}</span>
            {selectedRun === run.name && <ChevronRight className="h-3 w-3 shrink-0" />}
          </button>
        ))}
      </div>
    </div>
  );
};

export default ExperimentSelector;
