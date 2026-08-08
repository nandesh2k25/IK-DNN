import React, { useState } from 'react';
import { ImageIcon, Maximize2, X } from 'lucide-react';
import { getPlotUrl } from '../utils/api';

const PlotsViewer = ({ selectedRun }) => {
  const [selectedPlot, setSelectedPlot] = useState(null);

  if (!selectedRun) return (
    <div className="flex flex-col items-center justify-center h-64 text-slate-400">
      <ImageIcon className="h-12 w-12 mb-4 opacity-20" />
      <p>Select an experiment to view plots</p>
    </div>
  );

  const plots = [
    { id: 'training_loss_curve.png', title: 'Training Loss' },
    { id: 'nll_fk_loss_curve.png', title: 'NLL vs FK Loss' },
    { id: 'workspace_xy_distribution.png', title: 'Workspace (XY)' },
    { id: 'workspace_xz_distribution.png', title: 'Workspace (XZ)' },
    { id: 'test_fk_error_histogram.png', title: 'FK Error Histogram' },
    { id: 'test_angle_error_histogram.png', title: 'Angle Error Histogram' },
    { id: 'test_component_usage.png', title: 'Component Usage' },
    { id: 'test_workspace_error_xy.png', title: 'Workspace Error (XY)' },
  ];

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {plots.map((plot) => (
          <div 
            key={plot.id} 
            className="group bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm hover:shadow-md transition-all cursor-pointer"
            onClick={() => setSelectedPlot(plot)}
          >
            <div className="p-3 border-b border-slate-100 flex justify-between items-center">
              <span className="text-xs font-bold text-slate-700 uppercase tracking-tight">{plot.title}</span>
              <Maximize2 className="h-3 w-3 text-slate-400 group-hover:text-blue-600 transition-colors" />
            </div>
            <div className="aspect-[4/3] bg-slate-50 relative overflow-hidden">
              <img 
                src={getPlotUrl(selectedRun, plot.id)} 
                alt={plot.title}
                className="w-full h-full object-contain p-2"
                onError={(e) => {
                  e.target.src = "https://via.placeholder.com/400x300?text=Plot+Not+Found";
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {selectedPlot && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="relative bg-white rounded-2xl w-full max-w-5xl max-h-[90vh] overflow-hidden shadow-2xl flex flex-col animate-in zoom-in-95 duration-200">
            <div className="p-4 border-b border-slate-200 flex justify-between items-center bg-slate-50">
              <h3 className="text-lg font-bold text-slate-900">{selectedPlot.title}</h3>
              <button 
                onClick={() => setSelectedPlot(null)}
                className="p-2 hover:bg-slate-200 rounded-full transition-colors"
              >
                <X className="h-6 w-6 text-slate-500" />
              </button>
            </div>
            <div className="flex-1 overflow-auto bg-white p-6 flex items-center justify-center">
              <img 
                src={getPlotUrl(selectedRun, selectedPlot.id)} 
                alt={selectedPlot.title}
                className="max-w-full max-h-full object-contain shadow-sm border border-slate-100"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PlotsViewer;
