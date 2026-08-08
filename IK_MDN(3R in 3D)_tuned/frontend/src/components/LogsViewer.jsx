import React, { useState, useEffect, useRef } from 'react';
import { Terminal, RefreshCw, Download, ChevronDown, Filter } from 'lucide-react';
import { getExperimentLogs } from '../utils/api';

const LogsViewer = ({ selectedRun }) => {
  const [logs, setLogs] = useState('');
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (selectedRun) {
      fetchLogs();
    }
  }, [selectedRun]);

  useEffect(() => {
    let interval;
    if (autoRefresh && selectedRun) {
      interval = setInterval(fetchLogs, 5000);
    }
    return () => clearInterval(interval);
  }, [autoRefresh, selectedRun]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const fetchLogs = async () => {
    try {
      const res = await getExperimentLogs(selectedRun);
      setLogs(res.data.logs);
    } catch (err) {
      console.error("Failed to fetch logs", err);
    }
  };

  if (!selectedRun) return (
    <div className="flex flex-col items-center justify-center h-64 text-slate-400">
      <Terminal className="h-12 w-12 mb-4 opacity-20" />
      <p>Select an experiment to view logs</p>
    </div>
  );

  const logLines = logs.split('\n');

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="bg-slate-900 rounded-xl border border-slate-800 shadow-2xl overflow-hidden flex flex-col h-[700px]">
        <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex gap-1.5 px-2">
              <div className="w-3 h-3 rounded-full bg-red-500/80" />
              <div className="w-3 h-3 rounded-full bg-amber-500/80" />
              <div className="w-3 h-3 rounded-full bg-green-500/80" />
            </div>
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest ml-2 flex items-center">
              <Terminal className="h-4 w-4 mr-2" />
              Training Logs
            </h3>
          </div>
          
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer group">
              <input 
                type="checkbox" 
                checked={autoRefresh} 
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="w-3 h-3 rounded border-slate-700 bg-slate-800 text-blue-600 focus:ring-0 focus:ring-offset-0 transition-all"
              />
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter group-hover:text-slate-300 transition-colors">Auto-Refresh</span>
            </label>
            <div className="h-4 w-px bg-slate-800" />
            <button 
              onClick={fetchLogs}
              className="p-1.5 hover:bg-slate-800 rounded-md transition-colors text-slate-400 hover:text-white"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        <div 
          ref={scrollRef}
          className="flex-1 overflow-auto p-6 font-mono text-[13px] leading-relaxed selection:bg-blue-500/30"
        >
          {logLines.map((line, idx) => {
            const isError = line.includes('ERROR');
            const isWarning = line.includes('WARNING');
            const isInfo = line.includes('INFO');
            
            return (
              <div key={idx} className="group flex gap-4 hover:bg-white/5 px-2 -mx-2 rounded transition-colors">
                <span className="text-slate-700 select-none w-8 text-right shrink-0">{idx + 1}</span>
                <span className={`
                  ${isError ? 'text-red-400' : isWarning ? 'text-amber-400' : isInfo ? 'text-blue-400' : 'text-slate-300'}
                  break-all
                `}>
                  {line}
                </span>
              </div>
            );
          })}
          {logs === '' && (
            <div className="h-full flex items-center justify-center text-slate-600 italic">
              No logs available for this run
            </div>
          )}
        </div>
        
        <div className="px-6 py-2 bg-slate-950 border-t border-slate-800 flex justify-between items-center">
          <p className="text-[10px] text-slate-600">
            {logLines.length} lines • {(new Blob([logs]).size / 1024).toFixed(1)} KB
          </p>
          <div className="flex items-center gap-2 text-[10px] text-slate-600 font-bold uppercase tracking-tighter">
            <span className="flex h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
            Live Stream
          </div>
        </div>
      </div>
    </div>
  );
};

export default LogsViewer;
