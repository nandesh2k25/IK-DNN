import React, { useState, useEffect } from 'react';
import { Table as TableIcon, Download, Loader2, ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { getTableData } from '../utils/api';

const TablesViewer = ({ selectedRun }) => {
  const [selectedTable, setSelectedTable] = useState('final_metrics.csv');
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const tables = [
    { id: 'final_metrics.csv', label: 'Final Metrics' },
    { id: 'train_history.csv', label: 'Train History' },
    { id: 'test_prediction_summary.csv', label: 'Test Predictions' },
    { id: 'test_component_usage.csv', label: 'Component Usage' },
    { id: 'test_error_summary.csv', label: 'Error Summary' },
  ];

  useEffect(() => {
    if (selectedRun && selectedTable) {
      fetchTable();
    }
  }, [selectedRun, selectedTable]);

  const fetchTable = async () => {
    setLoading(true);
    try {
      const res = await getTableData(selectedRun, selectedTable);
      setData(res.data);
    } catch (err) {
      console.error("Failed to fetch table", err);
      setData([]);
    } finally {
      setLoading(false);
    }
  };

  const filteredData = data.filter(row => 
    Object.values(row).some(val => 
      String(val).toLowerCase().includes(searchQuery.toLowerCase())
    )
  );

  if (!selectedRun) return (
    <div className="flex flex-col items-center justify-center h-64 text-slate-400">
      <TableIcon className="h-12 w-12 mb-4 opacity-20" />
      <p>Select an experiment to view tables</p>
    </div>
  );

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col h-[700px]">
        <div className="p-4 border-b border-slate-200 bg-slate-50 flex flex-wrap items-center justify-between gap-4">
          <div className="flex bg-slate-200/50 p-1 rounded-lg border border-slate-200">
            {tables.map(table => (
              <button
                key={table.id}
                onClick={() => setSelectedTable(table.id)}
                className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${
                  selectedTable === table.id
                    ? 'bg-white text-blue-700 shadow-sm'
                    : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                {table.label}
              </button>
            ))}
          </div>
          
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                type="text"
                placeholder="Filter table..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="bg-white border border-slate-200 rounded-lg py-1.5 pl-9 pr-3 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500/20 w-48 transition-all"
              />
            </div>
            <button className="p-1.5 hover:bg-slate-200 rounded-md transition-colors text-slate-500">
              <Download className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto relative">
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center bg-white/50">
              <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
            </div>
          ) : (
            <table className="w-full text-xs text-left border-collapse">
              <thead className="bg-white sticky top-0 z-10">
                <tr>
                  {data.length > 0 && Object.keys(data[0]).map(key => (
                    <th key={key} className="px-4 py-3 font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200 bg-slate-50">
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredData.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-50 transition-colors">
                    {Object.values(row).map((val, i) => (
                      <td key={i} className="px-4 py-3 font-mono text-slate-600 truncate max-w-[200px]">
                        {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(6)) : String(val)}
                      </td>
                    ))}
                  </tr>
                ))}
                {filteredData.length === 0 && !loading && (
                  <tr>
                    <td colSpan="100%" className="px-4 py-12 text-center text-slate-400 italic">
                      No data matches your search
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
};

export default TablesViewer;
