import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Database, 
  Activity, 
  Target, 
  Image as ImageIcon, 
  Table as TableIcon, 
  Terminal,
  ChevronRight,
  RefreshCw,
  Search,
  CheckCircle2,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { 
  listExperiments, 
  getExperimentConfig, 
  getExperimentMetrics,
  getTrainStatus
} from './utils/api';

// Components
import Header from './components/Header';
import ExperimentSelector from './components/ExperimentSelector';
import DatasetPanel from './components/DatasetPanel';
import TrainingPanel from './components/TrainingPanel';
import EvaluationPanel from './components/EvaluationPanel';
import PredictionPanel from './components/PredictionPanel';
import PlotsViewer from './components/PlotsViewer';
import TablesViewer from './components/TablesViewer';
import LogsViewer from './components/LogsViewer';

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [experiments, setExperiments] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [config, setConfig] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [trainStatus, setTrainStatus] = useState({ is_training: false });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchExperiments();
    const interval = setInterval(fetchTrainStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedRun) {
      fetchRunDetails(selectedRun);
    }
  }, [selectedRun]);

  const fetchExperiments = async () => {
    try {
      const res = await listExperiments();
      setExperiments(res.data);
      if (res.data.length > 0 && !selectedRun) {
        setSelectedRun(res.data[0].name);
      }
    } catch (err) {
      console.error("Failed to fetch experiments", err);
    }
  };

  const fetchTrainStatus = async () => {
    try {
      const res = await getTrainStatus();
      setTrainStatus(res.data);
      // If training just finished, refresh experiments
      if (!res.data.is_training && trainStatus.is_training) {
        fetchExperiments();
      }
    } catch (err) {
      console.error("Failed to fetch train status", err);
    }
  };

  const fetchRunDetails = async (runName) => {
    setLoading(true);
    try {
      const [configRes, metricsRes] = await Promise.all([
        getExperimentConfig(runName),
        getExperimentMetrics(runName)
      ]);
      setConfig(configRes.data);
      setMetrics(metricsRes.data);
    } catch (err) {
      console.error("Failed to fetch run details", err);
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'dataset', label: 'Dataset', icon: Database },
    { id: 'train', label: 'Train', icon: Activity },
    { id: 'predict', label: 'Predict', icon: Target },
    { id: 'plots', label: 'Plots', icon: ImageIcon },
    { id: 'tables', label: 'Tables', icon: TableIcon },
    { id: 'logs', label: 'Logs', icon: Terminal },
  ];

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <Header />
      
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className="w-64 bg-white border-r border-slate-200 flex flex-col">
          <div className="p-4 border-b border-slate-200">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
              Navigation
            </h2>
            <nav className="space-y-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    activeTab === tab.id
                      ? 'bg-blue-50 text-blue-700'
                      : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                  }`}
                >
                  <tab.icon className={`mr-3 h-5 w-5 ${
                    activeTab === tab.id ? 'text-blue-700' : 'text-slate-400'
                  }`} />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
              Active Run
            </h2>
            <ExperimentSelector 
              experiments={experiments} 
              selectedRun={selectedRun} 
              onSelect={setSelectedRun} 
            />
            
            {config && (
              <div className="mt-6 space-y-4">
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
                  <p className="text-xs text-slate-500 mb-1">Status</p>
                  <div className="flex items-center text-sm font-medium">
                    {metrics?.final ? (
                      <span className="flex items-center text-green-600">
                        <CheckCircle2 className="h-4 w-4 mr-1.5" /> Completed
                      </span>
                    ) : (
                      <span className="flex items-center text-amber-600">
                        <Activity className="h-4 w-4 mr-1.5" /> In Progress
                      </span>
                    )}
                  </div>
                </div>
                
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
                  <p className="text-xs text-slate-500 mb-1">Components</p>
                  <p className="text-sm font-medium">{config.n_components}</p>
                </div>
                
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
                  <p className="text-xs text-slate-500 mb-1">Hidden Size</p>
                  <p className="text-sm font-medium">{config.hidden_size}</p>
                </div>
              </div>
            )}
          </div>
          
          <div className="p-4 border-t border-slate-200 bg-slate-50">
            <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
              <span>Training Status</span>
              {trainStatus.is_training && <Loader2 className="h-3 w-3 animate-spin text-blue-600" />}
            </div>
            <div className="text-sm font-medium truncate">
              {trainStatus.is_training 
                ? `Training: ${trainStatus.current_run}`
                : 'System Idle'}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-8 relative">
          {loading && (
            <div className="absolute inset-0 bg-white/50 backdrop-blur-[2px] z-50 flex items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
            </div>
          )}
          
          <div className="max-w-6xl mx-auto">
            {activeTab === 'dashboard' && (
              <DashboardView config={config} metrics={metrics} />
            )}
            {activeTab === 'dataset' && (
              <DatasetPanel config={config} />
            )}
            {activeTab === 'train' && (
              <TrainingPanel trainStatus={trainStatus} />
            )}
            {activeTab === 'predict' && (
              <PredictionPanel selectedRun={selectedRun} config={config} />
            )}
            {activeTab === 'plots' && (
              <PlotsViewer selectedRun={selectedRun} />
            )}
            {activeTab === 'tables' && (
              <TablesViewer selectedRun={selectedRun} />
            )}
            {activeTab === 'logs' && (
              <LogsViewer selectedRun={selectedRun} />
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

const DashboardView = ({ config, metrics }) => {
  if (!config) return (
    <div className="flex flex-col items-center justify-center h-64 text-slate-400">
      <LayoutDashboard className="h-12 w-12 mb-4 opacity-20" />
      <p>Select an experiment to view the dashboard</p>
    </div>
  );

  const finalMetrics = metrics?.final || {};

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard 
          label="Angle MSE" 
          value={finalMetrics.best_of_k_angle_mse?.toFixed(6) || '-'} 
          icon={Target}
          color="blue"
        />
        <StatCard 
          label="Pose RMSE" 
          value={finalMetrics.best_of_k_pose_rmse?.toFixed(6) || '-'} 
          icon={Activity}
          color="indigo"
        />
        <StatCard 
          label="Epochs" 
          value={finalMetrics.total_epochs_trained || '-'} 
          icon={ChevronRight}
          color="slate"
        />
        <StatCard 
          label="Components" 
          value={config.n_components} 
          icon={Database}
          color="emerald"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
            <Target className="h-5 w-5 mr-2 text-blue-600" />
            Model Performance (Best-of-K)
          </h3>
          <div className="space-y-4">
            <MetricRow label="Angle RMSE" value={finalMetrics.best_of_k_angle_rmse?.toFixed(6)} />
            <MetricRow label="Pose MSE" value={finalMetrics.best_of_k_pose_mse?.toFixed(6)} />
            <div className="pt-2 border-t border-slate-100">
              <h4 className="text-sm font-medium text-slate-500 mb-2 uppercase tracking-wider">Joint MAE (deg)</h4>
              <MetricRow label="Theta 1 MAE" value={(finalMetrics.theta1_mae * 180 / Math.PI)?.toFixed(4)} />
              <MetricRow label="Theta 2 MAE" value={(finalMetrics.theta2_mae * 180 / Math.PI)?.toFixed(4)} />
              <MetricRow label="Theta 3 MAE" value={(finalMetrics.theta3_mae * 180 / Math.PI)?.toFixed(4)} />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
            <Activity className="h-5 w-5 mr-2 text-indigo-600" />
            Training Configuration
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <ConfigItem label="Learning Rate" value={config.learning_rate} />
            <ConfigItem label="Batch Size" value={config.batch_size} />
            <ConfigItem label="Lambda FK" value={config.lambda_fk} />
            <ConfigItem label="Patience" value={config.patience} />
            <ConfigItem label="Max Epochs" value={config.max_epochs} />
            <ConfigItem label="Hidden Size" value={config.hidden_size} />
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, icon: Icon, color }) => {
  const colors = {
    blue: 'bg-blue-50 text-blue-600',
    indigo: 'bg-indigo-50 text-indigo-600',
    slate: 'bg-slate-50 text-slate-600',
    emerald: 'bg-emerald-50 text-emerald-600',
  };

  return (
    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className={`p-2 rounded-lg ${colors[color]}`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <p className="text-2xl font-bold text-slate-900">{value}</p>
    </div>
  );
};

const MetricRow = ({ label, value }) => (
  <div className="flex justify-between items-center py-2">
    <span className="text-sm text-slate-600">{label}</span>
    <span className="text-sm font-mono font-semibold text-slate-900 bg-slate-50 px-2 py-0.5 rounded border border-slate-100">
      {value || '-'}
    </span>
  </div>
);

const ConfigItem = ({ label, value }) => (
  <div>
    <p className="text-xs text-slate-400 mb-0.5 uppercase tracking-wider">{label}</p>
    <p className="text-sm font-medium text-slate-900">{value}</p>
  </div>
);

export default App;
