import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

export const listExperiments = () => api.get('/experiments');
export const getExperimentConfig = (runName) => api.get(`/experiments/${runName}/config`);
export const getExperimentMetrics = (runName) => api.get(`/experiments/${runName}/metrics`);
export const getExperimentLogs = (runName) => api.get(`/experiments/${runName}/logs`);
export const generateDataset = (data) => api.post('/dataset/generate', data);
export const trainModel = (data) => api.post('/train', data);
export const getTrainStatus = () => api.get('/train/status');
export const predictJoints = (data) => api.post('/predict', data);
export const getPlotUrl = (runName, plotName) => `/api/experiments/${runName}/plots/${plotName}`;
export const getTableData = (runName, tableName) => api.get(`/experiments/${runName}/tables/${tableName}`);

export default api;
