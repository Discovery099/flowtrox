import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

const client = axios.create({ baseURL: API, timeout: 240000 });

export const getInstruments = () => client.get("/instruments").then((r) => r.data);
export const getStrategyInfo = () => client.get("/strategy/info").then((r) => r.data);
export const getModelStatus = (symbol) =>
  client.get(`/model/status`, { params: { symbol } }).then((r) => r.data);
export const runSingle = (body) => client.post("/backtest/single", body).then((r) => r.data);
export const startOptimize = (symbol) =>
  client.post("/optimize/start", { symbol }).then((r) => r.data);
export const getJobStatus = (jobId) =>
  client.get(`/optimize/status/${jobId}`).then((r) => r.data);
export const downloadUrl = (runId, kind) => `${API}/runs/${runId}/download/${kind}`;

export default client;
