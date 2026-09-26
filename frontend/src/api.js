import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 60000,
});

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function checkHealth() {
  const res = await api.get("/health");
  return res.data;
}

export function reportDownloadUrl(reportId) {
  return `/api/report/${reportId}`;
}

export async function downloadReport(reportId) {
  const res = await api.post(`/report/${reportId}`, null, {
    responseType: "blob",
  });
  return res.data;
}

export function extractErrorMessage(err) {
  if (err?.response?.data?.detail) return err.response.data.detail;
  if (err?.message === "Network Error") {
    return "Can't reach the analysis server. Make sure the backend is running.";
  }
  return "Something went wrong. Please try again.";
}
