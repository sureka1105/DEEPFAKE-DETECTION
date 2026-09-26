import { useState } from "react";
import { FiAlertCircle } from "react-icons/fi";
import Uploader from "./components/Uploader.jsx";
import LoadingPanel from "./components/LoadingPanel.jsx";
import ResultDashboard from "./components/ResultDashboard.jsx";
import { analyzeImage, extractErrorMessage } from "./api.js";

export default function App() {
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  function handleFileSelected(selectedFile, err) {
    setError(err);
    setResult(null);
    if (selectedFile) setFile(selectedFile);
  }

  function handleClear() {
    setFile(null);
    setResult(null);
    setError(null);
  }

  async function handleAnalyze() {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyzeImage(file);
      setResult(data);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <div className="shell">
      <div className="topbar">
        <span className="mark" />
        <span className="wordmark">Deepfake Detection<span>Image Authenticity &amp; Forensics</span></span>
      </div>

      <div className="hero">
        <h1>Deepfake &amp; Synthetic Image Detection</h1>
        <p>
          Upload an image to get instant AI authenticity assessment, deep
          multi-signal forensics (FFT spectral analysis, ELA, sensor noise),
          camera profiling, and downloadable PDF reports.
        </p>
      </div>

      <Uploader
        selectedFile={file}
        onFileSelected={handleFileSelected}
        onClear={handleClear}
        onAnalyze={handleAnalyze}
        analyzing={analyzing}
      />

      {error && (
        <div className="error-banner">
          <FiAlertCircle style={{ marginTop: 2 }} />
          <span>{error}</span>
        </div>
      )}

      {analyzing && <LoadingPanel />}

      {result && !analyzing && <ResultDashboard result={result} />}
    </div>
  );
}
