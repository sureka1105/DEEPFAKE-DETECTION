import { useState } from "react";
import { FiDownload, FiAlertCircle } from "react-icons/fi";
import ScanFrame from "./ScanFrame.jsx";
import { downloadReport, extractErrorMessage } from "../api.js";

function MetaRow({ label, value }) {
  return (
    <tr>
      <td>{label}</td>
      <td>{value ?? "—"}</td>
    </tr>
  );
}

function ForensicCard({ label, value, note, flagged }) {
  return (
    <div className="forensic-card">
      <div className="forensic-label">{label}</div>
      <div className="forensic-value">{value ?? "—"}</div>
      <span className={`forensic-tag ${flagged ? "flagged" : "clear"}`}>{note}</span>
    </div>
  );
}

export default function ResultDashboard({ result }) {
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState(null);

  const isFake = result.prediction === "FAKE";
  const meta = result.metadata || {};
  const forensics = result.forensics || {};
  const cameraProfile = result.camera_profile || {};
  const isHeuristic = result.model_source === "heuristic_fallback";

  async function handleDownload() {
    setDownloading(true);
    setDownloadError(null);
    try {
      const blob = await downloadReport(result.report_id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `authenticity_report_${result.report_id.slice(0, 8)}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadError(extractErrorMessage(err));
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="result-section">
      <div className={`verdict-card ${isFake ? "fake" : "real"}`}>
        <div className="verdict-label">{result.prediction}</div>
        <div className="verdict-confidence">
          Confidence <b>{result.confidence}%</b>
        </div>
        <span className="verdict-assessment">{result.authenticity_assessment}</span>
      </div>

      <div className="section-block">
        <div className="section-title">Forensic Assessment &amp; AI Explanation</div>
        <p>{result.explanation}</p>
      </div>

      <div className="section-block">
        <div className="section-title">Visual Explanation (Grad-CAM Attention Heatmap)</div>
        <div className="gradcam-grid">
          <div className="gradcam-item">
            <ScanFrame src={result.gradcam.original} alt="Original image" />
            <div className="label">Original</div>
          </div>
          <div className="gradcam-item">
            <ScanFrame src={result.gradcam.heatmap} alt="Grad-CAM heatmap" />
            <div className="label">Heatmap</div>
          </div>
          <div className="gradcam-item">
            <ScanFrame src={result.gradcam.overlay} alt="Grad-CAM overlay" />
            <div className="label">Overlay</div>
          </div>
        </div>
        <p className="gradcam-note">
          {result.gradcam.available === false
            ? "Visual explanation was temporarily unavailable for this image."
            : "Highlighted regions indicate areas that contributed strongly to the model's prediction. This is not proof of manipulation."}
        </p>
      </div>

      {isHeuristic && (
        <div className="heuristic-banner">
          No trained model is loaded yet - this result used the transparent
          forensic-signal fallback below, not a trained classifier. Treat it
          with extra caution.
        </div>
      )}

      {forensics.ela_mean_error !== undefined && forensics.ela_mean_error !== null && (
        <div className="section-block">
          <div className="section-title">Multi-Signal Forensic Breakdown</div>
          <div className="forensic-grid">
            <ForensicCard
              label="Error Level Analysis (ELA)"
              value={`Error: ${forensics.ela_mean_error}`}
              note={forensics.ela_note}
              flagged={forensics.ela_mean_error > 0.08}
            />
            <ForensicCard
              label="2D-FFT Frequency Spectrum"
              value={`Ratio: ${forensics.fft_high_low_ratio}`}
              note={forensics.fft_note}
              flagged={forensics.fft_high_low_ratio > 0.35}
            />
            <ForensicCard
              label="Sensor Noise Residuals"
              value={`Var: ${forensics.noise_residual_variance}`}
              note={forensics.noise_note}
              flagged={forensics.noise_residual_variance < 8}
            />
            <ForensicCard
              label="Edge Sharpness (Laplacian)"
              value={`Var: ${forensics.edge_sharpness_variance}`}
              note={forensics.edge_note}
              flagged={forensics.edge_sharpness_variance > 600}
            />
          </div>
          <p className="gradcam-note">
            These are supporting, individually-inconclusive signals computed
            directly from this image - not proof by themselves.
          </p>
        </div>
      )}

      {cameraProfile.capture_source && (
        <div className="section-block">
          <div className="section-title">Camera &amp; Device Detection</div>
          <table className="meta-table">
            <tbody>
              <MetaRow label="Capture Source" value={cameraProfile.capture_source} />
              <MetaRow label="Sensor Profile" value={cameraProfile.sensor_profile} />
              {cameraProfile.camera_details && (
                <>
                  <MetaRow label="Camera Make" value={cameraProfile.camera_details.camera_make} />
                  <MetaRow label="Camera Model" value={cameraProfile.camera_details.camera_model} />
                  <MetaRow label="Device Category" value={cameraProfile.camera_details.device_category} />
                  <MetaRow label="Lens Model" value={cameraProfile.camera_details.lens_model} />
                  <MetaRow label="Focal Length / Aperture" value={cameraProfile.camera_details.focal_length_aperture} />
                  <MetaRow label="Exposure / ISO" value={cameraProfile.camera_details.exposure_iso} />
                  <MetaRow label="Capture Software" value={cameraProfile.camera_details.capture_software} />
                </>
              )}
              <MetaRow label="Dimensions & Size" value={cameraProfile.dimensions_size} />
              <MetaRow label="Color Gamut & Depth" value={cameraProfile.color_gamut_depth} />
              <MetaRow label="GPS Location Tags" value={cameraProfile.gps_location} />
            </tbody>
          </table>
          {cameraProfile.no_exif_note && (
            <p className="gradcam-note">{cameraProfile.no_exif_note}</p>
          )}
          {meta.observations && meta.observations.length > 0 && (
            <div className="meta-observation">
              {meta.observations.map((obs, i) => (
                <div key={i}>• {obs}</div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="section-block">
        <details className="tech-details">
          <summary>Technical Details</summary>
          <pre>{JSON.stringify(
            {
              probability_fake: result.probability_fake,
              confidence_percent: result.confidence,
              prediction: result.prediction,
            },
            null,
            2,
          )}</pre>
        </details>
      </div>

      <button className="btn btn-primary btn-block" onClick={handleDownload} disabled={downloading}>
        <FiDownload style={{ marginRight: 8, verticalAlign: -2 }} />
        {downloading ? "Preparing PDF…" : "Download Full PDF Forensic Report"}
      </button>
      {downloadError && (
        <div className="error-banner">
          <FiAlertCircle style={{ marginTop: 2 }} />
          <span>{downloadError}</span>
        </div>
      )}

      <p className="disclaimer">{result.disclaimer}</p>
    </div>
  );
}
