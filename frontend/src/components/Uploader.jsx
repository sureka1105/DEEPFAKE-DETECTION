import { useRef, useState } from "react";
import { FiUploadCloud, FiX } from "react-icons/fi";
import ScanFrame from "./ScanFrame.jsx";

const ACCEPTED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

function formatBytes(bytes) {
  if (!bytes) return "0 KB";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(0)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

export default function Uploader({ onFileSelected, selectedFile, onClear, onAnalyze, analyzing }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [dims, setDims] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  function handleFiles(files) {
    const file = files?.[0];
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      onFileSelected(null, "Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image.");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    const img = new Image();
    img.onload = () => setDims({ w: img.width, h: img.height });
    img.src = url;
    onFileSelected(file, null);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  }

  function clear() {
    setPreviewUrl(null);
    setDims(null);
    onClear();
  }

  if (selectedFile) {
    return (
      <div>
        <div className="preview-card">
          <ScanFrame src={previewUrl} alt="Selected preview" style={{ width: 84, height: 84 }} />
          <div className="preview-info">
            <div className="name">{selectedFile.name}</div>
            <div className="meta">
              {formatBytes(selectedFile.size)}
              {dims ? ` · ${dims.w}×${dims.h}px` : ""}
            </div>
          </div>
          <div className="preview-actions">
            <button className="btn btn-ghost" onClick={clear} disabled={analyzing} aria-label="Remove image">
              <FiX />
            </button>
          </div>
        </div>
        <button className="btn btn-primary btn-block" onClick={onAnalyze} disabled={analyzing}>
          {analyzing ? "Analyzing…" : "Analyze Image"}
        </button>
      </div>
    );
  }

  return (
    <div
      className={`dropzone${dragging ? " dragging" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <div className="icon"><FiUploadCloud /></div>
      <div className="primary">Drag & drop an image, or click to browse</div>
      <div className="secondary">JPG, PNG, or WEBP · up to 10 MB</div>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png,image/webp"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
