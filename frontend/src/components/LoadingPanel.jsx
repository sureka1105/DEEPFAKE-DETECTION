import { useEffect, useState } from "react";

const STEPS = [
  "Running AI model",
  "Generating visual explanation",
  "Analyzing metadata",
  "Preparing authenticity result",
];

export default function LoadingPanel() {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((i) => Math.min(i + 1, STEPS.length - 1));
    }, 900);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="loading-panel" role="status" aria-live="polite">
      <div className="lead">Analyzing image…</div>
      <ul className="loading-steps">
        {STEPS.map((step, i) => (
          <li key={step} className={i < activeIndex ? "done" : i === activeIndex ? "active" : ""}>
            <span className="step-dot" />
            {step}
          </li>
        ))}
      </ul>
    </div>
  );
}
