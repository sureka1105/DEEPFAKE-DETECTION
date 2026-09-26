export default function ScanFrame({ src, alt, scanning = false, style }) {
  return (
    <div className={`scan-frame${scanning ? " scanning" : ""}`} style={style}>
      {src && <img src={src} alt={alt} />}
      <span className="corner tl" />
      <span className="corner tr" />
      <span className="corner bl" />
      <span className="corner br" />
    </div>
  );
}
