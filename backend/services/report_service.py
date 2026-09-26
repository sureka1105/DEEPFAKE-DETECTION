"""
Generates a downloadable PDF authenticity report using ReportLab.
All values come from the actual analysis result passed in - nothing is
fabricated here.
"""
import base64
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
)


def _data_uri_to_image_reader(data_uri: str) -> io.BytesIO:
    header, b64data = data_uri.split(",", 1)
    return io.BytesIO(base64.b64decode(b64data))


def build_pdf_report(analysis: dict) -> bytes:
    """
    analysis: the same dict structure returned by /api/analyze (as a Python
    dict, not yet serialized to JSON).
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=4,
    )
    h2 = ParagraphStyle("H2Custom", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph("Deepfake Authenticity Analysis Report", title_style))
    story.append(Paragraph(
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small,
    ))
    story.append(Spacer(1, 10))

    prediction = analysis["prediction"]
    confidence = analysis["confidence"]
    assessment = analysis["authenticity_assessment"]

    result_color = colors.HexColor("#B00020") if prediction == "FAKE" else colors.HexColor("#1B7A3D")
    result_style = ParagraphStyle(
        "Result", parent=styles["Heading1"], fontSize=26,
        textColor=result_color, alignment=1, spaceAfter=2,
    )
    story.append(Paragraph(prediction, result_style))
    story.append(Paragraph(
        f"<para alignment='center'>Confidence: <b>{confidence}%</b></para>", body,
    ))
    story.append(Paragraph(
        f"<para alignment='center'>Authenticity Assessment: <b>{assessment}</b></para>", body,
    ))

    story.append(Paragraph("AI Explanation", h2))
    story.append(Paragraph(analysis["explanation"], body))

    # Visual explanation
    story.append(Paragraph("Visual Explanation (Grad-CAM)", h2))
    gc = analysis["gradcam"]
    if gc.get("available", True):
        try:
            imgs = []
            labels = []
            for key, label in (("original", "Original"), ("heatmap", "Heatmap"), ("overlay", "Overlay")):
                reader = _data_uri_to_image_reader(gc[key])
                imgs.append(RLImage(reader, width=4.8 * cm, height=4.8 * cm))
                labels.append(Paragraph(f"<para alignment='center'>{label}</para>", small))
            table = Table([imgs, labels], colWidths=[5.2 * cm] * 3)
            table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story.append(table)
        except Exception:
            story.append(Paragraph("Visual explanation images could not be embedded.", body))
    else:
        story.append(Paragraph("Visual explanation was unavailable for this analysis.", body))
    story.append(Paragraph(
        "Highlighted regions indicate areas that contributed strongly to the "
        "model's prediction. This is an attention visualization, not proof of manipulation.",
        small,
    ))

    meta = analysis["metadata"]

    # Camera & Device Detection - built from real EXIF/heuristics, never templated.
    cam = analysis.get("camera_profile") or {}
    if cam:
        story.append(Paragraph("Camera &amp; Device Detection", h2))
        cam_rows = [
            ["Capture Source", cam.get("capture_source", "")],
            ["Sensor Profile", cam.get("sensor_profile", "")],
        ]
        details = cam.get("camera_details")
        if details:
            cam_rows += [
                ["Camera Make", details.get("camera_make", "")],
                ["Camera Model", details.get("camera_model", "")],
                ["Device Category", details.get("device_category", "")],
                ["Lens Model", details.get("lens_model", "")],
                ["Focal Length / Aperture", details.get("focal_length_aperture", "")],
                ["Exposure / ISO", details.get("exposure_iso", "")],
                ["Capture Software", details.get("capture_software", "")],
            ]
        cam_rows += [
            ["Dimensions & Size", cam.get("dimensions_size", "")],
            ["Color Gamut & Depth", cam.get("color_gamut_depth", "")],
            ["GPS Location Tags", cam.get("gps_location", "")],
        ]
        cam_table = Table(cam_rows, colWidths=[5 * cm, 10 * cm])
        cam_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ]))
        story.append(cam_table)
        if cam.get("no_exif_note"):
            story.append(Spacer(1, 6))
            story.append(Paragraph(cam["no_exif_note"], small))
        if meta.get("observations"):
            story.append(Spacer(1, 6))
            for obs in meta["observations"]:
                story.append(Paragraph(f"&bull; {obs}", small))

    # Multi-signal forensic breakdown - real, computed-per-image values only.
    forensics = analysis.get("forensics") or {}
    if forensics.get("ela_mean_error") is not None:
        story.append(Paragraph("Multi-Signal Forensic Breakdown", h2))
        story.append(Paragraph(
            "These are supporting, individually-inconclusive pixel-level signals "
            "computed directly from this image. They do not by themselves prove "
            "authenticity or manipulation.", small,
        ))
        forensic_rows = [
            ["Signal", "Value", "Note"],
            ["Error Level Analysis (ELA)", str(forensics.get("ela_mean_error")), forensics.get("ela_note", "")],
            ["2D-FFT High/Low Ratio", str(forensics.get("fft_high_low_ratio")), forensics.get("fft_note", "")],
            ["Noise Residual Variance", str(forensics.get("noise_residual_variance")), forensics.get("noise_note", "")],
            ["Edge Sharpness (Laplacian Var.)", str(forensics.get("edge_sharpness_variance")), forensics.get("edge_note", "")],
        ]
        forensic_table = Table(forensic_rows, colWidths=[4.5 * cm, 2.5 * cm, 8 * cm])
        forensic_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(Spacer(1, 4))
        story.append(forensic_table)

    if analysis.get("model_source") == "heuristic_fallback":
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "&bull; No trained neural network was loaded for this analysis. The "
            "verdict above came from the forensic-signal fallback, not a trained "
            "classifier, and should be treated with extra caution.", small,
        ))

    # Disclaimer
    story.append(Paragraph("Disclaimer", h2))
    story.append(Paragraph(analysis["disclaimer"], small))

    doc.build(story)
    return buf.getvalue()
