"""PDF Report Generator for Routio AI / RoadZone.

Generates official engineering & civic inspection PDF reports using ReportLab.
Powered by Google Gemma 4 Multimodal Vision.
"""
from __future__ import annotations

import io
from datetime import datetime
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
)


def generate_flood_comparison_pdf(
    comp_res: dict,
    img_a: Image.Image | None = None,
    img_b: Image.Image | None = None,
) -> bytes:
    """Generate a formal municipal flood delta engineering PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1c1917"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#78716c"),
        spaceAfter=12,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#854d0e"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#292524"),
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#991b1b"),
    )

    story = []

    # Title & Header
    story.append(Paragraph("ROUTIO AI / ROADZONE CIVIL DEFENSE", title_style))
    story.append(
        Paragraph(
            f"Hydrological Road Inundation & Flood Delta Assessment &bull; Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#eab308"), spaceAfter=14))

    # AI Model Badge
    model_badge_data = [
        [
            Paragraph("<b>Diagnostic Engine:</b> Google Gemma 4 (gemma-4-26b-a4b-it)", body_style),
            Paragraph("<b>Vision Track:</b> Multimodal Spatial Hydrology", body_style),
        ]
    ]
    t_badge = Table(model_badge_data, colWidths=[270, 260])
    t_badge.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fefce8")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#fef08a")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_badge)
    story.append(Spacer(1, 14))

    # Metrics Summary Table
    sev_a = comp_res.get("image_a", {}).get("severity", 0.0)
    sev_b = comp_res.get("image_b", {}).get("severity", 0.0)
    delta = comp_res.get("severity_delta", 0.0)
    transit = comp_res.get("transit_status", "UNKNOWN")

    metrics_table_data = [
        [
            Paragraph("<b>Condition</b>", body_style),
            Paragraph("<b>Severity (0-10)</b>", body_style),
            Paragraph("<b>Status / Delta</b>", body_style),
        ],
        [
            Paragraph("Baseline Road (Dry / Pre-Event)", body_style),
            Paragraph(f"<b>{sev_a}/10</b>", body_style),
            Paragraph("Reference Baseline", body_style),
        ],
        [
            Paragraph("Inundated Road (Post-Rainfall)", body_style),
            Paragraph(f"<b>{sev_b}/10</b>", body_style),
            Paragraph(f"<font color='#dc2626'><b>+{delta} Severity Spike</b></font>", body_style),
        ],
        [
            Paragraph("<b>Transit Passability</b>", body_style),
            Paragraph(f"<b>{transit}</b>", callout_style),
            Paragraph("Emergency Vehicle Clearance", body_style),
        ],
    ]
    t_metrics = Table(metrics_table_data, colWidths=[200, 150, 180])
    t_metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f4")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e7e5e4")),
                ("PADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(Paragraph("1. Hydrological Telemetry Summary", section_heading))
    story.append(t_metrics)
    story.append(Spacer(1, 14))

    # Drainage & Action Assessment
    drainage_text = comp_res.get("drainage_assessment", "No drainage assessment provided.")
    action_text = comp_res.get("action_required", "Immediate drainage clearance required.")

    story.append(Paragraph("2. Gemma 4 Multimodal Drainage Assessment", section_heading))
    drainage_box = [
        [Paragraph(f"<b>Diagnostic Findings:</b><br/>{drainage_text}", body_style)],
    ]
    t_drainage = Table(drainage_box, colWidths=[530])
    t_drainage.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(t_drainage)
    story.append(Spacer(1, 14))

    story.append(Paragraph("3. Recommended Municipal Emergency Protocol", section_heading))
    action_box = [
        [
            Paragraph(
                f"<font color='#991b1b'><b>CIVIC INTERVENTION DIRECTIVE:</b></font><br/>{action_text}",
                body_style,
            )
        ]
    ]
    t_action = Table(action_box, colWidths=[530])
    t_action.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff1f2")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#fecdd3")),
                ("PADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(t_action)
    story.append(Spacer(1, 18))

    # Footer note
    footer_text = (
        "<i>This official document is generated automatically by Routio AI utilizing Google Gemma 4 Multimodal Vision "
        "and spatial intelligence telemetry. Valid for municipal submission, PWD escalation, and civil defense filing.</i>"
    )
    story.append(Paragraph(footer_text, subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_road_inspection_pdf(
    image_name: str,
    pothole_count: int,
    area_ratio: float,
    is_flooded: bool,
    flood_severity: float,
    calculated_score: float,
    zone_name: str = "Metropolitan Sector",
) -> bytes:
    """Generate an official road inspection damage assessment PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1c1917"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#78716c"),
        spaceAfter=12,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#854d0e"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#292524"),
    )

    story = []

    # Title & Header
    story.append(Paragraph("ROUTIO AI / ROADZONE INSPECTION REPORT", title_style))
    story.append(
        Paragraph(
            f"Autonomous Road Surface Quality Assessment &bull; Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#eab308"), spaceAfter=14))

    # Target Metadata
    meta_data = [
        [
            Paragraph(f"<b>Target Sector:</b> {zone_name}", body_style),
            Paragraph(f"<b>Image File:</b> {image_name}", body_style),
        ],
        [
            Paragraph("<b>Vision Model:</b> Google Gemma 4 (gemma-4-26b-a4b-it)", body_style),
            Paragraph("<b>Spatial Radius:</b> 1.4 km Catchment Zone", body_style),
        ],
    ]
    t_meta = Table(meta_data, colWidths=[270, 260])
    t_meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fefce8")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#fef08a")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_meta)
    story.append(Spacer(1, 14))

    # Inspection Findings
    hazard_color = "#dc2626" if calculated_score >= 12 else ("#d97706" if calculated_score >= 6 else "#16a34a")
    status_label = "CRITICAL HAZARD" if calculated_score >= 12 else ("MODERATE DAMAGE" if calculated_score >= 6 else "NOMINAL")

    findings = [
        [Paragraph("<b>Inspection Metric</b>", body_style), Paragraph("<b>Gemma 4 Measurement</b>", body_style)],
        [Paragraph("Detected Asphalt Potholes / Craters", body_style), Paragraph(f"<b>{pothole_count} detected</b>", body_style)],
        [Paragraph("Structural Cavity Area Ratio", body_style), Paragraph(f"<b>{area_ratio:.2%}</b> of total road surface", body_style)],
        [Paragraph("Waterlogging Condition", body_style), Paragraph("<b>Flooded / Submerged</b>" if is_flooded else "Dry / Normal", body_style)],
        [Paragraph("Flood Severity Index", body_style), Paragraph(f"<b>{flood_severity:.1f} / 10.0</b>", body_style)],
        [
            Paragraph("<b>Calculated Hazard Risk Score</b>", body_style),
            Paragraph(f"<font color='{hazard_color}'><b>{calculated_score} ({status_label})</b></font>", body_style),
        ],
    ]
    t_findings = Table(findings, colWidths=[270, 260])
    t_findings.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f4")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e7e5e4")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(Paragraph("1. Surface Damage Telemetry", section_heading))
    story.append(t_findings)
    story.append(Spacer(1, 14))

    # Legal / Municipal Citation
    story.append(Paragraph("2. Statutory Municipal Duty Citation", section_heading))
    legal_box = [
        [
            Paragraph(
                "Under the Municipal Corporation Act & Public Works Department (PWD) road safety directives, "
                "local authorities maintain a non-delegable duty of care to repair road craters exceeding hazardous thresholds "
                "within 48 hours of telemetry logging. This document serves as prima facie evidence for municipal remediation.",
                body_style,
            )
        ]
    ]
    t_legal = Table(legal_box, colWidths=[530])
    t_legal.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(t_legal)
    story.append(Spacer(1, 18))

    footer_text = (
        "<i>Generated by Routio AI RoadZone Autonomous Infrastructure Monitor. "
        "Verified with Google Gemma 4 Extended Multimodal Vision.</i>"
    )
    story.append(Paragraph(footer_text, subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_complaint_letter_pdf(
    complaint_markdown: str,
    zone_name: str,
    authority_name: str,
) -> bytes:
    """Convert Gemma 4 generated municipal complaint notice into a formal PDF."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "NoticeTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#7f1d1d"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "NoticeSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#78716c"),
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "NoticeBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14.5,
        textColor=colors.HexColor("#1c1917"),
        spaceAfter=8,
    )

    story = []

    story.append(Paragraph("FORMAL STATUTORY MUNICIPAL NOTICE", title_style))
    story.append(
        Paragraph(
            f"Public Hazard & Infrastructure Grievance &bull; Authority: {authority_name} &bull; Zone: {zone_name}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#b91c1c"), spaceAfter=12))

    # Parse markdown paragraphs safely
    paragraphs = complaint_markdown.split("\n\n")
    for p in paragraphs:
        cleaned = p.strip().replace("\n", "<br/>")
        # Replace markdown bold **text** with <b>text</b>
        import re
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", cleaned)
        cleaned = re.sub(r"\*(.*?)\*", r"<i>\1</i>", cleaned)
        cleaned = re.sub(r"^#+\s*", "", cleaned)
        if cleaned:
            story.append(Paragraph(cleaned, body_style))

    story.append(Spacer(1, 14))
    footer_text = (
        "<i>Notice prepared with Google Gemma 4 Extended Thinking Mode for civil safety enforcement. "
        "Verified with telemetry evidence from Routio AI Autonomous Sensors.</i>"
    )
    story.append(Paragraph(footer_text, subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

