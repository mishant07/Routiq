"""RoadZone AI - Real-Time Road Health Zone Mapping, Multimodal Flood Intelligence, and Civil Defense.

Built for DigitalOcean Open-Source AI Track & Best Use of Gemma 4.
"""
from __future__ import annotations

import io
import os
import random
import base64
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_folium import st_folium

load_dotenv()


def get_logo_base64() -> str:
    logo_path = Path(__file__).parent / "assets" / "routio_logo.png"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    return ""


# Page configuration
st.set_page_config(
    page_title="RoadZone AI | Road Health & Hazard Spatial Defense",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Award-Winning Hackathon Aesthetic
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Warm obsidian / stone background with soft cream ambient glow */
    .stApp {
        background: radial-gradient(circle at 18% 18%, #1c1a17 0%, #141210 55%, #0d0c0a 100%);
        color: #fefce8;
    }

    /* Branded Header Logo */
    .header-logo-img {
        height: 56px;
        width: auto;
        filter: drop-shadow(0 0 14px rgba(254, 240, 138, 0.35));
    }

    /* Hero Header Card with Soft Cream & Warm Stone */
    .hero-container {
        background: linear-gradient(135deg, rgba(38, 35, 30, 0.85) 0%, rgba(24, 22, 19, 0.95) 100%);
        border: 1px solid rgba(254, 240, 138, 0.22);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 25px rgba(254, 240, 138, 0.08);
        backdrop-filter: blur(14px);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #ffffff 0%, #fef9c3 40%, #fef08a 75%, #fde047 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #d6d3cb;
        font-size: 0.92rem;
        margin-bottom: 12px;
    }

    /* Soft Pastel Pill Badges (Cream, Yellow, Warm White) */
    .tag-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .tag-gemma {
        background: rgba(254, 240, 138, 0.14);
        color: #fef08a;
        border: 1px solid rgba(254, 240, 138, 0.35);
    }
    .tag-db {
        background: rgba(245, 245, 244, 0.12);
        color: #f5f5f4;
        border: 1px solid rgba(245, 245, 244, 0.28);
    }
    .tag-live {
        background: rgba(253, 230, 138, 0.16);
        color: #fde68a;
        border: 1px solid rgba(253, 230, 138, 0.35);
    }

    /* Soft Frosted Metric Cards */
    .metric-card {
        background: rgba(38, 35, 30, 0.65);
        border: 1px solid rgba(254, 240, 138, 0.18);
        border-radius: 14px;
        padding: 16px 18px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(254, 240, 138, 0.45);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), 0 0 16px rgba(254, 240, 138, 0.15);
    }
    .metric-val {
        font-size: 1.75rem;
        font-weight: 800;
        margin-bottom: 2px;
        color: #ffffff;
    }
    .metric-lbl {
        color: #d6d3cb;
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.4px;
    }

    /* Soft Tone Zone Cards */
    .zone-card {
        background: rgba(38, 35, 30, 0.6);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(254, 240, 138, 0.15);
        backdrop-filter: blur(10px);
        transition: all 0.2s ease;
    }
    .zone-card:hover {
        background: rgba(48, 44, 38, 0.8);
        border-color: rgba(254, 240, 138, 0.35);
    }
    .zone-red { 
        border-left: 5px solid #f87171 !important; 
        box-shadow: -4px 0 18px rgba(248, 113, 113, 0.15); 
    }
    .zone-orange { 
        border-left: 5px solid #fde047 !important; 
        box-shadow: -4px 0 18px rgba(253, 224, 71, 0.15); 
    }
    .zone-green { 
        border-left: 5px solid #86efac !important; 
        box-shadow: -4px 0 18px rgba(134, 239, 172, 0.15); 
    }

    /* Soft Comparison Containers */
    .soft-card-cream {
        background: rgba(38, 35, 30, 0.7);
        border: 1px solid rgba(254, 240, 138, 0.3);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .soft-card-rose {
        background: rgba(38, 35, 30, 0.7);
        border: 1px solid rgba(254, 205, 211, 0.3);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .badge-cream {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(254, 240, 138, 0.18);
        color: #fef08a;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid rgba(254, 240, 138, 0.4);
        margin-bottom: 10px;
    }
    .badge-rose {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(254, 205, 211, 0.16);
        color: #fecdd3;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid rgba(254, 205, 211, 0.35);
        margin-bottom: 10px;
    }

    /* Soft Cream & Pastel Yellow Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #fefce8 0%, #fef08a 45%, #fde047 100%);
        color: #1c1917 !important;
        border: 1px solid rgba(254, 240, 138, 0.6);
        border-radius: 10px;
        font-weight: 700;
        padding: 9px 22px;
        box-shadow: 0 4px 18px rgba(250, 204, 21, 0.22);
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ffffff 0%, #fef9c3 40%, #fef08a 100%);
        box-shadow: 0 6px 24px rgba(250, 204, 21, 0.38);
        transform: translateY(-1px);
    }
</style>
    """,
    unsafe_allow_html=True,
)

from detect import detect_potholes
from flood import assess_flood_severity
from flood_compare import compare_flood_images
from zones import (
    initialize_zones,
    evaluate_all_zones,
    build_folium_map,
    build_red_zone_detail_map,
    load_zones_fast,
    ZoneRecord,
)
from complaint import generate_complaint_letter
from database import (
    get_all_zones,
    get_all_scans,
    get_incident_reports,
    record_scan_in_db,
    DB_PATH,
)
from pdf_report import (
    generate_flood_comparison_pdf,
    generate_road_inspection_pdf,
    generate_complaint_letter_pdf,
)

# Load persistent zones from SQLite instantly
if "zones_data" not in st.session_state:
    st.session_state.zones_data = load_zones_fast()

zones: list[ZoneRecord] = st.session_state.zones_data
worst_zone = zones[0] if zones else None

# --- HERO SECTION (BRANDED WITH ROUTIO LOGO & SOFT CREAM ACCENTS) ---
logo_b64 = get_logo_base64()
st.markdown(
    f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="data:image/png;base64,{logo_b64}" class="header-logo-img" alt="Routio Logo">
            <div>
                <div class="hero-title" style="margin: 0; line-height: 1.1;">Routio</div>
                <div class="hero-subtitle" style="margin: 4px 0 10px 0;">
                    Autonomous Road-Health Spatial Mapping & Civil Hazard Defense System
                </div>
                <div>
                    <span class="tag-pill tag-live">● LIVE SENSING</span>
                    <span class="tag-pill tag-gemma">Google Gemma 4 Multimodal Vision & Reasoning</span>
                    <span class="tag-pill tag-db">1.4 km Ward Radius</span>
                </div>
            </div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 8px;">
            <div style="background: rgba(41, 37, 36, 0.85); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(254, 240, 138, 0.25); font-size: 0.78rem; color: #d6d3d1; text-align: right;">
                <strong style="color: #fef08a;">Database:</strong> SQLite Persistent Layer (<code>roadzone.db</code>)<br>
                <strong style="color: #fef08a;">Catchment Radius:</strong> 1.4 km per Sector
            </div>
        </div>
    </div>
</div>
    """,
    unsafe_allow_html=True,
)

# --- KPI METRICS STRIP ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

total_zones = len(zones)
critical_count = sum(1 for z in zones if z.color == "red")
total_potholes = sum(z.pothole_count for z in zones)
max_flood = max((z.flood_severity for z in zones), default=0.0)

with col_m1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Monitored Zones</div>
        <div class="metric-val" style="color: #fef08a;">{total_zones} Sectors</div>
        <div style="font-size: 0.75rem; color: #d6d3d1;">Continuous Telemetry Ingestion</div>
    </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-lbl">Critical Hazard Zones</div>
        <div class="metric-val" style="color: #f87171;">{critical_count} Red</div>
        <div style="font-size: 0.75rem; color: #fecdd3;">Severe Craters & Flooding</div>
    </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-lbl">Potholes Detected</div>
        <div class="metric-val" style="color: #fde047;">{total_potholes} Craters</div>
        <div style="font-size: 0.75rem; color: #d6d3d1;">Google Gemma 4 Vision</div>
    </div>
        """,
        unsafe_allow_html=True,
    )

with col_m4:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-lbl">Peak Flood Severity</div>
        <div class="metric-val" style="color: #fefce8;">{max_flood}/10</div>
        <div style="font-size: 0.75rem; color: #d6d3d1;">Gemma 4 Multimodal Index</div>
    </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

with st.expander("🧠 WHERE IS GOOGLE GEMMA 4 USED IN THIS SYSTEM? (Click to view judging criteria)", expanded=True):
    st.markdown(
        """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; font-size: 0.83rem; margin-bottom: 4px;">
        <div style="background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.35); border-radius: 8px; padding: 12px;">
            <strong style="color: #c084fc;">1. Multimodal Road Verification (detect.py)</strong><br>
            <span style="color: #cbd5e1;">Gemma 4 verifies genuine outdoor road surfaces vs non-road UI screenshots, eliminating false positives and pinpointing exact crater coordinates.</span>
        </div>
        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 8px; padding: 12px;">
            <strong style="color: #38bdf8;">2. Multimodal Flood Severity (flood.py)</strong><br>
            <span style="color: #cbd5e1;">Analyzes standing water depth, lane submergence, and drainage overflow, returning a structured JSON 0–10 severity index.</span>
        </div>
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; padding: 12px;">
            <strong style="color: #34d399;">3. Flood Delta Comparison (flood_compare.py)</strong><br>
            <span style="color: #cbd5e1;">Compares baseline dry roads vs flooded roads to calculate flood severity spikes and vehicle transit passability.</span>
        </div>
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; padding: 12px;">
            <strong style="color: #fbbf24;">4. High-Thinking Legal Notice (complaint.py)</strong><br>
            <span style="color: #cbd5e1;">Gemma 4 Extended Thinking Mode synthesizes sector telemetry to draft formal statutory municipal grievance letters.</span>
        </div>
    </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# --- TABS FOR WORKFLOW ---
tab_map, tab_red_zone, tab_detect, tab_flood_compare, tab_letter, tab_database = st.tabs([
    "🗺️ Spatial Health Map & Leaderboard",
    "🚨 Red Zone Tactical Map & Incident Feed",
    "📸 Live On-the-Fly Road Inspector",
    "🌊 Flood Road Comparison Engine",
    "⚖️ Auto-Municipal Complaint Generator",
    "🗄️ Database & Dataset Explorer",
])

# ==========================================
# TAB 1: SPATIAL MAP & LEADERBOARD
# ==========================================
with tab_map:
    col_map_view, col_board = st.columns([1.8, 1.2], gap="medium")

    with col_map_view:
        st.subheader("📍 Real-Time Zone Health Folium Map")
        st.caption("🔴 Red = Critical Hazard (Top Third) | 🟡 Orange = Moderate Wear | 🟢 Green = Clean / Resurfaced (Auto-fit viewport)")
        folium_map = build_folium_map(zones)
        st_folium(folium_map, width="100%", height=530)

    with col_board:
        st.subheader("🏆 Top-3 Worst Hazardous Zones")
        st.caption("Ranked by Combined Score = Pothole Score + (1.5 × Flood Severity)")

        for z in zones[:3]:
            card_class = f"zone-{z.color}"
            badge_color = "#ef4444" if z.color == "red" else "#f59e0b"

            st.markdown(
                f"""
            <div class="zone-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.05rem; color: #f3f4f6;">
                        #{z.rank} {z.name}
                    </span>
                    <span style="background: {badge_color}22; color: {badge_color}; font-weight: 700; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; border: 1px solid {badge_color}44;">
                        RISK SCORE: {z.final_score}
                    </span>
                </div>
                <div style="color: #94a3b8; font-size: 0.8rem; margin: 4px 0 8px 0;">
                    {z.description}
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 8px 10px; border-radius: 6px; margin-bottom: 6px; font-size: 0.82rem;">
                    🕳️ <strong style="color: #f87171;">{z.pothole_count} Potholes Identified</strong> across this sector<br>
                    🌊 Flood Severity: <strong style="color: #38bdf8;">{z.flood_severity}/10</strong> (Gemma 4) • Damage Area: <strong>{z.avg_area_ratio:.1%}</strong>
                </div>
            </div>
                """,
                unsafe_allow_html=True,
            )


        st.write("")
        if st.button("🔄 Re-sample & Re-evaluate Zones (Random Seed)"):
            new_seed = random.randint(10, 999)
            new_zones = initialize_zones(seed=new_seed)
            st.session_state.zones_data = evaluate_all_zones(new_zones, use_gemma_flood=False)
            st.rerun()

# ==========================================
# TAB 2: RED ZONE TACTICAL MAP & INCIDENT FEED
# ==========================================
with tab_red_zone:
    st.subheader("🚨 Critical Red Zone Tactical Deep-Dive")
    st.caption("High-resolution spatial telemetry displaying exact pothole craters, waterlogged choke points, and citizen reports.")

    red_candidates = [z for z in zones if z.color == "red"] or zones[:2]
    selected_red = st.selectbox(
        "Select High-Risk Sector to Inspect:",
        red_candidates,
        format_func=lambda z: f"🔴 {z.name} (Risk Score: {z.final_score}/25 | {z.pothole_count} Potholes | Flood {z.flood_severity}/10)",
    )

    col_rz_map, col_rz_feed = st.columns([1.7, 1.3], gap="medium")

    with col_rz_map:
        st.markdown(
            f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
            <strong style="color: #f87171; font-size: 1rem;">📍 Tactical Waypoint Map: {selected_red.name}</strong>
            <div style="color: #94a3b8; font-size: 0.8rem;">
                Zoomed into sector perimeter showing verified 🕳️ Pothole Clusters and 🌊 Submerged Choke Points.
            </div>
        </div>
            """,
            unsafe_allow_html=True,
        )
        red_detail_map = build_red_zone_detail_map(selected_red)
        st_folium(red_detail_map, width="100%", height=510, key=f"red_map_{selected_red.id}")

    with col_rz_feed:
        st.markdown("#### 📢 Citizen & Telemetry Incident Feed")
        st.caption(f"Active unresolved civic reports for {selected_red.name}")

        for rep in selected_red.incident_reports:
            status_color = "#ef4444" if "CRITICAL" in rep["severity"] else "#f59e0b"
            st.markdown(
                f"""
            <div style="background: rgba(22, 30, 46, 0.9); border: 1px solid rgba(51, 65, 85, 0.8); border-left: 4px solid {status_color}; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 4px;">
                    <span style="font-weight: 700; color: #f3f4f6;">{rep['id']} • {rep['type']}</span>
                    <span style="color: {status_color}; font-weight: 700; background: {status_color}22; padding: 1px 6px; border-radius: 3px;">
                        {rep['severity']}
                    </span>
                </div>
                <div style="color: #cbd5e1; font-size: 0.82rem; margin-bottom: 6px;">
                    {rep['description']}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #94a3b8; border-top: 1px dashed rgba(51, 65, 85, 0.6); padding-top: 4px;">
                    <span>Source: {rep['reporter']}</span>
                    <span>{rep['timestamp']}</span>
                </div>
            </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("➕ Submit New Citizen Road Incident Report"):
            rep_type = st.selectbox("Hazard Category:", ["Deep Pothole Crater", "Flooded / Waterlogged Lane", "Sewer Cover Displaced", "Two-Wheeler Accident Hazard"])
            rep_desc = st.text_area("Incident Description / Landmark:", placeholder="e.g. 2 ft deep waterlogging near underpass exit, traffic completely stalled.")
            if st.button("📢 Broadcast Report to RoadZone Network", type="primary"):
                if rep_desc:
                    new_rep = {
                        "id": f"REP-NEW{random.randint(100, 999)}",
                        "zone_id": selected_red.id,
                        "type": rep_type,
                        "severity": "CRITICAL",
                        "timestamp": "Just now",
                        "reporter": "Live Citizen Submission (Web)",
                        "description": rep_desc,
                        "status": "VERIFIED & BROADCASTED",
                    }
                    selected_red.incident_reports.insert(0, new_rep)
                    st.success("✅ Incident report logged and mapped to this hazard zone!")
                    st.rerun()

# ==========================================
# TAB 3: LIVE ON-THE-FLY ROAD INSPECTOR
# ==========================================
with tab_detect:
    st.subheader("📸 Live Road Inspection (Google Gemma 4 Multimodal Vision)")
    st.caption("Upload any road photo or pick from the curated testbed to run instant pothole bounding-box detection and flood severity assessment.")

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        input_source = st.radio(
            "Select Inspection Input Mode:",
            ["📤 Upload Custom Photo / Dashcam", "📁 Curated Road Dataset Gallery"],
            horizontal=True,
        )

        chosen_image = None
        chosen_filename = "custom_upload.jpg"

        if input_source == "📤 Upload Custom Photo / Dashcam":
            uploaded_file = st.file_uploader(
                "Upload Road Image (JPG / PNG):",
                type=["jpg", "jpeg", "png", "webp"],
                help="Upload dashcam captures or road photos for real-time Gemma 4 inspection.",
            )
            if uploaded_file is not None:
                chosen_image = Image.open(uploaded_file)
                chosen_filename = uploaded_file.name
        else:
            sample_options = {
                "dataset/images/road_potholes_severe.jpg": "🕳️ Heavy Potholes & Structural Cavities",
                "dataset/images/road_cracked_pavement.jpg": "⚠️ Moderate Asphalt Wear & Fissures",
                "dataset/images/road_dry_clean.jpg": "🟢 Pristine Resurfaced Highway",
                "dataset/flood/flood_street_submerged.jpg": "🌊 Severe Standing Flood (Submerged Traffic)",
                "dataset/flood/flood_urban_waterlogged.jpg": "💧 Urban Waterlogging & Drain Choke",
            }
            selected_path = st.selectbox(
                "Pick Road Scenario:",
                options=list(sample_options.keys()),
                format_func=lambda p: sample_options[p],
            )
            if Path(selected_path).exists():
                chosen_image = Image.open(selected_path)
                chosen_filename = Path(selected_path).name

        inspect_btn = st.button("⚡ Run Instant Multimodal Inspection", type="primary")

    with col_result:
        if inspect_btn and chosen_image:
            with st.spinner("Analyzing road infrastructure with Google Gemma 4 Multimodal Vision..."):
                det_res = detect_potholes(chosen_image)
                flood_res = assess_flood_severity(chosen_image)

            st.image(det_res["annotated_image"], caption="Annotated Road Inspection Visual", use_container_width=True)

            is_road = det_res.get("is_road", True)
            p_cnt = det_res["pothole_count"]
            tot_area = det_res["total_area_ratio"]
            f_sev = flood_res.get("severity", 0)
            is_fl = flood_res.get("flooded", False)
            calc_score = round(p_cnt + (10.0 * tot_area) + (1.5 * f_sev), 2)

            if not is_road:
                st.warning("⚠️ Non-road image detected (e.g. software window / indoor document). Potholes marked as 0.")

            st.markdown(
                f"""
            <div style="background: rgba(38, 35, 30, 0.7); padding: 20px; border-radius: 14px; border: 1px solid rgba(254, 240, 138, 0.22); margin-top: 10px; backdrop-filter: blur(12px);">
                <h4 style="margin: 0 0 12px 0; color: #fef08a; font-weight: 700;">📊 Instant Telemetry Assessment</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.92rem;">
                    <div><span style="color:#d6d3cb;">Detected Potholes:</span> <span style="color: #fecdd3; font-weight: 700;">{p_cnt}</span></div>
                    <div><span style="color:#d6d3cb;">Damage Area Ratio:</span> <span style="color:#fefce8; font-weight:600;">{tot_area:.1%}</span></div>
                    <div><span style="color:#d6d3cb;">Flood Status:</span> <span style="color:#fef08a; font-weight:600;">{'🌊 Flooded' if is_fl else '☀️ Dry / Normal'}</span></div>
                    <div><span style="color:#d6d3cb;">Severity Index:</span> <span style="color: #fde047; font-weight: 700;">{f_sev} / 10</span></div>
                </div>
                <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed rgba(254, 240, 138, 0.25); font-size: 0.95rem; display: flex; justify-content: space-between; align-items: center;">
                    <span style="color:#d6d3cb;">Calculated Hazard Score:</span>
                    <span style="color: #fef08a; font-weight: 800; font-size: 1.35rem;">{calc_score}</span>
                </div>
            </div>
                """,
                unsafe_allow_html=True,
            )

            # Direct PDF Report Download
            insp_pdf = generate_road_inspection_pdf(
                image_name=chosen_filename,
                pothole_count=p_cnt,
                area_ratio=tot_area,
                is_flooded=is_fl,
                flood_severity=f_sev,
                calculated_score=calc_score,
                zone_name="Monitored Sector",
            )
            st.download_button(
                label="📥 Download Official Inspection Assessment (PDF Report)",
                data=insp_pdf,
                file_name=f"Road_Inspection_{chosen_filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.write("")
            st.markdown("##### 💾 Persist Scan to SQLite Database (`roadzone.db`)")
            zone_options = {z.id: f"{z.name} (Current Score: {z.final_score})" for z in zones}
            target_save_zone_id = st.selectbox(
                "Assign this Scan to Sector:",
                options=list(zone_options.keys()),
                format_func=lambda zid: zone_options[zid],
                key="scan_target_zone_save",
            )
            if st.button("📥 Commit Scan to Database & Recalculate Sector Score", type="primary"):
                db_res = record_scan_in_db(
                    zone_id=target_save_zone_id,
                    image_name=chosen_filename,
                    pothole_count=p_cnt,
                    area_ratio=tot_area,
                    is_flooded=is_fl,
                    flood_severity=f_sev,
                )
                st.session_state.zones_data = load_zones_fast()
                st.success(f"✅ Scan saved to SQLite DB! Sector score updated to {db_res['new_zone_final_score']}. Map recalculated!")
                st.rerun()

        elif chosen_image:
            st.image(chosen_image, caption="Selected Image Preview", use_container_width=True)
        else:
            st.info("Pick a curated sample or upload an image on the left to run inspection.")

# ==========================================
# TAB 4: FLOOD ROAD COMPARISON ENGINE
# ==========================================
with tab_flood_compare:
    st.subheader("🌊 Multimodal Flood Road Comparison Engine")
    st.caption("Upload road surface captures before & after rainfall to detect waterlogging spikes, lane submergence, and impassable transit choke points using Google Gemma 4.")

    col_fc1, col_fc2 = st.columns(2, gap="medium")

    img_a = None
    img_b = None
    name_a = ""
    name_b = ""

    with col_fc1:
        st.markdown(
            """
            <div class="soft-card-cream">
                <span class="badge-cream">☀️ Baseline Road (Dry Condition)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_fc_a = st.file_uploader(
            "Upload Baseline Dry Road Image:",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_fc_a",
            help="Upload a photo of the dry, pre-event road surface",
        )
        if uploaded_fc_a is not None:
            img_a = Image.open(uploaded_fc_a)
            name_a = uploaded_fc_a.name
            st.image(img_a, caption=f"Uploaded Baseline: {uploaded_fc_a.name}", use_container_width=True)
        else:
            st.markdown(
                """
                <div style="border: 2px dashed rgba(254, 240, 138, 0.35); border-radius: 12px; padding: 36px 18px; text-align: center; background: rgba(38, 35, 30, 0.4); margin-bottom: 12px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">📷</div>
                    <strong style="color: #fef08a; font-size: 0.95rem;">Upload Baseline Road Image</strong>
                    <div style="font-size: 0.8rem; color: #a8a29e; margin-top: 4px;">Drag & drop or browse a dry roadway photo (JPG, PNG)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("📁 Or test with a sample baseline photo"):
                sample_a = st.selectbox(
                    "Choose sample:",
                    [
                        "dataset/images/road_dry_clean.jpg (Pristine Paved Highway)",
                        "dataset/images/road_dry_arterial.jpg (Dry Arterial Corridor)",
                        "dataset/images/road_cracked_pavement.jpg (Dry Pre-Event Cracked Road)",
                    ],
                    key="sample_select_a",
                )
                if st.button("Use Sample Baseline", key="btn_use_sample_a"):
                    st.session_state["active_sample_a"] = sample_a.split(" ")[0]

            if st.session_state.get("active_sample_a") and uploaded_fc_a is None:
                img_a = Image.open(st.session_state["active_sample_a"])
                name_a = Path(st.session_state["active_sample_a"]).name
                st.image(img_a, caption=f"Selected Sample: {name_a}", use_container_width=True)

    with col_fc2:
        st.markdown(
            """
            <div class="soft-card-rose">
                <span class="badge-rose">🌊 Inundated Road (Flooded Condition)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_fc_b = st.file_uploader(
            "Upload Flooded Target Image:",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_fc_b",
            help="Upload a photo of the waterlogged / inundated road surface",
        )
        if uploaded_fc_b is not None:
            img_b = Image.open(uploaded_fc_b)
            name_b = uploaded_fc_b.name
            st.image(img_b, caption=f"Uploaded Flooded Target: {uploaded_fc_b.name}", use_container_width=True)
        else:
            st.markdown(
                """
                <div style="border: 2px dashed rgba(254, 205, 211, 0.35); border-radius: 12px; padding: 36px 18px; text-align: center; background: rgba(38, 35, 30, 0.4); margin-bottom: 12px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">🌊</div>
                    <strong style="color: #fecdd3; font-size: 0.95rem;">Upload Flooded Road Image</strong>
                    <div style="font-size: 0.8rem; color: #a8a29e; margin-top: 4px;">Drag & drop or browse a waterlogged road photo (JPG, PNG)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("📁 Or test with a sample flooded photo"):
                sample_b = st.selectbox(
                    "Choose sample:",
                    [
                        "dataset/flood/flood_street_submerged.jpg (Submerged Roadway - Stranded Vehicles)",
                        "dataset/flood/flood_urban_waterlogged.jpg (Urban Street Waterlogged)",
                        "dataset/flood/flood_junction_deep.jpg (Deep Standing Water at Major Junction)",
                        "dataset/flood/flood_heavy_torrent.jpg (Torrential Flash Flood Overflow)",
                    ],
                    key="sample_select_b",
                )
                if st.button("Use Sample Flooded", key="btn_use_sample_b"):
                    st.session_state["active_sample_b"] = sample_b.split(" ")[0]

            if st.session_state.get("active_sample_b") and uploaded_fc_b is None:
                img_b = Image.open(st.session_state["active_sample_b"])
                name_b = Path(st.session_state["active_sample_b"]).name
                st.image(img_b, caption=f"Selected Sample: {name_b}", use_container_width=True)

    st.write("")
    if img_a and img_b:
        if st.button("⚡ Run Side-by-Side Flood Delta Analysis with Google Gemma 4", type="primary"):
            with st.spinner("Google Gemma 4 is comparing hydrological telemetry across both road surfaces..."):
                st.session_state["flood_comp_data"] = compare_flood_images(img_a, img_b)

        if "flood_comp_data" in st.session_state and st.session_state["flood_comp_data"]:
            comp_res = st.session_state["flood_comp_data"]

            col_c_res1, col_c_res2, col_c_res3 = st.columns(3)
            with col_c_res1:
                st.metric("Baseline Severity", f"{comp_res.get('image_a', {}).get('severity', 0)}/10")
            with col_c_res2:
                st.metric("Flooded Severity", f"{comp_res.get('image_b', {}).get('severity', 8)}/10")
            with col_c_res3:
                delta = comp_res.get("severity_delta", 8.0)
                st.metric("Flood Severity Delta", f"+{delta}", delta_color="inverse")

            st.markdown(
                f"""
            <div style="background: rgba(38, 35, 30, 0.7); border: 1px solid rgba(254, 240, 138, 0.22); border-radius: 14px; padding: 22px; margin-top: 16px; backdrop-filter: blur(14px);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                    <h4 style="margin: 0; color: #fef08a; font-weight: 700;">🌊 Google Gemma 4 Hydrological Diagnostic</h4>
                    <span style="background: rgba(254, 205, 211, 0.16); color: #fecdd3; border: 1px solid rgba(254, 205, 211, 0.35); padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem;">
                        TRANSIT STATUS: {comp_res.get('transit_status', 'CAUTION')}
                    </span>
                </div>
                <p style="font-size: 0.92rem; color: #fefce8; margin-bottom: 14px; line-height: 1.6;">
                    <strong style="color: #fde047;">Drainage Assessment:</strong> {comp_res.get('drainage_assessment', '')}
                </p>
                <div style="background: rgba(24, 22, 19, 0.75); padding: 14px; border-radius: 10px; border-left: 4px solid #facc15; font-size: 0.88rem; color: #e7e5e4; line-height: 1.5;">
                    <strong style="color: #fef08a;">Recommended Municipal Action:</strong> {comp_res.get('action_required', 'Deploy dewatering pumps and clear drainage grates.')}
                </div>
            </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            # PDF Generation
            pdf_bytes = generate_flood_comparison_pdf(
                comp_res,
                img_a=img_a,
                img_b=img_b,
            )
            st.download_button(
                label="📥 Download Official Flood Comparison Assessment (PDF Report)",
                data=pdf_bytes,
                file_name=f"Routio_Flood_Delta_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.info("👆 Please upload both a Baseline Road Image and a Flooded Road Image above to enable Gemma 4 side-by-side comparison.")

# ==========================================
# TAB 5: MUNICIPAL COMPLAINT GENERATOR
# ==========================================
with tab_letter:
    st.subheader("⚖️ Formal Municipal Grievance & Complaint Generator")
    st.caption("Powered by Google Gemma 4 Extended Thinking Mode for legal, civic, and road safety advocacy.")

    zone_names = [f"#{z.rank} - {z.name} (Score: {z.final_score})" for z in zones]
    selected_zone_str = st.selectbox("Select Target Sector for Official Complaint Notice:", zone_names, index=0)

    selected_rank = int(selected_zone_str.split(" - ")[0].replace("#", ""))
    target_zone_obj = next((z for z in zones if z.rank == selected_rank), worst_zone)

    col_target, col_gen = st.columns([1, 2], gap="large")

    with col_target:
        st.markdown(
            f"""
        <div style="background: rgba(22, 30, 46, 0.85); padding: 18px; border-radius: 12px; border: 1px solid rgba(51, 65, 85, 0.8);">
            <h4 style="color: #ef4444; margin-top: 0;">Target: {target_zone_obj.name}</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">{target_zone_obj.description}</p>
            <hr style="border-color: rgba(51, 65, 85, 0.8);">
            <ul style="font-size: 0.85rem; padding-left: 20px; color: #cbd5e1;">
                <li>Hazard Score: <strong>{target_zone_obj.final_score}/25</strong></li>
                <li>Pothole Count: <strong>{target_zone_obj.pothole_count}</strong></li>
                <li>Flood Severity: <strong>{target_zone_obj.flood_severity}/10</strong></li>
                <li>Surface Area Damage: <strong>{target_zone_obj.avg_area_ratio:.1%}</strong></li>
            </ul>
        </div>
            """,
            unsafe_allow_html=True,
        )

        authority_input = st.text_input(
            "Addressed Civic Authority:",
            value="Municipal Commissioner & Chief Engineer (PWD Road Safety Division)",
        )

        gen_btn = st.button("📝 Draft Legal Complaint with Gemma 4", type="primary")

    with col_gen:
        if "complaint_text" not in st.session_state:
            st.session_state.complaint_text = ""

        if gen_btn:
            with st.spinner("Google Gemma 4 is analyzing civil safety statutes and drafting formal notice..."):
                target_dict = {
                    "name": target_zone_obj.name,
                    "rank": target_zone_obj.rank,
                    "final_score": target_zone_obj.final_score,
                    "pothole_count": target_zone_obj.pothole_count,
                    "flood_severity": target_zone_obj.flood_severity,
                    "avg_area_ratio": target_zone_obj.avg_area_ratio,
                    "description": target_zone_obj.description,
                }
                st.session_state.complaint_text = generate_complaint_letter(
                    target_dict,
                    authority_name=authority_input,
                )

        if st.session_state.complaint_text:
            st.success("✅ Formal Grievance Letter Generated via Gemma 4!")
            st.markdown(
                f"""
            <div style="background: rgba(38, 35, 30, 0.85); padding: 22px; border-radius: 12px; border: 1px solid rgba(254, 240, 138, 0.25); max-height: 480px; overflow-y: auto; color: #fefce8; line-height: 1.6;">
            """,
                unsafe_allow_html=True,
            )
            st.markdown(st.session_state.complaint_text)
            st.markdown("</div>", unsafe_allow_html=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download Complaint (.md)",
                    data=st.session_state.complaint_text,
                    file_name=f"municipal_complaint_{target_zone_obj.id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_dl2:
                complaint_pdf = generate_complaint_letter_pdf(
                    st.session_state.complaint_text,
                    zone_name=target_zone_obj.name,
                    authority_name=authority_input,
                )
                st.download_button(
                    label="📄 Download Official Notice (PDF)",
                    data=complaint_pdf,
                    file_name=f"formal_notice_{target_zone_obj.id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.info("Click the button to draft a formal grievance letter for the selected zone.")

# ==========================================
# TAB 6: DATABASE & DATASET EXPLORER
# ==========================================
with tab_database:
    st.subheader("🗄️ RoadZone Database & Dataset Storage Explorer")
    st.caption("Inspect live SQLite tables (`roadzone.db`), raw image datasets, and historical scans.")

    col_storage1, col_storage2 = st.columns(2)
    with col_storage1:
        st.markdown(
            f"""
        <div style="background: rgba(22, 30, 46, 0.85); border: 1px solid rgba(51, 65, 85, 0.8); border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;">
            <strong style="color: #38bdf8;">📁 Persistent SQLite Database:</strong>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">
                {DB_PATH}
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                Stores all zones, historical scans, and citizen grievance reports.
            </div>
        </div>
            """,
            unsafe_allow_html=True,
        )

    with col_storage2:
        dataset_p = Path(__file__).parent / "dataset"
        st.markdown(
            f"""
        <div style="background: rgba(22, 30, 46, 0.85); border: 1px solid rgba(51, 65, 85, 0.8); border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;">
            <strong style="color: #34d399;">📁 Image Datasets Location:</strong>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">
                {dataset_p}/images (Potholes) | {dataset_p}/flood (Floods)
            </div>
            <div style="font-size: 0.75rem; color: #a8a29e; margin-top: 4px;">
                Raw JPEG/PNG road imagery analyzed by Google Gemma 4 Multimodal Vision.
            </div>
        </div>
            """,
            unsafe_allow_html=True,
        )

    db_zones = get_all_zones()
    db_scans = get_all_scans()
    db_reports = get_incident_reports()

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Monitored Sectors Table (`zones`)", "📸 Scans Log Table (`scans`)", "📢 Citizen Reports Table (`incident_reports`)"])

    with sub_tab1:
        if db_zones:
            df_zones = pd.DataFrame(db_zones)
            st.dataframe(df_zones, use_container_width=True)
        else:
            st.info("No zones stored in database yet.")

    with sub_tab2:
        if db_scans:
            df_scans = pd.DataFrame(db_scans)
            st.dataframe(df_scans, use_container_width=True)
            csv_data = df_scans.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Scans to CSV", data=csv_data, file_name="roadzone_scans.csv", mime="text/csv")
        else:
            st.info("No scans recorded yet. Upload and commit a scan in Tab 3 to see records appear here!")

    with sub_tab3:
        if db_reports:
            df_rep = pd.DataFrame(db_reports)
            st.dataframe(df_rep, use_container_width=True)
        else:
            st.info("No reports logged yet.")
