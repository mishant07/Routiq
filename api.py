"""FastAPI REST Backend for RoadZone AI.

Provides endpoints for IoT Dashcams, Mobile Apps, and Municipal Integrations.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

load_dotenv()

from database import (
    get_all_zones,
    get_zone_by_id,
    record_scan_in_db,
    get_all_scans,
    add_incident_report,
    get_incident_reports,
    get_hotspots_by_zone,
)
from detect import detect_potholes
from flood import assess_flood_severity
from complaint import generate_complaint_letter

app = FastAPI(
    title="RoadZone AI - REST Backend API",
    description="Backend engine for road hazard spatial mapping, civil defense, and Google Gemma 4 multimodal intelligence.",
    version="1.0.0",
)

# Enable CORS for external frontends & mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static asset folders
BASE_DIR = Path(__file__).parent
app.mount("/dataset", StaticFiles(directory=str(BASE_DIR / "dataset")), name="dataset")
app.mount("/assets", StaticFiles(directory=str(BASE_DIR / "assets")), name="assets")


class IncidentReportRequest(BaseModel):
    zone_id: str
    type: str
    severity: str
    description: str
    reporter: str


class ComplaintRequest(BaseModel):
    authority_name: Optional[str] = "Municipal Commissioner & Public Works Department (PWD)"


@app.get("/")
def root():
    return {
        "project": "Routio AI / RoadZone",
        "description": "Autonomous Road-Health Spatial Hazard Defense System",
        "models": {
            "pothole_vision": f"Google Gemma 4 Vision ({os.getenv('GEMMA_MODEL', 'gemma-4-26b-a4b-it')})",
            "flood_and_complaints": f"Google Gemma 4 Extended Thinking ({os.getenv('GEMMA_MODEL', 'gemma-4-26b-a4b-it')})",
        },
        "endpoints": {
            "docs": "/docs",
            "zones": "/api/zones",
            "scans": "/api/scans",
            "reports": "/api/reports",
        },
    }



@app.get("/api/zones")
def list_zones():
    """Retrieve all monitored metropolitan zones with live risk scores and colors."""
    zones = get_all_zones()
    return {"count": len(zones), "zones": zones}


@app.get("/api/zones/{zone_id}")
def get_zone_details(zone_id: str):
    """Get full details for a zone including historical scans, hotspots, and citizen grievances."""
    zone = get_zone_by_id(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    scans = get_all_scans(zone_id=zone_id)
    reports = get_incident_reports(zone_id=zone_id)
    hotspots = get_hotspots_by_zone(zone_id)
    return {
        "zone": zone,
        "scans": scans,
        "reports": reports,
        "hotspots": hotspots,
    }


@app.post("/api/scans")
async def process_road_scan(
    zone_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a road image: runs Google Gemma 4 Multimodal road damage & flood analysis,
    persists in SQLite, and updates target zone status in real time.
    """
    zone = get_zone_by_id(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")

    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))

    # 1. Google Gemma 4 road damage & cavity detection
    det_res = detect_potholes(pil_image)
    pothole_count = det_res["pothole_count"]
    area_ratio = det_res["total_area_ratio"]

    # 2. Gemma 4 Flood analysis
    flood_res = assess_flood_severity(pil_image)
    is_flooded = flood_res.get("flooded", False)
    flood_severity = flood_res.get("severity", 0.0)

    # 3. Save to database & recalculate zone
    db_res = record_scan_in_db(
        zone_id=zone_id,
        image_name=file.filename or "upload.jpg",
        pothole_count=pothole_count,
        area_ratio=area_ratio,
        is_flooded=is_flooded,
        flood_severity=flood_severity,
    )

    return {
        "status": "success",
        "detection": {
            "pothole_count": pothole_count,
            "area_ratio": round(area_ratio, 4),
            "boxes_count": len(det_res["boxes"]),
        },
        "flood": {
            "is_flooded": is_flooded,
            "severity": flood_severity,
            "explanation": flood_res.get("explanation", ""),
        },
        "database_update": db_res,
    }


@app.get("/api/reports")
def list_reports(zone_id: Optional[str] = None):
    """List citizen incident reports."""
    reports = get_incident_reports(zone_id)
    return {"count": len(reports), "reports": reports}


@app.post("/api/reports")
def create_report(payload: IncidentReportRequest):
    """Submit a citizen incident report into the database."""
    import random
    rep_dict = {
        "id": f"REP-API{random.randint(100, 999)}",
        "zone_id": payload.zone_id,
        "type": payload.type,
        "severity": payload.severity,
        "description": payload.description,
        "reporter": payload.reporter,
        "status": "UNRESOLVED - ACTIVE DANGER",
    }
    add_incident_report(rep_dict)
    return {"status": "created", "report": rep_dict}


@app.post("/api/complaint/{zone_id}")
def generate_complaint_endpoint(zone_id: str, payload: ComplaintRequest):
    """Generate a formal municipal complaint letter using Gemma 4 for a specific zone."""
    zone = get_zone_by_id(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")

    letter = generate_complaint_letter(
        zone_data=zone,
        authority_name=payload.authority_name or "Municipal Commissioner",
    )
    return {
        "zone_id": zone_id,
        "zone_name": zone["name"],
        "complaint_letter": letter,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
