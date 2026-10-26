"""Municipal complaint letter generation powered by Google Gemma 4."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger("complaint")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


def generate_complaint_letter(
    zone_data: Dict[str, Any],
    authority_name: str = "Chief Engineer, Municipal Corporation & Public Works Department (PWD)",
    model_name: Optional[str] = None,
) -> str:
    """Generate a formal municipal complaint letter using Gemma 4.
    
    Args:
        zone_data: Dictionary containing zone details:
            - name (str)
            - rank (int)
            - final_score (float)
            - pothole_count (int)
            - flood_severity (float)
            - avg_area_ratio (float)
            - description (str)
        authority_name: Target civic authority department.
        model_name: Optional Gemma model override.
        
    Returns:
        Formatted markdown complaint letter string.
    """
    model = model_name or os.getenv("GEMMA_MODEL", DEFAULT_MODEL)
    current_date = datetime.now().strftime("%B %d, %Y")

    name = zone_data.get("name", "Metropolitan Transit Sector")
    pothole_count = zone_data.get("pothole_count", 0)
    flood_severity = zone_data.get("flood_severity", 0.0)
    final_score = zone_data.get("final_score", 0.0)
    avg_area = zone_data.get("avg_area_ratio", 0.0)
    desc = zone_data.get("description", "High-volume arterial road corridor.")

    prompt = f"""
You are an expert civic rights and municipal infrastructure advocate representing citizens and commuters.
Write an urgent, formal, and authoritative Municipal Grievance and Public Safety Complaint Letter regarding severe road hazards detected by RoadZone AI telemetry.

Telemetry Data:
- Date of Inspection: {current_date}
- Target Zone: {name}
- Zone Description: {desc}
- Overall Road Hazard Risk Score: {final_score}/25 (Categorized as CRITICAL HAZARD - Rank #1 worst road)
- Confirmed Pothole Count: {pothole_count} deep structural fissures (Average surface area rupture: {avg_area:.1%})
- Monitored Flood & Waterlogging Severity: {flood_severity}/10 (Dangerous standing water)
- Addressed To: {authority_name}

Instructions:
1. Use an official, professional, and firm tone demanding statutory compliance with urban road safety guidelines.
2. Clearly cite the quantitative telemetry data above to prove systemic neglect.
3. Detail the specific risks: vehicular accidents, two-wheeler skidding fatalities, waterborne contamination, and emergency response delays.
4. Demand immediate 72-hour emergency interventions: deploying pothole cold-mix asphalt patching, unclogging stormwater drains, and erecting reflective caution barriers.
5. Format with proper professional letter layout: Subject line, formal salutation, numbered evidence points, action timeline, and sign-off block from the "RoadZone Citizen Safety Alliance & Commuter Forum".
"""

    try:
        client = _get_client()

        # Call Gemma 4 with Thinking mode
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            system_instruction="You are a legal and civic engineering consultant. Draft formal, rigorous, and legally structured public grievance documents.",
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        if response.text and response.text.strip():
            logger.info("Successfully generated complaint letter with Gemma 4.")
            return response.text.strip()
        else:
            raise ValueError("Empty response received from Gemma model.")

    except Exception as e:
        logger.warning(f"Gemma complaint generation fallback due to: {e}")
        # High quality deterministic template fallback
        return f"""# URGENT CIVIC GRIEVANCE & STATUTORY REPAIR NOTICE

**Date:** {current_date}  
**To:** {authority_name}  
**From:** RoadZone Civic Safety Committee & Citizen Alliance  
**Subject:** Formal Grievance regarding Hazardous Road Conditions & Flooding at **{name}**  
**Classification:** URGENT / HIGH-PRIORITY PUBLIC SAFETY HAZARD  

---

Dear Municipal Commissioner and Public Works Executive,

We are submitting this formal statutory notice under the Public Safety and Municipal Governance Act regarding the alarming deterioration of **{name}** ({desc}).

Automated multimodal sensor telemetry and aerial imaging gathered on **{current_date}** have classified this transit corridor as a **Category-1 Critical Risk Zone (Hazard Score: {final_score}/25)**.

### Verified Infrastructure Telemetry:
1. **Severe Potholes & Structural Fractures:** **{pothole_count}** confirmed structural ruptures with an average surface damage ratio of **{avg_area:.1%}**, posing lethal rollover risks to two-wheelers and severe chassis damage to transit buses.
2. **Critical Waterlogging & Submergence:** Water hazard severity index of **{flood_severity}/10**, where stagnant surface drainage prevents pedestrian passage and masks submerged chasms.

### Demanded Civic Remediation (72-Hour Statutory Timeline):
- **Immediate Cordons:** Erect illuminated, reflective hazard barriers around all active craters exceeding 0.3m depth.
- **Surface Resurfacing:** Deploy rapid-curing bitumen/cold-mix asphalt crews to seal craters.
- **Drainage Clearance:** Desilt municipal stormwater catch basins along this arterial link.

Failure to initiate remedial measures within 72 hours will compel our coalition to file an urgent writ before the District Consumer Disputes & Grievance Redressal Forum for criminal negligence.

Respectfully submitted,

**RoadZone AI Public Safety Task Force**  
*Citizen Infrastructure Oversight Committee*  
*Contact: public-safety@roadzone.org*
"""


if __name__ == "__main__":
    print("Testing complaint.py...")
    sample_zone = {
        "name": "Okhla Industrial Sector 3",
        "rank": 1,
        "final_score": 19.8,
        "pothole_count": 8,
        "flood_severity": 7.5,
        "avg_area_ratio": 0.18,
        "description": "Heavy transit & logistics corridor with chronic deep waterlogging.",
    }
    letter = generate_complaint_letter(sample_zone)
    print("\n--- GENERATED LETTER PREVIEW ---\n")
    print(letter[:400] + "\n...")
    print("\ncomplaint.py test completed.")
