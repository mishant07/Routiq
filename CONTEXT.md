# CONTEXT.md: RoadZone AI

## PROJECT
**RoadZone AI**: A road-health zone map for potholes and flooded roads.

## GOAL
Take road images/video, detect potholes and flooding, score each area, and show an interactive map where:
- 🔴 **Red** = Worst roads (severe damage/flooding)
- 🟡 **Orange/Yellow** = Moderate damage
- 🟢 **Green** = Minimal damage

Also auto-generate a formal municipal complaint letter for the worst zones.

## HACKATHON CONTEXT
- **Tracks**: DigitalOcean "Best Open-Source AI Project" & "Best Use of Gemma 4"
- **Priority**: Speed and a robust, working demo matter more than polish.

## CONSTRAINTS & STACK
- **Open-source/open-weight models only**:
  - **Google Gemma 4 (`gemma-4-26b-a4b-it`)**:
    - Multimodal Vision: Pothole & cavity bounding-box detection, asphalt surface damage ratio, and road authenticity verification.
    - Multimodal Hydrology: Flood severity assessment (0-10) and side-by-side road inundation delta comparison.
    - Extended Thinking Mode: Formal statutory municipal complaint notices citing telemetry evidence.
- **Python Stack**: Python 3.10+, OpenCV, Folium, Streamlit, google-genai, ReportLab.
- **Environment**: Gemma model name and API key come from env vars: `GEMMA_MODEL`, `GEMINI_API_KEY`. Never hardcode keys.
- **Simulated Coordinates**: Dataset images (RDD2022 / Roboflow pothole set) have NO GPS. Assign each image to a zone with a simulated location, and label this clearly in the UI as *"simulated locations for proof of concept"*.
- **Deployment**: Must be deployable on a DigitalOcean Droplet or App Platform.
- **Execution**: Simple, well-commented, runnable with `streamlit run app.py`.

## DATA PATHS
- `./dataset/images/*.jpg` (pothole images)
- `./dataset/flood/*.jpg` (flood images)

