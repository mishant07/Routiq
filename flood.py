"""Flood severity analysis using Google Gemma 4 multimodal vision."""
from __future__ import annotations

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()
logger = logging.getLogger("flood")
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


def assess_flood_severity(
    image_input: str | Path | Image.Image | bytes,
    model_name: str | None = None,
) -> Dict[str, Any]:
    """Assess if a road is flooded and determine severity using Gemma 4.

    Prompt specification:
        "Is this road flooded? Reply only JSON: {flooded: bool, severity: 0-10}"

    Returns:
        dict: {"flooded": bool, "severity": int, "raw_response": str, "explanation": str}
    """
    model = model_name or os.getenv("GEMMA_MODEL", DEFAULT_MODEL)
    fallback_result = {
        "flooded": False,
        "severity": 0,
        "explanation": "No flooding detected or assessment fallback.",
        "raw_response": "",
    }

    try:
        # Prepare PIL Image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            logger.error(f"Unsupported image input type: {type(image_input)}")
            return fallback_result

        # Resize large images to keep inference fast
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim))

        client = _get_client()

        prompt = (
            "Is this road flooded? Reply only JSON: {\"flooded\": bool, \"severity\": 0-10} "
            "where severity is 0 for dry/normal, 1-3 for small puddles/light waterlogging, "
            "4-7 for moderate standing water covering road lanes, and 8-10 for deep, impassable flood water."
        )

        response = client.models.generate_content(
            model=model,
            contents=[pil_img, prompt],
        )

        response_text = response.text or ""
        logger.info(f"Gemma 4 flood raw response: {response_text.strip()}")

        # Extract JSON safely using regex
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            flooded = bool(data.get("flooded", False))
            severity = int(data.get("severity", 0))
            severity = max(0, min(10, severity))
            return {
                "flooded": flooded,
                "severity": severity,
                "explanation": f"Flooded: {flooded}, Severity: {severity}/10",
                "raw_response": response_text.strip(),
            }
        else:
            # Fallback simple textual heuristic if JSON extraction failed
            is_flooded = "true" in response_text.lower()
            return {
                "flooded": is_flooded,
                "severity": 5 if is_flooded else 0,
                "explanation": response_text.strip(),
                "raw_response": response_text.strip(),
            }

    except Exception as e:
        logger.warning(f"Gemma flood evaluation error: {e}. Using fallback.")
        fallback_result["error"] = str(e)
        return fallback_result


if __name__ == "__main__":
    print("Testing flood.py with synthetic road image...")
    # Create a small blue-ish image representing standing water
    test_img = Image.new("RGB", (300, 200), color=(40, 70, 120))
    result = assess_flood_severity(test_img)
    print("Flood assessment output:", result)
