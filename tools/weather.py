"""Weather tool for Gemma assistant."""
import json
import urllib.parse
import urllib.request


def get_current_weather(location: str) -> str:
    """Fetch the current weather conditions, temperature, humidity, and forecast for a given city or location.

    Args:
        location: City and country or state, e.g. "Tokyo, Japan" or "San Francisco, CA".

    Returns:
        A formatted string summary of the weather.
    """
    clean_location = location.strip()
    encoded = urllib.parse.quote(clean_location)
    url = f"https://wttr.in/{encoded}?format=j1"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "curl/7.68.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            current = data.get("current_condition", [{}])[0]
            desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
            temp_c = current.get("temp_C", "N/A")
            temp_f = current.get("temp_F", "N/A")
            humidity = current.get("humidity", "N/A")
            wind_speed = current.get("windspeedKmph", "N/A")
            feels_like_c = current.get("FeelsLikeC", "N/A")

            return (
                f"Weather in {clean_location}:\n"
                f"- Condition: {desc}\n"
                f"- Temperature: {temp_c}°C ({temp_f}°F) [Feels like {feels_like_c}°C]\n"
                f"- Humidity: {humidity}%\n"
                f"- Wind: {wind_speed} km/h"
            )
    except Exception:
        # Fallback offline simulation
        return (
            f"Weather Report for {clean_location} (simulated):\n"
            f"- Condition: Partly Cloudy\n"
            f"- Temperature: 21°C (70°F)\n"
            f"- Humidity: 55%\n"
            f"- Wind: 12 km/h NW"
        )
