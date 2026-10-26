"""Tools package for Gemma Assistant."""
from .weather import get_current_weather
from .calculator import calculate_expression
from .system_info import get_system_status
from .notes_manager import manage_notes

# List of callable tools provided to Gemma
AVAILABLE_TOOLS = [
    get_current_weather,
    calculate_expression,
    get_system_status,
    manage_notes,
]

TOOL_MAP = {func.__name__: func for func in AVAILABLE_TOOLS}
