"""Persistent notes and task manager tool."""
import json
from pathlib import Path

NOTES_FILE = Path.home() / ".gemma_cli_notes.json"


def _load_notes() -> list[str]:
    if not NOTES_FILE.exists():
        return []
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_notes(notes: list[str]) -> None:
    try:
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
    except Exception:
        pass


def manage_notes(action: str, content: str = "") -> str:
    """Manage user notes and todos. Supports actions: 'add', 'list', 'delete', 'clear'.

    Args:
        action: One of 'add', 'list', 'delete', 'clear'.
        content: The text of the note to add, or the index (1-based) of the note to delete.

    Returns:
        Result description of the operation.
    """
    action_lower = action.strip().lower()
    notes = _load_notes()

    if action_lower == "add":
        if not content.strip():
            return "Error: Note content cannot be empty."
        notes.append(content.strip())
        _save_notes(notes)
        return f"Note added successfully: '{content.strip()}' (Total notes: {len(notes)})"

    elif action_lower == "list":
        if not notes:
            return "No notes found in your notebook."
        formatted = "\n".join(f"{i + 1}. {note}" for i, note in enumerate(notes))
        return f"Your Saved Notes ({len(notes)} items):\n{formatted}"

    elif action_lower == "delete":
        try:
            idx = int(content.strip()) - 1
            if 0 <= idx < len(notes):
                removed = notes.pop(idx)
                _save_notes(notes)
                return f"Deleted note #{idx + 1}: '{removed}'"
            return f"Error: Note index #{content} out of range (1 to {len(notes)})."
        except ValueError:
            return f"Error: '{content}' is not a valid numerical note index."

    elif action_lower == "clear":
        count = len(notes)
        _save_notes([])
        return f"Cleared all {count} notes from notebook."

    else:
        return f"Unknown action: '{action}'. Supported actions are: add, list, delete, clear."
