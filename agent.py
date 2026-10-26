"""Gemma Assistant Agent managing conversation, thinking mode, and tool execution."""
import functools
import inspect
import os
from typing import Any, Callable, Dict, List, Optional
from google import genai
from google.genai import types
from tools import AVAILABLE_TOOLS, TOOL_MAP


class GemmaAgent:
    """Agent that communicates with Gemma models using Gemini API SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemma-4-26b-a4b-it",
        thinking_level: str = "high",
        tool_callback: Optional[Callable[[str, dict, Any], None]] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.thinking_level = thinking_level  # "high" or "minimal"
        self.tool_callback = tool_callback
        self.client: Optional[genai.Client] = None
        self.chat = None
        self.system_instruction = (
            "You are Gemma Assistant, a powerful and thoughtful AI companion powered by Google Gemma 4. "
            "You have access to a variety of tools: get_current_weather, calculate_expression, "
            "get_system_status, and manage_notes. Always choose the appropriate tool when user queries "
            "involve math, current weather, system stats, or user notes/todos. "
            "Explain your reasoning concisely, accurately, and politely."
        )

        if self.api_key:
            self._init_client()

    def _init_client(self):
        """Initialize the Google GenAI client."""
        self.client = genai.Client(api_key=self.api_key)
        self.reset_chat()

    def update_api_key(self, api_key: str):
        """Update API key and reinitialize client."""
        self.api_key = api_key.strip()
        os.environ["GEMINI_API_KEY"] = self.api_key
        self._init_client()

    def _wrap_tool(self, func):
        """Wrap tool functions to intercept tool calls for CLI visual feedback while preserving signature."""
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                call_args = bound.arguments
            except Exception:
                call_args = kwargs

            if self.tool_callback:
                self.tool_callback(func.__name__, call_args, None)
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                result = f"Tool error: {str(e)}"
            if self.tool_callback:
                self.tool_callback(func.__name__, call_args, result)
            return result

        return wrapped

    def reset_chat(self):
        """Create or reset the multi-turn chat session with tools and thinking configuration."""
        if not self.client:
            return

        wrapped_tools = [self._wrap_tool(t) for t in AVAILABLE_TOOLS]

        thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level)
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=wrapped_tools,
            thinking_config=thinking_config,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=5
            ),
        )

        self.chat = self.client.chats.create(
            model=self.model_name,
            config=config,
        )

    def set_thinking_level(self, level: str):
        """Set thinking level to 'high' or 'minimal'."""
        if level not in ["high", "minimal"]:
            raise ValueError("Thinking level must be 'high' or 'minimal'")
        self.thinking_level = level
        self.reset_chat()

    def set_model(self, model_name: str):
        """Change the active Gemma model."""
        self.model_name = model_name
        self.reset_chat()

    def send_message(self, prompt: str) -> str:
        """Send a message to Gemma and return the final generated response text.

        Args:
            prompt: User message text.

        Returns:
            The model's text response.
        """
        if not self.client or not self.chat:
            raise ValueError("API key not set. Please provide your Gemini API key.")

        response = self.chat.send_message(prompt)
        if response.text:
            return response.text

        # Extract text from candidate parts if any
        collected_texts = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            collected_texts.append(part.text)
        if collected_texts:
            return "\n".join(collected_texts)

        return "Task completed."

    def get_history_summary(self) -> List[Dict[str, str]]:
        """Return a simplified summary of turns in the current chat."""
        if not self.chat:
            return []
        turns = []
        for content in getattr(self.chat, "_curated_history", []):
            role = content.role
            parts_text = []
            for part in getattr(content, "parts", []):
                if hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    parts_text.append(f"[Called tool: {part.function_call.name}]")
                elif hasattr(part, "function_response") and part.function_response:
                    parts_text.append(f"[Tool response: {part.function_response.name}]")
            if parts_text:
                turns.append({"role": role, "text": "\n".join(parts_text)})
        return turns
