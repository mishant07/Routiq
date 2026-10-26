#!/usr/bin/env python3
"""Interactive Rich CLI for Gemma 4 with Tool Calling and Thinking Mode."""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# Load environment variables
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from agent import GemmaAgent
from tools import AVAILABLE_TOOLS

console = Console()


def print_banner(agent: GemmaAgent):
    """Render welcome header banner."""
    banner_text = Text()
    banner_text.append("✦ GEMMA 4 SMART ASSISTANT ✦\n", style="bold cyan")
    banner_text.append("Powered by Google GenAI API • Tool Calling • Thinking Process\n\n", style="italic")
    banner_text.append(f"• Active Model:    ", style="dim")
    banner_text.append(f"{agent.model_name}\n", style="bold green")
    banner_text.append(f"• Thinking Mode:   ", style="dim")
    thinking_style = "bold magenta" if agent.thinking_level == "high" else "dim yellow"
    banner_text.append(f"{agent.thinking_level.upper()}\n", style=thinking_style)
    banner_text.append(f"• Registered Tools: ", style="dim")
    banner_text.append(f"{len(AVAILABLE_TOOLS)} loaded\n", style="bold blue")
    banner_text.append("Type your question or use /help for available commands.", style="dim")

    console.print(Panel(banner_text, border_style="cyan", title="[bold white]Google Gemma[/bold white]", subtitle="[dim]CLI Interface[/dim]"))


def print_tools_table():
    """Print table of all available tools."""
    table = Table(title="Available Function Calling Tools", border_style="blue")
    table.add_column("Tool Name", style="bold cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for tool in AVAILABLE_TOOLS:
        doc = (tool.__doc__ or "No description").strip().split("\n")[0]
        table.add_row(tool.__name__, doc)

    console.print(table)


def print_help():
    """Print help information."""
    help_table = Table(title="CLI Commands", border_style="green")
    help_table.add_column("Command", style="bold yellow")
    help_table.add_column("Description", style="white")

    help_table.add_row("/help", "Show this help table")
    help_table.add_row("/tools", "List all registered function-calling tools")
    help_table.add_row("/thinking [on|off]", "Toggle Gemma's high-reasoning thinking process")
    help_table.add_row("/model [name]", "Switch model (gemma-4-26b-a4b-it or gemma-4-31b-it)")
    help_table.add_row("/history", "View chat turns in current session")
    help_table.add_row("/clear", "Clear session memory and start fresh")
    help_table.add_row("/key", "Update Gemini API key")
    help_table.add_row("/exit, /quit", "Quit the assistant")

    console.print(help_table)


def tool_execution_callback(tool_name: str, args: dict, result: any):
    """Render real-time tool invocation in Rich console."""
    if result is None:
        # Tool call started
        args_str = ", ".join(f"{k}='{v}'" for k, v in args.items())
        console.print(
            Panel(
                f"[bold yellow]Invoking Tool:[/bold yellow] [cyan]{tool_name}[/cyan]({args_str})",
                border_style="yellow",
                title="[bold yellow]⚙ Tool Call[/bold yellow]",
            )
        )
    else:
        # Tool call completed
        result_preview = str(result)
        if len(result_preview) > 300:
            result_preview = result_preview[:300] + "... [truncated]"
        console.print(
            Panel(
                f"[bold green]Result:[/bold green]\n{result_preview}",
                border_style="green",
                title=f"[bold green]✓ {tool_name} Output[/bold green]",
            )
        )


def check_api_key(agent: GemmaAgent):
    """Ensure API key is available or prompt user."""
    if not agent.api_key:
        console.print(
            Panel(
                "[bold yellow]No GEMINI_API_KEY detected![/bold yellow]\n\n"
                "To use Gemma via the Gemini API, you need a free key from Google AI Studio:\n"
                "[link=https://aistudio.google.com/apikey]https://aistudio.google.com/apikey[/link]",
                border_style="yellow",
                title="[bold red]API Key Required[/bold red]",
            )
        )
        key = Prompt.ask("[bold cyan]Enter your Gemini API Key[/bold cyan]")
        if key.strip():
            agent.update_api_key(key.strip())
            # Offer to save to .env
            save = Prompt.ask("Save this key to .env for future runs?", choices=["y", "n"], default="y")
            if save == "y":
                try:
                    with open(ENV_PATH, "w", encoding="utf-8") as f:
                        f.write(f"GEMINI_API_KEY={key.strip()}\n")
                        f.write(f"GEMMA_MODEL={agent.model_name}\n")
                        f.write(f"GEMMA_THINKING_LEVEL={agent.thinking_level}\n")
                    console.print("[green]✓ Key saved to .env[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to save .env: {e}[/red]")
        else:
            console.print("[red]Cannot proceed without API key. Exiting.[/red]")
            sys.exit(1)


def handle_command(cmd: str, agent: GemmaAgent) -> bool:
    """Handle slash commands. Returns True if handled, False otherwise."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ["/exit", "/quit"]:
        console.print("[bold yellow]Goodbye! Happy building with Gemma![/bold yellow]")
        sys.exit(0)

    elif command == "/help":
        print_help()
        return True

    elif command == "/tools":
        print_tools_table()
        return True

    elif command == "/clear":
        agent.reset_chat()
        console.print("[green]✓ Chat session and memory cleared.[/green]")
        return True

    elif command == "/thinking":
        if len(parts) > 1:
            arg = parts[1].lower()
            if arg in ["on", "high", "true", "yes"]:
                agent.set_thinking_level("high")
                console.print("[bold magenta]✓ Gemma Thinking Mode enabled (level: high)[/bold magenta]")
            elif arg in ["off", "minimal", "false", "no"]:
                agent.set_thinking_level("minimal")
                console.print("[yellow]✓ Gemma Thinking Mode disabled (level: minimal)[/yellow]")
            else:
                console.print("[red]Usage: /thinking [on|off][/red]")
        else:
            current = agent.thinking_level
            console.print(f"Current thinking mode: [bold magenta]{current.upper()}[/bold magenta]")
        return True

    elif command == "/model":
        if len(parts) > 1:
            new_model = parts[1]
            agent.set_model(new_model)
            console.print(f"[bold green]✓ Switched model to: {new_model}[/bold green]")
        else:
            console.print(f"Current model: [bold green]{agent.model_name}[/bold green]")
            console.print("Supported models: [cyan]gemma-4-26b-a4b-it[/cyan], [cyan]gemma-4-31b-it[/cyan]")
        return True

    elif command == "/history":
        turns = agent.get_history_summary()
        if not turns:
            console.print("[dim]No messages in current history.[/dim]")
        else:
            table = Table(title="Conversation History", border_style="magenta")
            table.add_column("Turn", style="dim", width=6)
            table.add_column("Role", style="bold cyan", width=10)
            table.add_column("Content", style="white")
            for idx, turn in enumerate(turns):
                table.add_row(str(idx + 1), turn["role"].capitalize(), turn["text"])
            console.print(table)
        return True

    elif command == "/key":
        new_key = Prompt.ask("[bold cyan]Enter new Gemini API Key[/bold cyan]")
        if new_key.strip():
            agent.update_api_key(new_key.strip())
            console.print("[green]✓ API key updated successfully.[/green]")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Gemma Smart Assistant CLI")
    parser.add_argument("query", nargs="*", help="Optional single query to execute directly")
    parser.add_argument("--model", default=os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it"), help="Gemma model name")
    parser.add_argument("--thinking", default=os.getenv("GEMMA_THINKING_LEVEL", "high"), choices=["high", "minimal"], help="Thinking level")
    args = parser.parse_args()

    agent = GemmaAgent(
        model_name=args.model,
        thinking_level=args.thinking,
        tool_callback=tool_execution_callback,
    )

    check_api_key(agent)

    # One-shot mode if query is passed
    if args.query:
        query_text = " ".join(args.query)
        console.print(f"[bold cyan]User:[/bold cyan] {query_text}")
        with console.status("[bold green]Gemma is processing & thinking...[/bold green]", spinner="dots"):
            try:
                response = agent.send_message(query_text)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                return
        console.print(Panel(Markdown(response), border_style="cyan", title="[bold cyan]Gemma[/bold cyan]"))
        return

    # Interactive REPL mode
    print_banner(agent)

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                handled = handle_command(user_input, agent)
                if handled:
                    continue

            with console.status("[bold green]Gemma is thinking & evaluating tools...[/bold green]", spinner="dots"):
                try:
                    response_text = agent.send_message(user_input)
                except Exception as e:
                    console.print(f"[bold red]API Error:[/bold red] {e}")
                    continue

            console.print(
                Panel(
                    Markdown(response_text),
                    border_style="cyan",
                    title="[bold cyan]Gemma 4[/bold cyan]",
                    subtitle=f"[dim]{agent.model_name} • thinking: {agent.thinking_level}[/dim]",
                )
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting Gemma CLI. Goodbye![/yellow]")
            break


if __name__ == "__main__":
    main()
