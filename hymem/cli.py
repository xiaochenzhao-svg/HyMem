#!/usr/bin/env python3
"""
HyMem command-line chat interface.

Entry points:
    python -m hymem.cli                      # uses ./config.json
    python -m hymem.cli --config my.json
    python -m hymem.cli --persist mem_dir    # persist memories across runs
    python -m hymem.cli --verbose            # debug logging

Slash commands:
    /quit /exit /bye    end the conversation (archives current session)
    /new                start a fresh conversation (current session is archived)
    /clear              wipe ALL memories
    /status             show memory statistics
    /save <dir>         save memories to a directory
    /load <dir>         load memories from a directory
"""

import os
import sys
import logging
import argparse

from hymem.agent import HyMemAgent
from hymem.config.settings import ConfigManager

BANNER = """\
==================================================
  🧠  HyMem - Hybrid Memory Conversational Agent
==================================================
Type your message and press Enter to chat.
Slash commands: /new  /clear  /status  /save <dir>  /load <dir>  /quit
"""


def _clear_line():
    """Clear the previous in-place status line (handles wide emoji)."""
    print("\r" + " " * 80 + "\r", end="", flush=True)


class HyMemCLI:
    def __init__(self, config_path: str, persist_dir: str = "", verbose: bool = False):
        if verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )

        self.persist_dir = persist_dir
        cfg = ConfigManager()

        if config_path and os.path.exists(config_path):
            print(f"📁 Loading configuration from {config_path}")
            settings = cfg.load_from_file(config_path)
        else:
            print("🔧 No config file found, falling back to environment variables")
            settings = cfg.load_from_env()

        if not settings.llm.api_key:
            raise RuntimeError(
                "No LLM API key found. Either set 'llm.api_key' in your config "
                "file or export OPENAI_API_KEY."
            )

        print("🤖 Initializing agent...")
        self.agent = HyMemAgent.from_settings(settings)

        if self.persist_dir and os.path.isdir(self.persist_dir):
            try:
                self.agent.load_memories(self.persist_dir)
                print(f"💾 Loaded persisted memories from {self.persist_dir} -> {self.agent}")
            except Exception as e:
                print(f"⚠️  Failed to load memories: {e}")

        self.agent.start_conversation()
        print("✅ Agent ready\n")

    # ------------------------------------------------------------------ #
    # Slash commands
    # ------------------------------------------------------------------ #

    def _handle_command(self, raw: str) -> bool:
        if not raw.startswith("/"):
            return False

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/bye"):
            self._on_exit()
            sys.exit(0)

        elif cmd == "/new":
            self.agent.end_conversation()
            self.agent.start_conversation()
            print("🆕 Started a new conversation. Previous session has been archived to long-term memory.")

        elif cmd == "/clear":
            self.agent.clear_memories()
            self.agent.start_conversation()
            print("🧹 All memories cleared.")

        elif cmd == "/status":
            print(f"📊 {self.agent}")

        elif cmd == "/save":
            target = arg or self.persist_dir or "cached_memories"
            self.agent.save_memories(target)
            print(f"💾 Memories saved to {target}")

        elif cmd == "/load":
            target = arg or self.persist_dir or "cached_memories"
            self.agent.load_memories(target)
            print(f"📥 Memories loaded from {target} -> {self.agent}")

        else:
            print(f"❓ Unknown command: {cmd}")

        return True

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def chat_loop(self):
        print(BANNER)
        while True:
            try:
                user_input = input("👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self._on_exit()
                return

            if not user_input:
                continue

            if self._handle_command(user_input):
                continue

            print("🤔 Thinking...", end="\r", flush=True)
            try:
                answer, _ = self.agent.chat(user_input)
            except Exception as e:
                _clear_line()
                print(f"❌ Error: {e}")
                continue
            _clear_line()
            if not answer or not answer.strip():
                answer = (
                    "(I couldn't generate a reply for that — could you rephrase or add a bit more context?)"
                )
            print(f"🤖 AI : {answer}\n")

    def _on_exit(self):
        try:
            self.agent.end_conversation()
            if self.persist_dir:
                self.agent.save_memories(self.persist_dir)
                print(f"💾 Memories persisted to {self.persist_dir}")
        except Exception as e:
            print(f"⚠️  Failed to persist on exit: {e}")
        print("👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="HyMem chat CLI")
    parser.add_argument("--config", default="config.json", help="Path to JSON config file")
    parser.add_argument(
        "--persist",
        default="",
        help="Directory used to load/save long-term memories across runs",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    try:
        cli = HyMemCLI(args.config, args.persist, args.verbose)
        cli.chat_loop()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()