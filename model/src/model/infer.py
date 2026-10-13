"""
Operant GPU Interactive CLI — colorful terminal chat with arrow-key editing.

Usage:
    uv run python -m model.infer --chat
    uv run python -m model.infer --prompt "Hello"
    uv run infer-gpu --chat --model-file out
"""

import argparse
import sys
from .gpu_engine import GPUEngine

# ── ANSI helpers ───────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
ITALIC = "\033[3m"

GREEN  = "\033[38;5;114m"
CYAN   = "\033[38;5;117m"
YELLOW = "\033[38;5;222m"
PURPLE = "\033[38;5;141m"
GRAY   = "\033[38;5;245m"
WHITE  = "\033[38;5;255m"
RED    = "\033[38;5;204m"

BG_DARK = "\033[48;5;234m"


# ── System anchor ────────────────────────────────────────────────────
# Keeps the model in role across turns. Without this, filler replies get fed
# back as history and the session drifts into sycophancy.

SYSTEM_PROMPT = (
    "You are Operant, an agentic AI assistant. "
    "Answer directly and concisely. Do not flatter the user, do not pad "
    "with filler, and say when you do not know. "
    "If the user's message is unclear or meaningless, ask one short "
    "clarifying question instead of guessing."
)


def _banner(device: str, model_name: str):
    print()
    print(f"  {GREEN}{BOLD}╭─────────────────────────────────────────╮{RESET}")
    print(f"  {GREEN}{BOLD}│       ⚡  OPERANT  Interactive Chat     │{RESET}")
    print(f"  {GREEN}{BOLD}╰─────────────────────────────────────────╯{RESET}")
    print()
    print(f"  {GRAY}Model   {RESET}{WHITE}{model_name}{RESET}")
    print(f"  {GRAY}Device  {RESET}{CYAN}{device}{RESET}")
    print(f"  {GRAY}Exit    {RESET}{DIM}type {ITALIC}exit{RESET}{DIM}, {ITALIC}quit{RESET}{DIM}, or press {ITALIC}Ctrl+D{RESET}")
    print(f"  {GRAY}Reset   {RESET}{DIM}type {ITALIC}/reset{RESET}{DIM} to clear conversation history")
    print()


def _print_token(text: str):
    """Print a streamed token in the model's color (green)."""
    sys.stdout.write(f"{GREEN}{text}{RESET}")
    sys.stdout.flush()


# ── One-shot mode ──────────────────────────────────────────────────────

def _run_oneshot(engine: GPUEngine, prompt: str, temperature: float, max_tokens: int):
    formatted = engine.format_prompt_from_text(prompt)
    res = engine.generate(formatted, temperature=temperature, max_tokens=max_tokens)
    print(f"{GREEN}{res['text']}{RESET}")


# ── Interactive chat mode ──────────────────────────────────────────────

def _run_chat(engine: GPUEngine, temperature: float, max_tokens: int):
    _banner(str(engine.device), engine.model_name)
    # prompt_toolkit gives us readline-style arrow keys, history, multiline, etc.
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.formatted_text import ANSI

        history = InMemoryHistory()
        session = PromptSession(history=history)

        def _get_input() -> str:
            prompt_text = ANSI(f"{PURPLE}{BOLD}  You ❯ {RESET}")
            return session.prompt(prompt_text).strip()

    except ImportError:
        # Fallback to plain input() if prompt_toolkit missing
        def _get_input() -> str:
            try:
                return input(f"{PURPLE}{BOLD}  You ❯ {RESET}").strip()
            except EOFError:
                return ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Sliding window: system prompt + recent turns only, so long sessions
    # never overflow the model's context window (~32k tokens).
    MAX_HISTORY_MESSAGES = 40

    while True:
        try:
            user_input = _get_input()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GRAY}  Goodbye! 👋{RESET}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"\n{GRAY}  Goodbye! 👋{RESET}\n")
            break
        if user_input.strip() == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"{GRAY}  (conversation cleared){RESET}\n")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = engine.format_chat_prompt(messages)

        # Print the model response label
        sys.stdout.write(f"\n{CYAN}{BOLD}  Operant ❯ {RESET}")
        sys.stdout.flush()

        full_reply = ""
        try:
            for token in engine.generate_stream(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                _print_token(token)
                full_reply += token
        except KeyboardInterrupt:
            print(f"\n{GRAY}  (generation interrupted){RESET}")

        print(f"{RESET}\n")

        messages.append({"role": "assistant", "content": full_reply.strip()})
        if len(messages) > MAX_HISTORY_MESSAGES + 1:
            messages = [messages[0]] + messages[-MAX_HISTORY_MESSAGES:]

        # Show live context size so drift is visible (limit is ~32k tokens).
        ctx_tokens = len(engine.tokenizer.encode(prompt).ids)
        print(f"{GRAY}  (context: {ctx_tokens} tokens, history: {len(messages)} messages){RESET}\n")


# ── Entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Operant GPU Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-file", "-m", default="out", help="Path to model directory")
    parser.add_argument("--prompt", help="One-shot prompt (skips interactive mode)")
    parser.add_argument("--chat", action="store_true", help="Interactive terminal chat session")
    parser.add_argument("--device", "-d", default=None, help="Device (e.g. cuda:0, mps, cpu)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    engine = GPUEngine(model_path=args.model_file, device=args.device)

    if args.prompt:
        _run_oneshot(engine, args.prompt, args.temperature, args.max_tokens)
    else:
        _run_chat(engine, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()
