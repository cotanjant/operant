"""
Operant GPU worker — headless generation backend for the Rust CLI.

This is NOT a user interface. The Rust CLI (instruct/) spawns one worker
per `chat` / `run --gpu` session and speaks JSON lines over stdio:

  --> {"messages": [{"role": "user", "content": "..."}],
       "temperature": 0.7, "max_tokens": 512, "stream": true}
  <-- {"type": "token", "text": "..."}
  <-- {"type": "done", "text": "<full reply>", "finish_reason": "stop",
       "prompt_tokens": 12, "completion_tokens": 34}
  <-- {"type": "error", "message": "..."}   (single request failed)

EOF on stdin exits the worker cleanly.
"""

import argparse
import json
import sys

from .gpu_engine import GPUEngine


def _send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle(engine: GPUEngine, req: dict):
    messages = req.get("messages", [])
    temperature = float(req.get("temperature", 0.7))
    max_tokens = int(req.get("max_tokens", 512))
    stream = bool(req.get("stream", True))
    prompt = engine.format_chat_prompt(messages)

    if not stream:
        res = engine.generate(
            prompt, temperature=temperature, max_tokens=max_tokens
        )
        _send(
            {
                "type": "done",
                "text": res["text"],
                "finish_reason": res["finish_reason"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
            }
        )
        return

    full = ""
    for token in engine.generate_stream(
        prompt, temperature=temperature, max_tokens=max_tokens
    ):
        full += token
        _send({"type": "token", "text": token})

    _send(
        {
            "type": "done",
            "text": full.strip(),
            "finish_reason": "stop",
            "prompt_tokens": len(engine.tokenizer.encode(prompt)),
            "completion_tokens": len(engine.tokenizer.encode(full)),
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Operant GPU worker (internal; use the Rust CLI)"
    )
    parser.add_argument("--model-file", "-m", default="out")
    parser.add_argument("--device", "-d", default=None)
    args = parser.parse_args()

    engine = GPUEngine(model_path=args.model_file, device=args.device)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _send({"type": "error", "message": f"bad request: {exc}"})
            continue
        try:
            _handle(engine, req)
        except BrokenPipeError:
            break
        except Exception as exc:  # one bad request must not kill the session
            _send({"type": "error", "message": str(exc)})


if __name__ == "__main__":
    main()
