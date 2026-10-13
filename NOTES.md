# Operant — session notes & roadmap

## Where the project stands

Single Rust CLI (`instruct/`, clap). Python (`model/`) is engine internals only:
headless GPU worker (JSON over stdio), GPU API server process, training script.
No Python user interface anymore.

## What we did

- Training moved to `google/gemma-3-270m-it`. Dataset (`data.jsonl`, 25 samples)
  uses the Gemma chat template, matching inference prompts. Dropped the junk
  toy lines (`Who am i? -> king`) that poisoned early runs.
- Fixed the loops: stale `out/` weights were the cause, plus `no_repeat_ngram_size=3`
  guards and stop sequences on both paths. Retrained twice.
- Safety incident (model echoed a slur): now three layers. Hard slur blocklist on
  inputs and outputs in both engines, default system prompt with refusal orders
  injected by both API servers, refusal examples trained into the weights.
- Chat quality: system anchor every session, `/reset`, sliding window history
  (40 messages), live context/token display, graceful Ctrl+C.
- CLI: colorful readline chat (rustyline, history, arrow keys), `serve` / `chat` /
  `run` / `info`, `--gpu` flag everywhere. Fixed the missing `Parser` import.
- Python cleanup: deleted `infer.py`, added headless `worker.py`, one shared REPL
  in Rust for CPU and GPU. Dropped `infer-gpu`/`serve-gpu` scripts and
  `prompt-toolkit`. Docs (`AGENTS.md`, skill) show cargo commands only.
- GPU path accepts HF hub IDs directly (used to silently fall back to `out/`).
- Website split: `/` is the chat app (multi-turn history), landing page moved to
  `landing.html`.
- Git history rewritten in a human voice.

## Limits we accept right now

- 25 samples / ~7 steps barely nudges the base model. Identity ("I am Operant")
  doesn't stick; most answers fall back to stock Gemma behavior.
- Safety rests on the blocklist + system prompt, not deep alignment. Novel
  phrasings outside the list rely on the prompt and weights. Not certified safe.
- Web chat is request/response, no token streaming in the browser.
- Rust CPU inference works but is slow and lags the GPU path in guardrails.
- No evals: no prompt set, no pass/fail, quality judged by eyeballing.
- No persistent shell history, no multiline input in chat.
- Single user, single model, no auth on the API.

## What would bring it to the level we want

1. Data: hundreds of samples, real multi-turn sessions, more refusal coverage,
   red-team prompts. Version the dataset.
2. Training: more epochs, eval split during training, checkpoint selection on
   eval score instead of last-epoch. Track loss curves somewhere.
3. Eval harness: fixed prompt set (identity, refusal, agentic tasks) with
   expected behaviors, runnable via one command, committed next to the tests.
4. Identity: bake "Operant" in so it survives without the system prompt.
5. Web streaming: SSE token streaming in the browser chat.
6. CLI: history file, multiline paste, `/save` transcript export.
7. Serve: API key / rate limit, model hot-reload, serve either engine on :8000
   interchangeably (already close).
8. CPU speed: quantization (Q4/Q8) for the Candle path or document GPU-first.
9. Shipping: release binaries, Dockerfile, root README + model card.
10. CI: cargo test + worker smoke test + eval harness on every push.
