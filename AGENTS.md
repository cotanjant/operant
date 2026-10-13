# AGENTS.md

> Instructions for AI coding agents working on **Operant**.

**Operant** is a fine-tuned agentic native AI model + inference engine.
It consists of:
- A **website** for the project
- **Data collection / finding** + **model training** (PyTorch)
- A **Rust CLI** (must use `clap`)
- An **OpenAI-compatible inference API** (the inference engine)

## Skills

Before starting any non-trivial task, load skills from:

1. `~/.agents/` (user’s local skills on this machine)
2. `skills/` at the root of this repository (project-specific skills)

Prefer project skills when they exist. Always respect both sources.

## Project Layout
operant/
├── model/                  # Training pipeline (Python + uv + PyTorch)
│   ├── src/model/
│   │   ├── train.py        # Fine-tuning script (currently Gemma-3-270m)
│   │   ├── data.jsonl      # Training data
│   │   └── init.py
│   ├── pyproject.toml
│   └── uv.lock
├── instruct/               # Inference engine (Rust)
│   ├── src/
│   │   ├── main.rs
│   │   ├── cli.rs          # CLI (must use clap)
│   │   ├── api.rs          # OpenAI-compatible HTTP API (axum)
│   │   └── lib.rs
│   ├── Cargo.toml
│   └── Cargo.lock
├── website/                # Project website (to be built / expanded)
├── skills/                 # Project-specific agent skills
├── test.csv
└── LICENSE                 # GPL-2.0
## Core Components & Rules

### 1. Website
- Lives under `website/` (or top-level if preferred).
- Purpose: project landing page, docs, model cards, download / demo links, status of training & inference.
- Keep it modern, fast, and informative. Prefer static or lightly dynamic (e.g. Astro, Next.js, or plain HTML + Tailwind).
- Document how to run the CLI and the API.

### 2. Data Finding & Model Training (`model/`)
- Python project managed with **uv**.
- Current base model: `google/gemma-3-270m-it` (instruct-tuned).
- Training data lives in `model/src/model/data.jsonl` (JSONL with `"text"` field in Gemma-3 IT chat template: `<start_of_turn>user ... <end_of_turn><start_of_turn>model ... <end_of_turn>` — must match inference prompts or the model never learns EOS).
- Training script: `model/src/model/train.py` (uses `transformers` + `datasets` + `Trainer`).
- When adding or improving data:
  - Prefer high-quality, agentic / instruction-following examples.
  - Keep format consistent (`{"text": "### Question: ...\n### Answer: ..."}` or similar chat format).
- When changing training:
  - Keep GPU detection / single-GPU focus.
  - Preserve `bf16`, gradient checkpointing, and reasonable defaults unless deliberately changing them.
  - Output goes to `out/` (model + tokenizer).

### 3. Rust CLI (`instruct/`) – **must use clap**
- Entry point: `instruct/src/main.rs` → calls into `cli.rs`.
- **Always** use the `clap` crate with `#[derive(Parser)]` for argument parsing.
- Current argument: `--model-file` / `-m` (path to the model).
- Expand the CLI carefully:
  - Keep `clap` derive style.
  - Support loading the trained model, running inference, serving the API, etc.
  - Prefer subcommands if the surface grows (e.g. `operant serve`, `operant chat`, `operant info`).

### 4. OpenAI-Compatible Inference API (`instruct/src/api.rs`)
- This is the **inference engine**.
- Must remain **OpenAI-compatible**.
- Current endpoint: `POST /v1/chat/completions`
- Request shape already matches the OpenAI chat completions schema (`model`, `messages`, `stream`, …).
- Response should stay compatible (`object: "chat.completion"`, `choices`, `finish_reason`, etc.).
- Currently a stub that returns `"hello operant!"`. Wire it to the real model (via candle or whatever backend is chosen).
- Server currently binds to `0.0.0.0:8000` with axum.
- When extending:
  - Keep the `/v1/chat/completions` route and schema.
  - Support streaming if/when implemented.
  - Do not break OpenAI client compatibility.

## Coding Conventions

### Rust (`instruct/`)
- Edition 2024 (as in current `Cargo.toml`).
- Use `anyhow` for errors.
- Prefer `axum` for the HTTP server.
- Inference backend: `candle` is already a dependency — prefer it unless there is a strong reason to change.
- Keep modules clean: `cli`, `api`, and later `model` / `engine` etc.
- Always re-export / organize through `lib.rs`.

### Python (`model/`)
- Managed with **uv**.
- Target modern Python (currently `>=3.14`).
- Prefer `torch`, `transformers`, `datasets`.
- Keep training scripts focused and reproducible.

### General
- Prefer small, focused commits / PRs.
- Do not break the OpenAI-compatible API contract.
- Do not replace `clap` with another CLI library.
- When adding new features, update this `AGENTS.md` if the architecture changes significantly.

## Common Tasks for Agents

1. **Improve training data** → edit / expand `model/src/model/data.jsonl` and re-run training.
2. **Wire real inference** → connect `api.rs` (and CLI) to the model produced by `train.py` (via candle).
3. **Extend the CLI** → add subcommands / flags using `clap` derive.
4. **Build / improve the website** → make it the public face of Operant.
5. **Add project skills** → put reusable agent skills into `skills/`.

## Running Things (quick reference)

```bash
# Training
cd model
uv run model          # or python -m model

# Rust inference engine / CLI (CPU)
cd instruct
cargo run -- --model-file path/to/model
cargo run -- chat --model-file path/to/model

# GPU-accelerated inference & API (AMD ROCm / CUDA)
cargo run -- run --prompt "Hello" --gpu
cargo run -- serve --gpu --port 8000
# or direct via uv:
cd ../model && uv run serve-gpu
```

## Philosophy

Agentic native model + fast, self-hosted inference.
Training side is Python/PyTorch; serving side is Rust.
CLI is the human-friendly entry point (clap).
API is the machine-friendly entry point (OpenAI compatible).
Website is the public face.

Always prefer solutions that keep these pieces cleanly separated and interoperable.
