---
name: operant
description: >
  Operant agent skill: Instructions and procedures for running, serving, evaluating,
  and extending the Operant AI model and pure-Rust inference engine.
argument-hint: "[serve|chat|train|eval]"
license: GPL-2.0
---

# Operant Agent Skill

Operant is a fine-tuned agentic native AI model coupled with a pure-Rust inference engine.

## Core Operations

### 1. Running the Rust Inference Engine & API Server
```bash
cd instruct
# Serve OpenAI-compatible API on 0.0.0.0:8000 (CPU)
cargo run -- serve --model-file ../model/out --port 8000

# Serve with GPU acceleration (AMD ROCm / CUDA)
cargo run -- serve --model-file ../model/out --gpu --port 8000

# Interactive CLI chat in terminal (CPU or GPU)
cargo run -- chat --model-file ../model/out
cargo run -- chat --model-file ../model/out --gpu

# One-shot command completion
cargo run -- run --model-file ../model/out --prompt "Your prompt here" --gpu

# Model inspection
cargo run -- info --model-file ../model/out
```

### 2. Direct Python GPU Server (uv)
```bash
cd model
uv run serve-gpu --port 8000
uv run infer-gpu --prompt "Your prompt here"
```

### 2. Querying the Inference API
The inference API is 100% OpenAI compatible:
- Endpoint: `POST http://localhost:8000/v1/chat/completions`
- Models: `GET http://localhost:8000/v1/models`
- Health: `GET http://localhost:8000/health`

Example cURL:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "operant",
    "messages": [{"role": "user", "content": "Explain your architecture."}],
    "temperature": 0.7
  }'
```

### 3. Model Training Pipeline
The training pipeline lives in `model/` and is managed with `uv`:
```bash
cd model

# Generate / augment training data
uv run python src/model/generate_data.py

# Run fine-tuning (PyTorch + ROCm/CUDA + Transformers)
uv run model
```
Outputs are written to `model/out/` containing:
- `model.safetensors`
- `config.json`
- `tokenizer.json`

### 4. Running the Website
The website lives in `website/` and is completely static:
```bash
cd website
python3 -m http.server 3000
# Open http://localhost:3000
```
