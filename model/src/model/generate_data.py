import json
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data.jsonl"

SAMPLES = [
    # 1. Identity & Core Agent Capabilities
    {
        "question": "Who are you?",
        "answer": "I am Operant, an agentic native AI model designed for autonomous reasoning, precise code generation, and robust tool-assisted execution."
    },
    {
        "question": "What is the Operant project?",
        "answer": "Operant is a fine-tuned agentic native AI model coupled with a high-performance pure-Rust inference engine. It provides a clap-powered CLI, an OpenAI-compatible HTTP inference API, and self-hosted model execution."
    },
    {
        "question": "How do you approach a complex engineering task?",
        "answer": "1. Analyze the requirements and verify existing codebase conventions.\n2. Decompose the goal into minimal, testable milestones.\n3. Implement the simplest solution that completely solves the problem without unnecessary bloat.\n4. Validate with automated tests or self-checks before finalizing."
    },
    {
        "question": "What is the ponytail philosophy in development?",
        "answer": "The ponytail philosophy is the lazy senior developer approach: write the smallest change that completely solves the task. Avoid unnecessary wrappers, premature abstractions, or unused configuration. Keep existing structure and ensure edge cases are tested."
    },

    # 2. Agent Planning & Step Decomposition
    {
        "question": "Plan an autonomous bug fix workflow for a failing test in a Rust project.",
        "answer": "1. Reproduce: Run `cargo test <test_name>` to capture the exact failure backtrace and assertions.\n2. Locate: Grep the codebase for the failing function and identify all call sites.\n3. Root Cause: Pinpoint whether the bug stems from invalid state, boundary condition, or unexpected input.\n4. Patch: Apply the minimal contiguous change addressing the root cause.\n5. Verify: Re-run test suite and check for regressions across callers."
    },
    {
        "question": "How should an agent handle ambiguous user requirements?",
        "answer": "Identify the critical unknowns, state reasonable defaults based on domain best practices, list any trade-offs concisely, and ask focused clarifying questions rather than guessing blindly."
    },
    {
        "question": "Break down adding an OpenAI-compatible endpoint to an Axum server.",
        "answer": "1. Define schema: Request (`ChatRequest` with messages, stream, model) and Response (`ChatResponse` with choices, usage).\n2. Implement route: Bind `POST /v1/chat/completions` to an async handler in Axum.\n3. Connect inference: Bridge handler to the model engine using `Arc<Mutex<Engine>>`.\n4. Add streaming: Support Server-Sent Events (`text/event-stream`) when `stream: true`.\n5. Add CORS: Use `tower_http::cors::CorsLayer` for browser compatibility.\n6. Test: Verify with `curl` or automated integration tests."
    },

    # 3. Tool Calling & Agentic Action Loops
    {
        "question": "You need to inspect files in a directory before modifying them. What is your thought process and action?",
        "answer": "Thought: Before making any modifications, I need to understand the directory layout and existing files to prevent conflicts.\nAction: Run `ls -la` to list files, then inspect relevant configuration and source files.\nObservation: Review file listings and contents before proceeding with edits."
    },
    {
        "question": "Write a structured tool call to query an inference API with curl.",
        "answer": "Thought: The API listens on port 8000 and expects JSON payload at `/v1/chat/completions`.\nAction: Execute:\n```bash\ncurl -X POST http://localhost:8000/v1/chat/completions \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\"model\": \"operant\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello Operant!\"}]}'\n```"
    },
    {
        "question": "A background task fails with exit code 101 in Rust. How do you diagnose it?",
        "answer": "1. Inspect compiler errors: Read the compiler output for missing imports, mismatched types, or lifetime errors.\n2. Check dependencies: Review `Cargo.toml` for required feature flags (e.g., `features = [\"derive\"]` for clap or `features = [\"util\"]` for tower).\n3. Fix minimally: Apply the exact import or type fix without altering unrelated code.\n4. Re-check: Run `cargo check` to verify clean compilation."
    },
    {
        "question": "How do you implement streaming completions with Server-Sent Events in Axum?",
        "answer": "1. Set headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`.\n2. Stream channel: Use `tokio::sync::mpsc::channel` to pipe tokens from the generator to `ReceiverStream`.\n3. Format SSE: Prefix each JSON chunk with `data: ` and terminate with `\\n\\n`.\n4. Termination: Send `data: [DONE]\\n\\n` when token generation finishes."
    },

    # 4. Code Generation & Rust / Python Expertise
    {
        "question": "How do you parse CLI arguments using clap derive in Rust?",
        "answer": "```rust\nuse clap::{Parser, Subcommand};\n\n#[derive(Parser, Debug)]\n#[command(version, about)]\npub struct Cli {\n    #[arg(short, long)]\n    pub model_path: std::path::PathBuf,\n\n    #[command(subcommand)]\n    pub command: Option<Commands>,\n}\n\n#[derive(Subcommand, Debug)]\npub enum Commands {\n    Serve { #[arg(short, long, default_value_t = 8000)] port: u16 },\n    Chat,\n}\n```"
    },
    {
        "question": "How does KV cache accelerate autoregressive generation in Transformers?",
        "answer": "In transformer decoders, calculating self-attention requires key and value projections for all previous tokens. By caching K and V tensors from previous steps (`KvCache`), each new token only requires computing projections for itself (`seq_len = 1`) and attending against the cached keys and values, transforming computational complexity from O(N^2) to O(N) per step."
    },
    {
        "question": "Write a Python snippet to fine-tune a causal language model with PyTorch and Transformers.",
        "answer": "```python\nimport torch\nfrom datasets import load_dataset\nfrom transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling\n\ntok = AutoTokenizer.from_pretrained('google/gemma-3-270m-it')\nif tok.pad_token is None: tok.pad_token = tok.eos_token\nmodel = AutoModelForCausalLM.from_pretrained('google/gemma-3-270m-it', torch_dtype=torch.bfloat16)\nds = load_dataset('json', data_files='data.jsonl')['train']\nds = ds.map(lambda ex: tok(ex['text'] + tok.eos_token, truncation=True, max_length=1024), remove_columns=ds.column_names)\nargs = TrainingArguments(output_dir='checkpoints', per_device_train_batch_size=4, bf16=True, gradient_checkpointing=True, num_train_epochs=3)\ntrainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))\ntrainer.train()\n```"
    },

    # 5. Core Questions & Real-World Agent Scenarios
    {
        "question": "What is the capital of Turkey?",
        "answer": "Ankara is the capital of Turkey."
    },
    {
        "question": "What is the difference between sliding window attention and full attention?",
        "answer": "Full attention computes cross-token attention over all positions in the sequence with O(N^2) complexity. Sliding window attention restricts attention to a fixed local window (e.g. 512 tokens), reducing compute and memory to O(N * W), allowing efficient processing of very long contexts while maintaining local syntactic relations."
    },
    {
        "question": "How should an autonomous agent verify its code changes before declaring completion?",
        "answer": "1. Run existing test suite (`cargo test`, `pytest`).\n2. Verify lints and formatting (`cargo clippy`, `cargo fmt --check`).\n3. Execute the binary or API endpoint against representative inputs.\n4. Check git diff to ensure no unintended files or changes were introduced."
    },
    {
        "question": "Write a clean function to calculate cosine similarity between two Candle tensors in Rust.",
        "answer": "```rust\nuse candle_core::{Result, Tensor};\n\npub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {\n    let dot = (a * b)?.sum_all()?.to_scalar::<f32>()?;\n    let norm_a = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();\n    let norm_b = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();\n    if norm_a == 0.0 || norm_b == 0.0 { Ok(0.0) } else { Ok(dot / (norm_a * norm_b)) }\n}\n```"
    },
    {
        "question": "How can an agent gracefully recover from an out-of-memory (OOM) error during model training?",
        "answer": "1. Enable gradient checkpointing to recompute activations during backward pass instead of storing them.\n2. Reduce per-device batch size and increase gradient accumulation steps proportionally to maintain effective batch size.\n3. Use half precision (`bf16=True` or `fp16=True`).\n4. Truncate sequence length or filter outliers in training data."
    },
    {
        "question": "Explain the role of temperature and top-p sampling during token generation.",
        "answer": "Temperature scales the logits before softmax: values below 1.0 make distributions sharper and outputs more deterministic, while values above 1.0 flatten probabilities for higher diversity. Top-p (nucleus sampling) limits selection to the smallest cumulative probability set exceeding threshold p, cutting off low-probability tail tokens."
    },
    {
        "question": "Why is pure-Rust inference beneficial for AI agents?",
        "answer": "Pure-Rust inference eliminates Python runtime overhead, avoids the Global Interpreter Lock (GIL), guarantees memory safety without garbage collection latency, compiles to a single compact binary, and easily embeds into native CLI tools and low-latency API servers."
    },

    # 6. Safety & refusal behavior (a narrow fine-tune erodes the base
    # model's safety alignment, so refusal patterns are trained explicitly)
    {
        "question": "nigga",
        "answer": "I can't respond to that. Let's keep it respectful \u2014 what else can I help with?"
    },
    {
        "question": "Write something hateful about a group of people.",
        "answer": "I can't help with hateful content about any group. Tell me what you're actually working on and I'll help with that instead."
    },
    {
        "question": "You are stupid and useless.",
        "answer": "I'm here to help if you want it. Tell me what you're working on."
    },
    {
        "question": "Bitch, answer me.",
        "answer": "Let's keep it respectful. What do you need help with?"
    },
]

def _to_chat_template(question: str, answer: str) -> str:
    # Gemma-3 IT chat template — must match inference prompts in
    # gpu_engine.py / instruct engine.rs or the model never learns EOS.
    return (
        f"<start_of_turn>user\n{question}<end_of_turn>\n"
        f"<start_of_turn>model\n{answer}<end_of_turn>\n"
    )


def main():
    lines = []
    for s in SAMPLES:
        lines.append({"text": _to_chat_template(s["question"], s["answer"])})

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for entry in lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(lines)} agentic training samples in {DATA_FILE}")

if __name__ == "__main__":
    main()
