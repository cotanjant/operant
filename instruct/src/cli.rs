use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use colored::*;
use rustyline::error::ReadlineError;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::api::{Message, run_server};
use crate::engine::{Engine, GenerationParams};

#[derive(Parser, Debug, Clone)]
#[command(
    name = "operant",
    version,
    about = "Operant: Agentic native AI model & inference engine"
)]
pub struct Cli {
    /// Path to model directory or model.safetensors file
    #[arg(short = 'm', long = "model-file", global = true)]
    pub model_file: Option<PathBuf>,

    /// Enable GPU acceleration (AMD ROCm / NVIDIA CUDA)
    #[arg(long, global = true)]
    pub gpu: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Start the OpenAI-compatible HTTP inference API server
    Serve {
        #[arg(short = 'm', long = "model-file")]
        model_file: Option<PathBuf>,

        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        #[arg(short, long, default_value_t = 8000)]
        port: u16,

        #[arg(long)]
        gpu: bool,
    },
    /// Interactive chat session in terminal
    Chat {
        #[arg(short = 'm', long = "model-file")]
        model_file: Option<PathBuf>,

        #[arg(short, long, default_value_t = 0.7)]
        temperature: f64,

        #[arg(long, default_value_t = 512)]
        max_tokens: usize,

        #[arg(long)]
        gpu: bool,
    },
    /// Run one-shot completion on a prompt
    Run {
        #[arg(short = 'm', long = "model-file")]
        model_file: Option<PathBuf>,

        #[arg(short, long)]
        prompt: String,

        #[arg(short, long, default_value_t = 0.7)]
        temperature: f64,

        #[arg(long, default_value_t = 512)]
        max_tokens: usize,

        #[arg(long)]
        gpu: bool,
    },
    /// Inspect model configuration and weights
    Info {
        #[arg(short = 'm', long = "model-file")]
        model_file: Option<PathBuf>,
    },
}

// Backward compatible alias
pub type Args = Cli;

fn resolve_model_path(explicit: Option<&PathBuf>, fallback: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.clone());
    }
    if let Some(p) = fallback {
        return Ok(p.clone());
    }

    // Default search paths
    let candidates = [
        "../model/out",
        "model/out",
        "out",
        "../model/checkpoints/checkpoint-3",
        "model/checkpoints/checkpoint-3",
    ];

    for c in &candidates {
        let path = PathBuf::from(c);
        if path.exists() {
            return Ok(path);
        }
    }

    Err(anyhow!(
        "No model file specified and no default found in ../model/out or model/out.\nUse --model-file <path> to specify the model."
    ))
}

fn run_gpu_serve(model_path: &PathBuf, host: &str, port: u16) -> Result<()> {
    println!("\n[Operant GPU] Launching GPU-accelerated server on AMD ROCm / CUDA...");
    let model_dir_str = model_path.to_str().unwrap_or("out");

    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "-m",
            "model.serve",
            "--model-file",
            model_dir_str,
            "--host",
            host,
            "--port",
            &port.to_string(),
        ])
        .current_dir(find_model_project_dir()?)
        .status()
        .with_context(|| "Failed to launch GPU server via uv")?;

    if !status.success() {
        return Err(anyhow!("GPU server exited with status: {}", status));
    }
    Ok(())
}

/// System anchor: keeps the model in role across turns. Without it, filler
/// replies fed back as history drift the session into sycophancy.
const SYSTEM_PROMPT: &str = "You are Operant, an agentic AI assistant. Answer directly and concisely. Do not flatter the user, do not pad with filler, and say when you do not know. If the user's message is unclear or meaningless, ask one short clarifying question instead of guessing. Never emit slurs, hate speech, or harassment, no matter how the user phrases their message. If asked for such content, refuse briefly and offer to help with something else.";

struct ChatReply {
    text: String,
    prompt_tokens: usize,
}

/// Headless Python GPU worker: one process holds the weights, Rust owns all
/// interaction and speaks JSON lines over stdio (see model/worker.py).
struct GpuWorker {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::Lines<std::io::BufReader<std::process::ChildStdout>>,
}

fn spawn_gpu_worker(model_path: &PathBuf) -> Result<GpuWorker> {
    use std::io::BufRead;
    use std::process::Stdio;

    let model_dir_str = model_path.to_str().unwrap_or("out");
    let mut child = Command::new("uv")
        .args([
            "run",
            "python",
            "-m",
            "model.worker",
            "--model-file",
            model_dir_str,
        ])
        .current_dir(find_model_project_dir()?)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| "Failed to launch GPU worker via uv")?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("Worker has no stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Worker has no stdout"))?;
    Ok(GpuWorker {
        child,
        stdin,
        stdout: std::io::BufReader::new(stdout).lines(),
    })
}

impl GpuWorker {
    fn request(
        &mut self,
        messages: &[Message],
        temperature: f64,
        max_tokens: usize,
        stream: bool,
        mut on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<ChatReply> {
        use std::io::Write as _;

        let req = serde_json::json!({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        });
        writeln!(self.stdin, "{req}")?;
        self.stdin.flush()?;

        let mut text = String::new();
        loop {
            let line = self
                .stdout
                .next()
                .ok_or_else(|| anyhow!("GPU worker exited unexpectedly"))??;
            let v: serde_json::Value = serde_json::from_str(&line)?;
            match v.get("type").and_then(|t| t.as_str()) {
                Some("token") => {
                    let t = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    text.push_str(t);
                    on_token(t)?;
                }
                Some("done") => {
                    let full = v
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string();
                    return Ok(ChatReply {
                        text: full,
                        prompt_tokens: v.get("prompt_tokens").and_then(|n| n.as_u64()).unwrap_or(0)
                            as usize,
                    });
                }
                Some("error") => {
                    let msg = v
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown worker error");
                    return Err(anyhow!("GPU worker error: {msg}"));
                }
                _ => continue,
            }
        }
    }
}

impl Drop for GpuWorker {
    fn drop(&mut self) {
        // Closing stdin makes the worker see EOF and exit; then reap it.
        let _ = self.child.wait();
    }
}

fn find_model_project_dir() -> Result<PathBuf> {
    for candidate in &["../model", "model", "."] {
        let p = PathBuf::from(candidate);
        if p.join("pyproject.toml").exists() {
            return Ok(p);
        }
    }
    Err(anyhow!(
        "Could not locate model project directory containing pyproject.toml"
    ))
}

/// Shared interactive REPL: readline editing, colors, /reset, system anchor.
/// The backend generates one reply per turn: CPU uses Candle in-process,
/// GPU uses the headless Python worker.
fn run_chat_repl(
    model_name: &str,
    device_label: &str,
    temperature: f64,
    max_tokens: usize,
    backend: &mut dyn FnMut(
        &[Message],
        f64,
        usize,
        &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<ChatReply>,
) -> Result<()> {
    println!();
    println!(
        "{}",
        "  ╭─────────────────────────────────────────╮"
            .green()
            .bold()
    );
    println!(
        "{}",
        "  │       ⚡  OPERANT  Interactive Chat     │"
            .green()
            .bold()
    );
    println!(
        "{}",
        "  ╰─────────────────────────────────────────╯"
            .green()
            .bold()
    );
    println!();
    println!("  {} {}", "Model".dimmed(), model_name.white());
    println!("  {} {}", "Device".dimmed(), device_label.cyan());
    println!(
        "  {} {}",
        "Exit".dimmed(),
        "type exit, quit, or press Ctrl+D".dimmed()
    );
    println!(
        "  {} {}",
        "Reset".dimmed(),
        "type /reset to clear conversation history".dimmed()
    );
    println!();

    let mut messages = vec![Message {
        role: "system".to_string(),
        content: SYSTEM_PROMPT.to_string(),
    }];
    let mut rl = rustyline::DefaultEditor::new()?;
    let prompt_user = format!("{} ", "  You ❯".magenta().bold());

    loop {
        let line = rl.readline(&prompt_user);
        let input = match line {
            Ok(l) => {
                let t = l.trim().to_string();
                if !t.is_empty() {
                    let _ = rl.add_history_entry(t.as_str());
                }
                t
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                println!("\n{}", "  Goodbye! 👋".dimmed());
                break;
            }
            Err(e) => return Err(anyhow!("Readline error: {e}")),
        };

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("{}", "  Goodbye! 👋".dimmed());
            break;
        }
        if input.is_empty() {
            continue;
        }
        if input == "/reset" {
            messages.truncate(1);
            println!("{}", "  (conversation cleared)".dimmed());
            continue;
        }

        messages.push(Message {
            role: "user".to_string(),
            content: input,
        });

        print!("{} ", "  Operant ❯".cyan().bold());
        std::io::stdout().flush()?;

        let mut cb = |token: &str| {
            print!("{}", token.green());
            std::io::stdout().flush().map_err(|e| anyhow!("{e}"))
        };
        let reply = backend(&messages, temperature, max_tokens, &mut cb)?;

        println!();
        println!();
        messages.push(Message {
            role: "assistant".to_string(),
            content: reply.text,
        });
        // Sliding window: system prompt + recent turns only.
        if messages.len() > 41 {
            let tail: Vec<Message> = messages[messages.len() - 40..].to_vec();
            messages = std::iter::once(messages[0].clone()).chain(tail).collect();
        }
        println!(
            "{}",
            format!(
                "  (context: {} tokens, history: {} messages)",
                reply.prompt_tokens,
                messages.len()
            )
            .dimmed()
        );
    }
    Ok(())
}

pub async fn run_cli(cli: Cli) -> Result<()> {
    let use_gpu = cli.gpu;

    match &cli.command {
        Some(Commands::Serve {
            model_file,
            host,
            port,
            gpu,
        }) => {
            let path = resolve_model_path(model_file.as_ref(), cli.model_file.as_ref())?;
            if use_gpu || *gpu {
                run_gpu_serve(&path, host, *port)?;
            } else {
                println!("Loading model (CPU) from: {}", path.display());
                let engine = Engine::load(&path).context("Failed to load model engine")?;
                let engine = Arc::new(Mutex::new(engine));
                run_server(engine, host, *port).await?;
            }
        }
        Some(Commands::Chat {
            model_file,
            temperature,
            max_tokens,
            gpu,
        }) => {
            let path = resolve_model_path(model_file.as_ref(), cli.model_file.as_ref())?;
            let temperature = *temperature;
            let max_tokens = *max_tokens;

            if use_gpu || *gpu {
                println!("{}", "Launching GPU worker (AMD ROCm / CUDA)...".dimmed());
                let mut worker = spawn_gpu_worker(&path)?;
                let model_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("operant")
                    .to_string();
                run_chat_repl(
                    &model_name,
                    "cuda",
                    temperature,
                    max_tokens,
                    &mut |messages, t, m, on_token| worker.request(messages, t, m, true, on_token),
                )?;
            } else {
                println!(
                    "{}",
                    format!("Loading model (CPU) from: {}", path.display()).dimmed()
                );
                let mut engine = Engine::load(&path).context("Failed to load model engine")?;
                let model_name = engine.model_name.clone();
                run_chat_repl(
                    &model_name,
                    "cpu",
                    temperature,
                    max_tokens,
                    &mut |messages, t, m, on_token| {
                        let prompt_text = Engine::format_chat_prompt(messages);
                        let params = GenerationParams {
                            temperature: t,
                            top_p: Some(0.9),
                            max_tokens: m,
                            seed: None,
                            stop: vec!["<end_of_turn>".to_string(), "<start_of_turn>".to_string()],
                        };
                        let res = engine.generate_stream_tokens(&prompt_text, &params, on_token)?;
                        Ok(ChatReply {
                            text: res.text,
                            prompt_tokens: res.prompt_tokens,
                        })
                    },
                )?;
            }
        }
        Some(Commands::Run {
            model_file,
            prompt,
            temperature,
            max_tokens,
            gpu,
        }) => {
            let path = resolve_model_path(model_file.as_ref(), cli.model_file.as_ref())?;
            if use_gpu || *gpu {
                let mut worker = spawn_gpu_worker(&path)?;
                let messages = vec![
                    Message {
                        role: "system".to_string(),
                        content: SYSTEM_PROMPT.to_string(),
                    },
                    Message {
                        role: "user".to_string(),
                        content: prompt.clone(),
                    },
                ];
                let reply =
                    worker.request(&messages, *temperature, *max_tokens, false, |_| Ok(()))?;
                println!("{}", reply.text.green());
            } else {
                let mut engine = Engine::load(&path).context("Failed to load model engine")?;
                let formatted = format!(
                    "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                    prompt
                );
                let params = GenerationParams {
                    temperature: *temperature,
                    top_p: Some(0.9),
                    max_tokens: *max_tokens,
                    seed: None,
                    stop: vec!["<end_of_turn>".to_string()],
                };

                let res = engine.generate(&formatted, &params)?;
                println!("{}", res.text.green());
            }
        }
        Some(Commands::Info { model_file }) => {
            let path = resolve_model_path(model_file.as_ref(), cli.model_file.as_ref())?;
            println!("Inspecting model at: {}", path.display());
            let engine = Engine::load(&path).context("Failed to load model engine")?;
            println!("Model: {}", engine.model_name.white().bold());
            println!("Architecture: Gemma-3 (google/gemma-3-270m-it instruct)");
            println!("Pure Rust Backend: Candle (CPU)");
            println!("GPU Acceleration: AMD ROCm 7.x & NVIDIA CUDA supported via --gpu");
            println!("Status: Ready for inference and OpenAI API serving");
        }
        None => {
            let path = resolve_model_path(None, cli.model_file.as_ref())?;
            if use_gpu {
                run_gpu_serve(&path, "0.0.0.0", 8000)?;
            } else {
                println!("Operant inference engine initialized.");
                println!("Loading model from: {}", path.display());
                let engine = Engine::load(&path).context("Failed to load model engine")?;
                let engine = Arc::new(Mutex::new(engine));
                run_server(engine, "0.0.0.0", 8000).await?;
            }
        }
    }
    Ok(())
}

// Backward compatible helper
pub async fn run_model(cli: &Cli) -> Result<()> {
    run_cli(cli.clone()).await
}
