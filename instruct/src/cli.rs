use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use colored::*;
use rustyline::error::ReadlineError;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::api::run_server;
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

fn run_gpu_chat(model_path: &PathBuf, temperature: f64, max_tokens: usize) -> Result<()> {
    println!("\n[Operant GPU] Launching GPU-accelerated chat on AMD ROCm / CUDA...");
    let model_dir_str = model_path.to_str().unwrap_or("out");

    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "-m",
            "model.infer",
            "--model-file",
            model_dir_str,
            "--chat",
            "--temperature",
            &temperature.to_string(),
            "--max-tokens",
            &max_tokens.to_string(),
        ])
        .current_dir(find_model_project_dir()?)
        .status()
        .with_context(|| "Failed to launch GPU chat via uv")?;

    if !status.success() {
        return Err(anyhow!("GPU chat exited with status: {}", status));
    }
    Ok(())
}

fn run_gpu_completion(
    model_path: &PathBuf,
    prompt: &str,
    temperature: f64,
    max_tokens: usize,
) -> Result<()> {
    let model_dir_str = model_path.to_str().unwrap_or("out");

    let status = Command::new("uv")
        .args([
            "run",
            "python",
            "-m",
            "model.infer",
            "--model-file",
            model_dir_str,
            "--prompt",
            prompt,
            "--temperature",
            &temperature.to_string(),
            "--max-tokens",
            &max_tokens.to_string(),
        ])
        .current_dir(find_model_project_dir()?)
        .status()
        .with_context(|| "Failed to launch GPU inference via uv")?;

    if !status.success() {
        return Err(anyhow!("GPU inference exited with status: {}", status));
    }
    Ok(())
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
            if use_gpu || *gpu {
                run_gpu_chat(&path, *temperature, *max_tokens)?;
            } else {
                println!(
                    "{}",
                    format!("Loading model (CPU) from: {}", path.display()).dimmed()
                );
                let mut engine = Engine::load(&path).context("Failed to load model engine")?;
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
                println!("  {} {}", "Model".dimmed(), engine.model_name.white());
                println!("  {} {}", "Device".dimmed(), "cpu".cyan());
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

                // System anchor: merged into the first user turn (Gemma has no
                // system role). Without it, filler replies fed back as history
                // drift the session into sycophancy.
                const SYSTEM_PROMPT: &str = "You are Operant, an agentic AI assistant. Answer directly and concisely. Do not flatter the user, do not pad with filler, and say when you do not know. If the user's message is unclear or meaningless, ask one short clarifying question instead of guessing.";
                let mut conversation_prompt = String::new();
                let mut first_turn = true;
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
                        conversation_prompt.clear();
                        first_turn = true;
                        println!("{}", "  (conversation cleared)".dimmed());
                        continue;
                    }

                    conversation_prompt.push_str("<start_of_turn>user\n");
                    if first_turn {
                        conversation_prompt.push_str(SYSTEM_PROMPT);
                        conversation_prompt.push_str("\n\n");
                        first_turn = false;
                    }
                    conversation_prompt.push_str(&input);
                    conversation_prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");

                    print!("{} ", "  Operant ❯".cyan().bold());
                    std::io::stdout().flush()?;

                    let params = GenerationParams {
                        temperature: *temperature,
                        top_p: Some(0.9),
                        max_tokens: *max_tokens,
                        seed: None,
                        stop: vec!["<end_of_turn>".to_string(), "<start_of_turn>".to_string()],
                    };

                    let res =
                        engine.generate_stream_tokens(&conversation_prompt, &params, |token| {
                            print!("{}", token.green());
                            std::io::stdout().flush().map_err(|e| anyhow!("{e}"))
                        })?;

                    println!();
                    println!();
                    conversation_prompt.push_str(&res.text);
                    conversation_prompt.push_str("<end_of_turn>\n");
                }
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
                run_gpu_completion(&path, prompt, *temperature, *max_tokens)?;
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
