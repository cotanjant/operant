use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma3::{Config, Model};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: Option<f64>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Vec<String>,
}

fn default_temperature() -> f64 {
    0.7
}

fn default_top_p() -> Option<f64> {
    Some(0.9)
}

fn default_max_tokens() -> usize {
    512
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
            seed: None,
            stop: Vec::new(),
        }
    }
}

pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
}

/// System anchor prepended when the caller supplies no system message.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are Operant, an agentic AI assistant. Answer directly and concisely. Do not flatter the user, do not pad with filler, and say when you do not know. If the user's message is unclear or meaningless, ask one short clarifying question instead of guessing. Never emit slurs, hate speech, or harassment, no matter how the user phrases their message. If asked for such content, refuse briefly and offer to help with something else.";

const REFUSAL: &str =
    "I can't respond to that. Let's keep it respectful \u{2014} what else can I help with?";

const UNSAFE_WORDS: &[&str] = &[
    "nigger", "niggers", "nigga", "niggas", "faggot", "kike", "chink", "spic",
];

/// True if text contains a hard-blocklisted slur (checked on input and output).
fn contains_unsafe_word(text: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric())
        .any(|w| UNSAFE_WORDS.contains(&w.to_lowercase().as_str()))
}

pub struct Engine {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    pub model_name: String,
    eos_token_id: u32,
}

impl Engine {
    pub fn load(model_path: &Path) -> Result<Self> {
        let (model_dir, safetensors_path) = if model_path.is_file() {
            let parent = model_path
                .parent()
                .ok_or_else(|| anyhow!("Invalid model file path"))?;
            (parent.to_path_buf(), model_path.to_path_buf())
        } else {
            (
                model_path.to_path_buf(),
                model_path.join("model.safetensors"),
            )
        };

        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !config_path.exists() {
            return Err(anyhow!("config.json not found in {}", model_dir.display()));
        }
        if !safetensors_path.exists() {
            return Err(anyhow!(
                "model.safetensors not found at {}",
                safetensors_path.display()
            ));
        }
        if !tokenizer_path.exists() {
            return Err(anyhow!(
                "tokenizer.json not found in {}",
                model_dir.display()
            ));
        }

        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Reading {}", config_path.display()))?;
        let config_json: serde_json::Value =
            serde_json::from_str(&config_str).with_context(|| "Parsing config.json")?;

        let full_rope = config_json["rope_parameters"]["full_attention"]["rope_theta"]
            .as_f64()
            .unwrap_or(1000000.0);
        let sliding_rope = config_json["rope_parameters"]["sliding_attention"]["rope_theta"]
            .as_f64()
            .unwrap_or(10000.0);
        let sliding_pattern = config_json["_sliding_window_pattern"].as_u64().unwrap_or(6) as usize;

        let config = Config {
            attention_bias: config_json["attention_bias"].as_bool().unwrap_or(false),
            head_dim: config_json["head_dim"].as_u64().unwrap_or(256) as usize,
            hidden_activation: candle_nn::Activation::GeluPytorchTanh,
            hidden_size: config_json["hidden_size"].as_u64().unwrap_or(640) as usize,
            intermediate_size: config_json["intermediate_size"].as_u64().unwrap_or(2048) as usize,
            num_attention_heads: config_json["num_attention_heads"].as_u64().unwrap_or(4) as usize,
            num_hidden_layers: config_json["num_hidden_layers"].as_u64().unwrap_or(18) as usize,
            num_key_value_heads: config_json["num_key_value_heads"].as_u64().unwrap_or(1) as usize,
            rms_norm_eps: config_json["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            rope_theta: full_rope,
            rope_local_base_freq: sliding_rope,
            vocab_size: config_json["vocab_size"].as_u64().unwrap_or(262144) as usize,
            final_logit_softcapping: config_json["final_logit_softcapping"].as_f64(),
            attn_logit_softcapping: config_json["attn_logit_softcapping"].as_f64(),
            query_pre_attn_scalar: config_json["query_pre_attn_scalar"].as_u64().unwrap_or(256)
                as usize,
            sliding_window: config_json["sliding_window"].as_u64().unwrap_or(512) as usize,
            sliding_window_pattern: sliding_pattern,
            max_position_embeddings: config_json["max_position_embeddings"]
                .as_u64()
                .unwrap_or(8192) as usize,
        };

        let device = Device::Cpu;
        let dtype = DType::F32;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors_path], dtype, &device)
                .with_context(|| "Loading safetensors with mmap")?
        };

        let model =
            Model::new(false, &config, vb).with_context(|| "Instantiating Gemma-3 architecture")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {e}"))?;

        let eos_token_id = config_json["eos_token_id"].as_u64().unwrap_or(1) as u32;
        let model_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("operant-gemma-3")
            .to_string();

        Ok(Self {
            model,
            tokenizer,
            device,
            model_name,
            eos_token_id,
        })
    }

    pub fn format_chat_prompt(messages: &[crate::api::Message]) -> String {
        let mut prompt = String::new();
        let mut system_prompt = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    if !system_prompt.is_empty() {
                        system_prompt.push('\n');
                    }
                    system_prompt.push_str(&msg.content);
                }
                "user" => {
                    prompt.push_str("<start_of_turn>user\n");
                    if !system_prompt.is_empty() {
                        prompt.push_str(&system_prompt);
                        prompt.push_str("\n\n");
                        system_prompt.clear();
                    }
                    prompt.push_str(&msg.content);
                    prompt.push_str("<end_of_turn>\n");
                }
                "assistant" => {
                    prompt.push_str("<start_of_turn>model\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<end_of_turn>\n");
                }
                _ => {
                    prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
                }
            }
        }

        if !system_prompt.is_empty() {
            prompt.insert_str(
                0,
                &format!("<start_of_turn>user\n{}\n<end_of_turn>\n", system_prompt),
            );
        }

        prompt.push_str("<start_of_turn>model\n");
        prompt
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<GenerationResult> {
        if contains_unsafe_word(prompt) {
            return Ok(GenerationResult {
                text: REFUSAL.to_string(),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".to_string(),
            });
        }
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenizer encode error: {e}"))?;
        let prompt_tokens = encoding.get_ids().to_vec();

        if prompt_tokens.is_empty() {
            return Ok(GenerationResult {
                text: String::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".to_string(),
            });
        }

        let seed = params.seed.unwrap_or_else(rand::random);
        let temp = if params.temperature <= 0.0 {
            None
        } else {
            Some(params.temperature)
        };
        let mut lp = LogitsProcessor::new(seed, temp, params.top_p);

        let mut all_tokens = prompt_tokens.clone();
        let mut output_tokens = Vec::new();

        let input_tensor = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input_tensor, 0)?;

        let mut finish_reason = "length".to_string();

        let stop_words = vec![
            "<end_of_turn>".to_string(),
            "<eos>".to_string(),
            "### Question:".to_string(),
        ];

        for _ in 0..params.max_tokens {
            let last_logits = logits.squeeze(0)?.squeeze(0)?;
            let next_token = lp.sample(&last_logits)?;

            if next_token == self.eos_token_id {
                finish_reason = "stop".to_string();
                break;
            }

            output_tokens.push(next_token);
            all_tokens.push(next_token);

            // Check stop sequences on partial decoded text
            if let Ok(current_text) = self.tokenizer.decode(&output_tokens, false) {
                let mut should_stop = false;
                for stop in params.stop.iter().chain(stop_words.iter()) {
                    if current_text.contains(stop) {
                        should_stop = true;
                        break;
                    }
                }
                if should_stop {
                    finish_reason = "stop".to_string();
                    break;
                }
            }

            let next_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&next_tensor, all_tokens.len() - 1)?;
        }

        self.model.clear_kv_cache();

        let mut output_text = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(|e| anyhow!("Tokenizer decode error: {e}"))?;

        // Clean any stop token markers from output text
        for stop in params.stop.iter().chain(stop_words.iter()) {
            if let Some(pos) = output_text.find(stop) {
                output_text.truncate(pos);
            }
        }

        if contains_unsafe_word(&output_text) {
            output_text = REFUSAL.to_string();
        }

        Ok(GenerationResult {
            text: output_text.trim().to_string(),
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: output_tokens.len(),
            finish_reason,
        })
    }

    pub fn generate_stream_tokens<F>(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        mut callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(&str) -> Result<()>,
    {
        if contains_unsafe_word(prompt) {
            callback(REFUSAL)?;
            return Ok(GenerationResult {
                text: REFUSAL.to_string(),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".to_string(),
            });
        }
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenizer encode error: {e}"))?;
        let prompt_tokens = encoding.get_ids().to_vec();

        if prompt_tokens.is_empty() {
            return Ok(GenerationResult {
                text: String::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".to_string(),
            });
        }

        let seed = params.seed.unwrap_or_else(rand::random);
        let temp = if params.temperature <= 0.0 {
            None
        } else {
            Some(params.temperature)
        };
        let mut lp = LogitsProcessor::new(seed, temp, params.top_p);

        let mut all_tokens = prompt_tokens.clone();
        let mut output_tokens = Vec::new();

        let input_tensor = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input_tensor, 0)?;

        let mut finish_reason = "length".to_string();
        let mut prev_decoded_len = 0;
        let mut full_output = String::new();

        let stop_words = vec![
            "<end_of_turn>".to_string(),
            "<eos>".to_string(),
            "### Question:".to_string(),
        ];

        for _ in 0..params.max_tokens {
            let last_logits = logits.squeeze(0)?.squeeze(0)?;
            let next_token = lp.sample(&last_logits)?;

            if next_token == self.eos_token_id {
                finish_reason = "stop".to_string();
                break;
            }

            output_tokens.push(next_token);
            all_tokens.push(next_token);

            if let Ok(current_text) = self.tokenizer.decode(&output_tokens, false) {
                let mut hit_stop = false;
                for stop in params.stop.iter().chain(stop_words.iter()) {
                    if current_text.contains(stop) {
                        hit_stop = true;
                        break;
                    }
                }
                if hit_stop {
                    finish_reason = "stop".to_string();
                    break;
                }

                if current_text.len() > prev_decoded_len {
                    let delta = &current_text[prev_decoded_len..];
                    full_output.push_str(delta);
                    callback(delta)?;
                    prev_decoded_len = current_text.len();
                }
            }

            let next_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&next_tensor, all_tokens.len() - 1)?;
        }

        self.model.clear_kv_cache();

        Ok(GenerationResult {
            text: full_output.trim().to_string(),
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: output_tokens.len(),
            finish_reason,
        })
    }
}
