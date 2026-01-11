use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use anyhow::bail;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptFormat {
    Raw,
    Llama2,
    Llama3,
    ChatMarkupLanguage,
}

impl PromptFormat {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "raw" => Ok(Self::Raw),
            "llama2" => Ok(Self::Llama2),
            "llama3" => Ok(Self::Llama3),
            "chatml" => Ok(Self::ChatMarkupLanguage),
            other => {
                bail!("unknown prompt format {other:?}. Expected raw, llama2, llama3, or chatml.")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CommonParameters {
    pub model_path: String,

    pub temperature: f32,
    pub top_probability: f32,
    pub seed: u64,

    pub prompt_batch_size: usize,
    pub maximum_sequence_length: usize,
    pub maximum_generated_tokens: usize,

    pub prompt_format: PromptFormat,
    pub system_prompt: String,
    pub user_prompt: String,
}

#[derive(Debug, Clone)]
pub struct InferenceParameters {
    pub common: CommonParameters,
}

#[derive(Debug, Clone)]
pub struct ChatParameters {
    pub common: CommonParameters,
}

#[derive(Debug, Clone)]
pub enum Subcommand {
    Chat(ChatParameters),
    Inference(InferenceParameters),
}

pub fn print_usage(program_name: &str) {
    println!(
        r#"Usage:
  {program_name} <subcommand> [options]

Subcommands:
  chat        Interactive multi turn chat: reads user input from standard input and prints model output.
  inference   Single inference: runs once and exits.

Common options for chat and inference:
  --model-path <path>                     Path to the .gguf file (required for chat and inference)

  --temperature <float>                   Temperature. Use 0 for greedy sampling. Default: 0.7
  --top-probability <float>               Top probability (nucleus). Range: 0..=1. Default: 0.9
  --seed <unsigned integer>               Random seed. Default: current Unix time

  --prompt-batch-size <unsigned integer>  Prompt batch size for batched forward. Default: 32
  --maximum-sequence-length <integer>     Maximum sequence length. Set it to 0 to use the model's sequence length. Default: 8192
  --maximum-generated-tokens <integer>    Maximum number of generated tokens per assistant turn. Default: 256

  --prompt-format <raw|llama2|llama3|chatml> Prompt formatting. Default: llama3
  --system-prompt <string>                System prompt content. Default: "You are a helpful assistant."

Inference only options:
  --user-prompt <string>                  User prompt content (required for inference)

Help:
  --help                                  Print this help message

Examples:
  {program_name} chat --model-path ./model.gguf

  {program_name} inference --model-path ./model.gguf \
    --system-prompt "You are a helpful assistant." \
    --user-prompt "Explain rotary positional embeddings."

  {program_name} chat --model-path ./model.gguf --temperature 0.0 --top-probability 1.0
"#,
        program_name = program_name
    );
}

pub fn require_value(flag: &str, value: Option<String>) -> Result<String> {
    value.ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
}

pub fn parse_f32(flag: &str, value: &str) -> Result<f32> {
    value
        .parse::<f32>()
        .map_err(|e| anyhow::anyhow!("invalid float for {flag}: {value:?} ({e})"))
}

pub fn parse_u64(flag: &str, value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|e| anyhow::anyhow!("invalid unsigned integer for {flag}: {value:?} ({e})"))
}

pub fn parse_usize(flag: &str, value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|e| anyhow::anyhow!("invalid unsigned integer for {flag}: {value:?} ({e})"))
}

fn current_unix_time_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn parse_common_parameters(arguments: &[String], start_index: usize) -> Result<CommonParameters> {
    let mut model_path: Option<String> = None;

    let mut temperature: f32 = 0.7;
    let mut top_probability: f32 = 0.9;
    let mut seed: u64 = current_unix_time_seconds();

    let mut prompt_batch_size: usize = 32;
    let mut maximum_sequence_length: usize = 8192;
    let mut maximum_generated_tokens: usize = 8192;

    let mut prompt_format: PromptFormat = PromptFormat::ChatMarkupLanguage;
    let mut system_prompt: String = "You are a helpful assistant.".to_string();
    let mut user_prompt: String = "Why the sky blue?".to_string();

    let mut index = start_index;

    while index < arguments.len() {
        let flag = arguments[index].as_str();

        match flag {
            "--model-path" => {
                let value = require_value("--model-path", arguments.get(index + 1).cloned())?;

                model_path = Some(value);

                index += 2;
            }

            "--temperature" => {
                let value = require_value("--temperature", arguments.get(index + 1).cloned())?;

                temperature = parse_f32("--temperature", &value)?;

                index += 2;
            }

            "--top-probability" => {
                let value = require_value("--top-probability", arguments.get(index + 1).cloned())?;

                top_probability = parse_f32("--top-probability", &value)?;

                index += 2;
            }

            "--seed" => {
                let value = require_value("--seed", arguments.get(index + 1).cloned())?;

                seed = parse_u64("--seed", &value)?;

                index += 2;
            }

            "--prompt-batch-size" => {
                let value =
                    require_value("--prompt-batch-size", arguments.get(index + 1).cloned())?;

                prompt_batch_size = parse_usize("--prompt-batch-size", &value)?;

                index += 2;
            }

            "--maximum-sequence-length" => {
                let value = require_value(
                    "--maximum-sequence-length",
                    arguments.get(index + 1).cloned(),
                )?;

                maximum_sequence_length = parse_usize("--maximum-sequence-length", &value)?;

                index += 2;
            }

            "--maximum-generated-tokens" => {
                let value = require_value(
                    "--maximum-generated-tokens",
                    arguments.get(index + 1).cloned(),
                )?;

                maximum_generated_tokens = parse_usize("--maximum-generated-tokens", &value)?;

                index += 2;
            }

            "--prompt-format" => {
                let value = require_value("--prompt-format", arguments.get(index + 1).cloned())?;

                prompt_format = PromptFormat::parse(&value)?;

                index += 2;
            }

            "--system-prompt" => {
                let value = require_value("--system-prompt", arguments.get(index + 1).cloned())?;

                system_prompt = value;

                index += 2;
            }

            "--user-prompt" => {
                let value = require_value("--user-prompt", arguments.get(index + 1).cloned())?;

                user_prompt = value;

                index += 2;
            }

            other => {
                bail!(
                    "unknown command line parameter {other:?}. Use --help to see valid parameters."
                );
            }
        }
    }

    let Some(model_path) = model_path else {
        bail!("missing required parameter --model-path <path>.");
    };

    if !(0.0..=1.0).contains(&top_probability) {
        bail!("top probability must be within 0.0 and 1.0 (inclusive). Got {top_probability}.");
    }

    if temperature < 0.0 {
        bail!("temperature must be non negative. Got {temperature}.");
    }

    Ok(CommonParameters {
        model_path,
        temperature,
        top_probability,
        seed,
        prompt_batch_size,
        maximum_sequence_length,
        maximum_generated_tokens,
        prompt_format,
        system_prompt,
        user_prompt,
    })
}

pub fn parse_subcommand() -> Result<Subcommand> {
    let arguments: Vec<String> = std::env::args().collect();
    let program_name = arguments
        .get(0)
        .cloned()
        .unwrap_or_else(|| "program".to_string());

    if arguments.iter().any(|a| a == "--help") || arguments.len() < 2 {
        print_usage(&program_name);

        std::process::exit(0);
    }

    let subcommand = arguments[1].as_str();

    match subcommand {
        "chat" => {
            let common = parse_common_parameters(&arguments, 2)?;

            Ok(Subcommand::Chat(ChatParameters { common }))
        }

        "inference" => {
            let common = parse_common_parameters(&arguments, 2)?;

            Ok(Subcommand::Inference(InferenceParameters { common }))
        }

        other => bail!(
            "unknown subcommand {other:?}. Expected chat, or inference. Use --help for usage."
        ),
    }
}
