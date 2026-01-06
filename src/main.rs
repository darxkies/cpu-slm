#![allow(dead_code)]
#![allow(unused_variables)]

mod gguf;
mod math;
pub mod model;
pub mod quantized_key_value_cache;
mod quantized_tensor;
mod sampler;
mod tokenizer;
mod transformer;

use crate::{model::*, sampler::Sampler};
use anyhow::{Result, bail};
use std::io::{BufRead, Write};
use std::time::{SystemTime, UNIX_EPOCH};

fn current_unix_time_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptFormat {
    Raw,
    Llama2,
    Llama3,
    ChatMarkupLanguage,
}

impl PromptFormat {
    fn parse(value: &str) -> Result<Self> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
struct Message {
    role: Role,
    content: String,
}

#[derive(Debug, Clone)]
struct CommonParameters {
    model_path: String,

    temperature: f32,
    top_probability: f32,
    seed: u64,

    prompt_batch_size: usize,
    maximum_sequence_length: usize,
    maximum_generated_tokens: usize,

    prompt_format: PromptFormat,
    system_prompt: String,
}

#[derive(Debug, Clone)]
struct InferenceParameters {
    common: CommonParameters,
    user_prompt: String,
}

#[derive(Debug, Clone)]
struct ChatParameters {
    common: CommonParameters,
}

#[derive(Debug, Clone)]
enum Subcommand {
    Chat(ChatParameters),
    Inference(InferenceParameters),
}

fn print_usage(program_name: &str) {
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

fn require_value(flag: &str, value: Option<String>) -> Result<String> {
    value.ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
}

fn parse_f32(flag: &str, value: &str) -> Result<f32> {
    value
        .parse::<f32>()
        .map_err(|e| anyhow::anyhow!("invalid float for {flag}: {value:?} ({e})"))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|e| anyhow::anyhow!("invalid unsigned integer for {flag}: {value:?} ({e})"))
}

fn parse_usize(flag: &str, value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|e| anyhow::anyhow!("invalid unsigned integer for {flag}: {value:?} ({e})"))
}

fn build_prompt_from_messages(format: PromptFormat, messages: &[Message]) -> String {
    match format {
        PromptFormat::Raw => {
            let mut out = String::new();
            for message in messages {
                match message.role {
                    Role::System => {
                        out.push_str("system:\n");
                        out.push_str(&message.content);
                        out.push('\n');
                    }
                    Role::User => {
                        out.push_str("user:\n");
                        out.push_str(&message.content);
                        out.push('\n');
                    }
                    Role::Assistant => {
                        out.push_str("assistant:\n");
                        out.push_str(&message.content);
                        out.push('\n');
                    }
                }
            }
            out
        }

        PromptFormat::Llama3 => {
            let mut out = String::new();

            out.push_str("<|begin_of_text|>");

            for message in messages {
                let role = match message.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };

                out.push_str("<|start_header_id|>");
                out.push_str(role);
                out.push_str("<|end_header_id|>\n");
                out.push_str(&message.content);
                out.push_str("<|eot_id|>");
            }

            if !matches!(messages.last().map(|m| m.role), Some(Role::Assistant)) {
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n");
            }

            out
        }

        PromptFormat::ChatMarkupLanguage => {
            let mut out = String::new();

            for message in messages {
                let role = match message.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };

                out.push_str("<|im_start|>");
                out.push_str(role);
                out.push('\n');
                out.push_str(&message.content);
                out.push_str("<|im_end|>\n");
            }

            if !matches!(messages.last().map(|m| m.role), Some(Role::Assistant)) {
                out.push_str("<|im_start|>assistant\n");
            }

            out
        }

        PromptFormat::Llama2 => {
            let mut system_text = String::new();
            let mut turns: Vec<(String, Option<String>)> = Vec::new();

            for message in messages {
                match message.role {
                    Role::System => system_text = message.content.clone(),
                    Role::User => turns.push((message.content.clone(), None)),
                    Role::Assistant => {
                        if let Some(last) = turns.last_mut() {
                            last.1 = Some(message.content.clone());
                        }
                    }
                }
            }

            let mut out = String::new();

            for (turn_index, (user_text, assistant_text)) in turns.iter().enumerate() {
                if turn_index == 0 {
                    out.push_str("<s>[INST] <<SYS>>\n");
                    out.push_str(&system_text);
                    out.push_str("\n<</SYS>>\n\n");
                    out.push_str(user_text);
                    out.push_str(" [/INST]");
                } else {
                    out.push_str("<s>[INST] ");
                    out.push_str(user_text);
                    out.push_str(" [/INST]");
                }

                if let Some(assistant_text) = assistant_text {
                    out.push(' ');
                    out.push_str(assistant_text);
                    out.push_str(" </s>");
                } else {
                    out.push(' ');
                }
            }

            out
        }
    }
}

fn parse_common_parameters(
    arguments: &[String],
    start_index: usize,
) -> Result<(CommonParameters, usize)> {
    let mut model_path: Option<String> = None;

    let mut temperature: f32 = 0.7;
    let mut top_probability: f32 = 0.9;
    let mut seed: u64 = current_unix_time_seconds();

    let mut prompt_batch_size: usize = 32;
    let mut maximum_sequence_length: usize = 8192;
    let mut maximum_generated_tokens: usize = 2048;

    let mut prompt_format: PromptFormat = PromptFormat::ChatMarkupLanguage;
    let mut system_prompt: String = "You are a helpful assistant.".to_string();

    let mut index = start_index;

    while index < arguments.len() {
        let flag = arguments[index].as_str();

        if flag == "--user-prompt" {
            break;
        }

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

    Ok((
        CommonParameters {
            model_path,
            temperature,
            top_probability,
            seed,
            prompt_batch_size,
            maximum_sequence_length,
            maximum_generated_tokens,
            prompt_format,
            system_prompt,
        },
        index,
    ))
}

fn parse_subcommand() -> Result<Subcommand> {
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
            let (common, _index) = parse_common_parameters(&arguments, 2)?;

            Ok(Subcommand::Chat(ChatParameters { common }))
        }

        "inference" => {
            let (common, mut index) = parse_common_parameters(&arguments, 2)?;
            let mut user_prompt: Option<String> = None;

            while index < arguments.len() {
                let flag = arguments[index].as_str();

                match flag {
                    "--user-prompt" => {
                        let value =
                            require_value("--user-prompt", arguments.get(index + 1).cloned())?;

                        user_prompt = Some(value);

                        index += 2;
                    }
                    other => {
                        bail!(
                            "unknown command line parameter {other:?}. Use --help to see valid parameters."
                        );
                    }
                }
            }

            let Some(user_prompt) = user_prompt else {
                bail!("missing required parameter --user-prompt <string> for inference.");
            };

            Ok(Subcommand::Inference(InferenceParameters {
                common,
                user_prompt,
            }))
        }

        other => bail!(
            "unknown subcommand {other:?}. Expected chat, or inference. Use --help for usage."
        ),
    }
}

fn run_inference(parameters: &InferenceParameters) -> Result<()> {
    use std::time::Instant;

    println!("reading: {}", parameters.common.model_path);

    let mut model = Model::load_from_file(
        &parameters.common.model_path,
        parameters.common.maximum_sequence_length,
    )?;

    let mut sampler = Sampler::new(
        model.transformer.configuration.vocabulary_size as usize,
        parameters.common.temperature,
        parameters.common.top_probability,
        parameters.common.seed,
    );

    let messages = vec![
        Message {
            role: Role::System,
            content: parameters.common.system_prompt.clone(),
        },
        Message {
            role: Role::User,
            content: parameters.user_prompt.clone(),
        },
    ];

    let prompt = build_prompt_from_messages(parameters.common.prompt_format, &messages);
    let prompt_token_identifiers = model.tokenizer.encode(&prompt)?;

    model.transformer.state.key_value_cache.clear();

    let start_prompt = Instant::now();

    model.transformer.batched_forward(
        &prompt_token_identifiers,
        0,
        parameters.common.prompt_batch_size,
    )?;

    let stop_prompt = start_prompt.elapsed();

    let mut position = prompt_token_identifiers.len();
    let maximum_sequence_length = model.transformer.configuration.maximum_sequence_length as usize;

    let start_generation = Instant::now();

    let mut generated_token_identifiers: Vec<u32> = Vec::new();
    let mut printed_prefix_length_in_bytes: usize = 0;

    while position < maximum_sequence_length
        && position < prompt_token_identifiers.len() + parameters.common.maximum_generated_tokens
    {
        let next_token = sampler.sample_next_token(&mut model.transformer.state.logits);

        generated_token_identifiers.push(next_token);

        let decoded = model.tokenizer.decode(&generated_token_identifiers)?;
        let decoded_bytes_len = decoded.as_bytes().len();

        if decoded_bytes_len > printed_prefix_length_in_bytes {
            let new_text = &decoded[printed_prefix_length_in_bytes..];

            print!("{new_text}");

            let _ = std::io::stdout().flush();

            printed_prefix_length_in_bytes = decoded_bytes_len;
        }

        if next_token == model.tokenizer.end_of_sequence_token_identifier {
            break;
        }

        model.transformer.forward(next_token, position)?;

        position += 1;
    }

    {
        let prompt_seconds = (stop_prompt.as_millis() as f64) / 1000.0;
        let prompt_tokens_per_second = if prompt_seconds > 0.0 {
            prompt_token_identifiers.len() as f64 / prompt_seconds
        } else {
            f64::INFINITY
        };

        let generated_tokens = position.saturating_sub(prompt_token_identifiers.len());
        let generated_seconds = (start_generation.elapsed().as_millis() as f64) / 1000.0;
        let generation_tokens_per_second = if generated_seconds > 0.0 {
            generated_tokens as f64 / generated_seconds
        } else {
            f64::INFINITY
        };

        println!(
            "\n\ntokens: {position} | prompt tokens per second: {:.02} | generation tokens per second: {:.02}",
            prompt_tokens_per_second, generation_tokens_per_second
        );
    }

    Ok(())
}

fn run_chat(parameters: &ChatParameters) -> Result<()> {
    use std::time::Instant;

    println!("reading: {}", parameters.common.model_path);

    let mut model = Model::load_from_file(
        &parameters.common.model_path,
        parameters.common.maximum_sequence_length,
    )?;

    let mut sampler = Sampler::new(
        model.transformer.configuration.vocabulary_size as usize,
        parameters.common.temperature,
        parameters.common.top_probability,
        parameters.common.seed,
    );

    println!("enter messages. Press Control and D to exit.");
    println!();

    let mut messages: Vec<Message> = vec![Message {
        role: Role::System,
        content: parameters.common.system_prompt.clone(),
    }];

    let standard_input = std::io::stdin();
    let mut reader = std::io::BufReader::new(standard_input.lock());

    loop {
        print!("user> ");
        let _ = std::io::stdout().flush();

        let mut user_line = String::new();
        let bytes_read = reader.read_line(&mut user_line)?;

        if bytes_read == 0 {
            println!();
            break;
        }

        let user_line = user_line.trim_end_matches(['\r', '\n']).to_string();
        if user_line.is_empty() {
            continue;
        }

        messages.push(Message {
            role: Role::User,
            content: user_line,
        });

        let prompt = build_prompt_from_messages(parameters.common.prompt_format, &messages);
        let prompt_token_identifiers = model.tokenizer.encode(&prompt)?;

        model.transformer.state.key_value_cache.clear();

        let start_prompt = Instant::now();
        model.transformer.batched_forward(
            &prompt_token_identifiers,
            0,
            parameters.common.prompt_batch_size,
        )?;
        let stop_prompt = start_prompt.elapsed();

        let mut position = prompt_token_identifiers.len();
        let maximum_sequence_length =
            model.transformer.configuration.maximum_sequence_length as usize;

        print!("assistant> ");
        let _ = std::io::stdout().flush();

        let start_generation = Instant::now();

        // Fix: collect generated token identifiers and decode them as a stream.
        let mut generated_token_identifiers: Vec<u32> = Vec::new();
        let mut printed_prefix_length_in_bytes: usize = 0;

        while position < maximum_sequence_length
            && position
                < prompt_token_identifiers.len() + parameters.common.maximum_generated_tokens
        {
            let next_token = sampler.sample_next_token(&mut model.transformer.state.logits);

            generated_token_identifiers.push(next_token);

            let decoded = model.tokenizer.decode(&generated_token_identifiers)?;
            let decoded_bytes_len = decoded.as_bytes().len();

            if decoded_bytes_len > printed_prefix_length_in_bytes {
                let new_text = &decoded[printed_prefix_length_in_bytes..];
                print!("{new_text}");
                let _ = std::io::stdout().flush();
                printed_prefix_length_in_bytes = decoded_bytes_len;
            }

            if next_token == model.tokenizer.end_of_sequence_token_identifier {
                break;
            }

            model.transformer.forward(next_token, position)?;
            position += 1;
        }

        println!();

        let assistant_text = model.tokenizer.decode(&generated_token_identifiers)?;

        messages.push(Message {
            role: Role::Assistant,
            content: assistant_text,
        });

        {
            let prompt_seconds = (stop_prompt.as_millis() as f64) / 1000.0;
            let prompt_tokens_per_second = if prompt_seconds > 0.0 {
                prompt_token_identifiers.len() as f64 / prompt_seconds
            } else {
                f64::INFINITY
            };

            let generated_tokens = position.saturating_sub(prompt_token_identifiers.len());
            let generated_seconds = (start_generation.elapsed().as_millis() as f64) / 1000.0;
            let generation_tokens_per_second = if generated_seconds > 0.0 {
                generated_tokens as f64 / generated_seconds
            } else {
                f64::INFINITY
            };

            println!(
                "tokens: {position} | prompt tokens per second: {:.02} | generation tokens per second: {:.02}",
                prompt_tokens_per_second, generation_tokens_per_second
            );
            println!();
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let subcommand = parse_subcommand()?;

    match subcommand {
        Subcommand::Chat(parameters) => run_chat(&parameters),
        Subcommand::Inference(parameters) => run_inference(&parameters),
    }
}
