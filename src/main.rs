pub mod cli;
mod gguf;
mod math;
pub mod model;
pub mod quantized_key_value_cache;
mod quantized_tensor;
mod sampler;
mod tokenizer;
mod transformer;

use crate::cli::*;
use crate::{model::*, sampler::Sampler};
use anyhow::Result;
use std::io::{BufRead, Write};
use std::time::Duration;

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

fn build_prompt_from_messages(format: PromptFormat, messages: &[Message]) -> String {
    match format {
        PromptFormat::Raw => {
            let mut result = String::new();

            for message in messages {
                match message.role {
                    Role::System => {
                        result.push_str("system:\n");
                        result.push_str(&message.content);
                        result.push('\n');
                    }
                    Role::User => {
                        result.push_str("user:\n");
                        result.push_str(&message.content);
                        result.push('\n');
                    }
                    Role::Assistant => {
                        result.push_str("assistant:\n");
                        result.push_str(&message.content);
                        result.push('\n');
                    }
                }
            }

            result
        }

        PromptFormat::Llama3 => {
            let mut result = String::new();

            result.push_str("<|begin_of_text|>");

            for message in messages {
                let role = match message.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };

                result.push_str("<|start_header_id|>");
                result.push_str(role);
                result.push_str("<|end_header_id|>\n");
                result.push_str(&message.content);
                result.push_str("<|eot_id|>");
            }

            if !matches!(messages.last().map(|m| m.role), Some(Role::Assistant)) {
                result.push_str("<|start_header_id|>assistant<|end_header_id|>\n");
            }

            result
        }

        PromptFormat::ChatMarkupLanguage => {
            let mut result = String::new();

            for message in messages {
                let role = match message.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };

                result.push_str("<|im_start|>");
                result.push_str(role);
                result.push('\n');
                result.push_str(&message.content);
                result.push_str("<|im_end|>\n");
            }

            if !matches!(messages.last().map(|m| m.role), Some(Role::Assistant)) {
                result.push_str("<|im_start|>assistant\n");
            }

            result
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

            let mut result = String::new();

            for (turn_index, (user_text, assistant_text)) in turns.iter().enumerate() {
                if turn_index == 0 {
                    result.push_str("<s>[INST] <<SYS>>\n");
                    result.push_str(&system_text);
                    result.push_str("\n<</SYS>>\n\n");
                    result.push_str(user_text);
                    result.push_str(" [/INST]");
                } else {
                    result.push_str("<s>[INST] ");
                    result.push_str(user_text);
                    result.push_str(" [/INST]");
                }

                if let Some(assistant_text) = assistant_text {
                    result.push(' ');
                    result.push_str(assistant_text);
                    result.push_str(" </s>");
                } else {
                    result.push(' ');
                }
            }

            result
        }
    }
}

pub fn print_turn_statistics(
    total_prompt_tokens: usize,
    newly_processed_prompt_tokens: usize,
    prompt_duration: std::time::Duration,
    final_position: usize,
    generation_duration: std::time::Duration,
) {
    let prompt_seconds = (prompt_duration.as_millis() as f64) / 1000.0;
    let prompt_tokens_per_second = if prompt_seconds > 0.0 {
        (newly_processed_prompt_tokens as f64) / prompt_seconds
    } else {
        f64::INFINITY
    };

    let generated_tokens = final_position.saturating_sub(total_prompt_tokens);
    let generation_seconds = (generation_duration.as_millis() as f64) / 1000.0;
    let generation_tokens_per_second = if generation_seconds > 0.0 {
        (generated_tokens as f64) / generation_seconds
    } else {
        f64::INFINITY
    };

    println!();
    println!(
        "==================================================================================================================================="
    );
    println!(
        "prompt tokens: {total_prompt_tokens} | prompt new tokens: {newly_processed_prompt_tokens} | prompt tokens per second: {:.02} | generation tokens per second: {:.02}",
        prompt_tokens_per_second, generation_tokens_per_second
    );
    println!(
        "==================================================================================================================================="
    );
}

fn generate_streaming_to_stdout(
    model: &mut Model,
    sampler: &mut Sampler,
    start_position: usize,
    maximum_generated_tokens: usize,
) -> Result<(Vec<u32>, usize, Duration)> {
    use std::time::Instant;

    let maximum_sequence_length = model.transformer.configuration.maximum_sequence_length as usize;

    let start_generation = Instant::now();

    let mut position = start_position;
    let mut generated_token_identifiers: Vec<u32> = Vec::new();

    let mut pending_bytes: Vec<u8> = Vec::new();

    while position < maximum_sequence_length && position < start_position + maximum_generated_tokens
    {
        let next_token = sampler.sample_next_token(&mut model.transformer.state.logits);

        generated_token_identifiers.push(next_token);

        if next_token == model.tokenizer.end_of_sequence_token_identifier {
            break;
        }

        pending_bytes.extend_from_slice(&model.tokenizer.decode_token_to_bytes(next_token));

        loop {
            match std::str::from_utf8(&pending_bytes) {
                Ok(valid) => {
                    print!("{valid}");

                    pending_bytes.clear();

                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();

                    if valid_up_to == 0 {
                        break;
                    }

                    let valid_prefix =
                        unsafe { std::str::from_utf8_unchecked(&pending_bytes[..valid_up_to]) };

                    print!("{valid_prefix}");

                    pending_bytes.drain(..valid_up_to);
                }
            }
        }

        let _ = std::io::stdout().flush();

        model.transformer.forward(next_token, position)?;

        position += 1;
    }

    if !pending_bytes.is_empty() {
        print!("{}", String::from_utf8_lossy(&pending_bytes));

        let _ = std::io::stdout().flush();
    }

    let generation_duration = start_generation.elapsed();

    Ok((generated_token_identifiers, position, generation_duration))
}

fn run_inference(parameters: &InferenceParameters) -> Result<()> {
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
            content: parameters.common.user_prompt.clone(),
        },
    ];

    let prompt = build_prompt_from_messages(parameters.common.prompt_format, &messages);

    let prompt_token_identifiers = model.tokenizer.encode(&prompt)?;

    let (newly_processed_prompt_tokens, prompt_duration) =
        model.transformer.ingest_tokens_into_cache(
            &prompt_token_identifiers,
            0,
            parameters.common.prompt_batch_size,
        )?;

    let total_prompt_tokens = prompt_token_identifiers.len();
    let start_position = total_prompt_tokens;

    let (_generated_token_identifiers, final_position, generation_duration) =
        generate_streaming_to_stdout(
            &mut model,
            &mut sampler,
            start_position,
            parameters.common.maximum_generated_tokens,
        )?;

    print_turn_statistics(
        total_prompt_tokens,
        newly_processed_prompt_tokens,
        prompt_duration,
        final_position,
        generation_duration,
    );

    Ok(())
}

fn run_chat(parameters: &ChatParameters) -> Result<()> {
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

        println!("{prompt:?}");

        let prompt_token_identifiers = model.tokenizer.encode(&prompt)?;

        let (newly_processed_prompt_tokens, prompt_duration) =
            model.transformer.ingest_tokens_into_cache(
                &prompt_token_identifiers,
                0,
                parameters.common.prompt_batch_size,
            )?;

        let total_prompt_tokens = prompt_token_identifiers.len();
        let start_position = total_prompt_tokens;

        print!("assistant> ");
        let _ = std::io::stdout().flush();

        let (generated_token_identifiers, position, generation_duration) =
            generate_streaming_to_stdout(
                &mut model,
                &mut sampler,
                start_position,
                parameters.common.maximum_generated_tokens,
            )?;

        println!();

        let assistant_text = model.tokenizer.decode(&generated_token_identifiers)?;

        messages.push(Message {
            role: Role::Assistant,
            content: assistant_text,
        });

        print_turn_statistics(
            total_prompt_tokens,
            newly_processed_prompt_tokens,
            prompt_duration,
            position,
            generation_duration,
        );
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
