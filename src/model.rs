#![allow(unused_variables)]
use anyhow::Result;
use anyhow::bail;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::path::Path;
use std::path::PathBuf;

use crate::gguf::GgmlType;
use crate::gguf::GgufReader;
use crate::gguf::{GgufHeader, MetadataValue};
use crate::math::RotaryPositionalEmbeddingCache;
use crate::quantized_key_value_cache::QuantizedKeyValueCache;
use crate::quantized_tensor::QuantizedTensor;
use crate::tokenizer::BPETokenizer;
use crate::tokenizer::TokenizerType;
use crate::transformer::Transformer;
use crate::transformer::TransformerLayer;
use crate::transformer::TransformerRuntimeState;
use crate::transformer::TransformerWeights;

#[derive(Clone, Debug, Default)]
pub struct ModelConfiguration {
    pub architecture: String,
    pub model_hidden_dimension: i32,
    pub feed_forward_hidden_dimension: i32,
    pub number_of_layers: i32,
    pub number_of_attention_heads: i32,
    pub number_of_key_value_heads: i32,
    pub vocabulary_size: i32,
    pub maximum_sequence_length: i32,
    pub attention_head_dimension: i32,
    pub rotary_embedding_frequency_base: f32,
    pub use_query_and_key_normalization: bool,
    pub layer_normalization_root_mean_square_epsilon: f32,
}

pub struct Model {
    pub transformer: Transformer,
    pub tokenizer: BPETokenizer,
}

impl Model {
    fn load_configuration(header: &GgufHeader) -> Result<ModelConfiguration> {
        let mut result = ModelConfiguration::default();

        let value = |name: &'static str| -> Result<MetadataValue> {
            let Some(value) = header.metadata.get(name) else {
                bail!("missing key {name:?}");
            };

            Ok(value.clone())
        };

        let unsigned32 = |name: &'static str| -> Result<u32> {
            let value = value(name)?;

            let MetadataValue::Unsigned32(value) = value else {
                bail!("expected unsigned32 for {name:?} ({value:?})");
            };

            Ok(value)
        };

        let unsigned32_any = |names: &[&'static str]| -> Result<u32> {
            for &name in names {
                if let Some(value) = header.metadata.get(name) {
                    let MetadataValue::Unsigned32(value) = value else {
                        bail!("expected unsigned32 for {name:?} ({value:?})");
                    };

                    return Ok(*value);
                }
            }

            bail!("missing any of keys {names:?}");
        };

        let string = |name: &'static str| -> Result<String> {
            let value = value(name)?;

            let MetadataValue::String(value) = value else {
                bail!("expected string for {name:?} ({value:?})");
            };

            Ok(value)
        };

        let string_array = |name: &'static str| -> Result<Vec<String>> {
            let value = value(name)?;

            let MetadataValue::StringArray(value) = value else {
                bail!("expected a string array for {name:?} ({value:?})");
            };

            Ok(value)
        };

        let float32_any = |names: &[&'static str]| -> Result<f32> {
            for &name in names {
                if let Some(value) = header.metadata.get(name) {
                    match value {
                        MetadataValue::Float32(value) => return Ok(*value),
                        MetadataValue::Unsigned32(value) => return Ok(*value as f32),
                        _ => bail!("expected float32 for {name:?} ({value:?})"),
                    }
                }
            }

            bail!("missing any of keys {names:?}");
        };

        let has_tensor = |name: &str| -> bool { header.tensors.iter().any(|t| t.name == name) };

        let quantization_version = unsigned32("general.quantization_version")?;

        if quantization_version != 2 {
            bail!("unsupported quantization version {quantization_version}");
        }

        let file_type = GgmlType::from_u32(unsigned32("general.file_type")? + 1)?;

        if file_type != GgmlType::Quantization8_0 {
            bail!("only file type Q8_0 is supported ({file_type:?})");
        }

        result.use_query_and_key_normalization =
            has_tensor("blk.0.attn_q_norm.weight") && has_tensor("blk.0.attn_k_norm.weight");

        result.architecture = string("general.architecture")?;
        result.maximum_sequence_length = unsigned32_any(&[
            "mistral3.context_length",
            "llama.context_length",
            "qwen3.context_length",
        ])? as i32;
        result.model_hidden_dimension = unsigned32_any(&[
            "mistral3.embedding_length",
            "llama.embedding_length",
            "qwen3.embedding_length",
        ])? as i32;
        result.feed_forward_hidden_dimension = unsigned32_any(&[
            "mistral3.feed_forward_length",
            "llama.feed_forward_length",
            "qwen3.feed_forward_length",
        ])? as i32;
        result.attention_head_dimension = unsigned32_any(&[
            "mistral3.attention.key_length",
            "llama.attention.key_length",
            "qwen3.attention.key_length",
        ])? as i32;
        result.number_of_layers = unsigned32_any(&[
            "mistral3.block_count",
            "llama.block_count",
            "qwen3.block_count",
        ])? as i32;
        result.number_of_attention_heads = unsigned32_any(&[
            "mistral3.attention.head_count",
            "llama.attention.head_count",
            "qwen3.attention.head_count",
        ])? as i32;
        result.number_of_key_value_heads = unsigned32_any(&[
            "mistral3.attention.head_count_kv",
            "llama.attention.head_count_kv",
            "qwen3.attention.head_count_kv",
        ])? as i32;

        result.rotary_embedding_frequency_base = 10_000.0f32;

        if let Ok(value) = float32_any(&[
            "mistral3.rope.freq_base",
            "llama.rope.freq_base",
            "qwen3.rope.freq_base",
        ]) {
            result.rotary_embedding_frequency_base = value;
        }

        result.layer_normalization_root_mean_square_epsilon = 0.000001;

        if let Ok(value) = float32_any(&[
            "mistral3.attention.layer_norm_rms_epsilon",
            "llama.attention.layer_norm_rms_epsilon",
            "qwen3.attention.layer_norm_rms_epsilon",
        ]) {
            result.layer_normalization_root_mean_square_epsilon = value;
        }

        result.vocabulary_size = string_array("tokenizer.ggml.tokens")?.len() as i32;

        println!("architecture: {}", result.architecture);
        println!(
            "maximum sequence length: {}",
            result.maximum_sequence_length
        );
        println!("model hidden dimension: {}", result.model_hidden_dimension);
        println!(
            "feed forward hidden dimension: {}",
            result.feed_forward_hidden_dimension
        );
        println!(
            "attention head dimension: {}",
            result.attention_head_dimension
        );
        println!("number of layers: {}", result.number_of_layers);
        println!(
            "number of attention heads: {}",
            result.number_of_attention_heads
        );
        println!(
            "number of key value heads: {}",
            result.number_of_key_value_heads
        );
        println!("vocabulary size: {}", result.vocabulary_size);
        println!(
            "layer normalization root mean square epsilon: {}",
            result.layer_normalization_root_mean_square_epsilon
        );
        println!(
            "rotary embedding frequency base: {}",
            result.rotary_embedding_frequency_base
        );
        println!(
            "use query and key normalization: {}",
            result.use_query_and_key_normalization
        );

        Ok(result)
    }

    fn load_layer<R: Read + Seek>(
        reader: &mut GgufReader<R>,
        header: &GgufHeader,
        layer: i32,
    ) -> Result<TransformerLayer> {
        Ok(TransformerLayer {
            attention_normalization_weights: reader
                .read_f32_tensor(header, &format!("blk.{layer}.attn_norm.weight"))?,
            query_normalization_weights: reader
                .read_f32_tensor(header, &format!("blk.{layer}.attn_q_norm.weight"))
                .unwrap_or_default(),
            key_normalization_weights: reader
                .read_f32_tensor(header, &format!("blk.{layer}.attn_k_norm.weight"))
                .unwrap_or_default(),
            query_projection_weights: reader
                .read_quantized_tensor(header, &format!("blk.{layer}.attn_q.weight"))?,
            key_projection_weights: reader
                .read_quantized_tensor(header, &format!("blk.{layer}.attn_k.weight"))?,
            value_projection_weights: reader
                .read_quantized_tensor(header, &format!("blk.{layer}.attn_v.weight"))?,
            output_projection_weights: reader
                .read_quantized_tensor(header, &format!("blk.{layer}.attn_output.weight"))?,
            feed_forward_normalization_weights: reader
                .read_f32_tensor(header, &format!("blk.{layer}.ffn_norm.weight"))?,
            feed_forward_up_weights: reader
                .read_quantized_tensor(header, &format!("blk.{layer}.ffn_up.weight"))?,
            feed_forward_gate_weights: reader
                .read_quantized_tensor(header, &format!("blk.{layer}.ffn_gate.weight"))?,
            feed_forward_down_weights: reader
                .read_quantized_tensor(header, &format!("blk.{layer}.ffn_down.weight"))?,
        })
    }

    pub fn load_from_file(path: impl AsRef<Path>, maximum_sequence_length: usize) -> Result<Model> {
        let path: PathBuf = path.as_ref().to_path_buf();

        let mut reader = GgufReader::new(File::open(&path)?);
        let header = reader.read_header()?;
        let mut configuration = Self::load_configuration(&header)?;

        if maximum_sequence_length > 0 {
            configuration.maximum_sequence_length = configuration
                .maximum_sequence_length
                .min(maximum_sequence_length as i32)
        }

        let layers: Vec<TransformerLayer> = (0..configuration.number_of_layers)
            .into_par_iter()
            .map(|layer| -> Result<TransformerLayer> {
                let mut local_reader = GgufReader::new(File::open(&path)?);
                Self::load_layer(&mut local_reader, &header, layer)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut reader = GgufReader::new(File::open(&path)?);

        let token_embeddings = reader.read_quantized_tensor(&header, "token_embd.weight")?;

        let classifier_weights = match reader.read_quantized_tensor(&header, "output.weight") {
            Ok(classifier_weights) => classifier_weights,
            Err(_) => {
                println!("reusing token embeddings");
                token_embeddings.clone()
            }
        };

        let weights = TransformerWeights {
            layers,
            token_embeddings: token_embeddings.clone().dequantize(),
            final_normalization_weights: reader.read_f32_tensor(&header, "output_norm.weight")?,
            classifier_weights,
        };

        let attention_concatenated_dimension =
            configuration.number_of_attention_heads * configuration.attention_head_dimension;

        let quantized_buffer_value_capacity = configuration
            .model_hidden_dimension
            .max(attention_concatenated_dimension)
            .max(configuration.feed_forward_hidden_dimension);

        let attention_concatenated_dimension =
            configuration.number_of_attention_heads * configuration.attention_head_dimension;
        let key_value_concatenated_dimension =
            configuration.number_of_key_value_heads * configuration.attention_head_dimension;

        let rotary_positional_embedding_cache = RotaryPositionalEmbeddingCache::new(
            configuration.maximum_sequence_length as usize,
            configuration.attention_head_dimension as usize,
            configuration.rotary_embedding_frequency_base,
        );

        let key_value_cache = QuantizedKeyValueCache::new(
            configuration.number_of_layers as usize,
            configuration.number_of_key_value_heads as usize,
            configuration.maximum_sequence_length as usize,
            configuration.attention_head_dimension as usize,
        );

        let state = TransformerRuntimeState {
            rotary_positional_embedding_cache,

            quantized_activation_buffer: QuantizedTensor::new(
                quantized_buffer_value_capacity as usize,
            )?,

            residual_stream: vec![0f32; configuration.model_hidden_dimension as usize],
            attention_input_buffer: vec![0f32; configuration.model_hidden_dimension as usize],
            attention_output_buffer: vec![0f32; attention_concatenated_dimension as usize],

            attention_projected_output: vec![0f32; configuration.model_hidden_dimension as usize],
            key_for_position: vec![0f32; key_value_concatenated_dimension as usize],
            value_for_position: vec![0f32; key_value_concatenated_dimension as usize],

            feed_forward_input_buffer: vec![0f32; configuration.model_hidden_dimension as usize],
            feed_forward_hidden_buffer_gate: vec![
                0f32;
                configuration.feed_forward_hidden_dimension
                    as usize
            ],
            feed_forward_hidden_buffer_up: vec![
                0f32;
                configuration.feed_forward_hidden_dimension as usize
            ],
            feed_forward_output_buffer: vec![0f32; configuration.model_hidden_dimension as usize],

            query_activations: vec![0f32; attention_concatenated_dimension as usize],

            quantized_final: QuantizedTensor::new(configuration.model_hidden_dimension as usize)?,

            logits: vec![0f32; configuration.vocabulary_size as usize],

            key_value_cache,
        };

        let tokens = if let Some(MetadataValue::StringArray(elements)) =
            header.metadata.get("tokenizer.ggml.tokens")
        {
            elements.to_owned()
        } else {
            vec![]
        };

        let token_type = if let Some(MetadataValue::Signed32Array(elements)) =
            header.metadata.get("tokenizer.ggml.token_type")
        {
            elements.to_owned()
        } else {
            vec![]
        };

        let merges = if let Some(MetadataValue::StringArray(elements)) =
            header.metadata.get("tokenizer.ggml.merges")
        {
            elements
                .iter()
                .filter_map(|value| {
                    value
                        .split_once(" ")
                        .map(|(a, b)| (a.to_owned(), b.to_owned()))
                })
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        let token_type = if let Some(MetadataValue::Signed32Array(elements)) =
            header.metadata.get("tokenizer.ggml.token_type")
        {
            elements.to_owned()
        } else {
            vec![]
        };

        let scores = if let Some(MetadataValue::Float32Array(elements)) =
            header.metadata.get("tokenizer.ggml.scores")
        {
            elements.to_owned()
        } else {
            vec![]
        };

        let add_begin_of_sequence_token_token = if let Some(MetadataValue::Boolean(value)) =
            header.metadata.get("tokenizer.ggml.add_bos_token")
        {
            *value
        } else {
            false
        };

        let add_end_of_sequence_token = if let Some(MetadataValue::Boolean(value)) =
            header.metadata.get("tokenizer.ggml.add_eos_token")
        {
            *value
        } else {
            false
        };

        let add_space_prefix = if let Some(MetadataValue::Boolean(value)) =
            header.metadata.get("tokenizer.ggml.add_space_prefix")
        {
            *value
        } else {
            false
        };

        let begin_of_sequence_token_identifier =
            match header.metadata.get("tokenizer.ggml.bos_token_id") {
                Some(MetadataValue::Unsigned32(value)) => Some(*value),
                _ => None,
            };

        let Some(MetadataValue::Unsigned32(end_of_sequence_token_identifier)) =
            header.metadata.get("tokenizer.ggml.eos_token_id")
        else {
            bail!("missing end of sequence token identifier");
        };

        let Some(MetadataValue::String(tokenizer_model)) =
            header.metadata.get("tokenizer.ggml.model")
        else {
            bail!("missing tokenizer model");
        };

        let tokenizer = BPETokenizer::new(
            match tokenizer_model.as_str() {
                "gpt2" => TokenizerType::GPT2,
                "llama" => TokenizerType::LLAMA,
                _ => bail!("unknown tokenizer model: {tokenizer_model:?}"),
            },
            tokens,
            merges,
            token_type,
            scores,
            begin_of_sequence_token_identifier,
            *end_of_sequence_token_identifier,
            add_space_prefix,
            add_begin_of_sequence_token_token,
            add_end_of_sequence_token,
        )?;

        let transformer = Transformer {
            weights,
            configuration: configuration.clone(),
            state,
        };

        println!("loaded model\n");

        Ok(Model {
            transformer,
            tokenizer,
        })
    }
}
