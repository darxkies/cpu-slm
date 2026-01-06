#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(clippy::too_many_lines)]

use anyhow::Result;
use rayon::prelude::*;

use crate::math::*;
use crate::quantized_key_value_cache::QuantizedKeyValueCache;
use crate::quantized_tensor::QUANTIZATION_BLOCK_GROUP_SIZE;
use crate::{model::ModelConfiguration, quantized_tensor::*};

#[derive(Clone, Default)]
pub struct TransformerLayer {
    pub attention_normalization_weights: Vec<f32>,
    pub query_normalization_weights: Vec<f32>,
    pub key_normalization_weights: Vec<f32>,

    pub query_projection_weights: QuantizedTensor,
    pub key_projection_weights: QuantizedTensor,
    pub value_projection_weights: QuantizedTensor,
    pub output_projection_weights: QuantizedTensor,

    pub feed_forward_normalization_weights: Vec<f32>,
    pub feed_forward_up_weights: QuantizedTensor,
    pub feed_forward_gate_weights: QuantizedTensor,
    pub feed_forward_down_weights: QuantizedTensor,
}

#[derive(Clone)]
pub struct TransformerWeights {
    pub token_embeddings: Vec<f32>,
    pub layers: Vec<TransformerLayer>,
    pub final_normalization_weights: Vec<f32>,
    pub classifier_weights: QuantizedTensor,
}

pub struct TransformerRuntimeState {
    pub rotary_positional_embedding_cache: RotaryPositionalEmbeddingCache,

    pub quantized_activation_buffer: QuantizedTensor,

    pub residual_stream: Vec<f32>,
    pub attention_input_buffer: Vec<f32>,
    pub attention_output_buffer: Vec<f32>,

    pub attention_projected_output: Vec<f32>,
    pub key_for_position: Vec<f32>,
    pub value_for_position: Vec<f32>,

    pub feed_forward_input_buffer: Vec<f32>,
    pub feed_forward_hidden_buffer_gate: Vec<f32>,
    pub feed_forward_hidden_buffer_up: Vec<f32>,
    pub feed_forward_output_buffer: Vec<f32>,

    pub query_activations: Vec<f32>,

    pub quantized_final: QuantizedTensor,
    pub logits: Vec<f32>,

    pub key_value_cache: QuantizedKeyValueCache,
}

pub struct Transformer {
    pub configuration: ModelConfiguration,
    pub weights: TransformerWeights,
    pub state: TransformerRuntimeState,
}

impl Transformer {
    // Single token forward
    pub fn forward(&mut self, token_identifier: u32, position: usize) -> Result<&[f32]> {
        let model_hidden_dimension = self.configuration.model_hidden_dimension as usize;
        let feed_forward_hidden_dimension =
            self.configuration.feed_forward_hidden_dimension as usize;
        let number_of_layers = self.configuration.number_of_layers as usize;

        let number_of_attention_heads = self.configuration.number_of_attention_heads as usize;
        let number_of_key_value_heads = self.configuration.number_of_key_value_heads as usize;
        let attention_head_dimension = self.configuration.attention_head_dimension as usize;
        let maximum_sequence_length = self.configuration.maximum_sequence_length as usize;

        let attention_concatenated_dimension = number_of_attention_heads * attention_head_dimension;
        let key_value_concatenated_dimension = number_of_key_value_heads * attention_head_dimension;

        let key_value_head_sharing_multiplier =
            number_of_attention_heads / number_of_key_value_heads;

        let inverse_sqrt_attention_head_dimension =
            1.0f32 / (attention_head_dimension as f32).sqrt();

        let embedding_offset = token_identifier as usize * model_hidden_dimension;

        self.state.residual_stream.copy_from_slice(
            &self.weights.token_embeddings
                [embedding_offset..embedding_offset + model_hidden_dimension],
        );

        for layer_index in 0..number_of_layers {
            let layer = &self.weights.layers[layer_index];

            // Attention input normalization
            {
                self.state
                    .attention_input_buffer
                    .copy_from_slice(&self.state.residual_stream);

                root_mean_square_normalization_in_place(
                    &mut self.state.attention_input_buffer,
                    &layer.attention_normalization_weights,
                    self.configuration
                        .layer_normalization_root_mean_square_epsilon,
                );
            }

            // Query, key, value projections
            {
                self.state
                    .quantized_activation_buffer
                    .quantize_into(&self.state.attention_input_buffer)?;

                // Query
                layer.query_projection_weights.multiply_vector(
                    &mut self.state.query_activations,
                    &self.state.quantized_activation_buffer,
                    model_hidden_dimension,
                );

                self.state
                    .query_activations
                    .par_chunks_mut(attention_head_dimension)
                    .for_each(|query_head| {
                        if self.configuration.use_query_and_key_normalization {
                            root_mean_square_normalization_in_place(
                                query_head,
                                &layer.query_normalization_weights,
                                self.configuration
                                    .layer_normalization_root_mean_square_epsilon,
                            );
                            self.state
                                .rotary_positional_embedding_cache
                                .apply_in_place(query_head, position);
                        } else {
                            self.state
                                .rotary_positional_embedding_cache
                                .apply_interleaved_in_place(query_head, position);
                        }
                    });

                // Key
                layer.key_projection_weights.multiply_vector(
                    &mut self.state.key_for_position,
                    &self.state.quantized_activation_buffer,
                    model_hidden_dimension,
                );

                self.state
                    .key_for_position
                    .par_chunks_mut(attention_head_dimension)
                    .for_each(|key_head| {
                        if self.configuration.use_query_and_key_normalization {
                            root_mean_square_normalization_in_place(
                                key_head,
                                &layer.key_normalization_weights,
                                self.configuration
                                    .layer_normalization_root_mean_square_epsilon,
                            );
                            self.state
                                .rotary_positional_embedding_cache
                                .apply_in_place(key_head, position);
                        } else {
                            self.state
                                .rotary_positional_embedding_cache
                                .apply_interleaved_in_place(key_head, position);
                        }
                    });

                // Value
                layer.value_projection_weights.multiply_vector(
                    &mut self.state.value_for_position,
                    &self.state.quantized_activation_buffer,
                    model_hidden_dimension,
                );

                // Cache key and value for this position (quantized)
                for key_value_head_index in 0..number_of_key_value_heads {
                    let start = key_value_head_index * attention_head_dimension;
                    let end = start + attention_head_dimension;

                    self.state.key_value_cache.write_key_head_quantized(
                        layer_index,
                        key_value_head_index,
                        position,
                        &self.state.key_for_position[start..end],
                    );

                    self.state.key_value_cache.write_value_head_quantized(
                        layer_index,
                        key_value_head_index,
                        position,
                        &self.state.value_for_position[start..end],
                    );
                }
            }

            // Attention (streaming softmax, quantized key and value reads)
            {
                let head_dimension = attention_head_dimension;
                let key_value_share = key_value_head_sharing_multiplier;
                let scale = inverse_sqrt_attention_head_dimension;

                let key_value_cache = &self.state.key_value_cache;
                let query_activations = &self.state.query_activations;

                self.state
                    .attention_output_buffer
                    .par_chunks_mut(head_dimension)
                    .enumerate()
                    .for_each(|(head_index, attention_output_head)| {
                        let shared_key_value_head_index = head_index / key_value_share;

                        let query_head_start = head_index * head_dimension;
                        let query_head =
                            &query_activations[query_head_start..query_head_start + head_dimension];

                        let mut running_maximum = f32::NEG_INFINITY;
                        let mut running_sum_exponentials = 0.0f32;

                        attention_output_head.fill(0.0f32);

                        for time_index in 0..=position {
                            let (key_quantized, key_scales) = key_value_cache.key_head_at(
                                layer_index,
                                shared_key_value_head_index,
                                time_index,
                            );

                            let score = dot_product_f32_with_quantized_signed_eight_bit(
                                query_head,
                                key_quantized,
                                key_scales,
                            ) * scale;

                            let (value_quantized, value_scales) = key_value_cache.value_head_at(
                                layer_index,
                                shared_key_value_head_index,
                                time_index,
                            );

                            online_softmax_update_with_quantized_values(
                                score,
                                &mut running_maximum,
                                &mut running_sum_exponentials,
                                attention_output_head,
                                value_quantized,
                                value_scales,
                            );
                        }

                        online_softmax_finalize(attention_output_head, running_sum_exponentials);
                    });

                // Output projection and residual add
                self.state
                    .quantized_activation_buffer
                    .quantize_into(&self.state.attention_output_buffer)?;

                layer.output_projection_weights.multiply_vector(
                    &mut self.state.attention_projected_output,
                    &self.state.quantized_activation_buffer,
                    attention_concatenated_dimension,
                );

                for i in 0..model_hidden_dimension {
                    self.state.residual_stream[i] += self.state.attention_projected_output[i];
                }
            }

            // Feed forward normalization
            {
                self.state
                    .feed_forward_input_buffer
                    .copy_from_slice(&self.state.residual_stream);

                root_mean_square_normalization_in_place(
                    &mut self.state.feed_forward_input_buffer,
                    &layer.feed_forward_normalization_weights,
                    self.configuration
                        .layer_normalization_root_mean_square_epsilon,
                );
            }

            // Feed forward projections and gated activation
            {
                self.state
                    .quantized_activation_buffer
                    .quantize_into(&self.state.feed_forward_input_buffer)?;

                layer.feed_forward_up_weights.multiply_vector(
                    &mut self.state.feed_forward_hidden_buffer_up,
                    &self.state.quantized_activation_buffer,
                    model_hidden_dimension,
                );

                layer.feed_forward_gate_weights.multiply_vector(
                    &mut self.state.feed_forward_hidden_buffer_gate,
                    &self.state.quantized_activation_buffer,
                    model_hidden_dimension,
                );

                for i in 0..feed_forward_hidden_dimension {
                    let value = self.state.feed_forward_hidden_buffer_gate[i];
                    let sigmoid = 1.0f32 / (1.0f32 + (-value).exp());
                    self.state.feed_forward_hidden_buffer_gate[i] =
                        value * sigmoid * self.state.feed_forward_hidden_buffer_up[i];
                }

                self.state
                    .quantized_activation_buffer
                    .quantize_into(&self.state.feed_forward_hidden_buffer_gate)?;

                layer.feed_forward_down_weights.multiply_vector(
                    &mut self.state.feed_forward_output_buffer,
                    &self.state.quantized_activation_buffer,
                    feed_forward_hidden_dimension,
                );

                for i in 0..model_hidden_dimension {
                    self.state.residual_stream[i] += self.state.feed_forward_output_buffer[i];
                }
            }
        }

        // Final normalization and classifier
        root_mean_square_normalization_in_place(
            &mut self.state.residual_stream,
            &self.weights.final_normalization_weights,
            self.configuration
                .layer_normalization_root_mean_square_epsilon,
        );

        self.state
            .quantized_activation_buffer
            .quantize_into(&self.state.residual_stream)?;

        self.weights.classifier_weights.multiply_vector(
            &mut self.state.logits,
            &self.state.quantized_activation_buffer,
            model_hidden_dimension,
        );

        Ok(&self.state.logits)
    }

    // Batched forward
    pub fn batched_forward(
        &mut self,
        prompt_token_identifiers: &[u32],
        start_position: usize,
        batch_size: usize,
    ) -> Result<&[f32]> {
        if prompt_token_identifiers.is_empty() {
            return Ok(&self.state.logits);
        }

        let model_hidden_dimension = self.configuration.model_hidden_dimension as usize;
        let feed_forward_hidden_dimension =
            self.configuration.feed_forward_hidden_dimension as usize;

        let number_of_layers = self.configuration.number_of_layers as usize;
        let number_of_attention_heads = self.configuration.number_of_attention_heads as usize;
        let number_of_key_value_heads = self.configuration.number_of_key_value_heads as usize;

        let attention_head_dimension = self.configuration.attention_head_dimension as usize;
        let maximum_sequence_length = self.configuration.maximum_sequence_length as usize;

        let attention_concatenated_dimension = number_of_attention_heads * attention_head_dimension;
        let key_value_concatenated_dimension = number_of_key_value_heads * attention_head_dimension;

        let key_value_head_sharing_multiplier =
            number_of_attention_heads / number_of_key_value_heads;

        let inverse_sqrt_attention_head_dimension =
            1.0f32 / (attention_head_dimension as f32).sqrt();

        // Token-major buffers.
        let mut residual_batch: Vec<f32> = Vec::new();
        let mut attention_input_batch: Vec<f32> = Vec::new();
        let mut attention_projected_batch: Vec<f32> = Vec::new();

        let mut query_batch: Vec<f32> = Vec::new();
        let mut key_batch: Vec<f32> = Vec::new();
        let mut value_batch: Vec<f32> = Vec::new();

        let mut attention_output_batch: Vec<f32> = Vec::new();

        let mut feed_forward_input_batch: Vec<f32> = Vec::new();
        let mut feed_forward_up_batch: Vec<f32> = Vec::new();
        let mut feed_forward_gate_batch: Vec<f32> = Vec::new();
        let mut feed_forward_hidden_batch: Vec<f32> = Vec::new();
        let mut feed_forward_output_batch: Vec<f32> = Vec::new();

        // Packed buffers for attention.
        let mut query_by_head_token: Vec<f32> = Vec::new();
        let mut output_by_head_token: Vec<f32> = Vec::new();

        // Temporary quantized tensors for batch operations.
        let mut quantized_attention_input_batch = QuantizedTensor::new(batch_size)?;
        let mut quantized_attention_output_batch = QuantizedTensor::new(batch_size)?;
        let mut quantized_feed_forward_batch = QuantizedTensor::new(batch_size)?;
        let mut quantized_feed_forward_hidden_batch = QuantizedTensor::new(batch_size)?;

        let mut position = start_position;

        for token_batch in prompt_token_identifiers.chunks(batch_size) {
            let current_batch_size = token_batch.len();

            residual_batch.resize(current_batch_size * model_hidden_dimension, 0.0f32);
            attention_input_batch.resize(current_batch_size * model_hidden_dimension, 0.0f32);

            query_batch.resize(
                current_batch_size * attention_concatenated_dimension,
                0.0f32,
            );
            key_batch.resize(
                current_batch_size * key_value_concatenated_dimension,
                0.0f32,
            );
            value_batch.resize(
                current_batch_size * key_value_concatenated_dimension,
                0.0f32,
            );

            attention_output_batch.resize(
                current_batch_size * attention_concatenated_dimension,
                0.0f32,
            );
            attention_projected_batch.resize(current_batch_size * model_hidden_dimension, 0.0f32);

            feed_forward_input_batch.resize(current_batch_size * model_hidden_dimension, 0.0f32);
            feed_forward_up_batch
                .resize(current_batch_size * feed_forward_hidden_dimension, 0.0f32);
            feed_forward_gate_batch
                .resize(current_batch_size * feed_forward_hidden_dimension, 0.0f32);
            feed_forward_hidden_batch
                .resize(current_batch_size * feed_forward_hidden_dimension, 0.0f32);
            feed_forward_output_batch.resize(current_batch_size * model_hidden_dimension, 0.0f32);

            // Load embeddings
            for (batch_index, &token_identifier) in token_batch.iter().enumerate() {
                let start = batch_index * model_hidden_dimension;

                let destination = &mut residual_batch[start..start + model_hidden_dimension];

                let start = token_identifier as usize * model_hidden_dimension;

                destination.copy_from_slice(
                    &self.weights.token_embeddings[start..start + model_hidden_dimension],
                );
            }

            for layer_index in 0..number_of_layers {
                let layer = &self.weights.layers[layer_index];

                // Attention input normalization
                {
                    attention_input_batch.copy_from_slice(&residual_batch);
                    attention_input_batch
                        .par_chunks_mut(model_hidden_dimension)
                        .for_each(|values| {
                            root_mean_square_normalization_in_place(
                                values,
                                &layer.attention_normalization_weights,
                                self.configuration
                                    .layer_normalization_root_mean_square_epsilon,
                            )
                        });
                }

                // Attention projections
                {
                    quantized_attention_input_batch.quantize_batch_into(
                        &attention_input_batch,
                        current_batch_size,
                        model_hidden_dimension,
                    )?;

                    layer.query_projection_weights.multiply_batch(
                        &mut query_batch,
                        &quantized_attention_input_batch,
                        model_hidden_dimension,
                        attention_concatenated_dimension,
                        current_batch_size,
                    )?;

                    layer.key_projection_weights.multiply_batch(
                        &mut key_batch,
                        &quantized_attention_input_batch,
                        model_hidden_dimension,
                        key_value_concatenated_dimension,
                        current_batch_size,
                    )?;

                    layer.value_projection_weights.multiply_batch(
                        &mut value_batch,
                        &quantized_attention_input_batch,
                        model_hidden_dimension,
                        key_value_concatenated_dimension,
                        current_batch_size,
                    )?;
                }

                // Rotary embedding and optional normalization
                for batch_index in 0..current_batch_size {
                    let current_position = position + batch_index;

                    let start = batch_index * attention_concatenated_dimension;

                    let query_for_token =
                        &mut query_batch[start..start + attention_concatenated_dimension];

                    let start = batch_index * key_value_concatenated_dimension;

                    let key_for_token =
                        &mut key_batch[start..start + key_value_concatenated_dimension];

                    query_for_token
                        .par_chunks_mut(attention_head_dimension)
                        .for_each(|query_head| {
                            if self.configuration.use_query_and_key_normalization {
                                root_mean_square_normalization_in_place(
                                    query_head,
                                    &layer.query_normalization_weights,
                                    self.configuration
                                        .layer_normalization_root_mean_square_epsilon,
                                );

                                self.state
                                    .rotary_positional_embedding_cache
                                    .apply_in_place(query_head, current_position);
                            } else {
                                self.state
                                    .rotary_positional_embedding_cache
                                    .apply_interleaved_in_place(query_head, current_position);
                            }
                        });

                    key_for_token
                        .par_chunks_mut(attention_head_dimension)
                        .for_each(|key_head| {
                            if self.configuration.use_query_and_key_normalization {
                                root_mean_square_normalization_in_place(
                                    key_head,
                                    &layer.key_normalization_weights,
                                    self.configuration
                                        .layer_normalization_root_mean_square_epsilon,
                                );

                                self.state
                                    .rotary_positional_embedding_cache
                                    .apply_in_place(key_head, current_position);
                            } else {
                                self.state
                                    .rotary_positional_embedding_cache
                                    .apply_interleaved_in_place(key_head, current_position);
                            }
                        });
                }

                // Write keys and values (quantized) to cache for the block
                for batch_index in 0..current_batch_size {
                    let current_position = position + batch_index;

                    let start = batch_index * key_value_concatenated_dimension;

                    let key_for_token = &key_batch[start..start + key_value_concatenated_dimension];

                    let start = batch_index * key_value_concatenated_dimension;

                    let value_for_token =
                        &value_batch[start..start + key_value_concatenated_dimension];

                    for key_value_head_index in 0..number_of_key_value_heads {
                        let start = key_value_head_index * attention_head_dimension;
                        let end = start + attention_head_dimension;

                        self.state.key_value_cache.write_key_head_quantized(
                            layer_index,
                            key_value_head_index,
                            current_position,
                            &key_for_token[start..end],
                        );

                        self.state.key_value_cache.write_value_head_quantized(
                            layer_index,
                            key_value_head_index,
                            current_position,
                            &value_for_token[start..end],
                        );
                    }
                }

                {
                    let prefix_length = position;
                    let block_size = current_batch_size;

                    let head_dimension = attention_head_dimension;
                    let attention_stride = attention_concatenated_dimension;
                    let key_value_stride = key_value_concatenated_dimension;

                    let key_value_share = key_value_head_sharing_multiplier;
                    let scale = inverse_sqrt_attention_head_dimension;

                    let key_value_cache = &self.state.key_value_cache;
                    let blocks_per_head = key_value_cache.blocks_per_head();

                    // Pack queries: [head][token][head_dimension]
                    query_by_head_token.resize(
                        number_of_attention_heads * block_size * head_dimension,
                        0.0f32,
                    );

                    for token in 0..block_size {
                        let start = token * attention_stride;

                        let row = &query_batch[start..start + attention_stride];

                        for head_index in 0..number_of_attention_heads {
                            let start = head_index * head_dimension;

                            let head = &row[start..start + head_dimension];

                            let start = (head_index * block_size + token) * head_dimension;

                            query_by_head_token[start..start + head_dimension]
                                .copy_from_slice(head);
                        }
                    }

                    let mut key_block_quantized: Vec<i8> =
                        vec![0i8; number_of_key_value_heads * block_size * head_dimension];
                    let mut value_block_quantized: Vec<i8> =
                        vec![0i8; number_of_key_value_heads * block_size * head_dimension];

                    let mut key_block_scales: Vec<f32> =
                        vec![0.0f32; number_of_key_value_heads * block_size * blocks_per_head];
                    let mut value_block_scales: Vec<f32> =
                        vec![0.0f32; number_of_key_value_heads * block_size * blocks_per_head];

                    for token in 0..block_size {
                        let start = token * key_value_stride;
                        let key_row = &key_batch[start..start + key_value_stride];
                        let value_row = &value_batch[start..start + key_value_stride];

                        for key_value_head_index in 0..number_of_key_value_heads {
                            let head_start = key_value_head_index * head_dimension;
                            let head_end = head_start + head_dimension;

                            let key_head = &key_row[head_start..head_end];
                            let value_head = &value_row[head_start..head_end];

                            let quantized_base =
                                (key_value_head_index * block_size + token) * head_dimension;
                            let scale_base =
                                (key_value_head_index * block_size + token) * blocks_per_head;

                            let key_quantized_slice = &mut key_block_quantized
                                [quantized_base..quantized_base + head_dimension];
                            let value_quantized_slice = &mut value_block_quantized
                                [quantized_base..quantized_base + head_dimension];

                            let key_scale_slice =
                                &mut key_block_scales[scale_base..scale_base + blocks_per_head];
                            let value_scale_slice =
                                &mut value_block_scales[scale_base..scale_base + blocks_per_head];

                            for block_index in 0..blocks_per_head {
                                let start = block_index * QUANTIZATION_BLOCK_GROUP_SIZE;
                                let end = start + QUANTIZATION_BLOCK_GROUP_SIZE;

                                key_scale_slice[block_index] = quantize_block_signed_eight_bit(
                                    &mut key_quantized_slice[start..end],
                                    &key_head[start..end],
                                );

                                value_scale_slice[block_index] = quantize_block_signed_eight_bit(
                                    &mut value_quantized_slice[start..end],
                                    &value_head[start..end],
                                );
                            }
                        }
                    }

                    // Head-major output buffer: [head][token][head_dimension]
                    output_by_head_token.resize(
                        number_of_attention_heads * block_size * head_dimension,
                        0.0f32,
                    );

                    output_by_head_token
                        .par_chunks_mut(block_size * head_dimension)
                        .enumerate()
                        .for_each(|(head_index, out_head_tokens)| {
                            let key_value_head_index = head_index / key_value_share;
                            let query_head_base = head_index * block_size * head_dimension;

                            let mut accumulator = vec![0.0f32; head_dimension];

                            for token in 0..block_size {
                                let query_start = query_head_base + token * head_dimension;
                                let query =
                                    &query_by_head_token[query_start..query_start + head_dimension];

                                let mut running_maximum = f32::NEG_INFINITY;
                                let mut running_sum_exponentials = 0.0f32;

                                accumulator.fill(0.0f32);

                                // Prefix
                                for prefix_position in 0..prefix_length {
                                    let (key_quantized, key_scales) = key_value_cache.key_head_at(
                                        layer_index,
                                        key_value_head_index,
                                        prefix_position,
                                    );

                                    let score = dot_product_f32_with_quantized_signed_eight_bit(
                                        query,
                                        key_quantized,
                                        key_scales,
                                    ) * scale;

                                    let (value_quantized, value_scales) = key_value_cache
                                        .value_head_at(
                                            layer_index,
                                            key_value_head_index,
                                            prefix_position,
                                        );

                                    online_softmax_update_with_quantized_values(
                                        score,
                                        &mut running_maximum,
                                        &mut running_sum_exponentials,
                                        &mut accumulator,
                                        value_quantized,
                                        value_scales,
                                    );
                                }

                                // In-block causal
                                for block_position in 0..=token {
                                    let quantized_base = (key_value_head_index * block_size
                                        + block_position)
                                        * head_dimension;
                                    let scale_base = (key_value_head_index * block_size
                                        + block_position)
                                        * blocks_per_head;

                                    let key_quantized = &key_block_quantized
                                        [quantized_base..quantized_base + head_dimension];
                                    let key_scales =
                                        &key_block_scales[scale_base..scale_base + blocks_per_head];

                                    let score = dot_product_f32_with_quantized_signed_eight_bit(
                                        query,
                                        key_quantized,
                                        key_scales,
                                    ) * scale;

                                    let value_quantized = &value_block_quantized
                                        [quantized_base..quantized_base + head_dimension];
                                    let value_scales = &value_block_scales
                                        [scale_base..scale_base + blocks_per_head];

                                    online_softmax_update_with_quantized_values(
                                        score,
                                        &mut running_maximum,
                                        &mut running_sum_exponentials,
                                        &mut accumulator,
                                        value_quantized,
                                        value_scales,
                                    );
                                }

                                online_softmax_finalize(&mut accumulator, running_sum_exponentials);

                                let out_start = token * head_dimension;
                                out_head_tokens[out_start..out_start + head_dimension]
                                    .copy_from_slice(&accumulator);
                            }
                        });

                    // Transpose back to token-major for output projection
                    attention_output_batch.fill(0.0f32);

                    for token in 0..block_size {
                        let row_start = token * attention_stride;
                        for head_index in 0..number_of_attention_heads {
                            let source_start = (head_index * block_size + token) * head_dimension;
                            let destination_start = row_start + head_index * head_dimension;

                            attention_output_batch
                                [destination_start..destination_start + head_dimension]
                                .copy_from_slice(
                                    &output_by_head_token
                                        [source_start..source_start + head_dimension],
                                );
                        }
                    }
                }

                // Output projection
                quantized_attention_output_batch.quantize_batch_into(
                    &attention_output_batch,
                    current_batch_size,
                    attention_concatenated_dimension,
                )?;

                layer.output_projection_weights.multiply_batch(
                    &mut attention_projected_batch,
                    &quantized_attention_output_batch,
                    attention_concatenated_dimension,
                    model_hidden_dimension,
                    current_batch_size,
                )?;

                // Residual add
                residual_batch
                    .par_chunks_mut(model_hidden_dimension)
                    .zip(attention_projected_batch.par_chunks(model_hidden_dimension))
                    .for_each(|(residual, projected)| {
                        for i in 0..model_hidden_dimension {
                            residual[i] += projected[i];
                        }
                    });

                // Feed forward normalization
                feed_forward_input_batch.copy_from_slice(&residual_batch);
                feed_forward_input_batch
                    .par_chunks_mut(model_hidden_dimension)
                    .for_each(|values| {
                        root_mean_square_normalization_in_place(
                            values,
                            &layer.feed_forward_normalization_weights,
                            self.configuration
                                .layer_normalization_root_mean_square_epsilon,
                        )
                    });

                // Feed forward projections
                quantized_feed_forward_batch.quantize_batch_into(
                    &feed_forward_input_batch,
                    current_batch_size,
                    model_hidden_dimension,
                )?;

                layer.feed_forward_up_weights.multiply_batch(
                    &mut feed_forward_up_batch,
                    &quantized_feed_forward_batch,
                    model_hidden_dimension,
                    feed_forward_hidden_dimension,
                    current_batch_size,
                )?;

                layer.feed_forward_gate_weights.multiply_batch(
                    &mut feed_forward_gate_batch,
                    &quantized_feed_forward_batch,
                    model_hidden_dimension,
                    feed_forward_hidden_dimension,
                    current_batch_size,
                )?;

                // Gated activation
                feed_forward_hidden_batch
                    .par_chunks_mut(feed_forward_hidden_dimension)
                    .enumerate()
                    .for_each(|(batch_index, hidden_out)| {
                        let start = batch_index * feed_forward_hidden_dimension;
                        let end = start + feed_forward_hidden_dimension;

                        let gate = &feed_forward_gate_batch[start..end];
                        let up = &feed_forward_up_batch[start..end];

                        for i in 0..feed_forward_hidden_dimension {
                            let value = gate[i];
                            let sigmoid = 1.0f32 / (1.0f32 + (-value).exp());
                            hidden_out[i] = value * sigmoid * up[i];
                        }
                    });

                // Feed forward down projection
                quantized_feed_forward_hidden_batch.quantize_batch_into(
                    &feed_forward_hidden_batch,
                    current_batch_size,
                    feed_forward_hidden_dimension,
                )?;

                layer.feed_forward_down_weights.multiply_batch(
                    &mut feed_forward_output_batch,
                    &quantized_feed_forward_hidden_batch,
                    feed_forward_hidden_dimension,
                    model_hidden_dimension,
                    current_batch_size,
                )?;

                // Residual add
                residual_batch
                    .par_chunks_mut(model_hidden_dimension)
                    .zip(feed_forward_output_batch.par_chunks(model_hidden_dimension))
                    .for_each(|(residual, projected)| {
                        for i in 0..model_hidden_dimension {
                            residual[i] += projected[i];
                        }
                    });
            }

            // Logits for last token in batch
            let final_token_index = current_batch_size - 1;
            let start = final_token_index * model_hidden_dimension;

            let final_residual = &mut residual_batch[start..start + model_hidden_dimension];

            root_mean_square_normalization_in_place(
                final_residual,
                &self.weights.final_normalization_weights,
                self.configuration
                    .layer_normalization_root_mean_square_epsilon,
            );

            self.state.quantized_final.quantize_into(final_residual)?;

            self.weights.classifier_weights.multiply_vector(
                &mut self.state.logits,
                &self.state.quantized_final,
                model_hidden_dimension,
            );

            position += current_batch_size;

            let _ = maximum_sequence_length;
        }

        Ok(&self.state.logits)
    }
}
