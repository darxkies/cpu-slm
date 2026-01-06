use crate::{
    math::quantize_block_signed_eight_bit, quantized_tensor::QUANTIZATION_BLOCK_GROUP_SIZE,
};

#[derive(Clone, Debug)]
pub struct QuantizedKeyValueCache {
    number_of_layers: usize,
    number_of_key_value_heads: usize,
    maximum_sequence_length: usize,
    attention_head_dimension: usize,

    layer_stride: usize,
    head_stride: usize,

    key_quantized: Vec<i8>,
    value_quantized: Vec<i8>,

    blocks_per_head: usize,
    key_scales: Vec<f32>,
    value_scales: Vec<f32>,
}

impl QuantizedKeyValueCache {
    pub fn new(
        number_of_layers: usize,
        number_of_key_value_heads: usize,
        maximum_sequence_length: usize,
        attention_head_dimension: usize,
    ) -> Self {
        debug_assert!(number_of_layers > 0);
        debug_assert!(number_of_key_value_heads > 0);
        debug_assert!(maximum_sequence_length > 0);
        debug_assert!(attention_head_dimension > 0);
        debug_assert!(
            attention_head_dimension % QUANTIZATION_BLOCK_GROUP_SIZE == 0,
            "attention head dimension must be divisible by quantization block group size"
        );

        let head_stride = maximum_sequence_length * attention_head_dimension;
        let layer_stride = number_of_key_value_heads * head_stride;
        let total = number_of_layers * layer_stride;

        let blocks_per_head = attention_head_dimension / QUANTIZATION_BLOCK_GROUP_SIZE;
        let total_scales = number_of_layers
            * number_of_key_value_heads
            * maximum_sequence_length
            * blocks_per_head;

        Self {
            number_of_layers,
            number_of_key_value_heads,
            maximum_sequence_length,
            attention_head_dimension,
            layer_stride,
            head_stride,
            key_quantized: vec![0i8; total],
            value_quantized: vec![0i8; total],
            blocks_per_head,
            key_scales: vec![0.0f32; total_scales],
            value_scales: vec![0.0f32; total_scales],
        }
    }

    #[inline]
    fn base_element(&self, layer: usize, key_value_head: usize, position: usize) -> usize {
        debug_assert!(layer < self.number_of_layers);
        debug_assert!(key_value_head < self.number_of_key_value_heads);
        debug_assert!(position < self.maximum_sequence_length);

        layer * self.layer_stride
            + key_value_head * self.head_stride
            + position * self.attention_head_dimension
    }

    #[inline]
    fn base_scale(&self, layer: usize, key_value_head: usize, position: usize) -> usize {
        ((layer * self.number_of_key_value_heads + key_value_head) * self.maximum_sequence_length
            + position)
            * self.blocks_per_head
    }

    #[inline]
    pub fn clear(&mut self) {
        self.key_quantized.fill(0);
        self.value_quantized.fill(0);
        self.key_scales.fill(0.0f32);
        self.value_scales.fill(0.0f32);
    }

    #[inline]
    pub fn blocks_per_head(&self) -> usize {
        self.blocks_per_head
    }

    #[inline]
    pub fn write_key_head_quantized(
        &mut self,
        layer: usize,
        key_value_head: usize,
        position: usize,
        source: &[f32],
    ) {
        debug_assert_eq!(source.len(), self.attention_head_dimension);

        let base_element = self.base_element(layer, key_value_head, position);
        let base_scale = self.base_scale(layer, key_value_head, position);

        let quantized_slice =
            &mut self.key_quantized[base_element..base_element + self.attention_head_dimension];
        let scale_slice = &mut self.key_scales[base_scale..base_scale + self.blocks_per_head];

        for block_index in 0..self.blocks_per_head {
            let start = block_index * QUANTIZATION_BLOCK_GROUP_SIZE;
            let end = start + QUANTIZATION_BLOCK_GROUP_SIZE;

            let scale = quantize_block_signed_eight_bit(
                &mut quantized_slice[start..end],
                &source[start..end],
            );

            scale_slice[block_index] = scale;
        }
    }

    #[inline]
    pub fn write_value_head_quantized(
        &mut self,
        layer: usize,
        key_value_head: usize,
        position: usize,
        source: &[f32],
    ) {
        debug_assert_eq!(source.len(), self.attention_head_dimension);

        let base_element = self.base_element(layer, key_value_head, position);
        let base_scale = self.base_scale(layer, key_value_head, position);

        let quantized_slice =
            &mut self.value_quantized[base_element..base_element + self.attention_head_dimension];
        let scale_slice = &mut self.value_scales[base_scale..base_scale + self.blocks_per_head];

        for block_index in 0..self.blocks_per_head {
            let start = block_index * QUANTIZATION_BLOCK_GROUP_SIZE;
            let end = start + QUANTIZATION_BLOCK_GROUP_SIZE;

            let scale = quantize_block_signed_eight_bit(
                &mut quantized_slice[start..end],
                &source[start..end],
            );

            scale_slice[block_index] = scale;
        }
    }

    #[inline]
    pub fn key_head_at(
        &self,
        layer: usize,
        key_value_head: usize,
        position: usize,
    ) -> (&[i8], &[f32]) {
        let base_element = self.base_element(layer, key_value_head, position);
        let base_scale = self.base_scale(layer, key_value_head, position);

        let quantized_slice =
            &self.key_quantized[base_element..base_element + self.attention_head_dimension];
        let scale_slice = &self.key_scales[base_scale..base_scale + self.blocks_per_head];

        (quantized_slice, scale_slice)
    }

    #[inline]
    pub fn value_head_at(
        &self,
        layer: usize,
        key_value_head: usize,
        position: usize,
    ) -> (&[i8], &[f32]) {
        let base_element = self.base_element(layer, key_value_head, position);
        let base_scale = self.base_scale(layer, key_value_head, position);

        let quantized_slice =
            &self.value_quantized[base_element..base_element + self.attention_head_dimension];
        let scale_slice = &self.value_scales[base_scale..base_scale + self.blocks_per_head];

        (quantized_slice, scale_slice)
    }
}
