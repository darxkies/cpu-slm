use anyhow::{Result as AnyhowResult, bail};
use rayon::prelude::*;

pub const QUANTIZATION_BLOCK_GROUP_SIZE: usize = 32;

#[derive(Clone, Debug, Default)]
pub struct QuantizedTensor {
    pub scales: Vec<f32>,
    pub values: Vec<i8>,
}

impl QuantizedTensor {
    #[inline]
    pub fn block_count(&self) -> usize {
        self.scales.len()
    }

    #[inline]
    pub fn values_len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn new(size: usize) -> AnyhowResult<Self> {
        if size % QUANTIZATION_BLOCK_GROUP_SIZE != 0 {
            bail!(
                "q8_0 quantize requires size ({size}) to be multiple of {QUANTIZATION_BLOCK_GROUP_SIZE}"
            );
        }

        let blocks = size / QUANTIZATION_BLOCK_GROUP_SIZE;

        Ok(Self {
            scales: vec![0.0f32; blocks],
            values: vec![0i8; blocks * QUANTIZATION_BLOCK_GROUP_SIZE],
        })
    }

    #[inline]
    pub fn resize_blocks(&mut self, block_count: usize) {
        self.scales.resize(block_count, 0.0f32);
        self.values
            .resize(block_count * QUANTIZATION_BLOCK_GROUP_SIZE, 0i8);
    }

    #[inline]
    fn block_values_slice(&self, block_index: usize) -> &[i8] {
        let start = block_index * QUANTIZATION_BLOCK_GROUP_SIZE;
        &self.values[start..start + QUANTIZATION_BLOCK_GROUP_SIZE]
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let block_count = self.block_count();
        let mut result = Vec::with_capacity(block_count * QUANTIZATION_BLOCK_GROUP_SIZE);

        for block_index in 0..block_count {
            let scale = self.scales[block_index];
            let block_values = self.block_values_slice(block_index);

            for &q in block_values {
                result.push(scale * (q as f32));
            }
        }

        result
    }

    pub fn quantize_into(&mut self, input: &[f32]) -> AnyhowResult<()> {
        if input.len() % QUANTIZATION_BLOCK_GROUP_SIZE != 0 {
            bail!(
                "q8_0 quantize requires input length multiple of {QUANTIZATION_BLOCK_GROUP_SIZE}"
            );
        }

        self.quantize_batch_into(input, 1, input.len())
    }

    pub fn quantize_batch_into(
        &mut self,
        input: &[f32],
        batch_size: usize,
        vector_length: usize,
    ) -> AnyhowResult<()> {
        if input.len() != batch_size * vector_length {
            bail!("quantize_batch requires input length {batch_size} * {vector_length}");
        }

        if vector_length % QUANTIZATION_BLOCK_GROUP_SIZE != 0 {
            bail!(
                "quantize_batch requires vector_length multiple of {QUANTIZATION_BLOCK_GROUP_SIZE}"
            );
        }

        let blocks_per_vector = vector_length / QUANTIZATION_BLOCK_GROUP_SIZE;
        let expected_blocks = batch_size * blocks_per_vector;

        self.resize_blocks(expected_blocks);

        input
            .par_chunks_exact(vector_length)
            .zip(self.scales.par_chunks_mut(blocks_per_vector))
            .zip(self.values.par_chunks_mut(vector_length))
            .for_each(|((vector, scales_out), values_out)| {
                for (block_offset, chunk) in vector
                    .chunks_exact(QUANTIZATION_BLOCK_GROUP_SIZE)
                    .enumerate()
                {
                    let mut max_abs = 0.0f32;
                    for &value in chunk {
                        let a = value.abs();
                        if a > max_abs {
                            max_abs = a;
                        }
                    }

                    let out_start = block_offset * QUANTIZATION_BLOCK_GROUP_SIZE;
                    let out_end = out_start + QUANTIZATION_BLOCK_GROUP_SIZE;
                    let out_block = &mut values_out[out_start..out_end];

                    if max_abs == 0.0f32 {
                        scales_out[block_offset] = 0.0f32;
                        out_block.fill(0i8);
                        continue;
                    }

                    let scale = max_abs / 127.0f32;
                    let inverse_scale = 1.0f32 / scale;

                    for (i, &value) in chunk.iter().enumerate() {
                        let mut q = (value * inverse_scale).round() as i32;

                        if q > 127 {
                            q = 127;
                        } else if q < -127 {
                            q = -127;
                        }

                        out_block[i] = q as i8;
                    }

                    scales_out[block_offset] = scale;
                }
            });

        Ok(())
    }

    pub fn multiply_vector(
        &self,
        output: &mut [f32],
        quantized_input: &QuantizedTensor,
        input_length: usize,
    ) {
        let blocks_per_row = input_length / QUANTIZATION_BLOCK_GROUP_SIZE;

        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(output_index, out_value)| {
                let row_block_start = output_index * blocks_per_row;

                *out_value = dot_product_many_blocks(
                    &self.scales[row_block_start..row_block_start + blocks_per_row],
                    &self.values[row_block_start * QUANTIZATION_BLOCK_GROUP_SIZE
                        ..(row_block_start + blocks_per_row) * QUANTIZATION_BLOCK_GROUP_SIZE],
                    &quantized_input.scales[0..blocks_per_row],
                    &quantized_input.values[0..blocks_per_row * QUANTIZATION_BLOCK_GROUP_SIZE],
                );
            });
    }

    pub fn multiply_batch(
        &self,
        output: &mut [f32],
        quantized_inputs: &QuantizedTensor,
        input_length: usize,
        output_rows: usize,
        batch_size: usize,
    ) -> AnyhowResult<()> {
        let blocks_per_vector = input_length / QUANTIZATION_BLOCK_GROUP_SIZE;

        if output.len() != batch_size * output_rows {
            bail!("multiply_batch requires output length {batch_size} * {output_rows}");
        }

        if quantized_inputs.block_count() != batch_size * blocks_per_vector {
            bail!("multiply_batch requires input blocks length {batch_size} * {blocks_per_vector}");
        }

        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(flat_index, out_value)| {
                let batch_index = flat_index / output_rows;
                let row_index = flat_index - batch_index * output_rows;

                let input_block_start = batch_index * blocks_per_vector;
                let input_scales = &quantized_inputs.scales
                    [input_block_start..input_block_start + blocks_per_vector];
                let input_values = &quantized_inputs.values[input_block_start
                    * QUANTIZATION_BLOCK_GROUP_SIZE
                    ..(input_block_start + blocks_per_vector) * QUANTIZATION_BLOCK_GROUP_SIZE];

                let weight_row_block_start = row_index * blocks_per_vector;
                let weight_scales = &self.scales
                    [weight_row_block_start..weight_row_block_start + blocks_per_vector];
                let weight_values = &self.values[weight_row_block_start
                    * QUANTIZATION_BLOCK_GROUP_SIZE
                    ..(weight_row_block_start + blocks_per_vector) * QUANTIZATION_BLOCK_GROUP_SIZE];

                *out_value = dot_product_many_blocks(
                    weight_scales,
                    weight_values,
                    input_scales,
                    input_values,
                );
            });

        Ok(())
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_product_i32_lanes_advanced_vector_extensions_two(
    a_32: *const i8,
    b_32: *const i8,
) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;

    let a_bytes = unsafe { _mm256_loadu_si256(a_32 as *const __m256i) };
    let b_bytes = unsafe { _mm256_loadu_si256(b_32 as *const __m256i) };

    let a_lo = _mm256_castsi256_si128(a_bytes);
    let a_hi = _mm256_extracti128_si256(a_bytes, 1);
    let b_lo = _mm256_castsi256_si128(b_bytes);
    let b_hi = _mm256_extracti128_si256(b_bytes, 1);

    let a_lo_i16 = _mm256_cvtepi8_epi16(a_lo);
    let a_hi_i16 = _mm256_cvtepi8_epi16(a_hi);
    let b_lo_i16 = _mm256_cvtepi8_epi16(b_lo);
    let b_hi_i16 = _mm256_cvtepi8_epi16(b_hi);

    let sum_lo_i32 = _mm256_madd_epi16(a_lo_i16, b_lo_i16);
    let sum_hi_i32 = _mm256_madd_epi16(a_hi_i16, b_hi_i16);

    _mm256_add_epi32(sum_lo_i32, sum_hi_i32)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_i32_advanced_vector_extensions_two(v: core::arch::x86_64::__m256i) -> i32 {
    use core::arch::x86_64::*;

    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi32(lo, hi);

    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));

    _mm_cvtsi128_si32(sum32)
}

pub fn dot_product_many_blocks(
    scales_a: &[f32],
    values_a: &[i8],
    scales_b: &[f32],
    values_b: &[i8],
) -> f32 {
    assert_eq!(scales_a.len(), scales_b.len());
    assert_eq!(
        values_a.len(),
        scales_a.len() * QUANTIZATION_BLOCK_GROUP_SIZE
    );
    assert_eq!(
        values_b.len(),
        scales_b.len() * QUANTIZATION_BLOCK_GROUP_SIZE
    );

    let block_count = scales_a.len();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_f32 = 0.0f32;

                for block_index in 0..block_count {
                    let offset = block_index * QUANTIZATION_BLOCK_GROUP_SIZE;

                    let lanes = dot_product_i32_lanes_advanced_vector_extensions_two(
                        values_a.as_ptr().add(offset),
                        values_b.as_ptr().add(offset),
                    );

                    let dot_i32 = horizontal_sum_i32_advanced_vector_extensions_two(lanes);

                    sum_f32 += (dot_i32 as f32) * scales_a[block_index] * scales_b[block_index];
                }

                return sum_f32;
            }
        }
    }

    // Fallback
    let mut sum_f32 = 0.0f32;

    for block_index in 0..block_count {
        let offset = block_index * QUANTIZATION_BLOCK_GROUP_SIZE;

        let mut sum_i32: i32 = 0;
        for i in 0..QUANTIZATION_BLOCK_GROUP_SIZE {
            sum_i32 += (values_a[offset + i] as i32) * (values_b[offset + i] as i32);
        }

        sum_f32 += (sum_i32 as f32) * scales_a[block_index] * scales_b[block_index];
    }

    sum_f32
}
