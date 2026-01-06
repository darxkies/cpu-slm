#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(clippy::too_many_lines)]

use crate::quantized_tensor::QUANTIZATION_BLOCK_GROUP_SIZE;

pub fn root_mean_square_normalization_in_place(values: &mut [f32], weights: &[f32], epsilon: f32) {
    assert_eq!(values.len(), weights.len());

    let length = values.len();

    if length == 0 {
        return;
    }

    let mut sum_of_squares = 0.0f32;

    for value in values.iter() {
        sum_of_squares += value * value;
    }

    let mean_square = sum_of_squares / (length as f32);
    let inverse_root_mean_square = 1.0f32 / (mean_square + epsilon).sqrt();

    for i in 0..length {
        values[i] = weights[i] * inverse_root_mean_square * values[i];
    }
}

#[inline]
pub fn softmax_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    let mut maximum = f32::NEG_INFINITY;

    for &value in values.iter() {
        if value > maximum {
            maximum = value;
        }
    }

    if !maximum.is_finite() {
        let inverse = 1.0f32 / (values.len() as f32);
        for value in values.iter_mut() {
            *value = inverse;
        }
        return;
    }

    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }

    let inverse_sum = 1.0f32 / sum;
    for value in values.iter_mut() {
        *value *= inverse_sum;
    }
}

#[derive(Clone, Debug)]
pub struct RotaryPositionalEmbeddingCache {
    maximum_sequence_length: usize,
    half_head_dimension: usize,
    cosine: Vec<f32>,
    sine: Vec<f32>,
}

impl RotaryPositionalEmbeddingCache {
    pub fn new(
        maximum_sequence_length: usize,
        head_dimension: usize,
        rotary_embedding_frequency_base: f32,
    ) -> Self {
        debug_assert!(head_dimension % 2 == 0, "head dimension must be even");

        let half_head_dimension = head_dimension / 2;

        let mut frequency: Vec<f32> = vec![0.0f32; half_head_dimension];
        for pair_index in 0..half_head_dimension {
            frequency[pair_index] = rotary_embedding_frequency_base
                .powf(-(pair_index as f32) / (half_head_dimension as f32));
        }

        let table_size = maximum_sequence_length * half_head_dimension;
        let mut cosine = vec![0.0f32; table_size];
        let mut sine = vec![0.0f32; table_size];

        for position in 0..maximum_sequence_length {
            let row_offset = position * half_head_dimension;
            let position_f32 = position as f32;

            for index in 0..half_head_dimension {
                let angle = position_f32 * frequency[index];
                let (sine_value, cosine_value) = angle.sin_cos();
                cosine[row_offset + index] = cosine_value;
                sine[row_offset + index] = sine_value;
            }
        }

        Self {
            maximum_sequence_length,
            half_head_dimension,
            cosine,
            sine,
        }
    }

    #[inline]
    pub fn cosine_row(&self, position: usize) -> &[f32] {
        debug_assert!(position < self.maximum_sequence_length);
        let start = position * self.half_head_dimension;
        &self.cosine[start..start + self.half_head_dimension]
    }

    #[inline]
    pub fn sine_row(&self, position: usize) -> &[f32] {
        debug_assert!(position < self.maximum_sequence_length);
        let start = position * self.half_head_dimension;
        &self.sine[start..start + self.half_head_dimension]
    }

    #[inline]
    pub fn half_head_dimension(&self) -> usize {
        self.half_head_dimension
    }

    #[inline]
    pub fn apply_in_place(&self, values: &mut [f32], position: usize) {
        let head_dimension = values.len();
        let half = head_dimension / 2;

        debug_assert!(head_dimension % 2 == 0);
        debug_assert!(half == self.half_head_dimension());

        let cosine = self.cosine_row(position);
        let sine = self.sine_row(position);

        for index in 0..half {
            let c = cosine[index];
            let s = sine[index];

            let real = values[index];
            let imaginary = values[index + half];

            values[index] = real * c - imaginary * s;
            values[index + half] = real * s + imaginary * c;
        }
    }

    #[inline]
    pub fn apply_interleaved_in_place(&self, values: &mut [f32], position: usize) {
        let head_dimension = values.len();
        let half = head_dimension / 2;

        debug_assert!(head_dimension % 2 == 0);
        debug_assert!(half == self.half_head_dimension());

        let cosine = self.cosine_row(position);
        let sine = self.sine_row(position);

        for index in 0..half {
            let cosine = cosine[index];
            let sine = sine[index];

            let index = index * 2;

            let real = values[index];
            let imaginary = values[index + 1];

            values[index] = real * cosine - imaginary * sine;
            values[index + 1] = real * sine + imaginary * cosine;
        }
    }
}

#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                return dot_product_f32_avx2_fma(a, b);
            }
        }
    }

    dot_product_f32_scalar(a, b)
}

#[inline]
fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    use crate::quantized_tensor::QUANTIZATION_BLOCK_GROUP_SIZE;

    let n = a.len();
    let mut i = 0usize;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + QUANTIZATION_BLOCK_GROUP_SIZE <= n {
        let a0 = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let b0 = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };

        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        let a1 = unsafe { _mm256_loadu_ps(a.as_ptr().add(i + 8)) };
        let b1 = unsafe { _mm256_loadu_ps(b.as_ptr().add(i + 8)) };

        acc1 = _mm256_fmadd_ps(a1, b1, acc1);

        let a2 = unsafe { _mm256_loadu_ps(a.as_ptr().add(i + 16)) };
        let b2 = unsafe { _mm256_loadu_ps(b.as_ptr().add(i + 16)) };

        acc2 = _mm256_fmadd_ps(a2, b2, acc2);

        let a3 = unsafe { _mm256_loadu_ps(a.as_ptr().add(i + 24)) };
        let b3 = unsafe { _mm256_loadu_ps(b.as_ptr().add(i + 24)) };

        acc3 = _mm256_fmadd_ps(a3, b3, acc3);

        i += 32;
    }

    let mut accumulator = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    while i + 8 <= n {
        let av = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let bv = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };

        accumulator = _mm256_fmadd_ps(av, bv, accumulator);

        i += 8;
    }

    let hi128 = _mm256_extractf128_ps(accumulator, 1);
    let lo128 = _mm256_castps256_ps128(accumulator);
    let sum128 = _mm_add_ps(lo128, hi128);

    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut result = _mm_cvtss_f32(sums2);

    while i < n {
        result += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };

        i += 1;
    }

    result
}

#[inline(always)]
pub fn online_softmax_update(
    score: f32,
    running_maximum: &mut f32,
    running_sum_exponentials: &mut f32,
    accumulator: &mut [f32],
    value: &[f32],
) {
    if score > *running_maximum {
        let scale = (*running_maximum - score).exp();

        for a in accumulator.iter_mut() {
            *a *= scale;
        }

        *running_sum_exponentials *= scale;
        *running_maximum = score;
    }

    let weight = (score - *running_maximum).exp();

    *running_sum_exponentials += weight;

    for i in 0..accumulator.len() {
        accumulator[i] += weight * value[i];
    }
}

#[inline(always)]
pub fn online_softmax_finalize(accumulator: &mut [f32], running_sum_exponentials: f32) {
    let inverse = 1.0f32 / running_sum_exponentials.max(1.0e-20f32);

    for value in accumulator.iter_mut() {
        *value *= inverse;
    }
}

#[inline]
pub fn quantize_block_signed_eight_bit(destination: &mut [i8], source: &[f32]) -> f32 {
    debug_assert_eq!(source.len(), destination.len());
    debug_assert!(!source.is_empty());

    let mut maximum_absolute = 0.0f32;

    for &value in source.iter() {
        let absolute = value.abs();
        if absolute > maximum_absolute {
            maximum_absolute = absolute;
        }
    }

    if maximum_absolute == 0.0f32 {
        destination.fill(0);

        return 0.0f32;
    }

    let scale = maximum_absolute / 127.0f32;
    let inverse_scale = 1.0f32 / scale;

    for i in 0..source.len() {
        let mut value = (source[i] * inverse_scale).round() as i32;

        if value > 127 {
            value = 127;
        }

        if value < -127 {
            value = -127;
        }

        destination[i] = value as i8;
    }

    scale
}

#[inline]
pub fn dot_product_f32_with_quantized_signed_eight_bit(
    query: &[f32],
    key_quantized: &[i8],
    key_scales: &[f32],
) -> f32 {
    debug_assert_eq!(query.len(), key_quantized.len());
    debug_assert_eq!(query.len() % QUANTIZATION_BLOCK_GROUP_SIZE, 0);
    debug_assert_eq!(
        key_scales.len(),
        query.len() / QUANTIZATION_BLOCK_GROUP_SIZE
    );

    let mut sum = 0.0f32;

    for block_index in 0..key_scales.len() {
        let scale = key_scales[block_index];

        if scale == 0.0f32 {
            continue;
        }

        let start = block_index * QUANTIZATION_BLOCK_GROUP_SIZE;
        let end = start + QUANTIZATION_BLOCK_GROUP_SIZE;

        let mut block_sum = 0.0f32;

        for i in start..end {
            block_sum += query[i] * (key_quantized[i] as f32);
        }

        sum += block_sum * scale;
    }

    sum
}

#[inline(always)]
pub fn online_softmax_update_with_quantized_values(
    score: f32,
    running_maximum: &mut f32,
    running_sum_exponentials: &mut f32,
    accumulator: &mut [f32],
    value_quantized: &[i8],
    value_scales: &[f32],
) {
    debug_assert_eq!(accumulator.len(), value_quantized.len());
    debug_assert_eq!(accumulator.len() % QUANTIZATION_BLOCK_GROUP_SIZE, 0);
    debug_assert_eq!(
        value_scales.len(),
        accumulator.len() / QUANTIZATION_BLOCK_GROUP_SIZE
    );

    if score > *running_maximum {
        let rescale = (*running_maximum - score).exp();

        for a in accumulator.iter_mut() {
            *a *= rescale;
        }

        *running_sum_exponentials *= rescale;
        *running_maximum = score;
    }

    let weight = (score - *running_maximum).exp();

    *running_sum_exponentials += weight;

    for block_index in 0..value_scales.len() {
        let scale = value_scales[block_index];

        if scale == 0.0f32 {
            continue;
        }

        let start = block_index * QUANTIZATION_BLOCK_GROUP_SIZE;
        let end = start + QUANTIZATION_BLOCK_GROUP_SIZE;

        let weight_times_scale = weight * scale;

        for i in start..end {
            accumulator[i] += weight_times_scale * (value_quantized[i] as f32);
        }
    }
}
