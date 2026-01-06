use std::cmp::Ordering;

use crate::math::*;

#[derive(Clone, Copy)]
pub struct ProbabilityWithIndex {
    pub probability: f32,
    pub index: u32,
}

pub struct Sampler {
    pub temperature: f32,
    pub top_probability: f32,
    pub random_state: u64,
    pub probability_buffer: Vec<ProbabilityWithIndex>,
}

impl Sampler {
    pub fn new(vocabulary_size: usize, temperature: f32, top_probability: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_probability,
            random_state: seed,
            probability_buffer: vec![
                ProbabilityWithIndex {
                    probability: 0.0f32,
                    index: 0
                };
                vocabulary_size
            ],
        }
    }

    fn sample_greedy_maximum(probabilities: &[f32]) -> u32 {
        let mut best_index = 0u32;
        let mut best_value = probabilities[0];

        for i in 1..probabilities.len() {
            if probabilities[i] > best_value {
                best_value = probabilities[i];
                best_index = i as u32;
            }
        }

        best_index
    }

    fn sample_from_distribution(probabilities: &[f32], coin: f32) -> u32 {
        let mut cumulative = 0.0f32;

        for (index, &p) in probabilities.iter().enumerate() {
            cumulative += p;

            if coin < cumulative {
                return index as u32;
            }
        }

        probabilities.len().saturating_sub(1) as u32
    }

    fn sample_top_probability(
        probabilities: &[f32],
        top_probability: f32,
        buffer: &mut [ProbabilityWithIndex],
        coin: f32,
    ) -> u32 {
        let count = probabilities.len();
        let cutoff = (1.0f32 - top_probability) / ((count as f32) - 1.0f32);

        let mut candidate_count = 0usize;

        for i in 0..count {
            if probabilities[i] >= cutoff {
                buffer[candidate_count] = ProbabilityWithIndex {
                    probability: probabilities[i],
                    index: i as u32,
                };

                candidate_count += 1;
            }
        }

        buffer[..candidate_count].sort_by(|a, b| {
            if a.probability > b.probability {
                Ordering::Less
            } else if a.probability < b.probability {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });

        let mut cumulative = 0.0f32;
        let mut last_index = candidate_count.saturating_sub(1);

        for i in 0..candidate_count {
            cumulative += buffer[i].probability;

            if cumulative > top_probability {
                last_index = i;
                break;
            }
        }

        let target = coin * cumulative;

        let mut running = 0.0f32;

        for i in 0..=last_index {
            running += buffer[i].probability;

            if target < running {
                return buffer[i].index as u32;
            }
        }

        buffer[last_index].index as u32
    }

    fn random_u32(state: &mut u64) -> u32 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;

        let mixed = state.wrapping_mul(0x2545F4914F6CDD1Du64);

        (mixed >> 32) as u32
    }

    fn random_f32(state: &mut u64) -> f32 {
        (Self::random_u32(state) >> 8) as f32 / 16777216.0f32
    }

    pub fn sample_next_token(&mut self, logits: &mut [f32]) -> u32 {
        if self.temperature == 0.0f32 {
            return Self::sample_greedy_maximum(logits);
        }

        for value in logits.iter_mut() {
            *value /= self.temperature;
        }

        softmax_in_place(logits);

        let coin = Self::random_f32(&mut self.random_state);

        if self.top_probability <= 0.0f32 || self.top_probability >= 1.0f32 {
            Self::sample_from_distribution(logits, coin)
        } else {
            Self::sample_top_probability(
                logits,
                self.top_probability,
                &mut self.probability_buffer,
                coin,
            )
        }
    }
}
