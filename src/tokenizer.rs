use anyhow::{bail, Result};
use std::collections::HashMap;

pub const UNKNOWN_TOKEN: &str = "<unk>";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TokenizerType {
    GPT2,
    LLAMA,
}

pub struct BPETokenizer {
    pub r#type: TokenizerType,
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub merges: HashMap<(String, String), u32>,
    pub token_type: Vec<i32>,
    pub begin_of_sequence_token_identifier: Option<u32>,
    pub end_of_sequence_token_identifier: u32,
    pub special_tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub add_space_prefix: bool,
    pub add_begin_of_sequence_token: bool,
    pub add_end_of_sequence_token: bool,

    gpt2_bytes_to_unicode: Option<HashMap<u8, char>>,
    gpt2_unicode_to_bytes: Option<HashMap<char, u8>>,
}

impl BPETokenizer {
    fn gpt2_bytes_to_unicode_tables() -> (HashMap<u8, char>, HashMap<char, u8>) {
        let mut bytes_that_map_to_themselves: Vec<u32> = Vec::new();

        bytes_that_map_to_themselves.extend(33u32..=126u32);
        bytes_that_map_to_themselves.extend(161u32..=172u32);
        bytes_that_map_to_themselves.extend(174u32..=255u32);

        let mut unicode_code_points: Vec<u32> = bytes_that_map_to_themselves.clone();

        let mut count: u32 = 0;

        for byte in 0u32..=255u32 {
            if !bytes_that_map_to_themselves.contains(&byte) {
                bytes_that_map_to_themselves.push(byte);

                unicode_code_points.push(256 + count);

                count += 1;
            }
        }

        let mut byte_to_unicode: HashMap<u8, char> = HashMap::with_capacity(256);
        let mut unicode_to_byte: HashMap<char, u8> = HashMap::with_capacity(256);

        for (byte, code) in bytes_that_map_to_themselves
            .into_iter()
            .zip(unicode_code_points.into_iter())
        {
            let char = char::from_u32(code).expect("valid unicode scalar value");

            byte_to_unicode.insert(byte as u8, char);

            unicode_to_byte.insert(char, byte as u8);
        }

        (byte_to_unicode, unicode_to_byte)
    }

    pub fn new(
        r#type: TokenizerType,
        tokens: Vec<String>,
        merges: Vec<(String, String)>,
        token_type: Vec<i32>,
        scores: Vec<f32>,
        begin_of_sequence_token_identifier: Option<u32>,
        end_of_sequence_token_identifier: u32,
        add_space_prefix: bool,
        add_begin_of_sequence_token: bool,
        add_end_of_sequence_token: bool,
    ) -> Result<Self> {
        let id_to_token = tokens;

        let token_to_id = HashMap::from_iter(
            id_to_token
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, value)| (value, index as u32)),
        );

        let special_tokens = token_type
            .iter()
            .enumerate()
            .filter(|(_, token)| **token == 3 || **token == 4)
            .map(|(index, _)| {
                id_to_token
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| UNKNOWN_TOKEN.to_string())
            })
            .collect::<Vec<_>>();

        let (gpt2_bytes_to_unicode, gpt2_unicode_to_bytes) = if r#type == TokenizerType::GPT2 {
            let (bytes_to_unicode, unicode_to_bytes) = Self::gpt2_bytes_to_unicode_tables();

            (Some(bytes_to_unicode), Some(unicode_to_bytes))
        } else {
            (None, None)
        };

        Ok(Self {
            r#type,
            token_to_id,
            id_to_token,
            merges: HashMap::from_iter(
                merges
                    .into_iter()
                    .enumerate()
                    .map(|(index, (first, second))| ((first, second), index as u32)),
            ),
            token_type,
            begin_of_sequence_token_identifier,
            end_of_sequence_token_identifier,
            special_tokens,
            scores,
            add_space_prefix,
            add_begin_of_sequence_token,
            add_end_of_sequence_token,
            gpt2_bytes_to_unicode,
            gpt2_unicode_to_bytes,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self.r#type {
            TokenizerType::GPT2 => self.encode_gpt2(text),
            TokenizerType::LLAMA => self.encode_llama(text),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self.r#type {
            TokenizerType::GPT2 => self.decode_gpt2(ids),
            TokenizerType::LLAMA => Ok(self.decode_llama(ids)),
        }
    }

    pub fn decode_token(&self, id: u32) -> String {
        self.id_to_token
            .get(id as usize)
            .cloned()
            .unwrap_or_else(|| UNKNOWN_TOKEN.to_string())
    }

    fn encode_gpt2(&self, text: &str) -> Result<Vec<u32>> {
        let mut text = text.to_string();

        if self.add_space_prefix {
            text.insert(0, ' ');
        }

        let byte_to_unicode = self
            .gpt2_bytes_to_unicode
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing GPT2 bytes_to_unicode mapping"))?;

        let mut pieces: Vec<String> = text
            .as_bytes()
            .iter()
            .map(|&byte| {
                let char = *byte_to_unicode.get(&byte).expect("all bytes mapped");

                char.to_string()
            })
            .collect();

        // Collapse special tokens first.
        for special_token in &self.special_tokens {
            let tokens = special_token
                .chars()
                .map(|ch| ch.to_string())
                .collect::<Vec<String>>();

            loop {
                let mut found = None;
                for i in 0..=pieces.len().saturating_sub(tokens.len()) {
                    if tokens[..] == pieces[i..i + tokens.len()] {
                        found = Some(i);

                        break;
                    }
                }

                let Some(position) = found else {
                    break;
                };

                pieces.splice(position..position + tokens.len(), [special_token.clone()]);
            }
        }

        // Merge loop
        if pieces.len() > 1 {
            loop {
                let mut best_rank: Option<f32> = None;
                let mut best_index: usize = 0;

                for i in 0..pieces.len().saturating_sub(1) {
                    let left = &pieces[i];
                    let right = &pieces[i + 1];

                    let rank = if !self.merges.is_empty() {
                        self.merges
                            .get(&(left.clone(), right.clone()))
                            .map(|value| *value as f32)
                    } else {
                        let mut buffer = String::new();
                        buffer.push_str(left);
                        buffer.push_str(right);

                        if let Some(id) = self.token_to_id.get(&buffer) {
                            self.scores.get(*id as usize).copied()
                        } else {
                            None
                        }
                    };

                    if let Some(rank) = rank {
                        match best_rank {
                            None => {
                                best_rank = Some(rank);
                                best_index = i;
                            }
                            Some(current_best) => {
                                if rank < current_best {
                                    best_rank = Some(rank);
                                    best_index = i;
                                }
                            }
                        }
                    }
                }

                if best_rank.is_none() {
                    break;
                }

                let merged = format!("{}{}", pieces[best_index], pieces[best_index + 1]);

                pieces[best_index] = merged;

                pieces.remove(best_index + 1);

                if pieces.len() < 2 {
                    break;
                }
            }
        }

        let mut ids: Vec<u32> = Vec::with_capacity(pieces.len());

        for piece in &pieces {
            let Some(id) = self.token_to_id.get(piece).copied() else {
                bail!("unknown token {piece:?}");
            };

            ids.push(id);
        }

        Ok(ids)
    }

    fn decode_gpt2(&self, ids: &[u32]) -> Result<String> {
        let unicode_to_byte = self
            .gpt2_unicode_to_bytes
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing GPT2 unicode_to_bytes mapping"))?;

        let mut encoded = String::new();

        for &id in ids {
            let token = self.decode_token(id);

            encoded.push_str(&token);
        }

        let mut bytes: Vec<u8> = Vec::with_capacity(encoded.len());

        for ch in encoded.chars() {
            if let Some(b) = unicode_to_byte.get(&ch) {
                bytes.push(*b);
            } else {
                bytes.extend_from_slice(ch.to_string().as_bytes());
            }
        }

        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    fn encode_llama(&self, text: &str) -> Result<Vec<u32>> {
        let text = text.replace(' ', "▁");
        let mut ids: Vec<u32> = Vec::new();

        for ch in text.chars() {
            let s = ch.to_string();

            match self.token_to_id.get(&s) {
                Some(id) => ids.push(*id),
                None => {
                    for byte in s.as_bytes() {
                        // Skip <unk>, <s>, </s>
                        ids.push(*byte as u32 + 3);
                    }
                }
            }
        }

        // Collapse special tokens.
        for special_token in &self.special_tokens {
            let tokens = special_token
                .chars()
                .map(|char| {
                    let string = char.to_string();

                    let Some(id) = self.token_to_id.get(&string) else {
                        bail!("unknown token {string:?}");
                    };

                    Ok(*id)
                })
                .collect::<Result<Vec<_>>>()?;

            loop {
                let mut found = None;

                for i in 0..=ids.len().saturating_sub(tokens.len()) {
                    if tokens[..] == ids[i..i + tokens.len()] {
                        found = Some(i);

                        break;
                    }
                }

                let Some(position) = found else {
                    break;
                };

                let Some(special_token_id) = self.token_to_id.get(special_token) else {
                    bail!("special token {special_token:?} not found");
                };

                ids.splice(position..position + tokens.len(), [*special_token_id]);
            }
        }

        // Merge loop using scores and full token lookup.
        if ids.len() > 1 {
            loop {
                let mut best_rank: Option<f32> = None;
                let mut best_index: usize = 0;
                let mut best_id: u32 = 0;

                for i in 0..ids.len().saturating_sub(1) {
                    let Some(left) = self.id_to_token.get(ids[i] as usize) else {
                        bail!("{:?} has no token", ids[i]);
                    };

                    let Some(right) = self.id_to_token.get(ids[i + 1] as usize) else {
                        bail!("{:?} has no token", ids[i + 1]);
                    };

                    let mut buffer = String::new();

                    buffer.push_str(left);
                    buffer.push_str(right);

                    if let Some(id) = self.token_to_id.get(&buffer) {
                        if let Some(rank) = self.scores.get(*id as usize).copied() {
                            match best_rank {
                                None => {
                                    best_rank = Some(rank);
                                    best_index = i;
                                    best_id = *id;
                                }
                                Some(current_best) => {
                                    if rank < current_best {
                                        best_rank = Some(rank);
                                        best_index = i;
                                        best_id = *id;
                                    }
                                }
                            }
                        }
                    }
                }

                if best_rank.is_none() {
                    break;
                }

                ids[best_index] = best_id;
                ids.remove(best_index + 1);

                if ids.len() < 2 {
                    break;
                }
            }
        }

        if self.add_begin_of_sequence_token {
            if let Some(bos) = self.begin_of_sequence_token_identifier {
                if ids.first().is_none_or(|&first| first != bos) {
                    ids.insert(0, bos);
                }
            }
        }

        if self.add_end_of_sequence_token {
            if ids
                .last()
                .is_none_or(|&last| last != self.end_of_sequence_token_identifier)
            {
                ids.push(self.end_of_sequence_token_identifier);
            }
        }

        Ok(ids)
    }

    fn decode_llama(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();

        for &id in ids {
            let piece = self
                .id_to_token
                .get(id as usize)
                .cloned()
                .unwrap_or_else(|| UNKNOWN_TOKEN.to_string());

            let piece = piece.replace('▁', " ");

            if let Some(hex) = piece
                .strip_prefix("<0x")
                .and_then(|value| value.strip_suffix('>'))
            {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    bytes.push(byte);

                    continue;
                }
            }

            bytes.extend_from_slice(piece.as_bytes());
        }

        String::from_utf8_lossy(&bytes).to_string()
    }
}

