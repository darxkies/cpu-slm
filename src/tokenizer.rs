use anyhow::{Result, bail};
use std::collections::HashMap;

pub const UNKNOWN_TOKEN: &str = "<unk>";

#[derive(PartialEq)]
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
}

impl BPETokenizer {
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
        fn decode(string: String) -> String {
            string
                .chars()
                .filter_map(|char| {
                    let char = char as u32;

                    char::from_u32(if (256..298).contains(&char) {
                        char - 256
                    } else if (289..323).contains(&char) {
                        char - 256 + 127 - 161
                    } else {
                        char
                    })
                })
                .collect()
        }

        let token_to_id = HashMap::from_iter(
            tokens
                .clone()
                .into_iter()
                .enumerate()
                .map(|(index, value)| (value, index as u32)),
        );

        let tokens = tokens
            .into_iter()
            .map(|string| {
                if r#type == TokenizerType::GPT2 {
                    decode(string)
                } else {
                    string
                }
            })
            .collect::<Vec<_>>();

        let special_tokens = token_type
            .iter()
            .enumerate()
            .filter(|(_, token_type)| **token_type == 3 || **token_type == 4)
            .map(|(index, _)| tokens[index].clone())
            .collect::<Vec<_>>();

        //println!("special tokens: {special_tokens:?}");

        Ok(Self {
            r#type,
            token_to_id,
            id_to_token: tokens,
            special_tokens,
            token_type,
            merges: HashMap::from_iter(
                merges
                    .into_iter()
                    .enumerate()
                    .map(|(index, (first, second))| ((first, second), index as u32)),
            ),
            scores,
            begin_of_sequence_token_identifier,
            end_of_sequence_token_identifier,
            add_space_prefix,
            add_begin_of_sequence_token,
            add_end_of_sequence_token,
        })
    }

    pub fn encode_gpt2(&self, text: &str) -> Result<Vec<u32>> {
        let mut text = text.to_string();

        if self.add_space_prefix {
            text.insert(0, ' ');
        }

        let mut ids: Vec<u32> = Vec::new();

        let mut pieces: Vec<String> = text
            .as_bytes()
            .iter()
            .map(|&byte| {
                let mut value = byte as u32;

                if value < 33 {
                    value += 256;
                } else if value >= 127 && value < 161 {
                    value = value + 256 - 127 + 161;
                }

                char::from_u32(value)
                    .map(|char| char.to_string())
                    .unwrap_or_else(|| format!("<0x{:02X}>", byte))
            })
            .collect();

        for special_token in &self.special_tokens {
            let tokens = special_token
                .chars()
                .map(|char| char.to_string())
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

        if pieces.len() > 1 {
            loop {
                // Find the best merge (lowest rank) among adjacent pairs.
                let mut best_rank: Option<f32> = None;
                let mut best_index: usize = 0;

                for i in 0..pieces.len().saturating_sub(1) {
                    let left = &pieces[i];
                    let right = &pieces[i + 1];

                    let rank = if self.merges.is_empty() {
                        let mut buffer = String::new();

                        buffer.push_str(left);
                        buffer.push_str(right);

                        if let Some(id) = self.token_to_id.get(&buffer) {
                            self.scores.get(*id as usize).map(|value| *value)
                        } else {
                            None
                        }
                    } else {
                        self.merges
                            .get(&(left.clone(), right.clone()))
                            .map(|value| *value as f32)
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

                // No merge found, stop.
                if best_rank.is_none() {
                    break;
                }

                // Merge the best pair.
                let merged = format!("{}{}", pieces[best_index], pieces[best_index + 1]);

                pieces[best_index] = merged;

                pieces.remove(best_index + 1);

                if pieces.len() < 2 {
                    break;
                }
            }
        }

        for piece in &pieces {
            let Some(id) = self.token_to_id.get(piece).copied() else {
                bail!("unknown token {piece:?}");
            };

            ids.push(id);
        }

        Ok(ids)
    }

    pub fn encode_llama(&self, text: &str) -> Result<Vec<u32>> {
        let text = text.replace(" ", "▁");

        let mut ids: Vec<u32> = Vec::new();

        for char in text.chars() {
            let string = char.to_string();

            match self.token_to_id.get(&string) {
                Some(id) => ids.push(*id),
                None => {
                    for byte in string.as_bytes() {
                        // special tokens after <unk>, <s>, and </s>
                        ids.push(*byte as u32 + 3);
                    }
                }
            }
        }

        //println!( "pieces: {:?}", ids.iter() .map(|id| self .id_to_token .get(*id as usize) .cloned() .unwrap_or_default()) .collect::<Vec<_>>());

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
                    bail!("special token {special_token:?} not found ");
                };

                ids.splice(
                    position..position + tokens.len(),
                    [special_token_id.clone()],
                );
            }
        }

        //println!( "pieces: {:?}", ids.iter() .map(|id| self .id_to_token .get(*id as usize) .cloned() .unwrap_or_default()) .collect::<Vec<_>>());

        if ids.len() > 1 {
            loop {
                // Find the best merge (lowest rank) among adjacent pairs.
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
                        if let Some(rank) = self.scores.get(*id as usize).map(|value| *value) {
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

                // No merge found, stop.
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
            if let Some(begin_of_sequence_token_identifier) =
                self.begin_of_sequence_token_identifier
            {
                if ids
                    .first()
                    .is_none_or(|&first| first != begin_of_sequence_token_identifier)
                {
                    ids.insert(0, begin_of_sequence_token_identifier);
                }
            }
        }

        Ok(ids)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self.r#type {
            TokenizerType::GPT2 => self.encode_gpt2(text),
            TokenizerType::LLAMA => self.encode_llama(text),
        }
    }

    pub fn decode_token_gpt2(&self, id: u32) -> String {
        self.id_to_token
            .get(id as usize)
            .cloned()
            .unwrap_or(UNKNOWN_TOKEN.to_string())
    }

    pub fn decode_token_llama(&self, id: u32) -> String {
        let piece = self
            .id_to_token
            .get(id as usize)
            .cloned()
            .unwrap_or(UNKNOWN_TOKEN.to_string());

        if let Some(value) = piece.strip_prefix("<0x") {
            if let Some(value) = value.strip_suffix(">") {
                if let Ok(byte) = usize::from_str_radix(&value, 16) {
                    return match char::from_u32(byte as u32) {
                        Some(value) => value.to_string(),
                        None => piece,
                    };
                }
            }
        }

        piece.replace("▁", " ").to_string()
    }

    pub fn decode_token(&self, id: u32) -> String {
        match self.r#type {
            TokenizerType::GPT2 => self.decode_token_gpt2(id),
            TokenizerType::LLAMA => self.decode_token_llama(id),
        }
    }

    pub fn decode_tokens(&self, ids: &[u32]) -> Vec<String> {
        ids.iter().map(|id| self.decode_token(*id)).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();

        for id in ids {
            let token = self.decode_token(*id);

            result.push_str(&token);
        }

        result
    }
}
