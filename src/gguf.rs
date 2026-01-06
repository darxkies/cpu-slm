#![allow(dead_code)]
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{Result, bail};

use crate::quantized_tensor::{QUANTIZATION_BLOCK_GROUP_SIZE, QuantizedTensor};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION_SUPPORTED: u32 = 3;

const ALIGNMENT_BYTES: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataValueType {
    Unsigned8 = 0,
    Signed8 = 1,
    Unsigned16 = 2,
    Signed16 = 3,
    Unsigned32 = 4,
    Signed32 = 5,
    Float32 = 6,
    Boolean = 7,
    String = 8,
    Array = 9,
    Unsigned64 = 10,
    Signed64 = 11,
    Float64 = 12,
}

impl MetadataValueType {
    pub fn from_u32(value: u32) -> Result<Self> {
        let _type = match value {
            0 => Self::Unsigned8,
            1 => Self::Signed8,
            2 => Self::Unsigned16,
            3 => Self::Signed16,
            4 => Self::Unsigned32,
            5 => Self::Signed32,
            6 => Self::Float32,
            7 => Self::Boolean,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Unsigned64,
            11 => Self::Signed64,
            12 => Self::Float64,
            other => {
                bail!("invalid metadata value type {other:?}");
            }
        };

        Ok(_type)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Unsigned8(u8),
    Signed8(i8),
    Unsigned16(u16),
    Signed16(i16),
    Unsigned32(u32),
    Signed32(i32),
    Unsigned64(u64),
    Signed64(i64),
    Float32(f32),
    Float64(f64),
    Boolean(bool),
    String(String),
    Float32Array(Vec<f32>),
    Signed32Array(Vec<i32>),
    StringArray(Vec<String>),
    Array {
        element_type: MetadataValueType,
        elements: Vec<MetadataValue>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    Float32 = 0,
    Float16 = 1,
    Quantization4_0 = 2,
    Quantization4_1 = 3,
    Quantization5_0 = 6,
    Quantization5_1 = 7,
    Quantization8_0 = 8,
    Quantization8_1 = 9,
    Quantization2K = 10,
    Quantization3K = 11,
    Quantization4K = 12,
    Quantization5K = 13,
    Quantization6K = 14,
    Quantization8K = 15,
    Integer8 = 16,
    Integer16 = 17,
    Integer32 = 18,
    Integer64 = 19,
    Float64 = 20,
    Quantization1B = 21,
}

impl GgmlType {
    pub fn from_u32(value: u32) -> Result<Self> {
        let _type = match value {
            0 => Self::Float32,
            1 => Self::Float16,
            2 => Self::Quantization4_0,
            3 => Self::Quantization4_1,
            6 => Self::Quantization5_0,
            7 => Self::Quantization5_1,
            8 => Self::Quantization8_0,
            9 => Self::Quantization8_1,
            10 => Self::Quantization2K,
            11 => Self::Quantization3K,
            12 => Self::Quantization4K,
            13 => Self::Quantization5K,
            14 => Self::Quantization6K,
            15 => Self::Quantization8K,
            16 => Self::Integer8,
            17 => Self::Integer16,
            18 => Self::Integer32,
            19 => Self::Integer64,
            20 => Self::Float64,
            21 => Self::Quantization1B,
            other => {
                bail!("invalid ggml value type {other:?}");
            }
        };

        Ok(_type)
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Quantization4_0
                | Self::Quantization4_1
                | Self::Quantization5_0
                | Self::Quantization5_1
                | Self::Quantization8_0
                | Self::Quantization8_1
                | Self::Quantization2K
                | Self::Quantization3K
                | Self::Quantization4K
                | Self::Quantization5K
                | Self::Quantization6K
                | Self::Quantization8K
                | Self::Quantization1B
        )
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub number_of_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub position: u64,
}

#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_start: u64,
}

impl GgufHeader {
    pub fn tensor_by_name(&self, name: &str) -> Result<&TensorInfo> {
        let Some(tensor) = self.tensors.iter().find(|tensor| tensor.name == name) else {
            bail!("missing tensor {name:?}");
        };

        Ok(tensor)
    }
}

pub struct GgufReader<R> {
    inner: R,
    position: u64,
}

impl<R> GgufReader<R>
where
    R: Read + Seek,
{
    pub fn new(inner: R) -> Self {
        Self { inner, position: 0 }
    }

    pub fn into_inner(self) -> R {
        self.inner
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    fn seek_absolute(&mut self, absolute_position: u64) -> Result<()> {
        self.inner.seek(SeekFrom::Start(absolute_position))?;
        self.position = absolute_position;

        Ok(())
    }

    fn read_exact_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buffer = [0u8; N];

        self.inner.read_exact(&mut buffer)?;
        self.position += N as u64;

        Ok(buffer)
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        let buffer = self.read_exact_array::<1>()?;

        Ok(buffer[0])
    }

    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    pub fn read_u16_le(&mut self) -> Result<u16> {
        let buffer = self.read_exact_array::<2>()?;

        Ok(u16::from_le_bytes(buffer))
    }

    pub fn read_i16_le(&mut self) -> Result<i16> {
        let buffer = self.read_exact_array::<2>()?;

        Ok(i16::from_le_bytes(buffer))
    }

    pub fn read_u32_le(&mut self) -> Result<u32> {
        let buffer = self.read_exact_array::<4>()?;

        Ok(u32::from_le_bytes(buffer))
    }

    pub fn read_i32_le(&mut self) -> Result<i32> {
        let buffer = self.read_exact_array::<4>()?;

        Ok(i32::from_le_bytes(buffer))
    }

    pub fn read_u64_le(&mut self) -> Result<u64> {
        let buffer = self.read_exact_array::<8>()?;

        Ok(u64::from_le_bytes(buffer))
    }

    pub fn read_i64_le(&mut self) -> Result<i64> {
        let buffer = self.read_exact_array::<8>()?;

        Ok(i64::from_le_bytes(buffer))
    }

    pub fn read_f32_le(&mut self) -> Result<f32> {
        let buffer = self.read_exact_array::<4>()?;

        Ok(f32::from_le_bytes(buffer))
    }

    pub fn read_f64_le(&mut self) -> Result<f64> {
        let buffer = self.read_exact_array::<8>()?;

        Ok(f64::from_le_bytes(buffer))
    }

    pub fn read_bytes(&mut self, length: usize) -> Result<Vec<u8>> {
        let mut buffer = vec![0u8; length];

        self.inner.read_exact(&mut buffer)?;
        self.position += length as u64;

        Ok(buffer)
    }

    pub fn read_gguf_string(&mut self) -> Result<String> {
        let length_u64 = self.read_u64_le()?;

        let Ok(length_usize) = usize::try_from(length_u64) else {
            bail!("can't convert {length_u64} to usize");
        };

        let bytes = self.read_bytes(length_usize)?;
        Ok(String::from_utf8(bytes)?)
    }

    fn align_up(offset: u64, alignment: u64) -> u64 {
        if alignment == 0 {
            return offset;
        }

        let remainder = offset % alignment;

        if remainder == 0 {
            offset
        } else {
            offset + (alignment - remainder)
        }
    }

    fn align_reader_to(&mut self, alignment: u64) -> Result<()> {
        let aligned = Self::align_up(self.position, alignment);

        if aligned != self.position {
            self.seek_absolute(aligned)?;
        }

        Ok(())
    }

    fn read_boolean(&mut self) -> Result<bool> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => {
                bail!("invalid bool value {other}");
            }
        }
    }

    fn read_metadata_value(&mut self, value_type: MetadataValueType) -> Result<MetadataValue> {
        Ok(match value_type {
            MetadataValueType::Unsigned8 => MetadataValue::Unsigned8(self.read_u8()?),
            MetadataValueType::Signed8 => MetadataValue::Signed8(self.read_i8()?),
            MetadataValueType::Unsigned16 => MetadataValue::Unsigned16(self.read_u16_le()?),
            MetadataValueType::Signed16 => MetadataValue::Signed16(self.read_i16_le()?),
            MetadataValueType::Unsigned32 => MetadataValue::Unsigned32(self.read_u32_le()?),
            MetadataValueType::Signed32 => MetadataValue::Signed32(self.read_i32_le()?),
            MetadataValueType::Unsigned64 => MetadataValue::Unsigned64(self.read_u64_le()?),
            MetadataValueType::Signed64 => MetadataValue::Signed64(self.read_i64_le()?),
            MetadataValueType::Float32 => MetadataValue::Float32(self.read_f32_le()?),
            MetadataValueType::Float64 => MetadataValue::Float64(self.read_f64_le()?),
            MetadataValueType::Boolean => MetadataValue::Boolean(self.read_boolean()?),
            MetadataValueType::String => MetadataValue::String(self.read_gguf_string()?),

            MetadataValueType::Array => {
                let element_type_code = self.read_u32_le()?;
                let element_type = MetadataValueType::from_u32(element_type_code)?;

                let length = self.read_u64_le()?;
                let Ok(length_usize) = usize::try_from(length) else {
                    bail!("invalid array elements size {length}");
                };

                if element_type == MetadataValueType::Signed32 {
                    let mut elements = Vec::with_capacity(length_usize);

                    for _ in 0..length_usize {
                        let value = self.read_i32_le()?;

                        elements.push(value);
                    }

                    MetadataValue::Signed32Array(elements)
                } else if element_type == MetadataValueType::Float32 {
                    let mut elements = Vec::with_capacity(length_usize);

                    for _ in 0..length_usize {
                        let value = self.read_f32_le()?;

                        elements.push(value);
                    }

                    MetadataValue::Float32Array(elements)
                } else if element_type == MetadataValueType::String {
                    let mut elements = Vec::with_capacity(length_usize);

                    for _ in 0..length_usize {
                        let value = self.read_gguf_string()?;

                        elements.push(value);
                    }

                    MetadataValue::StringArray(elements)
                } else {
                    let mut elements = Vec::with_capacity(length_usize);

                    for _ in 0..length_usize {
                        let value = self.read_metadata_value(element_type)?;

                        elements.push(value);
                    }

                    MetadataValue::Array {
                        element_type,
                        elements,
                    }
                }
            }
        })
    }

    pub fn read_header(&mut self) -> Result<GgufHeader> {
        let magic = self.read_exact_array::<4>()?;

        if magic != GGUF_MAGIC {
            bail!("invalid gguf magic {magic:?}");
        }

        let version = self.read_u32_le()?;

        println!("version: {version}");

        if version != GGUF_VERSION_SUPPORTED {
            bail!("unsupported gguf version {version:?}");
        }

        let tensor_count = self.read_u64_le()?;
        let metadata_count = self.read_u64_le()?;

        println!("tensor count: {tensor_count}");
        println!("metadata count: {metadata_count}");

        let mut metadata = HashMap::new();

        for _ in 0..metadata_count {
            let key = self.read_gguf_string()?;
            let value_type_code = self.read_u32_le()?;
            let value_type = MetadataValueType::from_u32(value_type_code)?;
            let value = self.read_metadata_value(value_type)?;

            let string_value = format!("{value:?}");

            println!(
                "metadata: {key}={}",
                string_value.chars().take(128).collect::<String>()
            );

            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0));

        for _ in 0..tensor_count {
            let name = self.read_gguf_string()?;
            let name_length = name.len() as u64;

            if name_length > 64 {
                bail!("invalid tensor name length {name_length}");
            }

            let number_of_dimensions = self.read_u32_le()?;

            let Ok(number_of_dimensions_usize) = usize::try_from(number_of_dimensions) else {
                bail!("invalid tensor dimensions {number_of_dimensions}");
            };

            let mut dimensions = Vec::with_capacity(number_of_dimensions_usize);

            for _ in 0..number_of_dimensions_usize {
                dimensions.push(self.read_u64_le()?);
            }

            let ggml_type_code = self.read_u32_le()?;
            let ggml_type = GgmlType::from_u32(ggml_type_code)?;

            let offset_bytes = self.read_u64_le()?;

            if offset_bytes % ALIGNMENT_BYTES != 0 {
                bail!("invalid gguf allignment {offset_bytes}:{ALIGNMENT_BYTES}")
            }

            tensors.push(TensorInfo {
                name,
                number_of_dimensions,
                dimensions,
                ggml_type,
                position: offset_bytes,
            });
        }

        self.align_reader_to(ALIGNMENT_BYTES)?;

        let tensor_data_start = self.position;

        for tensor in &mut tensors {
            tensor.position += tensor_data_start;
        }

        for tensor in tensors.iter() {
            println!(
                "tensor: name={}, dimensions={:?}, ggml_type={:?}, offset_bytes={} number of dimensions={}",
                tensor.name,
                tensor.dimensions,
                tensor.ggml_type,
                tensor.position,
                tensor.number_of_dimensions
            );
        }

        Ok(GgufHeader {
            version,
            tensor_count,
            metadata_count,
            metadata,
            tensors,
            tensor_data_start,
        })
    }

    pub fn read_tensor_bytes(
        &mut self,
        tensor: &TensorInfo,
        length_bytes: usize,
    ) -> Result<Vec<u8>> {
        self.seek_absolute(tensor.position)?;

        self.read_bytes(length_bytes)
    }

    pub fn read_tensor_into(&mut self, tensor: &TensorInfo, buffer: &mut [u8]) -> Result<()> {
        self.seek_absolute(tensor.position)?;

        self.inner.read_exact(buffer)?;

        self.position += buffer.len() as u64;

        Ok(())
    }

    pub fn read_f32_tensor(&mut self, header: &GgufHeader, tensor_name: &str) -> Result<Vec<f32>> {
        let tensor = header.tensor_by_name(tensor_name)?;

        if tensor.ggml_type != GgmlType::Float32 {
            bail!("expected float32 and not {}", tensor.ggml_type.as_u32());
        }

        let mut element_count: u64 = 1;

        for dimension in &tensor.dimensions {
            element_count = match element_count.checked_mul(*dimension) {
                Some(value) => value,
                None => bail!("too many float32 elements {dimension}"),
            };
        }

        let Some(total_bytes) = element_count.checked_mul(4) else {
            bail!("too many float32 elements {element_count}");
        };

        let Ok(byte_length) = usize::try_from(total_bytes) else {
            bail!("too many float32 tensor elements {total_bytes}");
        };

        let bytes = self.read_tensor_bytes(tensor, byte_length)?;

        let Ok(element_len) = usize::try_from(element_count) else {
            bail!("too many float32 tensor elements {element_count}");
        };

        let mut values = Vec::with_capacity(element_len);

        for chunk in bytes.chunks_exact(4) {
            let bits = [chunk[0], chunk[1], chunk[2], chunk[3]];
            values.push(f32::from_le_bytes(bits));
        }

        Ok(values)
    }

    pub fn read_quantized_tensor(
        &mut self,
        header: &GgufHeader,
        tensor_name: &str,
    ) -> Result<QuantizedTensor> {
        let tensor = header.tensor_by_name(tensor_name)?;

        if tensor.ggml_type != GgmlType::Quantization8_0 {
            bail!("expected q8 instead of {}", tensor.ggml_type.as_u32());
        }

        let mut element_count: u64 = 1;

        for dimension in &tensor.dimensions {
            element_count = match element_count.checked_mul(*dimension) {
                Some(value) => value,
                None => bail!("invalid number of q8 elements {dimension}"),
            };
        }

        if element_count % QUANTIZATION_BLOCK_GROUP_SIZE as u64 != 0 {
            bail!("invalid q8 tensor length {element_count}");
        }

        let block_count_u64 = element_count / QUANTIZATION_BLOCK_GROUP_SIZE as u64;
        let total_bytes = block_count_u64 * 34;

        let byte_length = usize::try_from(total_bytes)
            .map_err(|_| anyhow::anyhow!("invalid number of bytes {total_bytes}"))?;

        let bytes = self.read_tensor_bytes(tensor, byte_length)?;

        let block_count = usize::try_from(block_count_u64)
            .map_err(|_| anyhow::anyhow!("invalid block count {block_count_u64}"))?;

        let mut scales = vec![0.0f32; block_count];
        let mut values = vec![0i8; block_count * QUANTIZATION_BLOCK_GROUP_SIZE];

        let mut i = 0usize;

        for block_index in 0..block_count {
            let scale_bits = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
            let scale = Self::fp16_bits_to_f32(scale_bits);
            scales[block_index] = scale;

            i += 2;

            let out_offset = block_index * QUANTIZATION_BLOCK_GROUP_SIZE;

            for j in 0..QUANTIZATION_BLOCK_GROUP_SIZE {
                values[out_offset + j] = bytes[i] as i8;
                i += 1;
            }
        }

        Ok(QuantizedTensor { scales, values })
    }

    fn fp16_bits_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exponent = ((bits >> 10) & 0x1f) as u32;
        let fraction = (bits & 0x03ff) as u32;

        let out: u32 = if exponent == 0 {
            if fraction == 0 {
                sign << 31
            } else {
                let mut fraction = fraction;
                let mut exponent: i32 = -14;

                while (fraction & 0x0400) == 0 {
                    fraction <<= 1;

                    exponent -= 1;
                }

                fraction &= 0x03ff;

                let exponent32 = (exponent + 127) as u32;

                (sign << 31) | (exponent32 << 23) | (fraction << 13)
            }
        } else if exponent == 31 {
            (sign << 31) | (0xff << 23) | (fraction << 13)
        } else {
            let exp32 = (exponent as i32 - 15 + 127) as u32;

            (sign << 31) | (exp32 << 23) | (fraction << 13)
        };

        f32::from_bits(out)
    }

    pub fn f32_to_f16_bits(value: f32) -> u16 {
        let bits = value.to_bits();

        let sign = ((bits >> 31) & 1) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32;
        let fraction = bits & 0x7fffff;

        if exponent == 255 {
            if fraction == 0 {
                return (sign << 15) | (0x1f << 10);
            }

            let payload = (fraction >> 13) as u16;

            return (sign << 15) | (0x1f << 10) | (payload | 1);
        }

        let exponent16 = exponent - 127 + 15;

        if exponent16 <= 0 {
            if exponent16 < -10 {
                return sign << 15; // signed zero
            }

            let mantisse = fraction | 0x800000;

            let shift = 14 - exponent16; // 24 - 10 - exp16
            let mut mantisse16 = (mantisse >> shift) as u16;

            let round_bit = (mantisse >> (shift - 1)) & 1;

            let sticky = mantisse & ((1u32 << (shift - 1)) - 1);

            if round_bit == 1 && (sticky != 0 || (mantisse16 & 1) == 1) {
                mantisse16 = mantisse16.wrapping_add(1);
            }

            return (sign << 15) | mantisse16;
        }

        if exponent16 >= 31 {
            return (sign << 15) | (0x1f << 10);
        }

        let mut mantisse16 = (fraction >> 13) as u16;

        let round_bit = (fraction >> 12) & 1;

        let sticky = fraction & 0x0fff;

        if round_bit == 1 && (sticky != 0 || (mantisse16 & 1) == 1) {
            mantisse16 = mantisse16.wrapping_add(1);

            if mantisse16 == 0x0400 {
                mantisse16 = 0;

                let exponent16_u = (exponent16 + 1) as u16;

                if exponent16_u >= 31 {
                    return (sign << 15) | (0x1f << 10);
                }

                return (sign << 15) | (exponent16_u << 10) | mantisse16;
            }
        }

        (sign << 15) | ((exponent16 as u16) << 10) | mantisse16
    }
}
