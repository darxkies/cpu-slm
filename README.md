# CPU-SLM

A holiday project to better understand the inner workings of SLM/LLM.

## Features

- Written in Rust
- Uses only CPU
- AVX2/SIMD accelerated
- Loads GGUF (only Q8_0 supported)
- About 10-15% slower than llama.cpp (10+ tokens/s for 3B/4B models).
- Few dependencies (anyhow and rayon)

# Supported Models

- Mistral 3 3B
- Llama 3.2 3B
- Qwen 3 4B & 0.6B
- Nanbeige 4 3B

# Launch

Download one of the supported models:

```sh
wget https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512-GGUF/resolve/main/Ministral-3-3B-Reasoning-2512-Q8_0.gguf
```

## Inference

```sh
RAYON_NUM_THREADS=16 RUSTFLAGS="-C target-cpu=native" cargo run --release -- inference --model-path Ministral-3-3B-Reasoning-2512-Q8_0.gguf --user-prompt "Did a UFO crash in Roswell?"
```

## Chat

```sh
RAYON_NUM_THREADS=16 RUSTFLAGS="-C target-cpu=native" cargo run --release -- chat --model-path Ministral-3-3B-Reasoning-2512-Q8_0.gguf
```
