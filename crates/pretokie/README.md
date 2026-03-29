<div align="center">

![tokie](https://raw.githubusercontent.com/chonkie-inc/tokie/main/assets/tokie.png)

# pretokie

[![Crates.io](https://img.shields.io/crates/v/pretokie)](https://crates.io/crates/pretokie)
[![Crates.io Downloads](https://img.shields.io/crates/d/pretokie)](https://crates.io/crates/pretokie)
[![docs.rs](https://img.shields.io/docsrs/pretokie)](https://docs.rs/pretokie)
[![License](https://img.shields.io/crates/l/pretokie)](LICENSE-MIT)
[![GitHub Stars](https://img.shields.io/github/stars/chonkie-inc/tokie)](https://github.com/chonkie-inc/tokie)

*Fast, zero-allocation pretokenizers for every major tokenizer — 4x faster than regex*

[Quick Start](#quick-start) •
[Pretokenizers](#pretokenizers) •
[Benchmarks](#benchmarks) •
[Regex Fallback](#regex-fallback) •
[Why Hand-Coded?](#why-hand-coded)

</div>

**pretokie** splits text into pieces before BPE/WordPiece/Unigram encoding. Each pretokenizer is a hand-coded, single-pass iterator — no regex, no allocation, just raw byte-level dispatch at 400 MB/s.

Part of the [tokie](https://github.com/chonkie-inc/tokie) tokenizer project.

## Quick Start

```toml
[dependencies]
pretokie = "0.0.2"
```

```rust
use pretokie::Gpt2;

let pieces: Vec<&str> = Gpt2::new("Hello world! It's a test.").collect();
assert_eq!(pieces, vec!["Hello", " world", "!", " It", "'s", " a", " test", "."]);
```

Every pretokenizer implements `Iterator<Item = &str>` — use `.collect()`, `.count()`, `.for_each()`, or any iterator combinator.

## Pretokenizers

| Name | Models | MB/s | Pieces* | cyc/B |
|------|--------|------|---------|-------|
| `Gpt2` | GPT-2, GPT-J, RoBERTa | **403** | 24.5M | 8.3 |
| `Cl100k` | GPT-3.5, GPT-4, Llama 3 | **407** | 23.9M | 8.2 |
| `O200k` | GPT-4o | **371** | 23.9M | 9.0 |
| `Bert` | BERT, DistilBERT, GTE, BGE, MiniLM | **387** | 26.1M | 8.6 |
| `Voyage` | Voyage 3, Voyage Code 3 | **403** | 25.0M | 8.3 |
| `SmolLM` | SmolLM2 | **398** | 26.2M | 8.4 |
| `DeepSeek` | DeepSeek-V3, DeepSeek-R1 | **399** | 23.9M | 8.4 |
| `Qwen` | Qwen3.5 | **384** | 25.0M | 8.7 |
| `Regex` | Any pattern (fallback) | 91 | 23.3M | 36.7 |

\* Pieces on 95 MB enwik8, Apple M3 Pro. Cycles/byte at 3.5 GHz.

## Benchmarks

All pretokenizers run at **370-407 MB/s** — 4x faster than the regex fallback at 91 MB/s. The fastest (CL100K at 407 MB/s) processes 95 MB of English text in 234ms, yielding 23.9 million pieces.

For comparison, HuggingFace tokenizers' regex-based pretokenizer runs at ~100 MB/s. pretokie's hand-coded iterators eliminate regex overhead entirely.

## Usage

```rust
use pretokie::{Cl100k, O200k, Bert};

// CL100K: case-insensitive contractions, 3-digit number chunks
let pieces: Vec<&str> = Cl100k::new("DON'T count 12345").collect();
assert_eq!(pieces, vec!["DON", "'T", " count", " ", "123", "45"]);

// O200K: CamelCase splitting, contractions merge into words
let pieces: Vec<&str> = O200k::new("XMLHttpRequest don't").collect();
assert_eq!(pieces, vec!["XMLHttp", "Request", " don't"]);

// BERT: whitespace-delimited, individual punctuation
let pieces: Vec<&str> = Bert::new("Hello, world!").collect();
assert_eq!(pieces, vec!["Hello", ",", "world", "!"]);
```

## Regex Fallback

For unknown tokenizer patterns, enable the `regex` feature:

```toml
[dependencies]
pretokie = { version = "0.0.2", features = ["regex"] }
```

```rust
use pretokie::Regex;

// Use built-in factories
let pretok = Regex::gpt2();
let pieces: Vec<&str> = pretok.split("Hello world").collect();

// Or compile a custom pattern
let pretok = Regex::new(&[
    (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{L}+|\p{N}+", false),
]).unwrap();
```

Without the `regex` feature, pretokie has only one dependency (`unicode-general-category` for BERT punctuation classification).

## Why Hand-Coded?

Regex-based pretokenizers run at ~91 MB/s. The hand-coded iterators run at ~400 MB/s — **4x faster** — because they eliminate:

- **Regex compilation** — no NFA/DFA construction at startup
- **Branch overhead** — specialized byte-level dispatch instead of generic regex engine
- **Allocation** — zero heap allocation per piece (iterators borrow from the input)

The pretokenizer runs before every encode call, on every piece of text. At 25 million pieces per 95 MB, even small per-piece overhead compounds. These iterators process each byte with a single `if`-chain dispatch, yielding `&str` slices directly from the input.

## License

MIT OR Apache-2.0
