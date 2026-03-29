<div align="center">

# pretokie

[![Crates.io](https://img.shields.io/crates/v/pretokie)](https://crates.io/crates/pretokie)
[![docs.rs](https://img.shields.io/docsrs/pretokie)](https://docs.rs/pretokie)
[![License](https://img.shields.io/crates/l/pretokie)](LICENSE-MIT)

*Fast, zero-allocation pretokenizers for every major tokenizer*

</div>

**pretokie** splits text into pieces before BPE/WordPiece/Unigram encoding. Each pretokenizer is a hand-coded, single-pass iterator — no regex, no allocation, no dependencies.

Part of the [tokie](https://github.com/chonkie-inc/tokie) project.

## Quick Start

```rust
use pretokie::Gpt2;

let pieces: Vec<&str> = Gpt2::new("Hello world! It's a test.").collect();
assert_eq!(pieces, vec!["Hello", " world", "!", " It", "'s", " a", " test", "."]);
```

Every pretokenizer implements `Iterator<Item = &str>` — use `.collect()`, `.count()`, `.for_each()`, or any iterator combinator.

## Pretokenizers

| Name | Models | Pattern | MB/s |
|------|--------|---------|------|
| `Gpt2` | GPT-2, GPT-J, RoBERTa | `'s\|'t\|...\| ?\p{L}+\| ?\p{N}+\|...` | ~370 |
| `Cl100k` | GPT-3.5, GPT-4, Llama 3 | Case-insensitive contractions, 3-digit chunks | ~380 |
| `O200k` | GPT-4o | CamelCase splitting, suffix contractions | ~340 |
| `Bert` | BERT, DistilBERT, GTE, BGE, MiniLM | Whitespace-delimited, individual punctuation | ~345 |
| `Voyage` | Voyage 3, Voyage Code 3 | Like CL100K, single-digit numbers | ~360 |
| `SmolLM` | SmolLM2 | Like GPT-2, single-digit isolation | ~365 |
| `DeepSeek` | DeepSeek-V3, DeepSeek-R1 | Like CL100K, marks stay with letters | ~370 |
| `Qwen` | Qwen3.5 | Like Voyage, marks stay with letters | ~370 |
| `Regex` | Any pattern (fallback) | `regex-automata` multi-pattern | ~88 |

Throughput measured on 95 MB of enwik8 on Apple Silicon. Hand-coded pretokenizers are **4x faster** than the regex fallback.

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
pretokie = { version = "0.0.1", features = ["regex"] }
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

Without the `regex` feature, pretokie has **zero dependencies**.

## Why Hand-Coded?

Regex-based pretokenizers run at ~88 MB/s. The hand-coded iterators run at ~370 MB/s — **4x faster** — because they eliminate:

- **Regex compilation** — no NFA/DFA construction
- **Branch mispredictions** — specialized byte-level dispatch instead of generic regex engine
- **Allocation** — zero heap allocation per piece (iterators borrow from the input)

The pretokenizer runs before every encode call, on every piece of text. At 25 million pieces per 95 MB, even small per-piece overhead adds up. These iterators process each byte with a single `if`-chain dispatch, yielding `&str` slices directly from the input.

## License

MIT OR Apache-2.0
