//! tokie - Fast BPE tokenizer using Aho-Corasick automata
//!
//! This crate implements Byte Pair Encoding (BPE) tokenization using the
//! algorithm from GitHub's rust-gems, which uses Aho-Corasick automata for
//! efficient suffix matching combined with compatibility checking.
//!
//! # Quick Start
//!
//! ```ignore
//! use tokie::{Tokenizer, PretokenizerType};
//!
//! // Load from HuggingFace tokenizer.json
//! let tokenizer = Tokenizer::from_json("tokenizer.json", PretokenizerType::Gpt2)?;
//!
//! // Encode text
//! let tokens = tokenizer.encode("Hello, world!");
//!
//! // Decode back
//! let text = tokenizer.decode(&tokens).unwrap();
//!
//! // Save/load binary format (fast)
//! tokenizer.to_file("model.tkz")?;
//! let tokenizer = Tokenizer::from_file("model.tkz")?;
//! ```
//!
//! # Architecture
//!
//! - [`Tokenizer`] - High-level API combining pre-tokenization + BPE encoding + decoding
//! - [`BytePairEncoder`] - Low-level BPE encoder (bytes → tokens)
//! - [`Decoder`] - Token ID to bytes decoder (can be shared across encoder types)
//! - [`Pretokenizer`] - Regex-based text splitter (GPT-2, cl100k, o200k patterns)

mod bpe;
mod compatibility;
mod decoder;
pub mod hf;
mod pretokenizer;
mod serde;
mod tokenizer;
mod types;

pub use bpe::{BytePairEncoder, EncodeIter};
pub use decoder::Decoder;
pub use hf::JsonLoadError;
pub use pretokenizer::{Pretokenizer, PretokenizerIter, PretokenizerType};
pub use serde::SerdeError;
pub use tokenizer::{Tokenizer, TokenizeIter};
pub use types::TokenId;
