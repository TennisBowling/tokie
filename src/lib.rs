//! tokie - Fast BPE tokenizer using Aho-Corasick automata
//!
//! This crate implements Byte Pair Encoding (BPE) tokenization using the
//! algorithm from GitHub's rust-gems, which uses Aho-Corasick automata for
//! efficient suffix matching combined with compatibility checking.
//!
//! # Quick Start
//!
//! ```ignore
//! use tokie::{Tokenizer, PretokType};
//!
//! // Load from HuggingFace tokenizer.json
//! let tokenizer = Tokenizer::from_json("tokenizer.json")?;
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
//! - [`pretok`] - Fast pretokenizers (GPT-2: 566 MiB/s, cl100k, o200k)

mod bpe;
mod compatibility;
mod decoder;
pub mod hf;
pub mod pretok;
mod serde;
mod tokenizer;
mod types;

pub use bpe::{BytePairEncoder, EncodeIter};
pub use decoder::Decoder;
pub use hf::JsonLoadError;
pub use pretok::{DynPretok, DynPretokIter, Gpt2Pretok, Pretok, PretokType, RegexPretok};
pub use serde::SerdeError;
pub use tokenizer::{TokenCount, Tokenizer, TokenizeIter};
pub use types::TokenId;

// Backward compatibility aliases
#[doc(hidden)]
#[deprecated(since = "0.2.0", note = "Use PretokType instead")]
pub type PretokenizerType = PretokType;
