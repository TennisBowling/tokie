//! Fast pre-tokenization for BPE tokenizers.
//!
//! This module provides hand-coded lexers for common pretokenization patterns,
//! achieving ~4x speedup over regex-based approaches while maintaining full
//! Unicode support.
//!
//! # Architecture
//!
//! Each pretokenizer type has an optimized hand-coded lexer:
//! - [`Gpt2Pretok`] - GPT-2, GPT-J, GPT-Neo (566 MiB/s)
//! - [`RegexPretok`] - Fallback for custom patterns (146 MiB/s)
//!
//! # Example
//!
//! ```
//! use tokie::pretok::{Pretok, Gpt2Pretok};
//!
//! let pretok = Gpt2Pretok::new();
//! let pieces: Vec<&str> = pretok.split("Hello world").collect();
//! assert_eq!(pieces, vec!["Hello", " world"]);
//! ```

mod gpt2;
mod regex;

pub use gpt2::{Gpt2Pretok, Gpt2PretokIter};
pub use regex::RegexPretok;

/// Type of pretokenizer for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PretokType {
    None = 0,
    Gpt2 = 1,
    Cl100k = 2,
    O200k = 3,
}

/// Pretokenizer trait - splits text into pieces before BPE encoding.
pub trait Pretok {
    /// Iterator type returned by `split`.
    type Iter<'a>: Iterator<Item = &'a str>
    where
        Self: 'a;

    /// Split text into pre-tokens.
    fn split<'a>(&'a self, text: &'a str) -> Self::Iter<'a>;

    /// Split text and collect into a Vec.
    fn split_to_vec<'a>(&'a self, text: &'a str) -> Vec<&'a str> {
        self.split(text).collect()
    }
}

/// Enum wrapper for dynamic dispatch when pretokenizer type is runtime-determined.
pub enum DynPretok {
    Gpt2(Gpt2Pretok),
    Cl100k(RegexPretok),
    O200k(RegexPretok),
}

impl DynPretok {
    /// Create a pretokenizer from type.
    pub fn from_type(typ: PretokType) -> Option<Self> {
        match typ {
            PretokType::None => None,
            PretokType::Gpt2 => Some(DynPretok::Gpt2(Gpt2Pretok::new())),
            PretokType::Cl100k => Some(DynPretok::Cl100k(RegexPretok::cl100k())),
            PretokType::O200k => Some(DynPretok::O200k(RegexPretok::o200k())),
        }
    }

    /// Split text into pre-tokens.
    pub fn split<'a>(&'a self, text: &'a str) -> DynPretokIter<'a> {
        match self {
            DynPretok::Gpt2(p) => DynPretokIter::Gpt2(p.split(text)),
            DynPretok::Cl100k(p) => DynPretokIter::Regex(p.split(text)),
            DynPretok::O200k(p) => DynPretokIter::Regex(p.split(text)),
        }
    }

    /// Split text and collect into a Vec.
    pub fn split_to_vec<'a>(&'a self, text: &'a str) -> Vec<&'a str> {
        self.split(text).collect()
    }
}

/// Iterator for dynamic pretokenizer dispatch.
pub enum DynPretokIter<'a> {
    Gpt2(Gpt2PretokIter<'a>),
    Regex(regex::RegexPretokIter<'a>),
}

impl<'a> Iterator for DynPretokIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DynPretokIter::Gpt2(iter) => iter.next(),
            DynPretokIter::Regex(iter) => iter.next(),
        }
    }
}
