//! Fast pretokenizers for BPE tokenizers.
//!
//! Each pretokenizer is a zero-allocation, single-pass iterator over text pieces.
//!
//! # Example
//!
//! ```
//! use pretokie::Gpt2;
//!
//! let pieces: Vec<&str> = Gpt2::new("Hello world").collect();
//! assert_eq!(pieces, vec!["Hello", " world"]);
//! ```

mod impls;
pub mod util;

pub use impls::gpt2::Gpt2;
pub use impls::cl100k::Cl100k;
pub use impls::bert::Bert;
pub use impls::o200k::O200k;
pub use impls::voyage::Voyage;
pub use impls::smollm::SmolLM;
pub use impls::deepseek::DeepSeek;
pub use impls::qwen::Qwen;
#[cfg(feature = "regex")]
pub mod regex {
    //! Regex-based pretokenizer (requires `regex` feature).
    pub use crate::impls::regex::{Regex, RegexIter};
}
#[cfg(feature = "regex")]
pub use impls::regex::Regex;
