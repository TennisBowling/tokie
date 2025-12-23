//! Pre-tokenization using the rust-gems regex approach.
//!
//! This module implements fast pre-tokenization by rewriting negative lookahead
//! patterns to work with standard regex-automata. This avoids the need for
//! fancy-regex or onig while achieving ~5x better performance.
//!
//! # How it works
//!
//! Patterns like `\s+(?!\S)` (whitespace not followed by non-whitespace) are
//! rewritten as multiple patterns:
//! - `\s+$` - whitespace at end of string
//! - `\s+\s` - whitespace followed by whitespace (drop last char)
//! - `\s+` - whitespace followed by non-whitespace
//!
//! The pretokenizer uses anchored matching and pattern priority to achieve
//! the same semantics as the original lookahead pattern.

use regex_automata::{
    meta::Regex,
    util::captures::Captures,
    Anchored, Input,
};

/// Type of pretokenizer for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PretokenizerType {
    None = 0,
    Gpt2 = 1,
    Cl100k = 2,
    O200k = 3,
}

impl PretokenizerType {
    /// Create a Pretokenizer from this type.
    pub fn build(self) -> Option<Pretokenizer> {
        match self {
            Self::None => None,
            Self::Gpt2 => Some(Pretokenizer::gpt2()),
            Self::Cl100k => Some(Pretokenizer::cl100k()),
            Self::O200k => Some(Pretokenizer::o200k()),
        }
    }
}

/// Pre-tokenizer that splits text before BPE encoding.
///
/// This is required for HuggingFace-compatible tokenization, as models like
/// GPT-2, cl100k (GPT-3.5/4), and o200k use pre-tokenization regex patterns.
pub struct Pretokenizer {
    regex: Regex,
    lookahead: Vec<bool>,
}

impl Pretokenizer {
    /// Create a new pretokenizer from patterns.
    ///
    /// Each pattern is a tuple of (pattern_str, is_lookahead).
    /// If is_lookahead is true, the last character of matches will be dropped.
    ///
    /// # Example
    /// ```
    /// use tokie::Pretokenizer;
    ///
    /// // GPT-2 style pretokenizer
    /// let pre = Pretokenizer::new(&[
    ///     (r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$", false),
    ///     (r"\s+\s", true),  // lookahead rewrite
    ///     (r"\s+", false),
    /// ]).unwrap();
    /// ```
    pub fn new(patterns: &[(&str, bool)]) -> Result<Self, regex_automata::meta::BuildError> {
        let pats: Vec<&str> = patterns.iter().map(|(p, _)| *p).collect();
        let lookahead: Vec<bool> = patterns.iter().map(|(_, l)| *l).collect();
        let regex = Regex::new_many(&pats)?;
        Ok(Self { regex, lookahead })
    }

    /// Create a GPT-2 compatible pretokenizer.
    ///
    /// Original pattern: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
    pub fn gpt2() -> Self {
        Self::new(&[
            (r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$", false),
            (r"\s+\s", true),
            (r"\s+", false),
        ]).expect("valid GPT-2 pattern")
    }

    /// Create a cl100k (GPT-3.5/GPT-4) compatible pretokenizer.
    ///
    /// Original pattern includes case-insensitive contractions and number chunking.
    pub fn cl100k() -> Self {
        Self::new(&[
            (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+$", false),
            (r"\s+\s", true),
            (r"\s+", false),
        ]).expect("valid cl100k pattern")
    }

    /// Create an o200k (GPT-4o) compatible pretokenizer.
    pub fn o200k() -> Self {
        let pat1 = [
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"\p{N}{1,3}",
            r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
            r"\s*[\r\n]+",
            r"\s+$",
        ].join("|");

        Self::new(&[
            (&pat1, false),
            (r"\s+\s", true),
            (r"\s+", false),
        ]).expect("valid o200k pattern")
    }

    /// Split text into pre-tokens.
    ///
    /// Returns an iterator over string slices of the original text.
    pub fn split<'a>(&'a self, text: &'a str) -> PretokenizerIter<'a> {
        PretokenizerIter {
            pretokenizer: self,
            text,
            pos: 0,
            caps: Captures::matches(self.regex.group_info().clone()),
        }
    }

    /// Split text and collect into a Vec.
    pub fn split_to_vec<'a>(&'a self, text: &'a str) -> Vec<&'a str> {
        self.split(text).collect()
    }
}

/// Iterator over pre-tokens.
pub struct PretokenizerIter<'a> {
    pretokenizer: &'a Pretokenizer,
    text: &'a str,
    pos: usize,
    caps: Captures,
}

impl<'a> Iterator for PretokenizerIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.text.len() {
            return None;
        }

        let input = Input::new(&self.text[self.pos..]).anchored(Anchored::Yes);
        self.caps.clear();
        self.pretokenizer.regex.captures(input, &mut self.caps);

        let m = self.caps.get_match()?;
        let start = self.pos;
        let mut end = self.pos + m.range().end;

        // If this is a lookahead pattern, drop the last character
        if self.pretokenizer.lookahead[m.pattern().as_usize()] {
            if let Some(last_char) = self.text[start..end].chars().next_back() {
                end -= last_char.len_utf8();
            }
        }

        // Safety: ensure we make progress
        if end <= start {
            self.pos = start + 1;
            return self.next();
        }

        self.pos = end;
        Some(&self.text[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_basic() {
        let pre = Pretokenizer::gpt2();

        let tokens = pre.split_to_vec("Hello world");
        assert_eq!(tokens, vec!["Hello", " world"]);
    }

    #[test]
    fn test_gpt2_multiple_spaces() {
        let pre = Pretokenizer::gpt2();

        // Multiple spaces between words
        let tokens = pre.split_to_vec("Hello  world");
        assert_eq!(tokens, vec!["Hello", " ", " world"]);

        let tokens = pre.split_to_vec("Hello   world");
        assert_eq!(tokens, vec!["Hello", "  ", " world"]);
    }

    #[test]
    fn test_gpt2_trailing_spaces() {
        let pre = Pretokenizer::gpt2();

        let tokens = pre.split_to_vec("test  ");
        assert_eq!(tokens, vec!["test", "  "]);
    }

    #[test]
    fn test_gpt2_leading_spaces() {
        let pre = Pretokenizer::gpt2();

        let tokens = pre.split_to_vec("  test");
        assert_eq!(tokens, vec![" ", " test"]);
    }

    #[test]
    fn test_gpt2_contractions() {
        let pre = Pretokenizer::gpt2();

        let tokens = pre.split_to_vec("How's it going?");
        assert_eq!(tokens, vec!["How", "'s", " it", " going", "?"]);
    }

    #[test]
    fn test_gpt2_punctuation() {
        let pre = Pretokenizer::gpt2();

        let tokens = pre.split_to_vec("Hello, world!");
        assert_eq!(tokens, vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn test_gpt2_newlines() {
        let pre = Pretokenizer::gpt2();

        // Individual newlines between words (each \n is separate when followed by non-ws)
        let tokens = pre.split_to_vec("Hello\n\nworld");
        assert_eq!(tokens, vec!["Hello", "\n", "\n", "world"]);

        // Trailing newlines are grouped together
        let tokens = pre.split_to_vec("test\n\n\n");
        assert_eq!(tokens, vec!["test", "\n\n\n"]);
    }

    #[test]
    fn test_gpt2_numbers() {
        let pre = Pretokenizer::gpt2();

        let tokens = pre.split_to_vec("test 123 hello");
        assert_eq!(tokens, vec!["test", " 123", " hello"]);
    }

    #[test]
    fn test_cl100k_numbers() {
        let pre = Pretokenizer::cl100k();

        // cl100k chunks numbers into groups of 1-3 digits
        let tokens = pre.split_to_vec("12345");
        assert_eq!(tokens, vec!["123", "45"]);
    }

    #[test]
    fn test_iterator() {
        let pre = Pretokenizer::gpt2();

        let mut iter = pre.split("a b c");
        assert_eq!(iter.next(), Some("a"));
        assert_eq!(iter.next(), Some(" b"));
        assert_eq!(iter.next(), Some(" c"));
        assert_eq!(iter.next(), None);
    }
}
