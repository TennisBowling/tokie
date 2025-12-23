//! High-level Tokenizer that combines pre-tokenization with BPE encoding.

use std::path::Path;
use std::thread;

use crate::bpe::{BytePairEncoder, EncodeIter};
use crate::decoder::Decoder;
use crate::hf::{self, JsonLoadError};
use crate::pretokenizer::{Pretokenizer, PretokenizerType};
use crate::types::TokenId;

/// High-level tokenizer combining pre-tokenization, BPE encoding, and decoding.
///
/// This is the main interface for tokenizing text. It handles:
/// 1. Pre-tokenization (splitting text using regex patterns)
/// 2. BPE encoding (converting each piece to token IDs)
/// 3. Decoding (converting token IDs back to text)
///
/// # Example
/// ```ignore
/// use tokie::Tokenizer;
///
/// let tokenizer = Tokenizer::from_json("tokenizer.json", PretokenizerType::Gpt2)?;
///
/// let tokens = tokenizer.encode("Hello, world!");
/// let text = tokenizer.decode(&tokens);
/// ```
pub struct Tokenizer {
    /// The underlying BPE encoder.
    encoder: BytePairEncoder,
    /// The decoder for converting tokens back to bytes.
    decoder: Decoder,
    /// Optional pre-tokenizer for splitting text before BPE.
    pretokenizer: Option<Pretokenizer>,
    /// The type of pretokenizer (for serialization).
    pretokenizer_type: PretokenizerType,
}

impl Tokenizer {
    /// Create a new tokenizer with an encoder, decoder, and pretokenizer type.
    pub fn new(
        encoder: BytePairEncoder,
        decoder: Decoder,
        pretokenizer_type: PretokenizerType,
    ) -> Self {
        let pretokenizer = pretokenizer_type.build();
        Self {
            encoder,
            decoder,
            pretokenizer,
            pretokenizer_type,
        }
    }

    /// Get the pretokenizer type.
    pub fn pretokenizer_type(&self) -> PretokenizerType {
        self.pretokenizer_type
    }

    /// Load a tokenizer from a HuggingFace tokenizer.json file.
    ///
    /// The pretokenizer type is auto-detected from the JSON.
    ///
    /// # Example
    /// ```ignore
    /// use tokie::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_json("tokenizer.json")?;
    /// let tokens = tokenizer.encode("Hello, world!");
    /// ```
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, JsonLoadError> {
        hf::from_json(path)
    }

    /// Get a reference to the underlying BPE encoder.
    pub fn encoder(&self) -> &BytePairEncoder {
        &self.encoder
    }

    /// Get a reference to the decoder.
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// Get a reference to the pre-tokenizer, if any.
    pub fn pretokenizer(&self) -> Option<&Pretokenizer> {
        self.pretokenizer.as_ref()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.decoder.vocab_size()
    }

    /// Minimum pieces to trigger parallel encoding.
    const PARALLEL_ENCODE_THRESHOLD: usize = 1000;

    /// Encode text into token IDs.
    ///
    /// If a pre-tokenizer is configured, the text is first split into pieces,
    /// then each piece is BPE-encoded (in parallel for large texts).
    ///
    /// # Example
    /// ```ignore
    /// let tokens = tokenizer.encode("Hello, world!");
    /// ```
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        match &self.pretokenizer {
            Some(pre) => {
                let pieces: Vec<&str> = pre.split(text).collect();

                if pieces.len() >= Self::PARALLEL_ENCODE_THRESHOLD {
                    self.encode_pretokens(&pieces)
                } else {
                    pieces
                        .iter()
                        .flat_map(|piece| self.encoder.encode(piece.as_bytes()))
                        .collect()
                }
            }
            None => self.encoder.encode(text.as_bytes()),
        }
    }

    /// Encode pretokenized pieces in parallel.
    fn encode_pretokens(&self, pieces: &[&str]) -> Vec<TokenId> {
        let num_cpus = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        let chunk_size = (pieces.len() + num_cpus - 1) / num_cpus;

        let results: Vec<Vec<TokenId>> = thread::scope(|s| {
            let handles: Vec<_> = pieces
                .chunks(chunk_size)
                .map(|chunk| {
                    let encoder = &self.encoder;
                    s.spawn(move || {
                        let mut tokens = Vec::new();
                        for piece in chunk {
                            tokens.extend(encoder.encode(piece.as_bytes()));
                        }
                        tokens
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Flatten results
        let total: usize = results.iter().map(|v| v.len()).sum();
        let mut output = Vec::with_capacity(total);
        for chunk in results {
            output.extend(chunk);
        }
        output
    }

    /// Encode bytes directly without pre-tokenization.
    ///
    /// This bypasses the pre-tokenizer and encodes raw bytes.
    /// Useful when you've already done your own text processing.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.encoder.encode(bytes)
    }

    /// Returns a streaming iterator over encoded tokens.
    ///
    /// Note: When a pre-tokenizer is configured, this currently collects
    /// all tokens (not truly streaming across pre-token boundaries).
    /// For true streaming without pre-tokenization, use `encode_bytes_iter`.
    pub fn encode_iter<'a>(&'a self, text: &'a str) -> TokenizeIter<'a> {
        TokenizeIter::new(self, text)
    }

    /// Returns a streaming iterator over encoded tokens from bytes.
    ///
    /// Bypasses pre-tokenization for true streaming.
    pub fn encode_bytes_iter<'a>(&'a self, bytes: &'a [u8]) -> EncodeIter<'a> {
        self.encoder.encode_iter(bytes)
    }

    /// Decode token IDs back to a string.
    ///
    /// Returns `None` if the decoded bytes are not valid UTF-8.
    pub fn decode(&self, tokens: &[TokenId]) -> Option<String> {
        self.decoder.decode_to_string(tokens)
    }

    /// Decode token IDs back to bytes.
    pub fn decode_bytes(&self, tokens: &[TokenId]) -> Vec<u8> {
        self.decoder.decode(tokens)
    }

    /// Get the byte sequence for a token.
    pub fn token_to_bytes(&self, token: TokenId) -> &[u8] {
        self.decoder.token_to_bytes(token)
    }

    /// Count tokens without storing them.
    ///
    /// More memory-efficient than `encode().len()` for large texts.
    pub fn count_tokens(&self, text: &str) -> usize {
        match &self.pretokenizer {
            Some(pre) => pre
                .split(text)
                .map(|piece| self.encoder.encode(piece.as_bytes()).len())
                .sum(),
            None => self.encoder.encode(text.as_bytes()).len(),
        }
    }

    /// Check if text exceeds a token limit.
    ///
    /// More efficient than `count_tokens() > limit` as it can stop early.
    pub fn exceeds_token_limit(&self, text: &str, limit: usize) -> bool {
        match &self.pretokenizer {
            Some(pre) => {
                let mut count = 0;
                for piece in pre.split(text) {
                    count += self.encoder.encode(piece.as_bytes()).len();
                    if count > limit {
                        return true;
                    }
                }
                false
            }
            None => self.encoder.encode_iter(text.as_bytes()).take(limit + 1).count() > limit,
        }
    }
}

/// Iterator over tokens from the high-level Tokenizer.
pub struct TokenizeIter<'a> {
    tokenizer: &'a Tokenizer,
    // Current state for pretokenized case
    pretokens: Option<Box<dyn Iterator<Item = &'a str> + 'a>>,
    current_encoder_iter: Option<EncodeIter<'a>>,
    // For non-pretokenized case
    bytes_iter: Option<EncodeIter<'a>>,
}

impl<'a> TokenizeIter<'a> {
    fn new(tokenizer: &'a Tokenizer, text: &'a str) -> Self {
        if tokenizer.pretokenizer.is_some() {
            let pretokens = tokenizer.pretokenizer.as_ref().unwrap().split(text);
            Self {
                tokenizer,
                pretokens: Some(Box::new(pretokens)),
                current_encoder_iter: None,
                bytes_iter: None,
            }
        } else {
            Self {
                tokenizer,
                pretokens: None,
                current_encoder_iter: None,
                bytes_iter: Some(tokenizer.encoder.encode_iter(text.as_bytes())),
            }
        }
    }
}

impl<'a> Iterator for TokenizeIter<'a> {
    type Item = TokenId;

    fn next(&mut self) -> Option<TokenId> {
        // Non-pretokenized case: just use the bytes iterator
        if let Some(ref mut iter) = self.bytes_iter {
            return iter.next();
        }

        // Pretokenized case: iterate through pretokens
        loop {
            // Try to get next token from current encoder iterator
            if let Some(ref mut encoder_iter) = self.current_encoder_iter {
                if let Some(token) = encoder_iter.next() {
                    return Some(token);
                }
            }

            // Current encoder exhausted, get next pretoken
            if let Some(ref mut pretokens) = self.pretokens {
                if let Some(piece) = pretokens.next() {
                    self.current_encoder_iter =
                        Some(self.tokenizer.encoder.encode_iter(piece.as_bytes()));
                    continue;
                }
            }

            // No more pretokens
            return None;
        }
    }
}

impl std::iter::FusedIterator for TokenizeIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> Tokenizer {
        // Simple encoder: a=0, b=1, c=2, space=3
        // Merges: a+b->ab(256)
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges = vec![(b'a' as u32, b'b' as u32)]; // a+b -> ab
        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(encoder, decoder, PretokenizerType::None)
    }

    fn make_test_tokenizer_with_pretokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges = vec![(b'a' as u32, b'b' as u32)]; // a+b -> ab
        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(encoder, decoder, PretokenizerType::Gpt2)
    }

    #[test]
    fn test_tokenizer_no_pretokenizer() {
        let tokenizer = make_test_tokenizer();

        let tokens = tokenizer.encode("abc");
        // 'a'+'b' merges to token 256, then 'c' is token 99
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_tokenizer_with_pretokenizer() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();

        // "Hello world" gets pre-tokenized to ["Hello", " world"]
        let tokens = tokenizer.encode("Hello world");
        assert!(!tokens.is_empty());

        // Verify decode roundtrip
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, "Hello world");
    }

    #[test]
    fn test_count_tokens() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();

        let text = "Hello world";
        let count = tokenizer.count_tokens(text);
        let tokens = tokenizer.encode(text);
        assert_eq!(count, tokens.len());
    }

    #[test]
    fn test_exceeds_token_limit() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();

        let text = "Hello world test";
        let total = tokenizer.count_tokens(text);

        assert!(!tokenizer.exceeds_token_limit(text, total));
        assert!(!tokenizer.exceeds_token_limit(text, total + 10));
        assert!(tokenizer.exceeds_token_limit(text, total - 1));
        assert!(tokenizer.exceeds_token_limit(text, 0));
    }

    #[test]
    fn test_encode_iter() {
        let tokenizer = make_test_tokenizer_with_pretokenizer();

        let text = "Hello world";
        let tokens: Vec<_> = tokenizer.encode_iter(text).collect();
        let expected = tokenizer.encode(text);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_decode_bytes() {
        let tokenizer = make_test_tokenizer();

        let text = b"abc";
        let tokens = tokenizer.encode_bytes(text);
        let decoded = tokenizer.decode_bytes(&tokens);
        assert_eq!(decoded, text);
    }
}
