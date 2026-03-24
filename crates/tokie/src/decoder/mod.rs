//! Decoders for converting token IDs back to text.
//!
//! - [`Decoder`] — high-level decoder combining vocab lookup with text post-processing
//! - [`VocabDecoder`] — maps token IDs to raw byte sequences (flat buffer, O(1) lookup)
//! - [`DecoderType`] — text-level post-processing strategy (WordPiece `##` stripping, metaspace `▁` → space, etc.)

mod vocab;

pub use vocab::VocabDecoder;

use crate::encoder::EncoderType;
use crate::postprocessor::PostProcessor;
use crate::types::TokenId;

/// Text-level decoder type, mirroring [`EncoderType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum DecoderType {
    /// Direct byte concatenation. Token bytes are already correct text.
    /// Used by byte-level BPE (GPT-2, Llama, Mistral, Phi, Qwen).
    #[default]
    ByteLevel = 0,

    /// WordPiece decoding (BERT, MiniLM, BGE, GTE).
    /// Strips `##` continuation prefixes and joins tokens with spaces.
    WordPiece = 1,

    /// Metaspace decoding (T5, XLM-R, Gemma, SentencePiece models).
    /// Replaces `▁` (U+2581) with spaces and strips the leading space.
    Metaspace = 2,
}

impl DecoderType {
    /// Infer the decoder type from the encoder type.
    pub fn from_encoder_type(encoder_type: EncoderType) -> Self {
        match encoder_type {
            EncoderType::WordPiece => DecoderType::WordPiece,
            EncoderType::SentencePiece | EncoderType::Unigram => DecoderType::Metaspace,
            _ => DecoderType::ByteLevel,
        }
    }

    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::ByteLevel),
            1 => Some(Self::WordPiece),
            2 => Some(Self::Metaspace),
            _ => None,
        }
    }
}

/// High-level decoder wrapping [`VocabDecoder`] (byte lookup) and [`DecoderType`] (text post-processing).
///
/// This is the primary decoder type used by [`Tokenizer`]. It provides:
/// - `decode()` — token IDs → String with text-level post-processing
/// - `decode_bytes()` — token IDs → raw bytes (no post-processing)
/// - `token_to_bytes()` — single token → byte slice
///
/// [`Tokenizer`]: crate::Tokenizer
#[derive(Clone)]
pub struct Decoder {
    vocab: VocabDecoder,
    decoder_type: DecoderType,
}

impl Decoder {
    /// Create a new decoder from token byte sequences.
    ///
    /// Decoder type defaults to [`DecoderType::ByteLevel`].
    pub fn new(token_bytes: Vec<Vec<u8>>) -> Self {
        Self {
            vocab: VocabDecoder::new(token_bytes),
            decoder_type: DecoderType::ByteLevel,
        }
    }

    /// Create a new decoder, inferring the decoder type from the encoder type.
    pub fn for_encoder(token_bytes: Vec<Vec<u8>>, encoder_type: EncoderType) -> Self {
        Self {
            vocab: VocabDecoder::new(token_bytes),
            decoder_type: DecoderType::from_encoder_type(encoder_type),
        }
    }

    /// Create a decoder with a specific decoder type.
    pub fn with_type(vocab: VocabDecoder, decoder_type: DecoderType) -> Self {
        Self { vocab, decoder_type }
    }

    /// Get the decoder type.
    pub fn decoder_type(&self) -> DecoderType {
        self.decoder_type
    }

    /// Get a reference to the underlying vocab decoder.
    pub fn vocab(&self) -> &VocabDecoder {
        &self.vocab
    }

    /// Consume self and return the inner VocabDecoder.
    pub fn into_vocab(self) -> VocabDecoder {
        self.vocab
    }

    /// Get the vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab.vocab_size()
    }

    /// Get the byte sequence for a token.
    #[inline]
    pub fn token_to_bytes(&self, token: TokenId) -> &[u8] {
        self.vocab.token_to_bytes(token)
    }

    /// Get the length of a token in bytes.
    #[inline]
    pub fn token_len(&self, token: TokenId) -> usize {
        self.vocab.token_len(token)
    }

    /// Decode token IDs back to raw bytes (no text post-processing).
    pub fn decode_bytes(&self, tokens: &[TokenId]) -> Vec<u8> {
        self.vocab.decode(tokens)
    }

    /// Decode token IDs to a string with text-level post-processing.
    ///
    /// Behavior depends on [`DecoderType`]:
    /// - **ByteLevel**: Direct byte concatenation (already correct)
    /// - **WordPiece**: Strips `##` prefixes, joins with spaces, skips special tokens
    /// - **Metaspace**: Replaces `▁` with spaces, strips leading space, skips special tokens
    ///
    /// Returns `None` if the result is not valid UTF-8.
    pub fn decode(&self, tokens: &[TokenId], post_processor: &PostProcessor) -> Option<String> {
        match self.decoder_type {
            DecoderType::ByteLevel => self.vocab.decode_to_string(tokens),
            DecoderType::WordPiece => decode_wordpiece(tokens, &self.vocab, post_processor),
            DecoderType::Metaspace => decode_metaspace(tokens, &self.vocab, post_processor),
        }
    }

    /// Decode token IDs to a UTF-8 string (raw, no text post-processing).
    ///
    /// Returns `None` if the decoded bytes are not valid UTF-8.
    pub fn decode_to_string(&self, tokens: &[TokenId]) -> Option<String> {
        self.vocab.decode_to_string(tokens)
    }

    // --- Delegated methods for serialization compatibility ---

    /// Create a decoder from pre-built parts (used for deserialization).
    pub fn from_parts(data: Vec<u8>, offsets: Vec<u32>, decoder_type: DecoderType) -> Self {
        Self {
            vocab: VocabDecoder::from_parts(data, offsets),
            decoder_type,
        }
    }

    /// Get references to the internal data and offsets.
    pub fn as_parts(&self) -> (&[u8], &[u32]) {
        self.vocab.as_parts()
    }

    /// Get token bytes as a Vec for compatibility.
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        self.vocab.token_bytes()
    }
}

fn decode_wordpiece(
    tokens: &[TokenId],
    vocab: &VocabDecoder,
    post_processor: &PostProcessor,
) -> Option<String> {
    let mut result = String::new();
    for &id in tokens {
        if post_processor.is_special_token(id) {
            continue;
        }
        let bytes = vocab.token_to_bytes(id);
        let token_str = std::str::from_utf8(bytes).ok()?;
        if let Some(stripped) = token_str.strip_prefix("##") {
            result.push_str(stripped);
        } else {
            if !result.is_empty() {
                result.push(' ');
            }
            result.push_str(token_str);
        }
    }
    Some(result)
}

fn decode_metaspace(
    tokens: &[TokenId],
    vocab: &VocabDecoder,
    post_processor: &PostProcessor,
) -> Option<String> {
    let mut result = String::new();
    for &id in tokens {
        if post_processor.is_special_token(id) {
            continue;
        }
        let bytes = vocab.token_to_bytes(id);
        let token_str = std::str::from_utf8(bytes).ok()?;
        result.push_str(&token_str.replace('\u{2581}', " "));
    }
    if result.starts_with(' ') {
        result.remove(0);
    }
    Some(result)
}
