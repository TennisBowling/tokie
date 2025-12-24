//! HuggingFace tokenizer.json loading support.

use std::path::Path;

use crate::bpe::BytePairEncoder;
use crate::decoder::Decoder;
use crate::pretok::PretokType;
use crate::tokenizer::Tokenizer;

/// Error loading from HuggingFace JSON format.
#[derive(Debug)]
pub enum JsonLoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidFormat(&'static str),
}

impl std::fmt::Display for JsonLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Json(e) => write!(f, "JSON error: {}", e),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for JsonLoadError {}

impl From<std::io::Error> for JsonLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for JsonLoadError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// Load a tokenizer from a HuggingFace tokenizer.json file.
///
/// Only GPT-2 style tokenizers (with `pre_tokenizer.type == "ByteLevel"`) are
/// auto-detected. For cl100k/o200k tokenizers that use Sequence pretokenizers,
/// use [`from_json_with_pretokenizer`] to explicitly specify the type.
///
/// # Example
/// ```ignore
/// use tokie::hf;
///
/// // GPT-2 (auto-detected)
/// let gpt2 = hf::from_json("gpt2_tokenizer.json")?;
///
/// // cl100k (requires explicit type)
/// use tokie::PretokenizerType;
/// let cl100k = hf::from_json_with_pretokenizer("cl100k_tokenizer.json", PretokenizerType::Cl100k)?;
/// ```
pub fn from_json(path: impl AsRef<Path>) -> Result<Tokenizer, JsonLoadError> {
    let json_str = std::fs::read_to_string(path)?;
    from_json_str(&json_str)
}

/// Load a tokenizer with a specific pretokenizer type (overriding auto-detection).
///
/// Use this when you want to use a different pretokenizer than what the JSON specifies.
pub fn from_json_with_pretokenizer(
    path: impl AsRef<Path>,
    pretokenizer_type: PretokType,
) -> Result<Tokenizer, JsonLoadError> {
    let json_str = std::fs::read_to_string(path)?;
    from_json_str_with_pretokenizer(&json_str, pretokenizer_type)
}

/// Load a tokenizer from a HuggingFace tokenizer.json string.
pub fn from_json_str(json_str: &str) -> Result<Tokenizer, JsonLoadError> {
    let data: serde_json::Value = serde_json::from_str(json_str)?;
    let pretokenizer_type = detect_pretokenizer_type(&data);
    load_from_json_value(&data, pretokenizer_type)
}

/// Load a tokenizer from JSON string with a specific pretokenizer type.
pub fn from_json_str_with_pretokenizer(
    json_str: &str,
    pretokenizer_type: PretokType,
) -> Result<Tokenizer, JsonLoadError> {
    let data: serde_json::Value = serde_json::from_str(json_str)?;
    load_from_json_value(&data, pretokenizer_type)
}

/// Internal: load tokenizer from parsed JSON value.
///
/// Detects whether this is a byte-level BPE tokenizer (GPT-2, cl100k, o200k, p50k)
/// or a vocab-defined BPE tokenizer (SentencePiece-style, some LLaMA variants).
fn load_from_json_value(
    data: &serde_json::Value,
    pretokenizer_type: PretokType,
) -> Result<Tokenizer, JsonLoadError> {
    let model = &data["model"];
    let vocab_map = model["vocab"]
        .as_object()
        .ok_or(JsonLoadError::InvalidFormat("vocab should be object"))?;
    let merges_arr = model["merges"]
        .as_array()
        .ok_or(JsonLoadError::InvalidFormat("merges should be array"))?;

    // Build vocabulary mapping sorted by id
    let mut vocab: Vec<(String, u32)> = vocab_map
        .iter()
        .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as u32))
        .collect();
    vocab.sort_by_key(|(_, id)| *id);

    // Detect tokenizer style based on merge order.
    // - Sequential merges (tiktoken): Merge N only references tokens 0 to 255+N-1
    // - Vocab-defined (LLaMA 3, etc.): Merges may reference "future" tokens from vocab
    //
    // We check if merges are in topological order (can be processed sequentially).
    // If not, we need to use vocab-first loading where all token bytes are pre-built.
    let num_base_tokens = 256; // Standard BPE base vocabulary size

    if are_merges_topological(vocab_map, merges_arr, num_base_tokens) {
        load_byte_level_bpe(data, &vocab, vocab_map, merges_arr, pretokenizer_type)
    } else {
        load_vocab_defined_bpe(data, &vocab, vocab_map, merges_arr, pretokenizer_type)
    }
}

/// Check if merges are in topological order (can be processed sequentially).
///
/// In tiktoken-style tokenizers, merge N only references tokens 0 to 255+N-1.
/// In vocab-defined tokenizers (like LLaMA 3), merges may reference "future" tokens
/// that exist in the vocab but would be created by later merges.
fn are_merges_topological(
    vocab_map: &serde_json::Map<String, serde_json::Value>,
    merges_arr: &[serde_json::Value],
    num_base_tokens: usize,
) -> bool {
    for (merge_idx, merge) in merges_arr.iter().enumerate() {
        let Some(s) = merge.as_str() else {
            continue;
        };
        let mut parts = s.split(' ');
        let Some(left_str) = parts.next() else {
            continue;
        };
        let Some(right_str) = parts.next() else {
            continue;
        };

        let left_id = vocab_map
            .get(left_str)
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let right_id = vocab_map
            .get(right_str)
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // At merge N, we have tokens 0 to (num_base_tokens + N - 1)
        let max_available = num_base_tokens + merge_idx;

        if left_id >= max_available || right_id >= max_available {
            // This merge references a token that hasn't been created yet
            return false;
        }
    }
    true
}

/// Load byte-level BPE tokenizer (GPT-2, cl100k, o200k, p50k).
///
/// These tokenizers have:
/// - First 256 tokens are bytes 0-255 (with GPT-2's encoding for non-printable bytes)
/// - Each merge creates the next sequential token ID
/// - Vocab IDs match: base_tokens + merge_order
fn load_byte_level_bpe(
    data: &serde_json::Value,
    vocab: &[(String, u32)],
    vocab_map: &serde_json::Map<String, serde_json::Value>,
    merges_arr: &[serde_json::Value],
    pretokenizer_type: PretokType,
) -> Result<Tokenizer, JsonLoadError> {
    // Build base tokens (first 256 are byte tokens)
    let base_tokens: Vec<Vec<u8>> = vocab
        .iter()
        .take(256)
        .map(|(s, _)| decode_bytelevel_token(s))
        .collect();

    // Collect added/special tokens that may occupy IDs in the merge range
    let added_tokens: Vec<(u32, Vec<u8>)> = data
        .get("added_tokens")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let id = t.get("id")?.as_u64()? as u32;
                    let content = t.get("content")?.as_str()?;
                    // Only include if in merge range (id >= 256)
                    if id >= 256 {
                        Some((id, content.as_bytes().to_vec()))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    // Build merges with proper ID mapping
    let merges: Vec<(u32, u32)> = merges_arr
        .iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let mut parts = s.split(' ');
            let left = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            let right = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            Some((left, right))
        })
        .collect();

    let (encoder, token_bytes) =
        BytePairEncoder::from_merges_with_added(&merges, &base_tokens, &added_tokens);
    let decoder = Decoder::new(token_bytes);

    Ok(Tokenizer::new(encoder, decoder, pretokenizer_type))
}

/// Load vocab-defined BPE tokenizer (LLaMA 3, Mistral, SentencePiece-style, etc.).
///
/// These tokenizers have:
/// - Vocab with pre-assigned IDs (not necessarily sequential with merges)
/// - Merges may reference "future" tokens that exist in vocab
/// - Need to pre-build all token bytes from vocab before processing merges
fn load_vocab_defined_bpe(
    data: &serde_json::Value,
    vocab: &[(String, u32)],
    vocab_map: &serde_json::Map<String, serde_json::Value>,
    merges_arr: &[serde_json::Value],
    pretokenizer_type: PretokType,
) -> Result<Tokenizer, JsonLoadError> {
    // Detect token encoding style from the decoder configuration
    let uses_bytelevel = is_bytelevel_decoder(data);

    // Build full vocab with decoded bytes using appropriate decoder
    let full_vocab: Vec<(u32, Vec<u8>)> = vocab
        .iter()
        .map(|(s, id)| {
            let bytes = if uses_bytelevel {
                decode_bytelevel_token(s)
            } else {
                decode_sentencepiece_token(s)
            };
            (*id, bytes)
        })
        .collect();

    // Find number of base tokens (single-byte tokens at the start)
    // For SentencePiece, this varies but we need at least byte coverage
    let num_base_tokens = full_vocab
        .iter()
        .take_while(|(_, bytes)| bytes.len() == 1)
        .count()
        .max(256);

    // Build merges with proper ID mapping
    let merges: Vec<(u32, u32)> = merges_arr
        .iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let mut parts = s.split(' ');
            let left = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            let right = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            Some((left, right))
        })
        .collect();

    let (encoder, token_bytes) =
        BytePairEncoder::from_vocab_and_merges(&full_vocab, &merges, num_base_tokens);
    let decoder = Decoder::new(token_bytes);

    Ok(Tokenizer::new(encoder, decoder, pretokenizer_type))
}

/// Detect the pretokenizer type from HuggingFace JSON.
///
/// Auto-detects based on:
/// - Pre-tokenizer type (ByteLevel = GPT-2)
/// - Vocabulary size (~100K = cl100k, ~200K = o200k)
///
/// For edge cases, use `from_json_with_pretokenizer` to explicitly specify.
fn detect_pretokenizer_type(data: &serde_json::Value) -> PretokType {
    let pre_tokenizer = &data["pre_tokenizer"];

    // Check pre_tokenizer type first
    if let Some(typ) = pre_tokenizer["type"].as_str() {
        // ByteLevel pre-tokenizer (GPT-2 style)
        if typ == "ByteLevel" {
            return PretokType::Gpt2;
        }
    }

    // For Sequence pretokenizers (cl100k, o200k), detect by vocab size
    if let Some(vocab) = data["model"]["vocab"].as_object() {
        let vocab_size = vocab.len();

        // o200k: ~199,998 to ~200,064 tokens
        if vocab_size >= 190_000 && vocab_size <= 210_000 {
            return PretokType::O200k;
        }

        // cl100k: ~100,256 to ~100,300 tokens
        if vocab_size >= 100_000 && vocab_size <= 110_000 {
            return PretokType::Cl100k;
        }

        // p50k/r50k: ~50,257 to ~50,281 tokens (GPT-2 family)
        if vocab_size >= 50_000 && vocab_size <= 52_000 {
            return PretokType::Gpt2;
        }
    }

    // Unknown - return None, BPE will still work but without pretokenization
    PretokType::None
}

/// Decode a HuggingFace ByteLevel token string to bytes.
///
/// HuggingFace's ByteLevel encoding maps bytes to visible Unicode characters:
/// - Printable bytes (33-126, 161-172, 174-255) map to their character
/// - Non-printable bytes (0-32, 127-160, 173) are encoded as U+0100+n
///
/// This encoding is used by GPT-2, LLaMA 3, and most modern BPE tokenizers.
fn decode_bytelevel_token(s: &str) -> Vec<u8> {
    // Non-printable bytes in the order GPT-2 encodes them (mapped to U+0100+)
    // This is: 0-32 (33 bytes), 127-160 (34 bytes), 173 (1 byte) = 68 bytes total
    static NON_PRINTABLE: [u8; 68] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, // 0-32
        127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
        145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, // 127-160
        173, // 173
    ];

    let mut bytes = Vec::with_capacity(s.len());
    for c in s.chars() {
        let code = c as u32;

        let b = if code >= 256 && code < 256 + NON_PRINTABLE.len() as u32 {
            // Non-printable byte encoded as U+0100+n
            NON_PRINTABLE[(code - 256) as usize]
        } else if code <= 255 {
            // Direct byte mapping (printable characters)
            code as u8
        } else {
            // Outside GPT-2 encoding range - shouldn't happen in valid tokens
            bytes.extend(c.to_string().as_bytes());
            continue;
        };
        bytes.push(b);
    }

    bytes
}

/// Check if the tokenizer uses ByteLevel decoding (vs ByteFallback/SentencePiece).
///
/// ByteLevel: Uses Ġ (U+0120) for space, characters map to bytes
/// ByteFallback: Uses ▁ (U+2581) for space, <0xXX> for raw bytes
fn is_bytelevel_decoder(data: &serde_json::Value) -> bool {
    let decoder = &data["decoder"];

    // Check decoder type directly
    if let Some(typ) = decoder["type"].as_str() {
        if typ == "ByteLevel" {
            return true;
        }
    }

    // Check for Sequence decoder containing ByteLevel
    if let Some(decoders) = decoder["decoders"].as_array() {
        for d in decoders {
            if let Some(typ) = d["type"].as_str() {
                if typ == "ByteLevel" {
                    return true;
                }
            }
        }
    }

    // Check pre_tokenizer for ByteLevel (often correlates)
    if let Some(pretoks) = data["pre_tokenizer"]["pretokenizers"].as_array() {
        for p in pretoks {
            if let Some(typ) = p["type"].as_str() {
                if typ == "ByteLevel" {
                    return true;
                }
            }
        }
    }

    false
}

/// Decode a SentencePiece token string to bytes.
///
/// SentencePiece uses different encoding than ByteLevel:
/// - ▁ (U+2581) represents a space/word boundary
/// - <0xXX> patterns represent raw bytes (e.g., <0x0A> for newline)
/// - Other characters are their UTF-8 representation
fn decode_sentencepiece_token(s: &str) -> Vec<u8> {
    // Handle <0xXX> byte patterns (e.g., "<0x0A>" for newline)
    if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
            return vec![byte];
        }
    }

    // Replace ▁ (U+2581) with space
    let s = s.replace('▁', " ");

    // Return as UTF-8 bytes
    s.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_bytelevel_token_ascii() {
        // Simple ASCII characters should decode to themselves
        assert_eq!(decode_bytelevel_token("Hello"), b"Hello".to_vec());
        assert_eq!(decode_bytelevel_token("world"), b"world".to_vec());
    }

    #[test]
    fn test_decode_bytelevel_token_space() {
        // Space (byte 32) is encoded as U+0120 (Ġ)
        assert_eq!(decode_bytelevel_token("Ġ"), vec![32]);
        assert_eq!(decode_bytelevel_token("Ġhello"), vec![32, 104, 101, 108, 108, 111]);
    }

    #[test]
    fn test_decode_bytelevel_token_newline() {
        // Newline (byte 10) is encoded as U+010A (Ċ)
        assert_eq!(decode_bytelevel_token("Ċ"), vec![10]);
    }

    #[test]
    fn test_decode_bytelevel_token_tab() {
        // Tab (byte 9) is encoded as U+0109 (ĉ)
        assert_eq!(decode_bytelevel_token("ĉ"), vec![9]);
    }

    #[test]
    fn test_decode_bytelevel_token_punctuation() {
        // Punctuation in printable ASCII range should decode directly
        assert_eq!(decode_bytelevel_token(","), vec![44]);
        assert_eq!(decode_bytelevel_token("."), vec![46]);
        assert_eq!(decode_bytelevel_token("!"), vec![33]);
    }
}
