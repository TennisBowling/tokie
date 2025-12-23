//! HuggingFace tokenizer.json loading support.

use std::path::Path;

use crate::bpe::BytePairEncoder;
use crate::decoder::Decoder;
use crate::pretokenizer::PretokenizerType;
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
    pretokenizer_type: PretokenizerType,
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
    pretokenizer_type: PretokenizerType,
) -> Result<Tokenizer, JsonLoadError> {
    let data: serde_json::Value = serde_json::from_str(json_str)?;
    load_from_json_value(&data, pretokenizer_type)
}

/// Internal: load tokenizer from parsed JSON value.
fn load_from_json_value(
    data: &serde_json::Value,
    pretokenizer_type: PretokenizerType,
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

    // Build base tokens (first 256 are byte tokens)
    let base_tokens: Vec<Vec<u8>> = vocab
        .iter()
        .take(256)
        .map(|(s, _)| decode_gpt2_token(s))
        .collect();

    // Build merges
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

    let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
    let decoder = Decoder::new(token_bytes);

    Ok(Tokenizer::new(encoder, decoder, pretokenizer_type))
}

/// Detect the pretokenizer type from HuggingFace JSON.
///
/// Only auto-detects simple cases (GPT-2 ByteLevel). For Sequence-based
/// pretokenizers (cl100k, o200k), use `from_json_with_pretokenizer` to
/// explicitly specify the type - this avoids silent mismatches if the
/// regex pattern differs slightly from what we expect.
fn detect_pretokenizer_type(data: &serde_json::Value) -> PretokenizerType {
    let pre_tokenizer = &data["pre_tokenizer"];

    if let Some(typ) = pre_tokenizer["type"].as_str() {
        // ByteLevel pre-tokenizer (GPT-2 style) - safe to auto-detect
        if typ == "ByteLevel" {
            return PretokenizerType::Gpt2;
        }
    }

    // For Sequence or other complex pretokenizers, return None.
    // User should use from_json_with_pretokenizer() to explicitly specify.
    PretokenizerType::None
}

/// Decode a GPT-2 token string to bytes.
///
/// GPT-2 uses byte-level BPE where:
/// - Printable bytes (33-126, 161-172, 174-255) map to their character
/// - Non-printable bytes (0-32, 127-160, 173) are encoded as U+0100+n
fn decode_gpt2_token(s: &str) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_gpt2_token_ascii() {
        // Simple ASCII characters should decode to themselves
        assert_eq!(decode_gpt2_token("Hello"), b"Hello".to_vec());
        assert_eq!(decode_gpt2_token("world"), b"world".to_vec());
    }

    #[test]
    fn test_decode_gpt2_token_space() {
        // Space (byte 32) is encoded as U+0120 (Ġ)
        assert_eq!(decode_gpt2_token("Ġ"), vec![32]);
        assert_eq!(decode_gpt2_token("Ġhello"), vec![32, 104, 101, 108, 108, 111]);
    }

    #[test]
    fn test_decode_gpt2_token_newline() {
        // Newline (byte 10) is encoded as U+010A (Ċ)
        assert_eq!(decode_gpt2_token("Ċ"), vec![10]);
    }

    #[test]
    fn test_decode_gpt2_token_tab() {
        // Tab (byte 9) is encoded as U+0109 (ĉ)
        assert_eq!(decode_gpt2_token("ĉ"), vec![9]);
    }

    #[test]
    fn test_decode_gpt2_token_punctuation() {
        // Punctuation in printable ASCII range should decode directly
        assert_eq!(decode_gpt2_token(","), vec![44]);
        assert_eq!(decode_gpt2_token("."), vec![46]);
        assert_eq!(decode_gpt2_token("!"), vec![33]);
    }
}
