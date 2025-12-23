//! Binary serialization for fast tokenizer loading.
//!
//! This module provides efficient save/load functionality using a custom binary format
//! that stores pre-built DAAC state, eliminating the need to rebuild the automaton.
//!
//! # File Format
//!
//! ```text
//! Header (56 bytes):
//!   - magic: "TOKI" (4 bytes)
//!   - version: u32 (4 bytes)
//!   - pretokenizer_type: u32 (4 bytes) - 0=None, 1=GPT2, 2=CL100K, 3=O200K
//!   - vocab_size: u32 (4 bytes)
//!   - num_merges: u32 (4 bytes)
//!   - num_base_tokens: u32 (4 bytes)
//!   - token_data_offset: u32, token_data_checksum: u32
//!   - merge_data_offset: u32, merge_data_checksum: u32
//!   - daac_data_offset: u32, daac_data_checksum: u32
//!   - prefix_data_offset: u32, prefix_data_checksum: u32
//!
//! Sections:
//!   - TOKEN_DATA: Decoder's flat buffer (offsets + data)
//!   - MERGE_DATA: split_table as raw bytes
//!   - DAAC_DATA: Pre-built DoubleArrayAhoCorasick state
//!   - PREFIX_DATA: next_prefix_match table (Vec<u32>)
//! ```

use core::mem::size_of;
use std::io::{Read, Write};

use crate::bpe::BytePairEncoder;
use crate::decoder::Decoder;
use crate::pretok::PretokType;
use crate::tokenizer::Tokenizer;
use crate::types::{Split, TokenId};
use daggrs::DoubleArrayAhoCorasick;
use fnv::FnvHashMap;

const MAGIC: &[u8; 4] = b"TOKI";
const VERSION: u32 = 2; // v2 adds prefix_data section
const HEADER_SIZE: usize = 56;

impl PretokType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Gpt2),
            2 => Some(Self::Cl100k),
            3 => Some(Self::O200k),
            _ => None,
        }
    }
}

/// Fast CRC32 checksum using hardware acceleration when available.
fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Error type for serialization/deserialization.
#[derive(Debug)]
pub enum SerdeError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidPretokenizer(u32),
    ChecksumMismatch { section: &'static str },
    InvalidData(&'static str),
}

impl std::fmt::Display for SerdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::InvalidMagic => write!(f, "Invalid magic bytes (not a TOKI file)"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            Self::InvalidPretokenizer(v) => write!(f, "Invalid pretokenizer type: {}", v),
            Self::ChecksumMismatch { section } => write!(f, "Checksum mismatch in {}", section),
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for SerdeError {}

impl From<std::io::Error> for SerdeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl Tokenizer {
    /// Save the tokenizer to a file.
    ///
    /// This saves the pre-built DAAC state, enabling fast loading without
    /// rebuilding the automaton.
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), SerdeError> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        self.save(&mut writer)
    }

    /// Save the tokenizer to a writer.
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<(), SerdeError> {
        let pretokenizer_type = self.pretokenizer_type();
        let encoder = self.encoder();
        let decoder = self.decoder();

        // Serialize sections
        let token_data = serialize_decoder(decoder);
        let merge_data = serialize_splits(encoder.split_table());
        let daac_data = encoder.matcher().serialize();
        let prefix_data = serialize_prefix_match(encoder.next_prefix_match());

        // Compute checksums
        let token_checksum = crc32(&token_data);
        let merge_checksum = crc32(&merge_data);
        let daac_checksum = crc32(&daac_data);
        let prefix_checksum = crc32(&prefix_data);

        // Compute offsets (after header)
        let token_offset = HEADER_SIZE as u32;
        let merge_offset = token_offset + token_data.len() as u32;
        let daac_offset = merge_offset + merge_data.len() as u32;
        let prefix_offset = daac_offset + daac_data.len() as u32;

        // Write header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(pretokenizer_type as u32).to_le_bytes())?;
        writer.write_all(&(decoder.vocab_size() as u32).to_le_bytes())?;
        writer.write_all(&((encoder.vocab_size() - encoder.num_base_tokens()) as u32).to_le_bytes())?;
        writer.write_all(&(encoder.num_base_tokens() as u32).to_le_bytes())?;

        // Interleaved offsets and checksums
        writer.write_all(&token_offset.to_le_bytes())?;
        writer.write_all(&token_checksum.to_le_bytes())?;
        writer.write_all(&merge_offset.to_le_bytes())?;
        writer.write_all(&merge_checksum.to_le_bytes())?;
        writer.write_all(&daac_offset.to_le_bytes())?;
        writer.write_all(&daac_checksum.to_le_bytes())?;
        writer.write_all(&prefix_offset.to_le_bytes())?;
        writer.write_all(&prefix_checksum.to_le_bytes())?;

        // Write sections
        writer.write_all(&token_data)?;
        writer.write_all(&merge_data)?;
        writer.write_all(&daac_data)?;
        writer.write_all(&prefix_data)?;

        Ok(())
    }

    /// Load a tokenizer from a file.
    ///
    /// This loads pre-built DAAC state for instant use without rebuilding.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, SerdeError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load(&mut reader)
    }

    /// Load a tokenizer from a reader.
    pub fn load<R: Read>(reader: &mut R) -> Result<Self, SerdeError> {
        // Read entire file
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        if data.len() < HEADER_SIZE {
            return Err(SerdeError::InvalidData("file too small"));
        }

        // Parse header
        if &data[0..4] != MAGIC {
            return Err(SerdeError::InvalidMagic);
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(SerdeError::UnsupportedVersion(version));
        }

        let pretokenizer_type = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let pretokenizer_type = PretokType::from_u32(pretokenizer_type)
            .ok_or(SerdeError::InvalidPretokenizer(pretokenizer_type))?;

        let vocab_size = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let _num_merges = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let num_base_tokens = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;

        let token_offset = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
        let token_checksum = u32::from_le_bytes(data[28..32].try_into().unwrap());
        let merge_offset = u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize;
        let merge_checksum = u32::from_le_bytes(data[36..40].try_into().unwrap());
        let daac_offset = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;
        let daac_checksum = u32::from_le_bytes(data[44..48].try_into().unwrap());
        let prefix_offset = u32::from_le_bytes(data[48..52].try_into().unwrap()) as usize;
        let prefix_checksum = u32::from_le_bytes(data[52..56].try_into().unwrap());

        // Extract and verify sections
        let token_data = &data[token_offset..merge_offset];
        if crc32(token_data) != token_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "token_data" });
        }

        let merge_data = &data[merge_offset..daac_offset];
        if crc32(merge_data) != merge_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "merge_data" });
        }

        let daac_data = &data[daac_offset..prefix_offset];
        if crc32(daac_data) != daac_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "daac_data" });
        }

        let prefix_data = &data[prefix_offset..];
        if crc32(prefix_data) != prefix_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "prefix_data" });
        }

        // Deserialize sections
        let (decoder_offsets, decoder_data) = deserialize_decoder(token_data, vocab_size)?;
        let split_table = deserialize_splits(merge_data)?;
        let (daac, _) = DoubleArrayAhoCorasick::deserialize(daac_data)
            .ok_or(SerdeError::InvalidData("failed to deserialize DAAC"))?;
        let next_prefix_match = deserialize_prefix_match(prefix_data)?;

        // Rebuild pair_lookup from split_table (this is fast - just hash insertions)
        let pair_lookup = rebuild_pair_lookup(&split_table, num_base_tokens);

        // Extract token lengths from decoder offsets
        let token_lengths: Vec<u8> = (0..vocab_size)
            .map(|i| {
                let start = decoder_offsets[i] as usize;
                let end = decoder_offsets[i + 1] as usize;
                (end - start).min(255) as u8
            })
            .collect();

        // Build encoder
        let encoder = BytePairEncoder::from_parts(
            split_table,
            pair_lookup,
            token_lengths,
            num_base_tokens,
            daac,
            next_prefix_match,
        );

        // Build decoder
        let decoder = Decoder::from_parts(decoder_data, decoder_offsets);

        Ok(Tokenizer::new(encoder, decoder, pretokenizer_type))
    }
}

/// Serialize the decoder's flat buffer.
fn serialize_decoder(decoder: &Decoder) -> Vec<u8> {
    let (data, offsets) = decoder.as_parts();

    // Format: num_offsets (u32) + offsets + data
    let mut buf = Vec::with_capacity(4 + offsets.len() * 4 + data.len());

    buf.extend_from_slice(&(offsets.len() as u32).to_le_bytes());
    for &offset in offsets {
        buf.extend_from_slice(&offset.to_le_bytes());
    }
    buf.extend_from_slice(data);

    buf
}

/// Deserialize the decoder's flat buffer.
/// Note: We read u32s manually because the slice may not be aligned.
fn deserialize_decoder(data: &[u8], vocab_size: usize) -> Result<(Vec<u32>, Vec<u8>), SerdeError> {
    if data.len() < 4 {
        return Err(SerdeError::InvalidData("decoder data too small"));
    }

    let num_offsets = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    if num_offsets != vocab_size + 1 {
        return Err(SerdeError::InvalidData("offset count mismatch"));
    }

    let offsets_end = 4 + num_offsets * 4;
    if data.len() < offsets_end {
        return Err(SerdeError::InvalidData("decoder data truncated"));
    }

    // Read offsets manually to handle unaligned data
    let mut offsets = Vec::with_capacity(num_offsets);
    for i in 0..num_offsets {
        let start = 4 + i * 4;
        offsets.push(u32::from_le_bytes(data[start..start + 4].try_into().unwrap()));
    }

    let token_data = data[offsets_end..].to_vec();

    Ok((offsets, token_data))
}

/// Serialize the split table.
fn serialize_splits(splits: &[Split]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(splits.len() * 8);
    for split in splits {
        buf.extend_from_slice(&split.left.to_le_bytes());
        buf.extend_from_slice(&split.right.to_le_bytes());
    }
    buf
}

/// Deserialize the split table.
/// Note: We read manually to handle unaligned data from file reads.
fn deserialize_splits(data: &[u8]) -> Result<Vec<Split>, SerdeError> {
    if data.len() % size_of::<Split>() != 0 {
        return Err(SerdeError::InvalidData("split data size not aligned"));
    }

    let num_splits = data.len() / size_of::<Split>();
    let mut splits = Vec::with_capacity(num_splits);

    for i in 0..num_splits {
        let start = i * 8;
        let left = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
        let right = u32::from_le_bytes(data[start + 4..start + 8].try_into().unwrap());
        splits.push(Split { left, right });
    }

    Ok(splits)
}

/// Serialize the next_prefix_match table.
fn serialize_prefix_match(prefixes: &[TokenId]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(prefixes.len() * 4);
    for &prefix in prefixes {
        buf.extend_from_slice(&prefix.to_le_bytes());
    }
    buf
}

/// Deserialize the next_prefix_match table.
fn deserialize_prefix_match(data: &[u8]) -> Result<Vec<TokenId>, SerdeError> {
    if data.len() % 4 != 0 {
        return Err(SerdeError::InvalidData("prefix data size not aligned"));
    }

    let num_prefixes = data.len() / 4;
    let mut prefixes = Vec::with_capacity(num_prefixes);

    for i in 0..num_prefixes {
        let start = i * 4;
        prefixes.push(u32::from_le_bytes(data[start..start + 4].try_into().unwrap()));
    }

    Ok(prefixes)
}

/// Rebuild pair_lookup from split_table.
fn rebuild_pair_lookup(
    splits: &[Split],
    num_base_tokens: usize,
) -> FnvHashMap<(TokenId, TokenId), TokenId> {
    let mut lookup = FnvHashMap::default();

    for (id, split) in splits.iter().enumerate().skip(num_base_tokens) {
        lookup.insert((split.left, split.right), id as TokenId);
    }

    lookup
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TokenId;

    #[test]
    fn test_crc32() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"hello"), crc32(b"hello"));
        assert_ne!(crc32(b"hello"), crc32(b"world"));
    }

    #[test]
    fn test_pretokenizer_type_roundtrip() {
        for typ in [
            PretokType::None,
            PretokType::Gpt2,
            PretokType::Cl100k,
            PretokType::O200k,
        ] {
            assert_eq!(PretokType::from_u32(typ as u32), Some(typ));
        }
    }

    fn make_test_tokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges: Vec<(TokenId, TokenId)> = vec![
            (b'a' as u32, b'b' as u32), // ab
            (b'c' as u32, b'd' as u32), // cd
            (256, 257),                  // abcd
        ];
        let (encoder, token_bytes) = crate::bpe::BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = crate::decoder::Decoder::new(token_bytes);
        Tokenizer::new(encoder, decoder, PretokType::Gpt2)
    }

    #[test]
    fn test_save_load_roundtrip() {
        let tokenizer = make_test_tokenizer();

        // Save to memory buffer
        let mut buf = Vec::new();
        tokenizer
            .save(&mut buf)
            .expect("save failed");

        // Load from buffer
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = Tokenizer::load(&mut cursor).expect("load failed");

        // Verify same vocab size
        assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());

        // Verify encoding matches
        let test_texts = ["Hello world", "abcd", "test 123", "abcdabcd"];
        for text in test_texts {
            let original_tokens = tokenizer.encode(text);
            let loaded_tokens = loaded.encode(text);
            assert_eq!(
                original_tokens, loaded_tokens,
                "encoding mismatch for '{}'",
                text
            );
        }

        // Verify decoding matches
        let tokens = tokenizer.encode("Hello world");
        let original_decoded = tokenizer.decode(&tokens);
        let loaded_decoded = loaded.decode(&tokens);
        assert_eq!(original_decoded, loaded_decoded);
    }

    #[test]
    fn test_save_load_file() {
        let tokenizer = make_test_tokenizer();

        let temp_path = std::env::temp_dir().join("tokie_test.bin");

        // Save to file
        tokenizer
            .to_file(&temp_path)
            .expect("to_file failed");

        // Load from file
        let loaded = Tokenizer::from_file(&temp_path).expect("from_file failed");

        // Verify encoding matches
        let text = "Hello world test";
        assert_eq!(tokenizer.encode(text), loaded.encode(text));

        // Cleanup
        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_load_invalid_magic() {
        let mut bad_data = vec![0u8; HEADER_SIZE + 100];
        bad_data[0..4].copy_from_slice(b"BADM");
        let mut cursor = std::io::Cursor::new(&bad_data);
        let result = Tokenizer::load(&mut cursor);
        assert!(matches!(result, Err(SerdeError::InvalidMagic)));
    }

    #[test]
    fn test_load_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes()); // Bad version
        data.resize(HEADER_SIZE + 100, 0);

        let mut cursor = std::io::Cursor::new(&data);
        let result = Tokenizer::load(&mut cursor);
        assert!(matches!(result, Err(SerdeError::UnsupportedVersion(99))));
    }
}
