//! Byte Pair Encoding implementation

use daggrs::{DoubleArrayAhoCorasick, MatchKind, Trie};
use fnv::FnvHashMap;
use memchunk::chunk;
use std::thread;

use crate::compatibility::is_valid_token_pair;
use crate::types::{Split, TokenId};

/// Minimum text size to use parallel processing (10KB).
const PARALLEL_THRESHOLD: usize = 10_000;

/// Split text into chunks at boundary characters (space/newline).
///
/// Uses memchunk's SIMD-accelerated search for fast boundary detection.
/// Splits text evenly across available CPU cores.
/// Uses prefix mode so delimiters stay with the next chunk (for BPE tokens that start with space).
#[inline]
fn split_at_boundaries(text: &[u8]) -> Vec<&[u8]> {
    let num_cpus = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let target_size = text.len() / num_cpus;
    chunk(text)
        .size(target_size)
        .delimiters(b" \n")
        .prefix()
        .collect()
}

use std::collections::VecDeque;

/// Buffer size for streaming iterator.
/// Based on empirical analysis: max cascade depth observed is 3 on GPT-2 vocabulary.
/// Using 8 for safety margin.
const ENCODE_ITER_BUFFER_SIZE: usize = 8;

/// Streaming iterator over encoded tokens.
///
/// Created by [`BytePairEncoder::encode_iter`]. Uses a small buffer (8 tokens)
/// to enable true streaming - tokens are yielded as they're confirmed safe,
/// without pre-computing the entire encoding.
///
/// # Memory Usage
/// - O(n/64) for the bitfield (tracking reachable positions)
/// - O(8) for the token buffer
/// - Much more memory-efficient than `encode()` for large texts
///
/// # Examples
/// ```ignore
/// // Count tokens without storing them all
/// let count = tokenizer.encode_iter(text).count();
///
/// // Get first N tokens (stops encoding early)
/// let first_10: Vec<_> = tokenizer.encode_iter(text).take(10).collect();
///
/// // Process tokens as they're yielded
/// for token in tokenizer.encode_iter(text) {
///     process(token);
/// }
/// ```
pub struct EncodeIter<'a> {
    tokenizer: &'a BytePairEncoder,
    text: &'a [u8],
    pos: usize,
    buffer: VecDeque<TokenId>,
    bitfield: Bitfield,
    next_token: Option<TokenId>,
    done: bool,
}

impl<'a> EncodeIter<'a> {
    fn new(tokenizer: &'a BytePairEncoder, text: &'a [u8]) -> Self {
        let n = text.len();
        let next_token = if text.is_empty() {
            None
        } else {
            tokenizer.next_match(text)
        };

        Self {
            tokenizer,
            text,
            pos: 0,
            buffer: VecDeque::with_capacity(ENCODE_ITER_BUFFER_SIZE + 1),
            bitfield: Bitfield::new(n + 1),
            next_token,
            done: text.is_empty(),
        }
    }

    /// Try to encode one token into the buffer.
    /// Returns true if a token was added, false if encoding is complete.
    fn encode_one_token(&mut self) -> bool {
        let Some(mut token) = self.next_token else {
            return false;
        };

        let last = self.buffer.back().copied();

        loop {
            let token_len = self.tokenizer.token_len(token);
            let end_pos = self.pos + token_len;

            let is_reachable = self.bitfield.is_set(end_pos);
            let is_compatible = last
                .map(|last_token| self.tokenizer.is_valid_pair(last_token, token))
                .unwrap_or(true);

            if is_reachable && is_compatible {
                // Accept this token
                self.buffer.push_back(token);
                self.pos = end_pos;
                self.next_token = self.tokenizer.next_match(&self.text[self.pos..]);
                return true;
            } else if let Some(shorter) = self.tokenizer.next_prefix(token) {
                // Try a shorter prefix token
                token = shorter;
            } else {
                // No shorter prefix works - backtrack
                self.bitfield.clear(self.pos);
                if let Some(last_token) = self.buffer.pop_back() {
                    self.pos -= self.tokenizer.token_len(last_token);
                    self.next_token = Some(last_token);
                    return false; // Didn't add a token, need to retry
                } else {
                    // Buffer empty and can't backtrack - this shouldn't happen
                    // with a valid vocabulary that covers all bytes
                    self.next_token = None;
                    return false;
                }
            }
        }
    }
}

impl Iterator for EncodeIter<'_> {
    type Item = TokenId;

    fn next(&mut self) -> Option<TokenId> {
        // If encoding is done, just drain the buffer
        if self.done {
            return self.buffer.pop_front();
        }

        // Try to fill buffer to BUFFER_SIZE
        while self.buffer.len() < ENCODE_ITER_BUFFER_SIZE {
            if !self.encode_one_token() {
                // Check if we're truly done (no more tokens to encode)
                if self.next_token.is_none() {
                    self.done = true;
                    break;
                }
                // Otherwise, backtracking happened - continue trying
            }
        }

        // Yield front token (or None if empty)
        self.buffer.pop_front()
    }
}

impl std::iter::FusedIterator for EncodeIter<'_> {}

/// BPE Tokenizer using Aho-Corasick for efficient suffix matching.
pub struct BytePairEncoder {
    /// For each token, stores the two tokens it was merged from.
    /// Base tokens point to themselves.
    split_table: Vec<Split>,

    /// Maps (left_token, right_token) -> merged_token.
    pair_lookup: FnvHashMap<(TokenId, TokenId), TokenId>,

    /// Length of each token in bytes (max 255).
    token_lengths: Vec<u8>,

    /// Number of base tokens (single bytes, typically 256).
    num_base_tokens: usize,

    /// Aho-Corasick automaton for pattern matching.
    matcher: DoubleArrayAhoCorasick,

    /// For each token, the ID of the longest prefix token (or u32::MAX if none).
    /// Used by the backtracking encoder to try shorter tokens.
    next_prefix_match: Vec<TokenId>,
}

impl BytePairEncoder {
    /// Create an encoder from pre-built parts (used for deserialization).
    pub fn from_parts(
        split_table: Vec<Split>,
        pair_lookup: FnvHashMap<(TokenId, TokenId), TokenId>,
        token_lengths: Vec<u8>,
        num_base_tokens: usize,
        matcher: DoubleArrayAhoCorasick,
        next_prefix_match: Vec<TokenId>,
    ) -> Self {
        Self {
            split_table,
            pair_lookup,
            token_lengths,
            num_base_tokens,
            matcher,
            next_prefix_match,
        }
    }

    /// Get a reference to the split table.
    pub fn split_table(&self) -> &[Split] {
        &self.split_table
    }

    /// Get a reference to the DAAC matcher.
    pub fn matcher(&self) -> &DoubleArrayAhoCorasick {
        &self.matcher
    }

    /// Get a reference to the next_prefix_match table.
    pub fn next_prefix_match(&self) -> &[TokenId] {
        &self.next_prefix_match
    }
}

impl BytePairEncoder {
    /// Create a new BPE encoder from merge rules.
    ///
    /// Returns the encoder and the token bytes table (for creating a Decoder).
    ///
    /// # Arguments
    /// * `merges` - List of (token1, token2) pairs in merge order.
    ///              Earlier merges have higher priority (lower token ID).
    /// * `base_tokens` - The base vocabulary (typically 256 single-byte tokens).
    ///
    /// # Example
    /// ```ignore
    /// let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
    /// let decoder = Decoder::new(token_bytes);
    /// ```
    pub fn from_merges(
        merges: &[(TokenId, TokenId)],
        base_tokens: &[Vec<u8>],
    ) -> (Self, Vec<Vec<u8>>) {
        let num_base_tokens = base_tokens.len();

        // Initialize split_table and token_bytes with base tokens
        let mut split_table: Vec<Split> = (0..num_base_tokens as TokenId)
            .map(Split::base)
            .collect();

        let mut token_bytes: Vec<Vec<u8>> = base_tokens.to_vec();
        let mut pair_lookup = FnvHashMap::default();

        // Process each merge
        for &(left, right) in merges {
            let new_id = split_table.len() as TokenId;

            // Record the split (how this token was formed)
            split_table.push(Split::merge(left, right));

            // Record the merge rule
            pair_lookup.insert((left, right), new_id);

            // Compute the byte sequence for this token
            let mut bytes = token_bytes[left as usize].clone();
            bytes.extend_from_slice(&token_bytes[right as usize]);
            token_bytes.push(bytes);
        }

        // Build the Aho-Corasick automaton for leftmost-longest matching
        let mut trie = Trie::new();
        for (id, bytes) in token_bytes.iter().enumerate() {
            trie.add(bytes, id as TokenId);
        }
        trie.build(MatchKind::LeftmostLongest);
        let matcher = trie.compile();

        // Build next_prefix_match table: for each token, find the longest prefix token
        let next_prefix_match: Vec<TokenId> = token_bytes
            .iter()
            .map(|token| {
                if token.len() <= 1 {
                    // Single-byte tokens have no shorter prefix
                    u32::MAX
                } else {
                    // Search for longest match in token[0..len-1]
                    let prefix = &token[0..token.len() - 1];
                    matcher
                        .find_iter(prefix)
                        .next()
                        .map(|m| m.pattern_id)
                        .unwrap_or(u32::MAX)
                }
            })
            .collect();

        // Extract token lengths (max 255 bytes per token)
        let token_lengths: Vec<u8> = token_bytes
            .iter()
            .map(|t| t.len().min(255) as u8)
            .collect();

        let encoder = Self {
            split_table,
            pair_lookup,
            token_lengths,
            num_base_tokens,
            matcher,
            next_prefix_match,
        };

        (encoder, token_bytes)
    }

    /// Check if two tokens can appear adjacent in a valid BPE encoding.
    pub fn is_valid_pair(&self, token1: TokenId, token2: TokenId) -> bool {
        is_valid_token_pair(&self.pair_lookup, &self.split_table, token1, token2)
    }

    /// Get the length of a token in bytes.
    #[inline]
    pub fn token_len(&self, token: TokenId) -> usize {
        self.token_lengths[token as usize] as usize
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.token_lengths.len()
    }

    /// Get the number of base tokens.
    pub fn num_base_tokens(&self) -> usize {
        self.num_base_tokens
    }

    /// Encode text into BPE tokens.
    ///
    /// Automatically uses parallel processing for texts larger than 10KB.
    /// For smaller texts, uses sequential encoding.
    ///
    /// # Example
    /// ```ignore
    /// let tokens = tokenizer.encode(b"Hello, world!");
    /// ```
    pub fn encode(&self, text: &[u8]) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        if text.len() < PARALLEL_THRESHOLD {
            return self.encode_sequential(text);
        }

        // Split at whitespace boundaries and encode in parallel
        let chunks = split_at_boundaries(text);

        if chunks.len() == 1 {
            return self.encode_sequential(chunks[0]);
        }

        // Encode chunks in parallel using scoped threads
        let results: Vec<Vec<TokenId>> = thread::scope(|s| {
            let handles: Vec<_> = chunks
                .iter()
                .map(|chunk| {
                    s.spawn(|| self.encode_sequential(chunk))
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

    /// Returns a streaming iterator over encoded tokens.
    ///
    /// Unlike `encode()`, this does not pre-compute all tokens. Instead, it
    /// encodes on-demand using a small buffer (8 tokens), yielding tokens
    /// as they're confirmed safe. This enables:
    ///
    /// - **Memory efficiency**: O(n/64) bitfield + O(8) buffer vs O(n) for full encoding
    /// - **Early termination**: `take(n)` stops encoding after n tokens
    /// - **Streaming**: Process tokens as they're produced
    ///
    /// # Example
    /// ```ignore
    /// // Count tokens without storing them
    /// let count = tokenizer.encode_iter(text).count();
    ///
    /// // Check if text exceeds token limit (stops early!)
    /// let exceeds_limit = tokenizer.encode_iter(text).take(4097).count() > 4096;
    ///
    /// // Stream tokens
    /// for token in tokenizer.encode_iter(text) {
    ///     process(token);
    /// }
    /// ```
    pub fn encode_iter<'a>(&'a self, text: &'a [u8]) -> EncodeIter<'a> {
        EncodeIter::new(self, text)
    }

    /// Encode multiple texts in parallel.
    ///
    /// Each text is encoded independently using available CPU cores.
    /// More efficient than calling `encode()` in a loop for batch processing.
    ///
    /// Uses sequential encoding per text to avoid nested parallelism -
    /// the parallelism comes from processing multiple texts concurrently.
    ///
    /// # Example
    /// ```ignore
    /// let texts = vec![b"Hello".as_slice(), b"World".as_slice()];
    /// let all_tokens = tokenizer.encode_batch(&texts);
    /// ```
    pub fn encode_batch(&self, texts: &[&[u8]]) -> Vec<Vec<TokenId>> {
        if texts.is_empty() {
            return Vec::new();
        }

        let num_cpus = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        if texts.len() <= num_cpus || num_cpus == 1 {
            // Few texts or single core: one thread per text (or sequential)
            if num_cpus == 1 {
                return texts.iter().map(|t| self.encode_sequential(t)).collect();
            }

            return thread::scope(|s| {
                let handles: Vec<_> = texts
                    .iter()
                    .map(|text| s.spawn(|| self.encode_sequential(text)))
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
        }

        // Many texts: chunk by CPU count, each thread processes a batch
        let chunk_size = (texts.len() + num_cpus - 1) / num_cpus;

        thread::scope(|s| {
            let handles: Vec<_> = texts
                .chunks(chunk_size)
                .map(|chunk| {
                    s.spawn(|| {
                        chunk.iter().map(|t| self.encode_sequential(t)).collect::<Vec<_>>()
                    })
                })
                .collect();

            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    }

    /// Sequential encoding using the backtracking algorithm.
    ///
    /// This is a lazy approach that greedily picks the longest match and only
    /// backtracks when compatibility checking fails. It's faster than the
    /// table-based approach for most inputs because it avoids processing
    /// positions that aren't part of the final encoding.
    fn encode_sequential(&self, text: &[u8]) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        let n = text.len();
        let mut tokens: Vec<TokenId> = Vec::with_capacity(n / 3);
        // Bitfield tracks positions that might lead to a valid encoding.
        // All positions start as potentially valid (bits = 1).
        // We clear bits when we prove a position is a dead end.
        let mut bitfield = Bitfield::new(n + 1);

        let mut pos = 0;
        let mut next_token = self.next_match(&text[pos..]);

        while let Some(mut token) = next_token {
            let last = tokens.last().copied();

            loop {
                let token_len = self.token_len(token);
                let end_pos = pos + token_len;

                // Check if this token is valid:
                // 1. The end position must still be potentially reachable
                // 2. The token must be compatible with the previous token
                let is_reachable = bitfield.is_set(end_pos);
                let is_compatible = last
                    .map(|last_token| self.is_valid_pair(last_token, token))
                    .unwrap_or(true);

                if is_reachable && is_compatible {
                    // Accept this token
                    tokens.push(token);
                    pos = end_pos;
                    next_token = self.next_match(&text[pos..]);
                    break;
                } else if let Some(shorter) = self.next_prefix(token) {
                    // Try a shorter prefix token
                    token = shorter;
                } else {
                    // No shorter prefix works - this position is a dead end
                    bitfield.clear(pos);
                    if let Some(last_token) = tokens.pop() {
                        pos -= self.token_len(last_token);
                    }
                    next_token = last;
                    break;
                }
            }
        }

        tokens
    }

    /// Find the longest match starting at the beginning of text.
    #[inline]
    fn next_match(&self, text: &[u8]) -> Option<TokenId> {
        self.matcher
            .find_iter(text)
            .next()
            .map(|m| m.pattern_id)
    }

    /// Get the next shorter prefix token for backtracking.
    #[inline]
    fn next_prefix(&self, token: TokenId) -> Option<TokenId> {
        let prefix = self.next_prefix_match[token as usize];
        if prefix == u32::MAX {
            None
        } else {
            Some(prefix)
        }
    }
}

/// Bitfield for tracking positions that might lead to a valid encoding.
/// All bits are initialized to 1 (all positions are potentially valid).
/// Bits are cleared when we prove a position is a dead end.
struct Bitfield {
    bits: Vec<u64>,
}

impl Bitfield {
    fn new(size: usize) -> Self {
        let num_words = (size + 63) / 64;
        Self {
            // Initialize all bits to 1 - all positions are potentially reachable
            bits: vec![u64::MAX; num_words],
        }
    }

    #[inline]
    fn clear(&mut self, pos: usize) {
        let word = pos / 64;
        let bit = pos % 64;
        self.bits[word] &= !(1 << bit);
    }

    #[inline]
    fn is_set(&self, pos: usize) -> bool {
        let word = pos / 64;
        let bit = pos % 64;
        (self.bits[word] >> bit) & 1 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::Decoder;

    #[test]
    fn test_from_merges() {
        // Base tokens: a=0, b=1, c=2
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];

        // Merges: a+b->ab(3), ab+c->abc(4)
        let merges = vec![(0, 1), (3, 2)];

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);

        assert_eq!(encoder.vocab_size(), 5);
        assert_eq!(encoder.num_base_tokens(), 3);

        // Check token bytes via decoder
        assert_eq!(decoder.token_to_bytes(0), b"a");
        assert_eq!(decoder.token_to_bytes(1), b"b");
        assert_eq!(decoder.token_to_bytes(2), b"c");
        assert_eq!(decoder.token_to_bytes(3), b"ab");
        assert_eq!(decoder.token_to_bytes(4), b"abc");
    }

    #[test]
    fn test_decode() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)]; // a+b->ab(3)

        let (_, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);

        assert_eq!(decoder.decode(&[3, 2]), b"abc");
        assert_eq!(decoder.decode(&[0, 1, 2]), b"abc");
    }

    #[test]
    fn test_is_valid_pair() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)]; // a+b->ab(3)

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        // (a, b) should NOT be valid - they merge to ab
        assert!(!encoder.is_valid_pair(0, 1));

        // (ab, c) should be valid - no merge rule
        assert!(encoder.is_valid_pair(3, 2));

        // (b, c) should be valid - no merge rule
        assert!(encoder.is_valid_pair(1, 2));
    }

    #[test]
    fn test_encode_empty() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)];

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        assert_eq!(encoder.encode(b""), Vec::<TokenId>::new());
    }

    #[test]
    fn test_encode_single_byte() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)];

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        assert_eq!(encoder.encode(b"a"), vec![0]);
        assert_eq!(encoder.encode(b"b"), vec![1]);
        assert_eq!(encoder.encode(b"c"), vec![2]);
    }

    #[test]
    fn test_encode_merged_token() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)]; // a+b->ab(3)

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        // "ab" should encode as [ab] not [a, b]
        assert_eq!(encoder.encode(b"ab"), vec![3]);

        // "abc" should encode as [ab, c]
        assert_eq!(encoder.encode(b"abc"), vec![3, 2]);
    }

    #[test]
    fn test_encode_with_all_merges() {
        // Base tokens: a=0, b=1, c=2
        // Merges: a+b->ab(3), ab+c->abc(4)
        //
        // For "abc": BPE applies all merges, so result is [abc]
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)]; // a+b->ab(3), ab+c->abc(4)

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);

        // "abc" should encode as [abc] because all merges apply
        let encoded = encoder.encode(b"abc");
        assert_eq!(encoded, vec![4]); // [abc]

        // Verify roundtrip
        assert_eq!(decoder.decode(&encoded), b"abc");
    }

    #[test]
    fn test_encode_partial_merges() {
        // Base tokens: a=0, b=1, c=2
        // Merges: a+b->ab(3) ONLY - no ab+c merge
        //
        // For "abc": BPE produces [ab, c] because there's no merge for ab+c
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)]; // a+b->ab(3) only

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);

        // "abc" should encode as [ab, c] - ab merges but no further merge with c
        let encoded = encoder.encode(b"abc");
        assert_eq!(encoded, vec![3, 2]); // [ab, c]

        // Verify roundtrip
        assert_eq!(decoder.decode(&encoded), b"abc");
    }

    #[test]
    fn test_encode_longer_sequence() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)]; // a+b->ab(3), ab+c->abc(4)

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);

        // "abcabc" should be [abc, abc]
        let encoded = encoder.encode(b"abcabc");
        assert_eq!(encoded, vec![4, 4]); // [abc, abc]

        // Verify roundtrip
        assert_eq!(decoder.decode(&encoded), b"abcabc");
    }

    #[test]
    fn test_encode_respects_merge_order() {
        // The KEY test for BPE semantics:
        // If we have overlapping potential merges, the EARLIER merge wins.
        //
        // Base tokens: a=0, b=1, c=2, d=3
        // Merges: b+c->bc(4), a+b->ab(5)
        //
        // For "abc":
        // - Greedy leftmost-longest would give [a, bc] (token bc is longer and matches first)
        // - But BPE should give [ab, c] because a+b merge (id=5) must happen before using bc
        //
        // Wait, that's not right either. Let me think...
        // Actually bc(4) was learned BEFORE ab(5), so bc has priority.
        // For "abc", at position 2 we have candidates [b, bc], and bc is valid.
        // Then at position 3, we need to continue from position where bc ended...
        //
        // Actually the key is: bc can only appear if the bytes before it don't form
        // a merge with b. If "ab" is a merge, then (a, bc) is incompatible because
        // a+b should merge.
        //
        // Let me construct a proper test:
        // Merges: a+b->ab(4) first, then b+c->bc(5)
        // For "abc": a+b merges first (lower ID), giving [ab, c]
        // The bc token (id=5) can't be used because that would require [a, bc],
        // and (a, bc) is incompatible - a+b should have merged.
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![
            (0, 1), // a+b->ab(3) - learned first, higher priority
            (1, 2), // b+c->bc(4) - learned second
        ];

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);

        // For "abc": even though bc exists as a token, we can't use it
        // because (a, bc) is incompatible - a and b should have merged.
        // So the result is [ab, c]
        let encoded = encoder.encode(b"abc");
        assert_eq!(encoded, vec![3, 2]); // [ab, c], not [a, bc]

        // Verify roundtrip
        assert_eq!(decoder.decode(&encoded), b"abc");
    }

    #[test]
    fn test_encode_no_merges() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges: Vec<(TokenId, TokenId)> = vec![]; // No merges

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        // Without merges, each byte is its own token
        assert_eq!(encoder.encode(b"abc"), vec![0, 1, 2]);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c'], vec![b'd']];
        let merges = vec![
            (0, 1), // a+b->ab(4)
            (2, 3), // c+d->cd(5)
            (4, 5), // ab+cd->abcd(6)
        ];

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);

        let texts = [b"abcd".as_slice(), b"ab", b"cd", b"abcdabcd", b"a", b""];

        for text in texts {
            let encoded = encoder.encode(text);
            let decoded = decoder.decode(&encoded);
            assert_eq!(decoded, text, "Roundtrip failed for {:?}", text);
        }
    }

    #[test]
    fn test_encode_iter_matches_encode() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c'], vec![b'd']];
        let merges = vec![
            (0, 1), // a+b->ab(4)
            (2, 3), // c+d->cd(5)
            (4, 5), // ab+cd->abcd(6)
        ];

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        let texts = [
            b"".as_slice(),
            b"a",
            b"ab",
            b"abc",
            b"abcd",
            b"abcdabcd",
            b"abcdabcdabcdabcdabcd", // longer than buffer size
        ];

        for text in texts {
            let encoded = encoder.encode(text);
            let iter_encoded: Vec<_> = encoder.encode_iter(text).collect();
            assert_eq!(
                encoded, iter_encoded,
                "encode_iter mismatch for {:?}",
                std::str::from_utf8(text).unwrap_or("<invalid utf8>")
            );
        }
    }

    #[test]
    fn test_encode_iter_empty() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)];
        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        let mut iter = encoder.encode_iter(b"");
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None); // FusedIterator behavior
    }

    #[test]
    fn test_encode_iter_single_token() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)];
        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        let mut iter = encoder.encode_iter(b"c");
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_encode_iter_early_termination() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c'], vec![b'd']];
        let merges = vec![(0, 1), (2, 3)];
        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        // "abcdabcd" = [ab, cd, ab, cd] = 4 tokens
        // take(2) should give us [ab, cd]
        let first_two: Vec<_> = encoder.encode_iter(b"abcdabcd").take(2).collect();
        assert_eq!(first_two.len(), 2);

        // Verify they're the right tokens
        let full = encoder.encode(b"abcdabcd");
        assert_eq!(first_two, full[..2]);
    }
}
