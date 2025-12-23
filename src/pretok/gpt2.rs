//! Hand-coded GPT-2 pretokenizer implemented as a lexer.
//!
//! This module implements the GPT-2 pretokenization pattern as a hand-coded lexer,
//! achieving ~400 MiB/s throughput (compared to ~150 MiB/s with regex).
//!
//! # The GPT-2 Pattern
//!
//! The original regex pattern is:
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! ```
//!
//! This pattern matches (in priority order):
//! 1. **Contractions**: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`
//! 2. **Letter runs**: Optional space + one or more Unicode letters
//! 3. **Number runs**: Optional space + one or more Unicode numbers
//! 4. **Other runs**: Optional space + one or more non-letter/non-number/non-whitespace
//! 5. **Whitespace**: One or more whitespace chars, with lookahead handling
//!
//! # Lexer Architecture
//!
//! The regex is decomposed into a state machine:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                         Main Dispatch                               │
//! │  byte_class(b) → Letter | Number | Space | Whitespace | Apostrophe  │
//! └─────────────────────────────────────────────────────────────────────┘
//!          │           │         │          │            │
//!          ▼           ▼         ▼          ▼            ▼
//!    scan_letters  scan_nums  space+X   scan_ws    check_contraction
//!                              │                         │
//!                     ┌────────┴────────┐          ┌─────┴─────┐
//!                     ▼        ▼        ▼          ▼           ▼
//!                  letter   number   other    contraction  punctuation
//! ```
//!
//! ## Regex → Lexer Mapping
//!
//! | Regex Part | Lexer Function | Description |
//! |------------|----------------|-------------|
//! | `'s\|'t\|...` | `check_contraction()` | Byte pattern matching |
//! | `\p{L}+` | `scan_letters()` | ASCII fast path + Unicode fallback |
//! | `\p{N}+` | `scan_numbers()` | ASCII digits + Unicode numbers |
//! | `[^\s\p{L}\p{N}]+` | `scan_other()` | Everything else (punctuation) |
//! | `\s+(?!\S)\|\s+` | `scan_whitespace()` | Whitespace with lookahead |
//! | ` ?` prefix | Main dispatch | Handled specially for space |
//!
//! # Optimizations
//!
//! 1. **256-byte lookup table**: Single table lookup classifies any byte in O(1)
//! 2. **ASCII fast path**: Most text is ASCII; avoid Unicode handling when possible
//! 3. **Bit manipulation**: `is_ascii_letter(b)` uses `(b | 0x20) - 'a' < 26`
//! 4. **Unsafe indexing**: Hot loops use `get_unchecked()` after bounds check
//! 5. **Streamlined space handling**: Optimizes common case (space + letter)
//!
//! # Example
//!
//! ```
//! use tokie::pretok::{Pretok, Gpt2Pretok};
//!
//! let pretok = Gpt2Pretok::new();
//! let pieces: Vec<&str> = pretok.split("Hello world").collect();
//! assert_eq!(pieces, vec!["Hello", " world"]);
//!
//! // Contractions are split
//! let pieces: Vec<&str> = pretok.split("don't").collect();
//! assert_eq!(pieces, vec!["don", "'t"]);
//!
//! // Multiple spaces use lookahead
//! let pieces: Vec<&str> = pretok.split("a  b").collect();
//! assert_eq!(pieces, vec!["a", " ", " b"]);
//! ```

use unicode_general_category::{get_general_category, GeneralCategory};

use super::Pretok;

// ============================================================================
// Character Classification
// ============================================================================

/// Character class for the GPT-2 lexer.
///
/// Each class corresponds to a different handling path in the main dispatch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum Class {
    /// ASCII letters (a-z, A-Z) and Unicode letters
    Letter = 0,
    /// ASCII digits (0-9) and Unicode numbers
    Number = 1,
    /// Regular space (U+0020) - special for optional space prefix
    Space = 2,
    /// Other whitespace (tab, newline, etc.)
    Whitespace = 3,
    /// ASCII apostrophe (') - triggers contraction check
    Apostrophe = 4,
    /// Everything else (punctuation, symbols)
    Other = 5,
}

/// Pre-computed lookup table for all bytes 0-255.
///
/// - Bytes 0-127 (ASCII) map to their character class
/// - Bytes 128-255 (UTF-8 continuation/start) map to 0xFF to trigger Unicode path
static BYTE_CLASS: [u8; 256] = {
    let mut table = [Class::Other as u8; 256];
    let mut i = 0u8;
    loop {
        table[i as usize] = match i {
            b'a'..=b'z' | b'A'..=b'Z' => Class::Letter as u8,
            b'0'..=b'9' => Class::Number as u8,
            b' ' => Class::Space as u8,
            b'\t' | b'\n' | b'\r' | 0x0B | 0x0C => Class::Whitespace as u8,
            b'\'' => Class::Apostrophe as u8,
            0x80..=0xFF => 0xFF, // UTF-8 continuation/start - needs special handling
            _ => Class::Other as u8,
        };
        if i == 255 {
            break;
        }
        i += 1;
    }
    table
};

/// Fast byte classification via table lookup.
#[inline(always)]
fn byte_class(b: u8) -> u8 {
    BYTE_CLASS[b as usize]
}

/// Check if a byte class can follow a space prefix (for ` ?X` pattern).
#[inline(always)]
fn can_follow_space(class: u8) -> bool {
    class == Class::Letter as u8
        || class == Class::Number as u8
        || class == Class::Other as u8
        || class == Class::Apostrophe as u8
}

/// Check if byte is an ASCII letter using bit manipulation.
///
/// This is faster than table lookup for the tight letter-scanning loop.
/// Works by: `(b | 0x20)` converts to lowercase, then check if in `[a-z]`.
#[inline(always)]
fn is_ascii_letter(b: u8) -> bool {
    let lower = b | 0x20;
    lower >= b'a' && lower <= b'z'
}

/// Classify a Unicode character (non-ASCII path).
#[inline]
fn classify_unicode(c: char) -> Class {
    if c.is_whitespace() {
        if c == ' ' {
            Class::Space
        } else {
            Class::Whitespace
        }
    } else {
        match get_general_category(c) {
            GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter => Class::Letter,
            GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber => Class::Number,
            _ => Class::Other,
        }
    }
}

/// Decode UTF-8 character and return (char, byte_length).
///
/// # Safety
/// Caller must ensure `bytes` contains a valid UTF-8 sequence.
#[inline(always)]
fn decode_utf8(bytes: &[u8]) -> (char, usize) {
    let b0 = bytes[0];
    if b0 < 0x80 {
        (b0 as char, 1)
    } else if b0 < 0xE0 {
        let c = ((b0 as u32 & 0x1F) << 6) | (bytes[1] as u32 & 0x3F);
        (unsafe { char::from_u32_unchecked(c) }, 2)
    } else if b0 < 0xF0 {
        let c = ((b0 as u32 & 0x0F) << 12)
            | ((bytes[1] as u32 & 0x3F) << 6)
            | (bytes[2] as u32 & 0x3F);
        (unsafe { char::from_u32_unchecked(c) }, 3)
    } else {
        let c = ((b0 as u32 & 0x07) << 18)
            | ((bytes[1] as u32 & 0x3F) << 12)
            | ((bytes[2] as u32 & 0x3F) << 6)
            | (bytes[3] as u32 & 0x3F);
        (unsafe { char::from_u32_unchecked(c) }, 4)
    }
}

// ============================================================================
// Pretokenizer
// ============================================================================

/// GPT-2 compatible pretokenizer.
///
/// Implements the GPT-2 pretokenization pattern as a hand-coded lexer.
/// Compatible with GPT-2, GPT-J, GPT-Neo, and other models using the same pattern.
#[derive(Debug, Clone, Default)]
pub struct Gpt2Pretok {
    _private: (),
}

impl Gpt2Pretok {
    /// Create a new GPT-2 pretokenizer.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Pretok for Gpt2Pretok {
    type Iter<'a> = Gpt2PretokIter<'a>;

    fn split<'a>(&'a self, text: &'a str) -> Self::Iter<'a> {
        Gpt2PretokIter {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }
}

// ============================================================================
// Iterator
// ============================================================================

/// Iterator over GPT-2 pre-tokens.
///
/// Each call to `next()` returns one pre-token slice.
pub struct Gpt2PretokIter<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for Gpt2PretokIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.bytes;
        if self.pos >= bytes.len() {
            return None;
        }

        let start = self.pos;
        let b = bytes[self.pos];
        let class = byte_class(b);

        // Fast path: ASCII (class != 0xFF)
        if class != 0xFF {
            match class {
                c if c == Class::Letter as u8 => {
                    // Letter run: \p{L}+
                    self.pos += 1;
                    self.scan_letters();
                }
                c if c == Class::Number as u8 => {
                    // Number run: \p{N}+
                    self.pos += 1;
                    self.scan_numbers();
                }
                c if c == Class::Space as u8 => {
                    // Space: handle ` ?` prefix pattern
                    // Most common case: single space followed by letter
                    if self.pos + 1 < bytes.len() {
                        let next_class = byte_class(bytes[self.pos + 1]);
                        if next_class == Class::Letter as u8 {
                            // ` \p{L}+` - space + letters
                            self.pos += 2;
                            self.scan_letters();
                        } else if next_class == Class::Number as u8 {
                            // ` \p{N}+` - space + numbers
                            self.pos += 2;
                            self.scan_numbers();
                        } else if can_follow_space(next_class) {
                            // ` [^\s\p{L}\p{N}]+` - space + other
                            self.pos += 2;
                            self.scan_other();
                        } else if next_class == 0xFF {
                            // Space + Unicode character
                            self.pos += 1;
                            self.handle_space_unicode();
                        } else {
                            // Space followed by more whitespace
                            self.scan_whitespace();
                        }
                    } else {
                        // Space at end of input
                        self.pos += 1;
                    }
                }
                c if c == Class::Whitespace as u8 => {
                    // Non-space whitespace: \s+
                    self.scan_whitespace();
                }
                c if c == Class::Apostrophe as u8 => {
                    // Apostrophe: check for contractions first
                    if self.pos + 1 < bytes.len() {
                        let contraction_len = self.check_contraction();
                        if contraction_len > 0 {
                            // Matched a contraction ('s, 't, 're, 've, 'm, 'll, 'd)
                            self.pos += contraction_len;
                        } else {
                            // Not a contraction - treat as punctuation
                            self.pos += 1;
                            self.scan_other();
                        }
                    } else {
                        // Apostrophe at end
                        self.pos += 1;
                    }
                }
                _ => {
                    // Other (punctuation/symbols): [^\s\p{L}\p{N}]+
                    self.pos += 1;
                    self.scan_other();
                }
            }
        } else {
            // Slow path: Unicode (non-ASCII byte)
            let (c, len) = decode_utf8(&bytes[self.pos..]);
            self.pos += len;
            let unicode_class = classify_unicode(c);

            match unicode_class {
                Class::Letter => self.scan_letters(),
                Class::Number => self.scan_numbers(),
                Class::Space | Class::Whitespace => self.scan_whitespace(),
                _ => self.scan_other(),
            }
        }

        // Safety: pos advances only on valid UTF-8 boundaries
        Some(unsafe { std::str::from_utf8_unchecked(&bytes[start..self.pos]) })
    }
}

// ============================================================================
// Scanning Functions (one per character class)
// ============================================================================

impl Gpt2PretokIter<'_> {
    /// Scan consecutive letters: `\p{L}+`
    ///
    /// Uses ASCII fast path with bit manipulation, falls back to Unicode.
    #[inline]
    fn scan_letters(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();

        // Fast path: ASCII letters in tight loop with unsafe indexing
        // Safety: we check pos < len before each access
        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if is_ascii_letter(b) {
                self.pos += 1;
            } else if b < 128 {
                // ASCII non-letter - done
                return;
            } else {
                // Unicode - check if letter
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                if classify_unicode(c) == Class::Letter {
                    self.pos += char_len;
                } else {
                    return;
                }
            }
        }
    }

    /// Scan consecutive numbers: `\p{N}+`
    #[inline]
    fn scan_numbers(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();

        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if b >= b'0' && b <= b'9' {
                self.pos += 1;
            } else if b < 128 {
                return;
            } else {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                if classify_unicode(c) == Class::Number {
                    self.pos += char_len;
                } else {
                    return;
                }
            }
        }
    }

    /// Scan consecutive "other" characters: `[^\s\p{L}\p{N}]+`
    ///
    /// Matches punctuation, symbols, and other non-letter/non-number characters.
    #[inline]
    fn scan_other(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();

        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            let class = byte_class(b);
            if class == Class::Other as u8 || class == Class::Apostrophe as u8 {
                self.pos += 1;
            } else if class == 0xFF {
                // Unicode - check if "other"
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                if classify_unicode(c) == Class::Other {
                    self.pos += char_len;
                } else {
                    return;
                }
            } else {
                return;
            }
        }
    }

    /// Scan whitespace with lookahead: `\s+(?!\S)|\s+`
    ///
    /// The GPT-2 pattern has a special lookahead rule:
    /// - If whitespace is followed by non-whitespace, leave one space for the ` ?` prefix
    /// - If whitespace is at end or followed by more whitespace, consume all
    ///
    /// Example: `"a  b"` → `["a", " ", " b"]` (not `["a", "  ", "b"]`)
    #[inline]
    fn scan_whitespace(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();
        let start = self.pos;

        // Consume all whitespace
        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            let class = byte_class(b);
            if class == Class::Space as u8 || class == Class::Whitespace as u8 {
                self.pos += 1;
            } else if class == 0xFF {
                let (c, char_len) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                if c.is_whitespace() {
                    self.pos += char_len;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Lookahead: if consumed >1 whitespace and next is non-whitespace, back up one
        let consumed = self.pos - start;
        if consumed > 1 && self.pos < len {
            let next_class = byte_class(unsafe { *bytes.get_unchecked(self.pos) });
            let needs_prefix = if next_class != 0xFF {
                can_follow_space(next_class)
            } else {
                let (c, _) = decode_utf8(unsafe { bytes.get_unchecked(self.pos..) });
                !c.is_whitespace()
            };

            if needs_prefix {
                // Back up one space character
                self.pos -= 1;
                // Handle multi-byte UTF-8 whitespace
                while self.pos > start
                    && (unsafe { *bytes.get_unchecked(self.pos) } & 0xC0) == 0x80
                {
                    self.pos -= 1;
                }
            }
        }
    }

    /// Handle space followed by Unicode character.
    #[inline]
    fn handle_space_unicode(&mut self) {
        let bytes = self.bytes;
        if self.pos >= bytes.len() {
            return;
        }

        let (c, len) = decode_utf8(&bytes[self.pos..]);
        let class = classify_unicode(c);

        match class {
            Class::Letter => {
                self.pos += len;
                self.scan_letters();
            }
            Class::Number => {
                self.pos += len;
                self.scan_numbers();
            }
            Class::Other => {
                self.pos += len;
                self.scan_other();
            }
            _ => {
                // Whitespace after space - back up and handle as whitespace run
                self.pos -= 1;
                self.scan_whitespace();
            }
        }
    }

    /// Check for contractions: `'s|'t|'re|'ve|'m|'ll|'d`
    ///
    /// Returns the length of the contraction (including apostrophe), or 0 if not matched.
    #[inline]
    fn check_contraction(&self) -> usize {
        let bytes = self.bytes;
        let pos = self.pos;

        if pos + 1 >= bytes.len() {
            return 0;
        }

        match bytes[pos + 1] {
            b's' | b't' | b'm' | b'd' => 2, // 's, 't, 'm, 'd
            b'r' if pos + 2 < bytes.len() && bytes[pos + 2] == b'e' => 3, // 're
            b'v' if pos + 2 < bytes.len() && bytes[pos + 2] == b'e' => 3, // 've
            b'l' if pos + 2 < bytes.len() && bytes[pos + 2] == b'l' => 3, // 'll
            _ => 0,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<String> {
        Gpt2Pretok::new()
            .split(text)
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn test_basic() {
        assert_eq!(split("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn test_multiple_spaces() {
        // Lookahead rule: leave one space for prefix
        assert_eq!(split("Hello  world"), vec!["Hello", " ", " world"]);
        assert_eq!(split("Hello   world"), vec!["Hello", "  ", " world"]);
    }

    #[test]
    fn test_trailing_spaces() {
        assert_eq!(split("test  "), vec!["test", "  "]);
    }

    #[test]
    fn test_leading_spaces() {
        assert_eq!(split("  test"), vec![" ", " test"]);
    }

    #[test]
    fn test_contractions() {
        assert_eq!(
            split("How's it going?"),
            vec!["How", "'s", " it", " going", "?"]
        );
        assert_eq!(split("I'm"), vec!["I", "'m"]);
        assert_eq!(split("don't"), vec!["don", "'t"]);
        assert_eq!(split("we're"), vec!["we", "'re"]);
        assert_eq!(split("they've"), vec!["they", "'ve"]);
        assert_eq!(split("you'll"), vec!["you", "'ll"]);
        assert_eq!(split("he'd"), vec!["he", "'d"]);
    }

    #[test]
    fn test_punctuation() {
        assert_eq!(split("Hello, world!"), vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn test_newlines() {
        // Each newline before non-whitespace gets its own token
        assert_eq!(split("Hello\n\nworld"), vec!["Hello", "\n", "\n", "world"]);
        // Trailing newlines are grouped
        assert_eq!(split("test\n\n\n"), vec!["test", "\n\n\n"]);
    }

    #[test]
    fn test_numbers() {
        assert_eq!(split("test 123 hello"), vec!["test", " 123", " hello"]);
    }

    #[test]
    fn test_unicode_letters() {
        // Russian
        assert_eq!(split("Привет мир"), vec!["Привет", " мир"]);
        // Chinese (all letters, no spaces)
        assert_eq!(split("你好世界"), vec!["你好世界"]);
        // Mixed ASCII and Unicode
        assert_eq!(split("Hello мир"), vec!["Hello", " мир"]);
    }

    #[test]
    fn test_smart_quotes() {
        // Smart quotes (U+2019 = ') are NOT treated as contractions
        // to match the original GPT-2 regex pattern
        assert_eq!(split("How\u{2019}s"), vec!["How", "\u{2019}", "s"]);
    }

    #[test]
    fn test_empty() {
        assert_eq!(split(""), Vec::<String>::new());
    }

    #[test]
    fn test_single_char() {
        assert_eq!(split("a"), vec!["a"]);
        assert_eq!(split(" "), vec![" "]);
        assert_eq!(split("!"), vec!["!"]);
    }
}
