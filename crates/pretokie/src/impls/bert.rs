//! BERT pretokenizer — single-pass, zero allocation.
//!
//! Rules:
//! - Whitespace is a delimiter only (stripped, never part of a piece)
//! - Each punctuation character is a separate piece: "..." → [".", ".", "."]
//! - Letters and digits stay together in words: "18th" → ["18th"]
//! - CJK ideographs are individual pieces: "你好" → ["你", "好"]
//! - Unicode marks attach to words (spacing combining marks)
//! - No contractions, no space prefix

use crate::util::decode_utf8;

/// Fast ASCII punctuation check (same ranges as HuggingFace's `_is_punctuation`).
#[inline(always)]
fn is_ascii_punct(b: u8) -> bool {
    matches!(b, 33..=47 | 58..=64 | 91..=96 | 123..=126)
}

/// Check if a character is a CJK ideograph.
#[inline]
fn is_cjk(c: char) -> bool {
    let cp = c as u32;
    matches!(cp,
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B820..=0x2CEAF
        | 0xF900..=0xFAFF
        | 0x2F800..=0x2FA1F
    )
}

/// Check if a Unicode character is punctuation (Unicode P* categories).
#[inline]
fn is_unicode_punct(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x80 {
        return is_ascii_punct(cp as u8);
    }
    matches!(cp,
        0x00A1 | 0x00A7 | 0x00AB | 0x00B6 | 0x00B7 | 0x00BB | 0x00BF |
        0x2000..=0x206F |
        0x2E00..=0x2E52 |
        0x3000..=0x303F |
        0xFE50..=0xFE6B |
        0xFF01..=0xFF0F | 0xFF1A..=0xFF20 | 0xFF3B..=0xFF40 | 0xFF5B..=0xFF65
    ) || {
        // Fallback: check Unicode categories via char properties
        // Punctuation characters that aren't alphanumeric, whitespace, or control
        !c.is_alphanumeric() && !c.is_whitespace() && !c.is_control()
            && !is_cjk(c) && !c.is_ascii()
            // Exclude marks (combining, spacing, etc.)
            && !is_unicode_mark(c)
    }
}

/// Check if a character is a Unicode mark (M category).
#[inline]
fn is_unicode_mark(c: char) -> bool {
    // Marks are combining characters (Mn, Mc, Me)
    // Common ranges:
    let cp = c as u32;
    matches!(cp,
        0x0300..=0x036F |   // Combining Diacritical Marks
        0x0483..=0x0489 |   // Cyrillic combining marks
        0x0591..=0x05BD |   // Hebrew marks
        0x05BF | 0x05C1..=0x05C2 | 0x05C4..=0x05C5 | 0x05C7 |
        0x0610..=0x061A |   // Arabic marks
        0x064B..=0x065F | 0x0670 |
        0x06D6..=0x06DC | 0x06DF..=0x06E4 | 0x06E7..=0x06E8 | 0x06EA..=0x06ED |
        0x0711 | 0x0730..=0x074A |
        0x07A6..=0x07B0 |
        0x0901..=0x0903 |   // Devanagari marks
        0x093A..=0x094F | 0x0951..=0x0957 | 0x0962..=0x0963 |
        0x0981..=0x0983 |   // Bengali marks
        0x09BC | 0x09BE..=0x09C4 | 0x09C7..=0x09C8 | 0x09CB..=0x09CD |
        0x09D7 | 0x09E2..=0x09E3 |
        0x0B82 | 0x0BBE..=0x0BC2 | 0x0BC6..=0x0BC8 | 0x0BCA..=0x0BCD | 0x0BD7 | // Tamil
        0x0C01..=0x0C03 | 0x0C3E..=0x0C44 | 0x0C46..=0x0C48 | 0x0C4A..=0x0C4D |
        0x0C55..=0x0C56 | 0x0C62..=0x0C63 |
        0x1DC0..=0x1DFF |   // Combining marks supplement
        0x20D0..=0x20FF |   // Combining marks for symbols
        0xFE00..=0xFE0F |   // Variation selectors
        0xFE20..=0xFE2F     // Combining half marks
    )
}

pub struct Bert<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> Bert<'a> {
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len() }
    }

    /// Skip whitespace (delimiter in BERT).
    #[inline(always)]
    fn skip_whitespace(&mut self) {
        while self.pos < self.len {
            let b = unsafe { *self.bytes.get_unchecked(self.pos) };
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else if b < 0x80 {
                return;
            } else {
                let (c, cl) = decode_utf8(unsafe { self.bytes.get_unchecked(self.pos..) });
                if c.is_whitespace() { self.pos += cl; } else { return; }
            }
        }
    }

    /// Scan word characters: ASCII letters, digits, and non-ASCII letters/numbers/marks.
    /// Stops at whitespace, ASCII punctuation, CJK, and Unicode punctuation.
    #[inline(always)]
    fn scan_word(&mut self) {
        while self.pos < self.len {
            let b = unsafe { *self.bytes.get_unchecked(self.pos) };
            if b.is_ascii_alphanumeric() {
                self.pos += 1;
            } else if b < 0x80 {
                return; // ASCII whitespace or punctuation — stop
            } else {
                let (c, cl) = decode_utf8(unsafe { self.bytes.get_unchecked(self.pos..) });
                if is_cjk(c) || c.is_whitespace() || is_unicode_punct(c) {
                    return;
                }
                // Letter, number, mark, or other non-punct non-CJK → continues word
                self.pos += cl;
            }
        }
    }
}

impl<'a> Iterator for Bert<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        // Skip whitespace
        self.skip_whitespace();
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let b = unsafe { *self.bytes.get_unchecked(self.pos) };

        if b < 0x80 {
            if is_ascii_punct(b) {
                // Single punctuation character
                self.pos += 1;
            } else {
                // ASCII letter or digit — scan word
                self.pos += 1;
                self.scan_word();
            }
        } else {
            let (c, cl) = decode_utf8(unsafe { self.bytes.get_unchecked(self.pos..) });
            if is_cjk(c) {
                // Single CJK character
                self.pos += cl;
            } else if is_unicode_punct(c) {
                // Single Unicode punctuation character
                self.pos += cl;
            } else {
                // Non-ASCII letter, number, mark, or other → start word
                self.pos += cl;
                self.scan_word();
            }
        }

        Some(unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) })
    }
}
