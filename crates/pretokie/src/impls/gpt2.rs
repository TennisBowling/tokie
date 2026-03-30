//! GPT-2 pretokenizer — single-pass, zero allocation.
//!
//! Pattern: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//! Contractions are case-sensitive and standalone.
//! Space prefixes both letters and numbers.

use crate::util::{decode_utf8, is_ascii_letter, is_digit, is_unicode_letter};

pub struct Gpt2<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> Gpt2<'a> {
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len() }
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> u8 {
        unsafe { *self.bytes.get_unchecked(pos) }
    }

    #[inline(always)]
    fn scan_letters(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if is_unicode_letter(ch) { self.pos += cl; } else { return; }
            } else {
                return;
            }
        }
    }

    #[inline(always)]
    fn scan_digits(&mut self) {
        while self.pos < self.len && is_digit(self.at(self.pos)) {
            self.pos += 1;
        }
    }

    /// Check for contraction ('s, 't, 'd, 'm, 'll, 've, 're).
    /// Returns byte length or 0. Case-sensitive for GPT-2.
    #[inline(always)]
    fn check_contraction(&self) -> usize {
        if self.pos >= self.len || self.bytes[self.pos] != b'\'' { return 0; }
        let rem = self.len - self.pos;
        if rem < 2 { return 0; }
        let b1 = self.bytes[self.pos + 1];
        if matches!(b1, b's' | b't' | b'd' | b'm') {
            if rem == 2 || !is_ascii_letter(self.bytes[self.pos + 2]) {
                return 2;
            }
        }
        if rem < 3 { return 0; }
        let b2 = self.bytes[self.pos + 2];
        if (b1 == b'l' && b2 == b'l')
            || (b1 == b'v' && b2 == b'e')
            || (b1 == b'r' && b2 == b'e')
        {
            return 3;
        }
        0
    }

    /// Scan `[^\s\p{L}\p{N}]+` — non-whitespace, non-letter, non-digit chars.
    #[inline(always)]
    fn scan_punct(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if !is_unicode_letter(ch) && !ch.is_numeric() && !ch.is_whitespace() {
                    self.pos += cl;
                } else { break; }
            } else if is_ascii_letter(b) || is_digit(b) || b == b' ' || b == b'\n'
                || b == b'\r' || b == b'\t'
            {
                break;
            } else {
                self.pos += 1;
            }
        }
    }

    /// Consume whitespace greedily, then apply \s+(?!\S) logic:
    /// if followed by non-WS and consumed > 1, back up one byte.
    #[inline(always)]
    fn scan_whitespace(&mut self) {
        let start = self.pos;
        let mut prev_pos = self.pos;
        while self.pos < self.len {
            let c = self.at(self.pos);
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' {
                prev_pos = self.pos;
                self.pos += 1;
            } else if c >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_whitespace() { prev_pos = self.pos; self.pos += cl; } else { break; }
            } else {
                break;
            }
        }
        // \s+(?!\S): if followed by non-WS, leave last char for next piece's prefix
        if self.pos < self.len && prev_pos > start {
            self.pos = prev_pos;
        }
    }

    #[inline(always)]
    fn emit(&self, start: usize) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) }
    }
}

impl<'a> Iterator for Gpt2<'a> {
    type Item = &'a str;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let b = self.at(self.pos);

        if is_ascii_letter(b) {
            self.pos += 1;
            self.scan_letters();
            if self.check_contraction() > 0 {
                return Some(self.emit(start));
            }
        } else if b == b' ' {
            // Peek at what follows the space
            if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    self.pos += 2;
                    self.scan_letters();
                    if self.check_contraction() > 0 {
                        return Some(self.emit(start));
                    }
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos + 1..]);
                    if is_unicode_letter(ch) {
                        self.pos += 1;
                        self.scan_letters();
                    } else if !ch.is_whitespace() && !ch.is_numeric() {
                        // Space + non-ASCII punct (e.g., em-dash): ` ?[^\s\p{L}\p{N}]+`
                        self.pos += 1;
                        self.scan_punct();
                    } else {
                        self.scan_whitespace();
                    }
                } else if is_digit(next) {
                    self.pos += 2;
                    self.scan_digits();
                } else if next == b' ' || next == b'\n' || next == b'\r' || next == b'\t' {
                    // Space + more whitespace → whitespace run with lookahead
                    self.scan_whitespace();
                } else {
                    // Space prefixes punctuation: ` ?[^\s\p{L}\p{N}]+`
                    self.pos += 2;
                    self.scan_punct();
                }
            } else {
                self.pos += 1;
            }
        } else if b == b'\'' {
            let clen = self.check_contraction();
            if clen > 0 {
                self.pos += clen;
            } else {
                self.pos += 1;
                self.scan_punct();
            }
        } else if is_digit(b) {
            self.pos += 1;
            self.scan_digits();
        } else if b == b'\n' || b == b'\r' || b == b'\t' {
            self.scan_whitespace();
        } else if b >= 0x80 {
            let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
            self.pos += cl;
            if is_unicode_letter(ch) {
                self.scan_letters();
            } else if ch.is_whitespace() {
                self.scan_whitespace();
            } else {
                // Non-ASCII symbol: scan punct group
                self.scan_punct();
            }
        } else {
            // Other ASCII punctuation
            self.pos += 1;
            self.scan_punct();
        }

        debug_assert!(self.pos > start, "no progress at pos {start}");
        Some(self.emit(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<&str> {
        Gpt2::new(text).collect()
    }

    #[test]
    fn basic_words() {
        assert_eq!(split("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn punctuation() {
        assert_eq!(split("Hello, world!"), vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn numbers() {
        assert_eq!(split("test 123"), vec!["test", " 123"]);
    }

    #[test]
    fn contraction() {
        assert_eq!(split("don't"), vec!["don", "'t"]);
    }

    #[test]
    fn contraction_ll() {
        assert_eq!(split("I'll"), vec!["I", "'ll"]);
    }

    #[test]
    fn contraction_ve() {
        assert_eq!(split("I've"), vec!["I", "'ve"]);
    }

    #[test]
    fn contraction_re() {
        assert_eq!(split("we're"), vec!["we", "'re"]);
    }

    #[test]
    fn space_prefix_digit() {
        assert_eq!(split("x 42"), vec!["x", " 42"]);
    }

    #[test]
    fn whitespace_double_newline() {
        // \s+(?!\S) matches first \n, then \s+ matches second \n
        assert_eq!(split("a\n\nb"), vec!["a", "\n", "\n", "b"]);
    }

    #[test]
    fn whitespace_newline_spaces() {
        // \n followed by spaces then non-WS: \s+(?!\S) leaves last space
        assert_eq!(split("a\n  b"), vec!["a", "\n ", " b"]);
    }

    #[test]
    fn multiple_spaces() {
        assert_eq!(split("a  b"), vec!["a", " ", " b"]);
    }

    #[test]
    fn space_prefix_punct() {
        assert_eq!(split("a <b"), vec!["a", " <", "b"]);
    }

    #[test]
    fn triple_spaces() {
        assert_eq!(split("a   b"), vec!["a", "  ", " b"]);
    }
}
