//! Qwen3.5 pretokenizer — single-pass, zero allocation.
//!
//! Used by: Qwen3.5 (all sizes)
//!
//! Like Voyage but:
//! - Combining marks stay with letters: `[\p{L}\p{M}]+` instead of `\p{L}+`
//! - Single digits: `12345` → `["1", "2", "3", "4", "5"]`
//! - Case-insensitive standalone contractions

use crate::util::{decode_utf8, is_ascii_letter, is_digit, is_unicode_mark};

pub struct Qwen<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> Qwen<'a> {
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len() }
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> u8 {
        unsafe { *self.bytes.get_unchecked(pos) }
    }

    /// Scan letters AND marks ([\p{L}\p{M}]+).
    #[inline(always)]
    fn scan_letters_with_marks(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_alphabetic() || is_unicode_mark(ch) {
                    self.pos += cl;
                } else { return; }
            } else {
                return;
            }
        }
    }

    #[inline(always)]
    fn check_contraction(&self) -> usize {
        if self.pos >= self.len || self.bytes[self.pos] != b'\'' { return 0; }
        let rem = self.len - self.pos;
        if rem < 2 { return 0; }
        let b1 = self.bytes[self.pos + 1] | 0x20;
        if matches!(b1, b's' | b't' | b'd' | b'm') {
            if rem == 2 || !is_ascii_letter(self.bytes[self.pos + 2]) { return 2; }
        }
        if rem < 3 { return 0; }
        let b2 = self.bytes[self.pos + 2] | 0x20;
        if (b1 == b'l' && b2 == b'l') || (b1 == b'v' && b2 == b'e') || (b1 == b'r' && b2 == b'e') {
            return 3;
        }
        0
    }

    #[inline(always)]
    fn is_punct_char(b: u8) -> bool {
        !is_ascii_letter(b) && !is_digit(b) && b != b' ' && b != b'\t' && b != b'\n' && b != b'\r' && b < 0x80
    }

    #[inline(always)]
    fn scan_punct_with_newlines(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if Self::is_punct_char(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if !ch.is_alphabetic() && !ch.is_numeric() && !ch.is_whitespace() && !is_unicode_mark(ch) {
                    self.pos += cl;
                } else { break; }
            } else { break; }
        }
        while self.pos < self.len {
            let b = self.at(self.pos);
            if b == b'\n' || b == b'\r' { self.pos += 1; }
            else { break; }
        }
    }

    #[inline(always)]
    fn scan_whitespace_to_newline(&mut self) {
        let start = self.pos;
        let mut last_newline_end = 0usize;
        let mut prev_pos = self.pos;
        while self.pos < self.len {
            let c = self.at(self.pos);
            if c == b'\n' || c == b'\r' {
                prev_pos = self.pos;
                self.pos += 1;
                last_newline_end = self.pos;
            } else if c == b' ' || c == b'\t' {
                prev_pos = self.pos;
                self.pos += 1;
            } else if c >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_whitespace() { prev_pos = self.pos; self.pos += cl; } else { break; }
            } else {
                break;
            }
        }
        if last_newline_end > 0 {
            self.pos = last_newline_end;
        } else {
            if self.pos < self.len && prev_pos > start {
                self.pos = prev_pos;
            }
        }
    }

    #[inline(always)]
    fn emit(&self, start: usize) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) }
    }
}

impl<'a> Iterator for Qwen<'a> {
    type Item = &'a str;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let b = self.at(self.pos);

        if is_ascii_letter(b) {
            self.pos += 1;
            self.scan_letters_with_marks();
        } else if b == b'\'' {
            let clen = self.check_contraction();
            if clen > 0 {
                self.pos += clen;
            } else {
                if self.pos + 1 < self.len {
                    let next = self.at(self.pos + 1);
                    if is_ascii_letter(next) {
                        self.pos += 1;
                        self.pos += 1;
                        self.scan_letters_with_marks();
                    } else if next >= 0x80 {
                        let (ch, _) = decode_utf8(&self.bytes[self.pos + 1..]);
                        if ch.is_alphabetic() || is_unicode_mark(ch) {
                            self.pos += 1;
                            self.scan_letters_with_marks();
                        } else {
                            self.pos += 1;
                            self.scan_punct_with_newlines();
                        }
                    } else {
                        self.pos += 1;
                        self.scan_punct_with_newlines();
                    }
                } else {
                    self.pos += 1;
                }
            }
        } else if is_digit(b) {
            // Single digit
            self.pos += 1;
        } else if b == b' ' || b == b'\t' {
            if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    self.pos += 2;
                    self.scan_letters_with_marks();
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos + 1..]);
                    if ch.is_alphabetic() || is_unicode_mark(ch) {
                        self.pos += 1;
                        self.scan_letters_with_marks();
                    } else if ch.is_whitespace() || ch.is_numeric() {
                        self.scan_whitespace_to_newline();
                    } else {
                        self.pos += 1;
                        self.scan_punct_with_newlines();
                    }
                } else if Self::is_punct_char(next) || next == b'\'' {
                    self.pos += 1;
                    self.scan_punct_with_newlines();
                } else {
                    self.scan_whitespace_to_newline();
                }
            } else {
                self.pos += 1;
            }
        } else if b == b'\n' || b == b'\r' {
            // \s*[\r\n]+ — consume all consecutive newlines
            self.pos += 1;
            while self.pos < self.len {
                let c = self.at(self.pos);
                if c == b'\n' || c == b'\r' { self.pos += 1; }
                else { break; }
            }
        } else if b >= 0x80 {
            let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
            if ch.is_alphabetic() || is_unicode_mark(ch) {
                self.pos += cl;
                self.scan_letters_with_marks();
            } else if ch.is_numeric() {
                self.pos += cl;
            } else if ch.is_whitespace() {
                self.scan_whitespace_to_newline();
            } else {
                // Non-ASCII symbol: try prefix letters, else punct group
                self.pos += cl;
                if self.pos < self.len {
                    let next = self.at(self.pos);
                    if is_ascii_letter(next) {
                        self.pos += 1;
                        self.scan_letters_with_marks();
                    } else if next >= 0x80 {
                        let (ch2, _) = decode_utf8(&self.bytes[self.pos..]);
                        if ch2.is_alphabetic() || is_unicode_mark(ch2) {
                            self.scan_letters_with_marks();
                        } else if !ch2.is_whitespace() && !ch2.is_numeric() {
                            self.scan_punct_with_newlines();
                        }
                    } else if Self::is_punct_char(next) {
                        self.scan_punct_with_newlines();
                    }
                }
            }
        } else {
            // Other ASCII punct — can prefix letters
            if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    self.pos += 1;
                    self.pos += 1;
                    self.scan_letters_with_marks();
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos + 1..]);
                    if ch.is_alphabetic() || is_unicode_mark(ch) {
                        self.pos += 1;
                        self.scan_letters_with_marks();
                    } else {
                        self.pos += 1;
                        self.scan_punct_with_newlines();
                    }
                } else {
                    self.pos += 1;
                    self.scan_punct_with_newlines();
                }
            } else {
                self.pos += 1;
            }
        }

        debug_assert!(self.pos > start, "no progress at pos {start}");
        Some(self.emit(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<&str> {
        Qwen::new(text).collect()
    }

    #[test]
    fn basic() { assert_eq!(split("Hello world"), vec!["Hello", " world"]); }

    #[test]
    fn single_digits() { assert_eq!(split("12345"), vec!["1", "2", "3", "4", "5"]); }

    #[test]
    fn contractions() {
        assert_eq!(split("don't"), vec!["don", "'t"]);
        assert_eq!(split("DON'T"), vec!["DON", "'T"]);
    }

    #[test]
    fn marks_with_letters() {
        assert_eq!(split("ก\u{0E31}น"), vec!["ก\u{0E31}น"]);
    }

    #[test]
    fn punct_prefix() { assert_eq!(split("$hello"), vec!["$hello"]); }

    #[test]
    fn whitespace_newline() {
        assert_eq!(split("a\nb"), vec!["a", "\n", "b"]);
        assert_eq!(split("a \nb"), vec!["a", " \n", "b"]);
    }
}
