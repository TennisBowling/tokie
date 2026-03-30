//! O200K pretokenizer — single-pass, zero allocation.
//!
//! Pattern: `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|...`
//!
//! Like CL100K but with:
//! - CamelCase splitting (case-aware letter scanning)
//! - Contractions as suffixes (merge into word: "don't" → ["don't"])
//! - Punct trailing slashes
//! - Case-insensitive contractions

use crate::util::{decode_utf8, is_ascii_letter, is_digit, is_unicode_letter, is_unicode_mark};

#[inline(always)]
fn is_lower(b: u8) -> bool { b.wrapping_sub(b'a') < 26 }

#[inline(always)]
fn is_upper(b: u8) -> bool { b.wrapping_sub(b'A') < 26 }

pub struct O200k<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> O200k<'a> {
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len() }
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> u8 {
        unsafe { *self.bytes.get_unchecked(pos) }
    }

    /// Case-aware letter scanning for CamelCase.
    /// First byte already consumed. Checks if it was lower or upper and dispatches.
    #[inline(always)]
    fn scan_letters_case_aware(&mut self, first: u8) {
        if first < 0x80 {
            if is_lower(first) {
                self.scan_lowercase();
            } else {
                self.scan_upper_then_lower();
            }
        } else {
            // Unicode: check case
            let start = self.pos - 1;
            let (ch, _) = decode_utf8(&self.bytes[start..]);
            if ch.is_lowercase() {
                self.scan_lowercase();
            } else {
                self.scan_upper_then_lower();
            }
        }
    }

    /// Scan lowercase letters and marks: [\p{Ll}\p{Lm}\p{Lo}\p{M}]+
    #[inline(always)]
    fn scan_lowercase(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_lower(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                // Ll, Lm, Lo, or M category
                if is_unicode_mark(ch) || ch.is_lowercase()
                    || (is_unicode_letter(ch) && !ch.is_uppercase())
                {
                    self.pos += cl;
                } else { return; }
            } else { return; }
        }
    }

    /// Scan uppercase/titlecase/modifier/other letters and marks, then lowercase+marks.
    #[inline(always)]
    fn scan_upper_then_lower(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_upper(b) {
                self.pos += 1;
            } else if is_lower(b) {
                self.pos += 1;
                self.scan_lowercase();
                return;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_uppercase() {
                    self.pos += cl;
                } else if ch.is_lowercase() || is_unicode_mark(ch) {
                    self.pos += cl;
                    self.scan_lowercase();
                    return;
                } else if is_unicode_letter(ch) {
                    // Lm, Lo — continue as "other"
                    self.pos += cl;
                } else { return; }
            } else { return; }
        }
    }

    /// Check for case-insensitive contraction suffix ('s, 't, 'd, 'm, 'll, 're, 've).
    /// Returns bytes to consume (0 if no contraction).
    #[inline(always)]
    fn check_contraction_suffix(&self) -> usize {
        if self.pos >= self.len || self.bytes[self.pos] != b'\'' { return 0; }
        let rem = self.len - self.pos;
        if rem < 2 { return 0; }
        let b1 = self.bytes[self.pos + 1] | 0x20;
        if matches!(b1, b's' | b't' | b'd' | b'm') { return 2; }
        if rem < 3 { return 0; }
        let b2 = self.bytes[self.pos + 2] | 0x20;
        if (b1 == b'l' && b2 == b'l') || (b1 == b'v' && b2 == b'e') || (b1 == b'r' && b2 == b'e') {
            return 3;
        }
        0
    }

    /// Scan up to 3 digits.
    #[inline(always)]
    fn scan_digits_chunked(&mut self) {
        let mut count = 0u8;
        while self.pos < self.len && count < 3 && is_digit(self.at(self.pos)) {
            self.pos += 1;
            count += 1;
        }
    }

    #[inline(always)]
    fn is_punct_char(b: u8) -> bool {
        !is_ascii_letter(b) && !is_digit(b) && b != b' ' && b != b'\t' && b != b'\n' && b != b'\r' && b < 0x80
    }

    /// Scan punct characters + trailing newlines/slashes.
    #[inline(always)]
    fn scan_punct(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if Self::is_punct_char(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if !is_unicode_letter(ch) && !is_unicode_mark(ch) && !ch.is_numeric() && !ch.is_whitespace() {
                    self.pos += cl;
                } else { break; }
            } else { break; }
        }
        // Trailing newlines + slashes
        while self.pos < self.len {
            let b = self.at(self.pos);
            if b == b'\n' || b == b'\r' || b == b'/' { self.pos += 1; }
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

impl<'a> Iterator for O200k<'a> {
    type Item = &'a str;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let b = self.at(self.pos);

        if is_ascii_letter(b) {
            self.pos += 1;
            self.scan_letters_case_aware(b);
            // Try contraction suffix
            let clen = self.check_contraction_suffix();
            if clen > 0 { self.pos += clen; }
        } else if b == b'\'' {
            // Apostrophe: can be a standalone contraction or prefix for letters
            // In O200K, contractions are suffixes, so a standalone 's at start is just punct+letters
            if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    // Apostrophe as prefix for letters
                    self.pos += 1;
                    self.pos += 1;
                    self.scan_letters_case_aware(next);
                    let clen = self.check_contraction_suffix();
                    if clen > 0 { self.pos += clen; }
                } else if next >= 0x80 {
                    let (ch, cl) = decode_utf8(&self.bytes[self.pos + 1..]);
                    if is_unicode_letter(ch) || is_unicode_mark(ch) {
                        self.pos += 1; // skip apostrophe prefix
                        self.pos += cl; // consume first unicode char
                        if ch.is_lowercase() || is_unicode_mark(ch) {
                            self.scan_lowercase();
                        } else {
                            self.scan_upper_then_lower();
                        }
                        let clen = self.check_contraction_suffix();
                        if clen > 0 { self.pos += clen; }
                    } else {
                        self.pos += 1;
                        self.scan_punct();
                    }
                } else {
                    self.pos += 1;
                    self.scan_punct();
                }
            } else {
                self.pos += 1;
            }
        } else if is_digit(b) {
            self.scan_digits_chunked();
        } else if b == b' ' || b == b'\t' {
            if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    self.pos += 2;
                    self.scan_letters_case_aware(next);
                    let clen = self.check_contraction_suffix();
                    if clen > 0 { self.pos += clen; }
                } else if next >= 0x80 {
                    let (ch, cl) = decode_utf8(&self.bytes[self.pos + 1..]);
                    if is_unicode_letter(ch) || is_unicode_mark(ch) {
                        self.pos += 1; // consume space prefix
                        self.pos += cl; // consume first unicode char
                        if ch.is_lowercase() || is_unicode_mark(ch) {
                            self.scan_lowercase();
                        } else {
                            self.scan_upper_then_lower();
                        }
                        let clen = self.check_contraction_suffix();
                        if clen > 0 { self.pos += clen; }
                    } else if ch.is_whitespace() || ch.is_numeric() {
                        self.scan_whitespace_to_newline();
                    } else {
                        self.pos += 1;
                        self.scan_punct();
                    }
                } else if Self::is_punct_char(next) || next == b'\'' {
                    self.pos += 1;
                    self.scan_punct();
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
            if is_unicode_letter(ch) {
                self.pos += cl;
                if ch.is_lowercase() { self.scan_lowercase(); } else { self.scan_upper_then_lower(); }
                let clen = self.check_contraction_suffix();
                if clen > 0 { self.pos += clen; }
            } else if is_unicode_mark(ch) {
                // Mark as start — scan as lowercase
                self.pos += cl;
                self.scan_lowercase();
                let clen = self.check_contraction_suffix();
                if clen > 0 { self.pos += clen; }
            } else if ch.is_numeric() {
                self.pos += cl;
                self.scan_digits_chunked();
            } else if ch.is_whitespace() {
                self.scan_whitespace_to_newline();
            } else {
                // Non-ASCII symbol: try prefix letters, else punct group
                self.pos += cl;
                if self.pos < self.len {
                    let next = self.at(self.pos);
                    if is_ascii_letter(next) {
                        self.pos += 1;
                        self.scan_letters_case_aware(next);
                        let clen = self.check_contraction_suffix();
                        if clen > 0 { self.pos += clen; }
                    } else if next >= 0x80 {
                        let (ch2, _) = decode_utf8(&self.bytes[self.pos..]);
                        if is_unicode_letter(ch2) || is_unicode_mark(ch2) {
                            self.scan_lowercase();
                            let clen = self.check_contraction_suffix();
                            if clen > 0 { self.pos += clen; }
                        } else if !ch2.is_whitespace() && !ch2.is_numeric() {
                            self.scan_punct();
                        }
                    } else if Self::is_punct_char(next) {
                        self.scan_punct();
                    }
                }
            }
        } else {
            // Other ASCII punct — can prefix letters only
            if self.pos + 1 < self.len {
                let next = self.at(self.pos + 1);
                if is_ascii_letter(next) {
                    self.pos += 1;
                    self.pos += 1;
                    self.scan_letters_case_aware(next);
                    let clen = self.check_contraction_suffix();
                    if clen > 0 { self.pos += clen; }
                } else if next >= 0x80 {
                    let (ch, cl) = decode_utf8(&self.bytes[self.pos + 1..]);
                    if is_unicode_letter(ch) || is_unicode_mark(ch) {
                        self.pos += 1; // skip punct prefix
                        self.pos += cl; // consume first unicode char
                        if ch.is_lowercase() || is_unicode_mark(ch) {
                            self.scan_lowercase();
                        } else {
                            self.scan_upper_then_lower();
                        }
                        let clen = self.check_contraction_suffix();
                        if clen > 0 { self.pos += clen; }
                    } else {
                        self.pos += 1;
                        self.scan_punct();
                    }
                } else {
                    self.pos += 1;
                    self.scan_punct();
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
        O200k::new(text).collect()
    }

    #[test]
    fn basic() {
        assert_eq!(split("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn contractions_suffix() {
        assert_eq!(split("don't"), vec!["don't"]);
        assert_eq!(split("DON'T"), vec!["DON'T"]);
        assert_eq!(split("I'm"), vec!["I'm"]);
        assert_eq!(split("we'll"), vec!["we'll"]);
        assert_eq!(split("they're"), vec!["they're"]);
        assert_eq!(split("you've"), vec!["you've"]);
        assert_eq!(split("she'd"), vec!["she'd"]);
    }

    #[test]
    fn camelcase() {
        assert_eq!(split("CamelCase"), vec!["Camel", "Case"]);
        assert_eq!(split("JSONParser"), vec!["JSONParser"]);
        assert_eq!(split("parseJSON"), vec!["parse", "JSON"]);
        assert_eq!(split("XMLHttpRequest"), vec!["XMLHttp", "Request"]);
    }

    #[test]
    fn numbers() {
        assert_eq!(split("12345"), vec!["123", "45"]);
        assert_eq!(split("test 123"), vec!["test", " ", "123"]);
    }

    #[test]
    fn apostrophe_prefix() {
        assert_eq!(split("'hello'"), vec!["'hello", "'"]);
    }
}
