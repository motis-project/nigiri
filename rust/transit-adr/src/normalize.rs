//! UTF-8 text normalization and phrase generation.
//!
//! Provides Unicode decomposition with diacritic stripping and case folding,
//! filler character removal, multi-token phrase generation with alternative
//! forms (e.g. "hbf" ↔ "hauptbahnhof", "str" ↔ "strasse").
//!
//! Abbreviation mappings are configurable via [`Abbreviations`]. The hardcoded
//! German defaults mirror the original C++ ADR implementation.

use crate::types::{TokenBitmask, MAX_INPUT_PHRASES, MAX_TOKENS};
use std::collections::HashMap;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;

/// Configurable abbreviation mappings for geocoder text matching.
///
/// Contains two kinds of substitution:
/// - **Exact**: full-token replacements (e.g. "hbf" ↔ "hauptbahnhof")
/// - **Postfix**: suffix-based replacements (e.g. "str" → "strasse")
///
/// Used by [`get_phrases_with_abbreviations`] and the geocoder's guess-string
/// builder to generate alternative search phrases.
#[derive(Debug, Clone, Default)]
pub struct Abbreviations {
    /// Exact token → replacement (e.g. "hbf" → "hauptbahnhof").
    /// Each pair is unidirectional; add both directions for bidirectional mapping.
    pub exact: HashMap<String, String>,
    /// Postfix suffix → replacement (e.g. "str" → "strasse").
    /// Matched against token suffixes. Each pair is unidirectional.
    pub postfix: Vec<(String, String)>,
}

impl Abbreviations {
    /// Create with the hardcoded German defaults matching the C++ ADR implementation.
    pub fn german_defaults() -> Self {
        let mut exact = HashMap::new();
        exact.insert("hbf".into(), "hauptbahnhof".into());
        exact.insert("hauptbahnhof".into(), "hbf".into());
        exact.insert("hauptbf".into(), "hbf".into());
        exact.insert("bahnhof".into(), "bhf".into());
        exact.insert("bhf".into(), "bahnhof".into());

        let postfix = vec![
            ("str".into(), "strasse".into()),
            ("str.".into(), "strasse".into()),
            ("strasse".into(), "str.".into()),
        ];

        Self { exact, postfix }
    }

    /// Get the exact alternative for a token, if one exists.
    pub fn get_exact_alt(&self, s: &str) -> Option<&str> {
        self.exact.get(s).map(|v| v.as_str())
    }

    /// Get the postfix-based alternative for a string, if one exists.
    pub fn get_postfix_alt(&self, s: &str) -> Option<(&str, &str)> {
        for (suffix, replacement) in &self.postfix {
            if s.ends_with(suffix.as_str()) {
                return Some((suffix.as_str(), replacement.as_str()));
            }
        }
        None
    }

    /// Get the alternative form of a string (exact match or postfix substitution).
    pub fn get_alt_string(&self, s: &str) -> Option<String> {
        if let Some(alt) = self.get_exact_alt(s) {
            return Some(alt.to_string());
        }
        if let Some((postfix, replacement)) = self.get_postfix_alt(s) {
            let prefix = &s[..s.len() - postfix.len()];
            return Some(format!("{prefix}{replacement}"));
        }
        None
    }
}

/// A phrase built from one or more input tokens, with a bitmask tracking
/// which tokens are included.
#[derive(Debug, Clone, PartialEq)]
pub struct Phrase {
    /// Bitmask indicating which input tokens are represented.
    pub token_bits: TokenBitmask,
    /// The combined, normalized phrase string.
    pub s: String,
}

/// Normalize a UTF-8 string: decompose, strip diacritics, case-fold.
///
/// Mirrors the C++ `normalize()` which uses `utf8proc` with
/// `DECOMPOSE | STRIPMARK | CASEFOLD`. Unicode case folding maps ß → ss,
/// which differs from simple lowercasing.
pub fn normalize(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for c in input.nfd() {
        match get_general_category(c) {
            GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark => continue,
            _ => {}
        }
        // Unicode case folding (CaseFolding.txt status C+F)
        match c {
            'ß' => result.push_str("ss"),
            'ẞ' => result.push_str("ss"),
            'ﬀ' => result.push_str("ff"),
            'ﬁ' => result.push_str("fi"),
            'ﬂ' => result.push_str("fl"),
            'ﬃ' => result.push_str("ffi"),
            'ﬄ' => result.push_str("ffl"),
            'ﬅ' => result.push_str("st"),
            'ﬆ' => result.push_str("st"),
            other => {
                for lc in other.to_lowercase() {
                    result.push(lc);
                }
            }
        }
    }
    result
}

/// Replace filler characters (`,;-/().`) with spaces, collapse consecutive
/// spaces, and trim leading/trailing whitespace.
pub fn erase_fillers(s: &mut String) {
    // Replace filler chars with spaces
    unsafe {
        for b in s.as_bytes_mut() {
            if matches!(*b, b',' | b';' | b'-' | b'/' | b'(' | b')' | b'.') {
                *b = b' ';
            }
        }
    }

    // Collapse consecutive spaces
    let bytes = s.as_bytes().to_vec();
    let mut result = Vec::with_capacity(bytes.len());
    let mut prev_space = false;
    for &b in &bytes {
        if b == b' ' {
            if !prev_space {
                result.push(b);
            }
            prev_space = true;
        } else {
            result.push(b);
            prev_space = false;
        }
    }

    // Trim trailing/leading spaces
    while result.last() == Some(&b' ') {
        result.pop();
    }
    while result.first() == Some(&b' ') {
        result.remove(0);
    }

    *s = String::from_utf8(result).expect("filler erasure preserves UTF-8");
}

/// Convert a token bitmask to its binary string representation (LSB first).
///
/// Example: `0b00000011` → `"11000000"`.
pub fn bit_mask_to_str(b: TokenBitmask) -> String {
    let mut r = String::with_capacity(MAX_TOKENS);
    for i in 0..MAX_TOKENS {
        r.push(if ((b >> i) & 0x1) == 0x1 { '1' } else { '0' });
    }
    r
}

/// Get the postfix-based alternative for a string, if one exists.
///
/// Uses the hardcoded German defaults. For configurable abbreviations,
/// use [`Abbreviations::get_postfix_alt`].
fn get_postfix_alt_string(s: &str) -> Option<(&str, &str)> {
    const POSTFIX_MAPPING: &[(&str, &str)] = &[
        ("str", "strasse"),
        ("str.", "strasse"),
        ("strasse", "str."),
    ];
    for &(postfix, replacement) in POSTFIX_MAPPING {
        if s.ends_with(postfix) {
            return Some((postfix, replacement));
        }
    }
    None
}

/// Get the exact alternative for a known abbreviation.
///
/// Uses the hardcoded German defaults. For configurable abbreviations,
/// use [`Abbreviations::get_exact_alt`].
pub fn get_exact_alt(s: &str) -> Option<&'static str> {
    match s {
        "hbf" => Some("hauptbahnhof"),
        "hauptbahnhof" => Some("hbf"),
        "hauptbf" => Some("hbf"),
        "bahnhof" => Some("bhf"),
        "bhf" => Some("bahnhof"),
        _ => None,
    }
}

/// Get the alternative form of a string (exact match or postfix substitution).
///
/// Uses the hardcoded German defaults. For configurable abbreviations,
/// use [`Abbreviations::get_alt_string`].
pub fn get_alt_string(s: &str) -> Option<String> {
    if let Some(alt) = get_exact_alt(s) {
        return Some(alt.to_string());
    }
    if let Some((postfix, replacement)) = get_postfix_alt_string(s) {
        let prefix = &s[..s.len() - postfix.len()];
        return Some(format!("{prefix}{replacement}"));
    }
    None
}

/// Generate phrase combinations using the hardcoded German abbreviation defaults.
///
/// For configurable abbreviations, use [`get_phrases_with_abbreviations`].
pub fn get_phrases(in_tokens: &[&str]) -> Vec<Phrase> {
    get_phrases_inner(in_tokens, |s| get_alt_string(s))
}

/// Generate phrase combinations using configurable abbreviations.
pub fn get_phrases_with_abbreviations(in_tokens: &[&str], abbrevs: &Abbreviations) -> Vec<Phrase> {
    get_phrases_inner(in_tokens, |s| abbrevs.get_alt_string(s))
}

/// Generate all phrase combinations from input tokens (up to length 4).
///
/// For each contiguous subsequence of tokens, builds the concatenated phrase
/// and, if an alternative form exists for any token, also generates the
/// alternative phrase. Results are sorted by descending length and limited
/// to [`MAX_INPUT_PHRASES`].
fn get_phrases_inner(in_tokens: &[&str], alt_fn: impl Fn(&str) -> Option<String>) -> Vec<Phrase> {
    let mut r: Vec<Phrase> = Vec::new();

    for from in 0..in_tokens.len() {
        let max_len = std::cmp::min(in_tokens.len() - from, 4);
        for length in 1..=max_len {
            let mut phrases = vec![Phrase {
                token_bits: 0,
                s: String::new(),
            }];

            for to in from..from + length {
                if to >= in_tokens.len() {
                    break;
                }

                let alt = alt_fn(in_tokens[to]);
                let mut alt_phrases: Vec<Phrase> = Vec::new();

                if let Some(ref alt_str) = alt {
                    alt_phrases = phrases.clone();
                    for p in &mut alt_phrases {
                        p.token_bits |= 1 << to;
                        if to != from {
                            p.s.push(' ');
                        }
                        p.s.push_str(alt_str);
                    }
                }

                for p in &mut phrases {
                    p.token_bits |= 1 << to;
                    if to != from {
                        p.s.push(' ');
                    }
                    p.s.push_str(in_tokens[to]);
                }

                phrases.extend(alt_phrases);
            }

            r.extend(phrases);
        }
    }

    // Sort by descending string length (matching C++ utl::sort)
    r.sort_by(|a, b| b.s.len().cmp(&a.s.len()));
    r.truncate(MAX_INPUT_PHRASES);
    r
}

/// Compute a bitmask of which tokens are predominantly numeric.
///
/// A token is considered numeric if at least half its characters are digits.
pub fn get_numeric_tokens_mask(tokens: &[&str]) -> u8 {
    let mut mask: u8 = 0;
    for (i, token) in tokens.iter().enumerate() {
        let digit_count = token.bytes().filter(|b| b.is_ascii_digit()).count();
        if digit_count > 0 && digit_count >= token.len().div_ceil(2) {
            mask |= 1 << i;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basic() {
        assert_eq!(normalize("Darmstadt"), "darmstadt");
    }

    #[test]
    fn normalize_diacritics() {
        // ß decomposes but has no combining mark → stays as ß
        // ä → a (combining diaeresis stripped)
        let result = normalize("Landwehrstraße");
        assert_eq!(result, "landwehrstrasse");
    }

    #[test]
    fn normalize_french_diacritics() {
        assert_eq!(normalize("Médecin"), "medecin");
        assert_eq!(normalize("Pyrénées"), "pyrenees");
    }

    #[test]
    fn erase_fillers_basic() {
        let mut s = "hello,world-test".to_string();
        erase_fillers(&mut s);
        assert_eq!(s, "hello world test");
    }

    #[test]
    fn erase_fillers_collapse_spaces() {
        let mut s = "a  -  b".to_string();
        erase_fillers(&mut s);
        assert_eq!(s, "a b");
    }

    #[test]
    fn erase_fillers_trim() {
        let mut s = " -hello- ".to_string();
        erase_fillers(&mut s);
        assert_eq!(s, "hello");
    }

    #[test]
    fn bit_mask_to_str_examples() {
        assert_eq!(bit_mask_to_str(0b00000001), "10000000");
        assert_eq!(bit_mask_to_str(0b00000011), "11000000");
        assert_eq!(bit_mask_to_str(0b00000111), "11100000");
    }

    /// C++ test: `TEST(adr, phrase)`
    ///
    /// 6 input tokens → generates phrase combinations with correct bitmasks.
    /// Output is sorted by descending phrase length (matching C++ `utl::sort`).
    /// Includes 4-token phrases per C++ `length != 5U`.
    #[test]
    fn phrase_generation() {
        let phrases = get_phrases(&[
            "willy", "brandt", "platz", "abert", "ainstein", "illme",
        ]);

        // Verify sorted by descending length
        for w in phrases.windows(2) {
            assert!(
                w[0].s.len() >= w[1].s.len(),
                "not sorted: '{}' ({}) before '{}' ({})",
                w[0].s,
                w[0].s.len(),
                w[1].s,
                w[1].s.len()
            );
        }

        // Verify all C++ expected phrases are present (content + bitmask)
        let expected_present: Vec<(&str, &str)> = vec![
            ("willy", "10000000"),
            ("willy brandt", "11000000"),
            ("willy brandt platz", "11100000"),
            ("willy brandt platz abert", "11110000"),
            ("brandt", "01000000"),
            ("brandt platz", "01100000"),
            ("brandt platz abert", "01110000"),
            ("brandt platz abert ainstein", "01111000"),
            ("platz", "00100000"),
            ("platz abert", "00110000"),
            ("platz abert ainstein", "00111000"),
            ("platz abert ainstein illme", "00111100"),
            ("abert", "00010000"),
            ("abert ainstein", "00011000"),
            ("abert ainstein illme", "00011100"),
            ("ainstein", "00001000"),
            ("ainstein illme", "00001100"),
            ("illme", "00000100"),
        ];

        assert_eq!(
            phrases.len(),
            expected_present.len(),
            "phrase count mismatch"
        );

        for (exp_s, exp_mask) in &expected_present {
            let found = phrases.iter().any(|p| {
                p.s == *exp_s && bit_mask_to_str(p.token_bits) == *exp_mask
            });
            assert!(found, "missing phrase: ({exp_s}, {exp_mask})");
        }
    }

    /// C++ test: `TEST(adr, alt_string)`
    ///
    /// Phrase generation with alternative forms (hauptbahnhof → hbf).
    #[test]
    fn alt_string_phrases() {
        let phrases =
            get_phrases(&["hauptbahnhof", "darmstadt", "abc"]);

        let expected: Vec<(&str, &str)> = vec![
            ("hauptbahnhof darmstadt abc", "11100000"),
            ("hauptbahnhof darmstadt", "11000000"),
            ("hbf darmstadt abc", "11100000"),
            ("hbf darmstadt", "11000000"),
            ("darmstadt abc", "01100000"),
            ("hauptbahnhof", "10000000"),
            ("darmstadt", "01000000"),
            ("hbf", "10000000"),
            ("abc", "00100000"),
        ];

        assert_eq!(
            phrases.len(),
            expected.len(),
            "phrase count mismatch: got {}, expected {}",
            phrases.len(),
            expected.len()
        );

        for (i, (phrase, expected_pair)) in
            phrases.iter().zip(expected.iter()).enumerate()
        {
            let actual_mask = bit_mask_to_str(phrase.token_bits);
            assert_eq!(
                (phrase.s.as_str(), actual_mask.as_str()),
                (expected_pair.0, expected_pair.1),
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn get_alt_string_hbf() {
        assert_eq!(get_alt_string("hbf"), Some("hauptbahnhof".to_string()));
        assert_eq!(get_alt_string("hauptbahnhof"), Some("hbf".to_string()));
        assert_eq!(get_alt_string("hauptbf"), Some("hbf".to_string()));
    }

    #[test]
    fn get_alt_string_postfix() {
        assert_eq!(
            get_alt_string("landwehrstr"),
            Some("landwehrstrasse".to_string())
        );
        assert_eq!(
            get_alt_string("strasse"),
            Some("str.".to_string())
        );
    }

    #[test]
    fn get_alt_string_none() {
        assert_eq!(get_alt_string("darmstadt"), None);
        assert_eq!(get_alt_string("hello"), None);
    }

    #[test]
    fn numeric_tokens_mask() {
        let mask =
            get_numeric_tokens_mask(&["abc", "98", "9a", "0aa"]);
        // "abc" -> 0 digits out of 3 -> not numeric
        // "98"  -> 2 digits out of 2 -> numeric (bit 1)
        // "9a"  -> 1 digit out of 2  -> numeric (1 >= (2+1)/2 = 1) (bit 2)
        // "0aa" -> 1 digit out of 3  -> not numeric (1 < (3+1)/2 = 2)
        assert_eq!(bit_mask_to_str(mask), "01100000");
    }
}
