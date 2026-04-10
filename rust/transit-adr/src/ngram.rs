//! N-gram (bigram/trigram) operations for fast string matching.
//!
//! Provides compression, decompression, extraction, and splitting of character
//! n-grams. These are used to build inverted indices for cosine-similarity
//! based candidate filtering.

use crate::types::Ngram;

/// Bit width of each character in a compressed ngram. C++ `kNgramCharBitWidth`.
pub const NGRAM_CHAR_BIT_WIDTH: u32 = 8;

/// Maximum value of a single compressed character. C++ `kMaxNgram`.
pub const MAX_NGRAM: u8 = 0xFF;

/// First character of a compressed bigram. C++ `kNgram0`.
pub const NGRAM0: u32 = 0;

/// Second character of a compressed bigram. C++ `kNgram1`.
pub const NGRAM1: u32 = 1;

/// Total number of possible bigrams. C++ `kNBigrams = numeric_limits<ngram_t>::max()`.
pub const N_BIGRAMS: usize = u16::MAX as usize;

/// Compress two bytes into a single `u16` bigram.
#[inline]
pub fn compress_bigram(s: &str) -> Ngram {
    debug_assert!(s.len() >= 2);
    let bytes = s.as_bytes();
    (bytes[0] as Ngram) | ((bytes[1] as Ngram) << NGRAM_CHAR_BIT_WIDTH)
}

/// Decompress a `u16` bigram back into a two-character string.
#[inline]
pub fn decompress_bigram(t: Ngram) -> String {
    let c0 = (t & 0xFF) as u8 as char;
    let c1 = ((t >> NGRAM_CHAR_BIT_WIDTH) & 0xFF) as u8 as char;
    let mut s = String::with_capacity(2);
    s.push(c0);
    s.push(c1);
    s
}

/// Call `f` for each trigram (3-byte window) in `s`.
///
/// Slides a 3-byte window over the string.  Positions that do not fall on
/// UTF-8 char boundaries are silently skipped, matching the C++ byte-level
/// `std::string_view::substr` semantics for ASCII while staying safe on
/// multi-byte codepoints.
pub fn for_each_trigram<F: FnMut(&str)>(s: &str, mut f: F) {
    let len = s.len();
    if len < 3 {
        return;
    }
    for i in 0..len - 2 {
        if let Some(sub) = s.get(i..i + 3) {
            f(sub);
        }
    }
}

/// Call `f` for each bigram (2-byte window) in `s`, skipping spaces.
///
/// Same sliding-window approach as [`for_each_trigram`] but with a 2-byte
/// window and an additional check that neither byte is a space.
pub fn for_each_bigram<F: FnMut(&str)>(s: &str, mut f: F) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len < 2 {
        return;
    }
    for i in 0..len - 1 {
        if bytes[i] != b' ' && bytes[i + 1] != b' ' {
            if let Some(sub) = s.get(i..i + 2) {
                f(sub);
            }
        }
    }
}

/// Split a normalized string into sorted compressed bigrams.
///
/// Returns a vector of compressed bigrams (sorted) suitable for cosine
/// similarity computation. Limited to 128 bigrams.
pub fn split_ngrams(normalized: &str) -> Vec<Ngram> {
    let mut bigrams = Vec::with_capacity(128);
    for_each_bigram(normalized, |bigram| {
        if bigrams.len() < 128 {
            bigrams.push(compress_bigram(bigram));
        }
    });
    bigrams.sort_unstable();
    bigrams
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize;

    /// C++ test: `TEST(adr, for_each_trigram)`
    ///
    /// Input: "Landwehrstraße" → normalized → extract trigrams.
    #[test]
    fn for_each_trigram_landwehrstrasse() {
        let normalized = normalize("Landwehrstraße");
        let mut v: Vec<String> = Vec::new();
        for_each_trigram(&normalized, |s| v.push(s.to_string()));
        assert_eq!(
            v,
            vec![
                "lan", "and", "ndw", "dwe", "weh", "ehr", "hrs", "rst", "str",
                "tra", "ras", "ass", "sse"
            ]
        );
    }

    /// C++ test: `TEST(adr, for_each_bigram)`
    ///
    /// Input: "Landwehrstraße" → normalized → extract bigrams → compress →
    /// decompress round-trip.
    #[test]
    fn for_each_bigram_landwehrstrasse() {
        let normalized = normalize("Landwehrstraße");
        let mut compressed: Vec<Ngram> = Vec::new();
        for_each_bigram(&normalized, |s| {
            compressed.push(compress_bigram(s));
        });
        let decompressed: Vec<String> =
            compressed.iter().map(|&ng| decompress_bigram(ng)).collect();
        assert_eq!(
            decompressed,
            vec![
                "la", "an", "nd", "dw", "we", "eh", "hr", "rs", "st", "tr",
                "ra", "as", "ss", "se"
            ]
        );
    }

    #[test]
    fn compress_decompress_roundtrip() {
        let pairs = ["ab", "zz", "01", "az"];
        for &pair in &pairs {
            let compressed = compress_bigram(pair);
            let decompressed = decompress_bigram(compressed);
            assert_eq!(pair, decompressed);
        }
    }

    #[test]
    fn bigram_skips_spaces() {
        let mut bigrams = Vec::new();
        for_each_bigram("a b c", |s| bigrams.push(s.to_string()));
        // "a " skipped, " b" skipped, "b " skipped, " c" skipped
        assert!(bigrams.is_empty());
    }

    #[test]
    fn split_ngrams_sorted() {
        let ngrams = split_ngrams("hello");
        // bigrams: "he", "el", "ll", "lo"
        assert_eq!(ngrams.len(), 4);
        // Verify sorted
        for w in ngrams.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    // ── Unicode safety tests ──

    #[test]
    fn bigram_cjk_does_not_panic() {
        // CJK chars are 3 bytes each in UTF-8; byte-level windows will land
        // inside codepoints and be safely skipped.
        let mut bigrams = Vec::new();
        for_each_bigram("图文巴", |s| bigrams.push(s.to_string()));
        // Most 2-byte windows fall on non-char boundaries → skipped
        // Only windows that happen to align produce output
        assert!(bigrams.iter().all(|b| b.len() == 2));
    }

    #[test]
    fn trigram_cjk_does_not_panic() {
        let mut trigrams = Vec::new();
        for_each_trigram("图文巴", |s| trigrams.push(s.to_string()));
        assert!(trigrams.iter().all(|t| t.len() == 3));
    }

    #[test]
    fn bigram_mixed_ascii_cjk() {
        // "a图b" — 'a' is 1 byte, '图' is 3 bytes, 'b' is 1 byte → 5 bytes total
        // Byte windows: [0..2] [1..3] [2..4] [3..5]
        // Only windows on char boundaries are emitted.
        let mut bigrams = Vec::new();
        for_each_bigram("a图b", |s| bigrams.push(s.to_string()));
        // No panic; any emitted bigrams are valid 2-byte str slices
        for b in &bigrams {
            assert_eq!(b.len(), 2);
        }
    }

    #[test]
    fn split_ngrams_cjk_does_not_panic() {
        // Should not panic and should return a sorted vec
        let ngrams = split_ngrams("图文巴士站");
        for w in ngrams.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn bigram_emoji_does_not_panic() {
        // Emoji are 4 bytes each in UTF-8
        let mut bigrams = Vec::new();
        for_each_bigram("🚌🚏🗺️", |s| bigrams.push(s.to_string()));
        assert!(bigrams.iter().all(|b| b.len() == 2));
    }

    #[test]
    fn bigram_two_byte_chars() {
        // Cyrillic chars are 2 bytes each — byte windows align on every
        // other position, so we get valid bigrams for each pair.
        let mut bigrams = Vec::new();
        for_each_bigram("мо", |s| bigrams.push(s.to_string()));
        // "мо" is 4 bytes: [0xd0 0xbc 0xd0 0xbe]
        // Windows: [0..2]="м", [1..3]=invalid, [2..4]="о"
        // Only the char-boundary windows survive
        assert!(!bigrams.is_empty());
        for b in &bigrams {
            assert_eq!(b.len(), 2);
        }
    }

    #[test]
    fn normalized_diacritics_produce_ascii_bigrams() {
        // After normalization, diacritics are stripped → pure ASCII → all
        // byte windows align. Verifies the happy path still works.
        let normalized = normalize("Zürich");
        let mut bigrams = Vec::new();
        for_each_bigram(&normalized, |s| bigrams.push(s.to_string()));
        assert_eq!(bigrams, vec!["zu", "ur", "ri", "ic", "ch"]);
    }
}
