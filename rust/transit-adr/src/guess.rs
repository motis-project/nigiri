//! Suggestion engine for geocoding.
//!
//! Implements the core bigram cosine-similarity candidate filtering and
//! SIFT4-based scoring pipeline, mirroring the C++ `adr::get_suggestions()`
//! / `typeahead::guess()` algorithm.

use crate::ngram::{compress_bigram, for_each_bigram, split_ngrams};
use crate::normalize::{normalize, Phrase};
use crate::score::get_match_score;
use crate::sift4::SiftOffset;
use crate::types::{Score, NO_MATCH};

/// Minimum cosine similarity threshold for bigram candidate filtering.
/// Matches the C++ constant (0.17).
const COS_SIM_THRESHOLD: f32 = 0.17;

/// A string entry in the index with its precomputed bigram count.
#[derive(Debug, Clone)]
pub struct StringEntry {
    /// The raw string value (normalized).
    pub value: String,
    /// Number of bigrams in this string.
    pub n_bigrams: u8,
}

/// A candidate found via cosine similarity on bigrams.
#[derive(Debug, Clone)]
pub struct CosSimilarityMatch {
    /// Index into the string database.
    pub string_idx: usize,
    /// Cosine similarity score (higher = more similar).
    pub cos_sim: f32,
}

/// A scored suggestion with location information.
#[derive(Debug, Clone)]
pub struct ScoredSuggestion {
    /// The matched string.
    pub name: String,
    /// ADR score (lower = better match). Negate or invert for ranking.
    pub score: Score,
    /// Index of the matched phrase from the input.
    pub phrase_idx: usize,
    /// Original index in the location database (place or address).
    pub location_idx: usize,
    /// Whether this is a place ("place") or address ("address").
    pub location_type: SuggestionLocationType,
}

/// Location type for a suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionLocationType {
    Place,
    Address,
}

/// Reusable scratch space for the suggestion engine, avoiding repeated allocations.
pub struct GuessContext {
    pub sift4_offsets: Vec<SiftOffset>,
    pub string_match_counts: Vec<u8>,
    pub cos_sim_matches: Vec<CosSimilarityMatch>,
}

impl GuessContext {
    pub fn new() -> Self {
        Self {
            sift4_offsets: Vec::new(),
            string_match_counts: Vec::new(),
            cos_sim_matches: Vec::new(),
        }
    }

    /// Resize internal buffers to match the given number of strings.
    pub fn resize(&mut self, n_strings: usize) {
        self.string_match_counts.resize(n_strings, 0);
        self.cos_sim_matches.clear();
    }
}

impl Default for GuessContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Inverted bigram index: maps each bigram (u16) to a list of string indices.
/// Mirrors C++ `typeahead::bigrams_`.
pub struct BigramIndex {
    /// For each bigram, the list of string indices containing that bigram.
    entries: Vec<Vec<usize>>,
    /// The number of bigrams for each string (mirrors `n_bigrams_`).
    n_bigrams: Vec<u8>,
}

impl BigramIndex {
    /// Build the bigram index from a list of normalized strings.
    pub fn build(strings: &[String]) -> Self {
        let capacity = u16::MAX as usize + 1;
        let mut entries: Vec<Vec<usize>> = vec![Vec::new(); capacity];
        let mut n_bigrams: Vec<u8> = vec![0; strings.len()];

        for (str_idx, s) in strings.iter().enumerate() {
            let normalized = normalize(s);
            // C++ stores min(255, normalized.size() - 1) as the denominator
            // for cosine similarity, NOT the actual bigram count.
            n_bigrams[str_idx] =
                (normalized.len().saturating_sub(1)).min(255) as u8;
            let mut count: u8 = 0;
            for_each_bigram(&normalized, |bigram| {
                if count < 128 {
                    let key = compress_bigram(bigram) as usize;
                    let list = &mut entries[key];
                    // Avoid duplicate entries for same string
                    if list.last() != Some(&str_idx) {
                        list.push(str_idx);
                    }
                    count = count.saturating_add(1);
                }
            });
        }

        Self { entries, n_bigrams }
    }

    /// Find candidate strings by bigram cosine similarity.
    ///
    /// This mirrors C++ `typeahead::guess()`:
    /// 1. Extract bigrams from the normalized input
    /// 2. Count shared bigrams for each indexed string
    /// 3. Compute cosine similarity: `match_count² / (n_input_bigrams × n_string_bigrams)`
    /// 4. Return candidates above the threshold, sorted by similarity (descending)
    pub fn find_candidates<'a>(
        &self,
        normalized_input: &str,
        ctx: &'a mut GuessContext,
    ) -> &'a [CosSimilarityMatch] {
        ctx.cos_sim_matches.clear();

        let input_bigrams = split_ngrams(normalized_input);
        let n_input_bigrams = input_bigrams.len();
        if n_input_bigrams == 0 {
            return &ctx.cos_sim_matches;
        }

        // Reset match counts
        for c in ctx.string_match_counts.iter_mut() {
            *c = 0;
        }

        // Minimum match count threshold (from C++):
        // min_match_count = 2 + n_input_ngrams / (4 + n_input_ngrams / 10)
        let min_match_count = 2 + n_input_bigrams / (4 + n_input_bigrams / 10);

        // Count shared bigrams for each string
        for &bigram in &input_bigrams {
            let key = bigram as usize;
            if let Some(string_indices) = self.entries.get(key) {
                for &str_idx in string_indices {
                    if let Some(count) = ctx.string_match_counts.get_mut(str_idx) {
                        *count = count.saturating_add(1);
                    }
                }
            }
        }

        // Compute cosine similarity and filter
        let n_in = n_input_bigrams as f32;
        for (str_idx, &match_count) in ctx.string_match_counts.iter().enumerate() {
            if (match_count as usize) < min_match_count {
                continue;
            }
            let n_str = self.n_bigrams.get(str_idx).copied().unwrap_or(0) as f32;
            if n_str == 0.0 {
                continue;
            }
            // Cosine similarity: match_count² / (n_input × n_string)
            let cos_sim = (match_count as f32 * match_count as f32) / (n_in * n_str);
            if cos_sim >= COS_SIM_THRESHOLD {
                ctx.cos_sim_matches.push(CosSimilarityMatch {
                    string_idx: str_idx,
                    cos_sim,
                });
            }
        }

        // Sort by cosine similarity descending
        ctx.cos_sim_matches
            .sort_unstable_by(|a, b| b.cos_sim.partial_cmp(&a.cos_sim).unwrap_or(std::cmp::Ordering::Equal));

        &ctx.cos_sim_matches
    }
}

/// Score a list of cosine-similarity candidates against input phrases using SIFT4.
///
/// For each candidate string, computes `get_match_score()` against each input phrase
/// and returns the best (lowest) score per candidate.
///
/// This mirrors the C++ `compute_string_phrase_match_scores()` function.
pub fn score_candidates(
    candidates: &[CosSimilarityMatch],
    strings: &[String],
    phrases: &[Phrase],
    sift4_offsets: &mut Vec<SiftOffset>,
) -> Vec<(usize, Score, usize)> {
    // Returns: (string_idx, best_score, best_phrase_idx)
    let mut results: Vec<(usize, Score, usize)> = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        let s = match strings.get(candidate.string_idx) {
            Some(s) => s.as_str(),
            None => continue,
        };

        let mut best_score = NO_MATCH;
        let mut best_phrase_idx = 0usize;

        for (phrase_idx, phrase) in phrases.iter().enumerate() {
            let score = get_match_score(s, &phrase.s, sift4_offsets);
            if score < best_score {
                best_score = score;
                best_phrase_idx = phrase_idx;
            }
        }

        if best_score < NO_MATCH {
            results.push((candidate.string_idx, best_score, best_phrase_idx));
        }
    }

    // Sort by score ascending (lower = better)
    results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Mapping from string index to location indices.
///
/// In the C++ typeahead, `string_to_location_` maps each string to the places/streets
/// that use it. This provides the same mapping for the Rust geocoder.
pub struct StringLocationMap {
    /// For each string index: list of (location_type, location_idx)
    entries: Vec<Vec<(SuggestionLocationType, usize)>>,
}

impl StringLocationMap {
    /// Build the string→location mapping.
    ///
    /// - `n_strings`: Total number of strings in the pool.
    /// - `place_name_indices`: For each place, the list of string indices it uses.
    /// - `address_street_indices`: For each address, its street string index.
    pub fn build(
        n_strings: usize,
        place_name_indices: impl Iterator<Item = (usize, Vec<usize>)>,
        address_street_indices: impl Iterator<Item = (usize, usize)>,
    ) -> Self {
        let mut entries: Vec<Vec<(SuggestionLocationType, usize)>> = vec![Vec::new(); n_strings];

        for (place_idx, string_indices) in place_name_indices {
            for str_idx in string_indices {
                if let Some(list) = entries.get_mut(str_idx) {
                    list.push((SuggestionLocationType::Place, place_idx));
                }
            }
        }

        for (addr_idx, str_idx) in address_street_indices {
            if let Some(list) = entries.get_mut(str_idx) {
                list.push((SuggestionLocationType::Address, addr_idx));
            }
        }

        Self { entries }
    }

    /// Get locations that use the given string.
    pub fn get(&self, string_idx: usize) -> &[(SuggestionLocationType, usize)] {
        self.entries.get(string_idx).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigram_index_basic() {
        let strings = vec![
            "darmstadt".to_string(),
            "frankfurt".to_string(),
            "mainz".to_string(),
        ];
        let index = BigramIndex::build(&strings);

        let mut ctx = GuessContext::new();
        ctx.resize(strings.len());

        let candidates = index.find_candidates("darmstadt", &mut ctx);
        assert!(!candidates.is_empty(), "should find darmstadt");
        assert_eq!(candidates[0].string_idx, 0);
    }

    #[test]
    fn bigram_index_fuzzy() {
        let strings = vec![
            "darmstadt".to_string(),
            "darmstaedter".to_string(),
            "berlin".to_string(),
        ];
        let index = BigramIndex::build(&strings);

        let mut ctx = GuessContext::new();
        ctx.resize(strings.len());

        let candidates = index.find_candidates("darmstad", &mut ctx);
        // Should match darmstadt and darmstaedter
        let matched_indices: Vec<usize> = candidates.iter().map(|c| c.string_idx).collect();
        assert!(matched_indices.contains(&0), "should match darmstadt");
    }

    #[test]
    fn score_candidates_basic() {
        let strings = vec![
            "darmstadt".to_string(),
            "frankfurt".to_string(),
        ];
        let phrases = vec![Phrase {
            token_bits: 1,
            s: "darmstadt".to_string(),
        }];

        let candidates = vec![
            CosSimilarityMatch { string_idx: 0, cos_sim: 0.9 },
            CosSimilarityMatch { string_idx: 1, cos_sim: 0.3 },
        ];

        let mut offsets = Vec::new();
        let results = score_candidates(&candidates, &strings, &phrases, &mut offsets);

        assert!(!results.is_empty());
        // darmstadt should score better (lower) than frankfurt
        assert_eq!(results[0].0, 0, "darmstadt should be best match");
    }

    #[test]
    fn string_location_map_basic() {
        let place_names = vec![
            (0usize, vec![0usize, 1]),
            (1, vec![2]),
        ];
        let addr_streets = vec![
            (0usize, 3usize),
        ];

        let map = StringLocationMap::build(
            4,
            place_names.into_iter(),
            addr_streets.into_iter(),
        );

        let locs = map.get(0);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], (SuggestionLocationType::Place, 0));

        let locs = map.get(3);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], (SuggestionLocationType::Address, 0));
    }
}
