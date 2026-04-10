//! String matching and scoring algorithms.
//!
//! Provides Levenshtein distance, token-level SIFT4-based scoring, and full
//! phrase-level match scoring. These mirror the C++ `adr::score.h` functions.

use crate::normalize::{erase_fillers, get_phrases, normalize, Phrase};
use crate::sift4::{self, SiftOffset};
use crate::types::{EditDist, Score, NO_MATCH};

/// Compute the Levenshtein edit distance between `source` and `target`.
///
/// Uses a single-row DP approach with early termination when the minimum
/// possible distance exceeds a limit of `(min_len / 2) + 2`.
pub fn levenshtein_distance(source: &str, target: &str) -> EditDist {
    let (source, target) = if source.len() > target.len() {
        (target, source)
    } else {
        (source, target)
    };

    let min_size = source.len();
    let max_size = target.len();
    let limit = (min_size / 2) + 2;

    let sb = source.as_bytes();
    let tb = target.as_bytes();

    let mut lev_dist: Vec<EditDist> = (0..=min_size as EditDist).collect();

    for j in 1..=max_size {
        let mut previous_diagonal = lev_dist[0];
        lev_dist[0] += 1;

        for i in 1..=min_size {
            let previous_diagonal_save = lev_dist[i];
            if sb[i - 1] == tb[j - 1] {
                lev_dist[i] = previous_diagonal;
            } else {
                lev_dist[i] = lev_dist[i - 1]
                    .min(lev_dist[i])
                    .min(previous_diagonal)
                    + 1;
            }
            previous_diagonal = previous_diagonal_save;
        }

        if lev_dist[min_size.min(j.saturating_sub(1))] as usize > limit {
            return EditDist::MAX;
        }
    }

    lev_dist[min_size]
}

/// Score how well a single dataset token matches an input phrase token.
///
/// Returns a float score (lower = better match). Exact matches get a large
/// negative score. Returns [`NO_MATCH`] if the match is too poor.
pub fn get_token_match_score(
    dataset_token: &str,
    p: &str,
    sift4_offset_arr: &mut Vec<SiftOffset>,
) -> Score {
    if dataset_token == p {
        return -2.0 - p.len() as Score * 0.75;
    }

    let cut_len = dataset_token.len().min(p.len());
    let cut_normalized_str = &dataset_token[..cut_len];

    let dist = sift4::sift4(
        cut_normalized_str,
        p,
        3,
        (dataset_token.len().min(p.len()) / 2 + 2) as EditDist,
        sift4_offset_arr,
    );

    if dist as usize >= cut_normalized_str.len() {
        return NO_MATCH;
    }

    let overhang_penalty = (dataset_token.len().saturating_sub(p.len()) as f32
        / 4.0)
        .min(4.0);

    let relative_coverage =
        6.0 * (dist as f32 / cut_normalized_str.len() as f32);

    let mut common_prefix_bonus: f32 = 0.0;
    let end = cut_normalized_str.len().min(p.len());
    let cb = cut_normalized_str.as_bytes();
    let pb = p.as_bytes();
    for i in 0..end {
        if cb[i] != pb[i] {
            break;
        }
        common_prefix_bonus -= 0.25;
    }

    let first_letter_penalty = if cb[0] != pb[0] { 2.0 } else { -0.5 };

    let second_letter_penalty = if cut_normalized_str.len() > 1 && p.len() > 1
    {
        if cb[1] != pb[1] {
            1.0
        } else {
            -0.25
        }
    } else {
        -0.25
    };

    let score = dist as Score
        + first_letter_penalty
        + second_letter_penalty
        + overhang_penalty
        + relative_coverage
        + common_prefix_bonus;

    let max = (cut_normalized_str.len() as f32 / 2.0).ceil();
    if score > max {
        NO_MATCH
    } else {
        score
    }
}

/// Tokenize a string by the given delimiter characters, calling `f` for each
/// non-empty token.
///
/// Mirrors the C++ `for_each_token()`. Returns `true` from `f` to continue,
/// `false` to break.
pub fn for_each_token<F: FnMut(&str) -> bool>(s: &str, delimiters: &[char], mut f: F) {
    let mut remaining = s;
    while !remaining.is_empty() {
        let token_end = remaining
            .find(|c: char| delimiters.contains(&c))
            .unwrap_or(remaining.len());
        let token = &remaining[..token_end];
        if !f(token) {
            break;
        }
        remaining = &remaining[token_end..];
        if !remaining.is_empty() {
            remaining = &remaining[1..]; // skip delimiter
        }
    }
}

/// Score how well a dataset string `s` matches an input phrase `p_token`.
///
/// The dataset string is normalized, tokenized, and all sub-phrases are
/// generated. The best-matching sub-phrase score is returned, with penalties
/// for unmatched dataset tokens.
///
/// Returns [`NO_MATCH`] if no adequate match is found.
///
/// Note: C++ passes a reusable `utf8_normalize_buf_t&` buffer to avoid per-call allocation.
/// The Rust version allocates per call in `normalize()`. This is a known PERF gap for
/// high-throughput search.
pub fn get_match_score(
    s: &str,
    p_token: &str,
    sift4_offset_arr: &mut Vec<SiftOffset>,
) -> Score {
    if s.is_empty() || p_token.is_empty() {
        return NO_MATCH;
    }

    let mut normalized_str = normalize(s);
    erase_fillers(&mut normalized_str);

    // Tokenize by space and hyphen (max 8 tokens)
    let mut s_tokens: Vec<&str> = Vec::new();
    let mut count = 0usize;
    {
        let ns = normalized_str.as_str();
        let mut remaining = ns;
        while !remaining.is_empty() && count < 8 {
            let token_end = remaining
                .find([' ', '-'])
                .unwrap_or(remaining.len());
            let token = &remaining[..token_end];
            if !token.is_empty() {
                s_tokens.push(token);
                count += 1;
            }
            remaining = &remaining[token_end..];
            if !remaining.is_empty() {
                remaining = &remaining[1..];
            }
        }
    }

    let fallback =
        get_token_match_score(&normalized_str, p_token, sift4_offset_arr);
    if s_tokens.len() == 1 {
        return fallback;
    }

    let s_phrases: Vec<Phrase> = get_phrases(&s_tokens);

    let mut best_s_score = NO_MATCH;
    let mut best_s_idx: usize = 0;
    for (s_idx, s_phrase) in s_phrases.iter().enumerate() {
        let s_p_match_score = get_token_match_score(
            &s_phrase.s,
            p_token,
            sift4_offset_arr,
        );
        if best_s_score > s_p_match_score {
            best_s_idx = s_idx;
            best_s_score = s_p_match_score;
        }
    }

    if best_s_score == NO_MATCH {
        return NO_MATCH;
    }

    let mut sum = best_s_score;
    let covered = s_phrases[best_s_idx].token_bits;
    let mut n_not_matched = 0u32;
    for (s_idx, s_token) in s_tokens.iter().enumerate() {
        if (covered & (1 << s_idx)) == 0 {
            n_not_matched += 1;
            let not_matched_penalty =
                (s_token.len() as f32 / 4.0).clamp(0.75, 3.0);
            sum += not_matched_penalty;
        }
    }

    if n_not_matched as usize == s_tokens.len() {
        return NO_MATCH;
    }

    let max = (s.len().min(p_token.len()) as f32 / 2.0).ceil();
    let score = fallback.min(sum);

    if score >= max {
        NO_MATCH
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Levenshtein tests ---

    #[test]
    fn levenshtein_identical() {
        assert_eq!(levenshtein_distance("kitten", "kitten"), 0);
    }

    #[test]
    fn levenshtein_single_edit() {
        assert_eq!(levenshtein_distance("kitten", "sitten"), 1);
    }

    #[test]
    fn levenshtein_empty() {
        // Empty vs empty = 0
        assert_eq!(levenshtein_distance("", ""), 0);
        // When one is empty and the other is short, the limit check
        // (min_size/2 + 2) triggers early exit returning MAX, matching
        // the C++ behavior.
        let d1 = levenshtein_distance("", "abc");
        let d2 = levenshtein_distance("abc", "");
        // Both return MAX due to limit: limit = 0/2+2 = 2, distance 3 > 2
        assert_eq!(d1, EditDist::MAX);
        assert_eq!(d2, EditDist::MAX);
    }

    // --- Token match score tests ---

    #[test]
    fn token_match_score_exact() {
        let mut offsets = Vec::new();
        let score = get_token_match_score("darmstadt", "darmstadt", &mut offsets);
        // Exact match: -2.0 - 9 * 0.75 = -8.75
        assert!((score - (-8.75)).abs() < f32::EPSILON);
    }

    #[test]
    fn token_match_score_no_match() {
        let mut offsets = Vec::new();
        let score =
            get_token_match_score("zzzzzzzzz", "aaaaaaaaa", &mut offsets);
        assert_eq!(score, NO_MATCH);
    }

    // --- for_each_token tests ---

    /// C++ test: `TEST(adr, for_each_token)`
    ///
    /// Tokenizes "Groß-Umstadt An der Pfalz" by space, hyphen, and 'z'.
    #[test]
    fn for_each_token_basic() {
        let mut tokens = Vec::new();
        for_each_token(
            "Groß-Umstadt An der Pfalz",
            &[' ', '-', 'z'],
            |s| {
                tokens.push(s.to_string());
                true
            },
        );
        // Should split on ' ', '-', and 'z'
        // "Groß-Umstadt An der Pfalz" with delimiters [' ', '-', 'z']:
        //   "Groß" '-' "Umstadt" ' ' "An" ' ' "der" ' ' "Pfal" 'z' (end)
        // Note: 'z' at the end of "Pfalz" splits to "Pfal" with no trailing empty token
        // because the loop exits when remaining is empty after skipping the delimiter.
        assert_eq!(
            tokens,
            vec!["Groß", "Umstadt", "An", "der", "Pfal"]
        );
    }

    // --- Full match score tests ---

    /// C++ test: active assertion from `TEST(adr, score_test)`:
    /// `get_match_score("Darmstadt", "darmstadt", ...)`
    ///
    /// "Darmstadt" normalizes to "darmstadt" → exact match → large negative.
    #[test]
    fn match_score_exact() {
        let mut offsets = Vec::new();
        let score = get_match_score("Darmstadt", "darmstadt", &mut offsets);
        // Exact match: -2.0 - 9 * 0.75 = -8.75
        assert!(score < 0.0, "exact match should be negative, got {score}");
        assert!(
            (score - (-8.75)).abs() < f32::EPSILON,
            "expected -8.75, got {score}"
        );
    }

    /// C++ test: active assertion from `TEST(adr, score_test)`:
    /// `get_match_score("Darmstadt,ZOB Zweifalltorweg", "darmstadt", ...)`
    ///
    /// Multi-token dataset string with "darmstadt" as one of the tokens.
    #[test]
    fn match_score_multi_token() {
        let mut offsets = Vec::new();
        let score = get_match_score(
            "Darmstadt,ZOB Zweifalltorweg",
            "darmstadt",
            &mut offsets,
        );
        assert_ne!(score, NO_MATCH, "should find a match for 'darmstadt'");
        assert!(
            score < 5.0,
            "match should have a good score, got {score}"
        );
    }

    /// C++ test: `TEST(adr, score_test)` — active (uncommented) assertions.
    ///
    /// The C++ uses `EXPECT_EQ(1, ...)` which appears to be a truthiness/placeholder
    /// check. We verify that both calls return an actual match (not NO_MATCH) with
    /// a reasonable negative score.
    #[test]
    fn score_test_cpp_parity() {
        let mut offsets = Vec::new();

        // Active C++ assertion 1:
        //   EXPECT_EQ(1, adr::get_match_score("Darmstadt", "darmstadt", sift4_dist, buf));
        let s1 = get_match_score("Darmstadt", "darmstadt", &mut offsets);
        assert_ne!(s1, NO_MATCH, "Darmstadt/darmstadt should match");
        assert!(s1 < 0.0, "exact match should be negative, got {s1}");

        // Active C++ assertion 2:
        //   EXPECT_EQ(1, adr::get_match_score("Darmstadt,ZOB Zweifalltorweg", "darmstadt", sift4_dist, buf));
        let s2 = get_match_score("Darmstadt,ZOB Zweifalltorweg", "darmstadt", &mut offsets);
        assert_ne!(s2, NO_MATCH, "Darmstadt,ZOB Zweifalltorweg/darmstadt should match");
        assert!(s2 < 5.0, "multi-token match should have a good score, got {s2}");
    }

    #[test]
    fn match_score_empty_inputs() {
        let mut offsets = Vec::new();
        assert_eq!(
            get_match_score("", "darmstadt", &mut offsets),
            NO_MATCH
        );
        assert_eq!(
            get_match_score("Darmstadt", "", &mut offsets),
            NO_MATCH
        );
    }
}
