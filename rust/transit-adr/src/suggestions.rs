//! Full suggestion pipeline — orchestrates tokenization, guessing, scoring,
//! matching, deduplication, and sorting.
//!
//! Mirrors C++ `adr/src/get_suggestions.cc`.

use crate::categories::get_category_score;
use crate::normalize::{erase_fillers, get_exact_alt, get_numeric_tokens_mask, get_phrases, normalize};
use crate::score::get_match_score;
use crate::suggestion::{Address, GuessContext, Suggestion, SuggestionLocation};
use crate::typeahead::{MatchItem, MatchItemType, ScoredMatch, Typeahead};
use crate::types::*;

/// Maximum number of scored matches to keep. C++ `kMaxScoredMatches`.
pub const MAX_SCORED_MATCHES: usize = 10000;

/// Insert into a sorted vec, maintaining sort order and capping at max_size.
fn insert_sorted<T: Ord>(vec: &mut Vec<T>, item: T, max_size: usize) {
    let pos = vec.binary_search_by(|probe| probe.cmp(&item)).unwrap_or_else(|e| e);
    if pos < max_size {
        vec.insert(pos, item);
        if vec.len() > max_size {
            vec.truncate(max_size);
        }
    }
}

/// Activate areas for an area set — compute per-phrase match scores for area names.
/// C++ `activate_areas()`.
fn activate_areas(
    t: &Typeahead,
    ctx: &mut GuessContext,
    numeric_tokens_mask: TokenBitmask,
    area_set_idx: AreaSetIdx,
    languages: &[LanguageIdx],
) {
    let area_set = &t.area_sets[area_set_idx.to_idx()];
    for &area in area_set {
        let area_idx = area.to_idx();
        if ctx.area_active[area_idx] {
            continue;
        }

        if t.area_admin_level[area_idx] == TIMEZONE_ADMIN_LEVEL {
            ctx.area_phrase_match_scores[area_idx] = NO_MATCH_SCORES;
            ctx.area_active[area_idx] = true;
            continue;
        }

        ctx.area_active[area_idx] = true;
        for (j, phrase) in ctx.phrases.iter().enumerate() {
            // Zip-code areas only match numeric tokens.
            let match_allowed = t.area_admin_level[area_idx] != POSTAL_CODE_ADMIN_LEVEL
                || (phrase.token_bits & numeric_tokens_mask) == phrase.token_bits;

            ctx.area_phrase_match_scores[area_idx][j] = NO_MATCH;
            if !match_allowed {
                continue;
            }

            // Find best match across all languages.
            for (_i, &lang) in languages.iter().enumerate() {
                let lang_idx = crate::typeahead::find_lang(
                    &t.area_name_lang[area_idx],
                    lang,
                );
                if lang_idx < 0 {
                    continue;
                }

                let name_str_idx =
                    t.area_names[area_idx][lang_idx as usize].to_idx();
                let area_name = &t.strings[name_str_idx];
                let lang_match_score = get_match_score(
                    area_name,
                    &phrase.s,
                    &mut ctx.sift4_offset_arr,
                );
                if lang_match_score < ctx.area_phrase_match_scores[area_idx][j] {
                    ctx.area_phrase_match_scores[area_idx][j] = lang_match_score;
                    ctx.area_phrase_lang[area_idx][j] = lang_idx as u8;
                }
            }
        }
    }
}

/// Compute string×phrase match score matrix.
/// C++ `compute_string_phrase_match_scores()`.
fn compute_string_phrase_match_scores(ctx: &mut GuessContext, t: &Typeahead) {
    ctx.string_phrase_match_scores
        .resize(ctx.string_matches.len(), NO_MATCH_SCORES);
    for (i, m) in ctx.string_matches.iter().enumerate() {
        for (j, p) in ctx.phrases.iter().enumerate() {
            ctx.string_phrase_match_scores[i][j] = get_match_score(
                &t.strings[m.idx.to_idx()],
                &p.s,
                &mut ctx.sift4_offset_arr,
            );
        }
    }
}

/// Build scored street and place match lists from string matches.
/// C++ `get_scored_matches()`.
fn get_scored_matches(
    t: &Typeahead,
    ctx: &mut GuessContext,
    filter: FilterType,
    place_filter: Option<&dyn Fn(PlaceIdx) -> bool>,
) {
    ctx.scored_street_matches.clear();
    ctx.scored_place_matches.clear();

    for (i, m) in ctx.string_matches.iter().enumerate() {
        for p_idx in 0..ctx.phrases.len() as PhraseIdx {
            let p_match_score = ctx.string_phrase_match_scores[i][p_idx as usize];
            if p_match_score == NO_MATCH {
                continue;
            }

            let str_idx = m.idx.to_idx();
            if str_idx >= t.string_to_location.len() {
                continue;
            }

            for (loc_idx, loc_type) in t.string_to_location[str_idx]
                .iter()
                .zip(t.string_to_type[str_idx].iter())
            {
                match loc_type {
                    LocationType::Street => {
                        let street_idx = StreetIdx(*loc_idx);
                        if filter != FilterType::None && filter != FilterType::Address {
                            continue;
                        }
                        insert_sorted(
                            &mut ctx.scored_street_matches,
                            ScoredMatch {
                                score: p_match_score,
                                phrase_idx: p_idx,
                                string_idx: m.idx,
                                idx: street_idx,
                            },
                            MAX_SCORED_MATCHES,
                        );
                    }
                    LocationType::Place => {
                        let place_idx = PlaceIdx(*loc_idx);
                        if filter != FilterType::None
                            && (filter == FilterType::Address
                                || ((filter == FilterType::Extra)
                                    != (t.place_type[place_idx.to_idx()]
                                        == crate::categories::AmenityCategory::Extra)))
                        {
                            continue;
                        }
                        if let Some(pf) = place_filter {
                            if !pf(place_idx) {
                                continue;
                            }
                        }
                        insert_sorted(
                            &mut ctx.scored_place_matches,
                            ScoredMatch {
                                score: p_match_score,
                                phrase_idx: p_idx,
                                string_idx: m.idx,
                                idx: place_idx,
                            },
                            MAX_SCORED_MATCHES,
                        );
                    }
                }
            }
        }
    }
}

/// Match streets against phrases and area names.
/// C++ `match_streets()`.
fn match_streets(
    all_tokens_mask: TokenBitmask,
    numeric_tokens_mask: TokenBitmask,
    t: &Typeahead,
    ctx: &mut GuessContext,
    tokens: &[String],
    languages: &[LanguageIdx],
) {
    let scored = ctx.scored_street_matches.clone();
    for sm in &scored {
        let street = sm.idx;
        let street_idx = street.to_idx();
        let street_edit_dist = sm.score;
        let street_p_idx = sm.phrase_idx as usize;
        let str_idx = sm.string_idx;

        ctx.area_match_items.clear();

        // Add street itself to each area set.
        for (index, &area_set) in t.street_areas[street_idx].iter().enumerate() {
            ctx.area_match_items
                .entry(area_set)
                .or_default()
                .push(MatchItem {
                    item_type: MatchItemType::Street,
                    score: 0.0,
                    index: index as u32,
                    house_number_p_idx: PhraseIdx::MAX,
                    matched_mask: ctx.phrases[street_p_idx].token_bits,
                });
        }

        // Match house numbers.
        let mut hn_index = 0u32;
        let n_house_numbers = t.house_numbers[street_idx].len();
        for hn_i in 0..n_house_numbers {
            let hn_str_idx = t.house_numbers[street_idx][hn_i].to_idx();
            let hn_areas = t.house_areas[street_idx][hn_i];

            for (hn_p_idx, p) in ctx.phrases.iter().enumerate() {
                if (p.token_bits & numeric_tokens_mask) != p.token_bits {
                    continue;
                }

                let hn_score = get_match_score(
                    &t.strings[hn_str_idx],
                    &p.s,
                    &mut ctx.sift4_offset_arr,
                );
                if hn_score == NO_MATCH {
                    continue;
                }

                let actual_score = if t.strings[hn_str_idx] == p.s {
                    -2.5
                } else {
                    hn_score as f64
                } as f32;

                ctx.area_match_items
                    .entry(hn_areas)
                    .or_default()
                    .push(MatchItem {
                        item_type: MatchItemType::HouseNumber,
                        score: actual_score,
                        index: hn_index,
                        house_number_p_idx: hn_p_idx as PhraseIdx,
                        matched_mask: ctx.phrases[street_p_idx].token_bits
                            | ctx.phrases[hn_p_idx].token_bits,
                    });
            }
            hn_index += 1;
        }

        // For each area set, greedily match area names.
        let area_match_items_snapshot: Vec<_> = ctx.area_match_items.drain().collect();
        for (area_set_idx, items) in &area_match_items_snapshot {
            activate_areas(t, ctx, numeric_tokens_mask, *area_set_idx, languages);

            let mut item_matched_masks = std::collections::HashSet::new();
            for item in items {
                item_matched_masks.insert(item.matched_mask);
            }

            for &item_matched_mask in &item_matched_masks {
                let mut matched_tokens_mask = item_matched_mask;
                let mut matched_areas_mask = 0u32;
                let mut area_lang = [0u8; 32];
                let mut areas_edit_dist = 0.0f32;

                let area_set = &t.area_sets[area_set_idx.to_idx()];
                for (area_p_idx, area_p) in ctx.phrases.iter().enumerate() {
                    if (area_p.token_bits & matched_tokens_mask) != 0 {
                        continue;
                    }

                    let mut best_edit_dist = f32::MAX;
                    let mut best_area_idx = 0usize;

                    for (area_idx, &area) in area_set.iter().enumerate() {
                        let match_allowed =
                            t.area_admin_level[area.to_idx()] != POSTAL_CODE_ADMIN_LEVEL
                                || (area_p.token_bits & numeric_tokens_mask)
                                    == area_p.token_bits;
                        if !match_allowed {
                            continue;
                        }

                        let edit_dist =
                            ctx.area_phrase_match_scores[area.to_idx()][area_p_idx];
                        if best_edit_dist > edit_dist {
                            best_edit_dist = edit_dist;
                            best_area_idx = area_idx;
                        }
                    }

                    if best_edit_dist != NO_MATCH {
                        let best_area = area_set[best_area_idx];
                        matched_areas_mask |= 1u32 << best_area_idx;
                        area_lang[best_area_idx] =
                            ctx.area_phrase_lang[best_area.to_idx()][area_p_idx];
                        areas_edit_dist += best_edit_dist;
                        areas_edit_dist -= (t.area_population[best_area.to_idx()].get()
                            as f32
                            / 10_000_000.0)
                            * 2.0;
                        matched_tokens_mask |= area_p.token_bits;
                    }
                }

                for item in items {
                    if item.matched_mask != item_matched_mask {
                        continue;
                    }

                    let mut total_score = street_edit_dist + areas_edit_dist + item.score;
                    for (t_idx, token) in tokens.iter().enumerate() {
                        if (matched_tokens_mask & (1u8 << t_idx)) == 0 {
                            total_score += token.len() as f32 * 3.0;
                        }
                    }

                    let house_number_score = if item.item_type == MatchItemType::HouseNumber {
                        5.0
                    } else {
                        0.0
                    };
                    let areas_bonus = matched_areas_mask.count_ones() as f32 * 2.0;
                    let no_area_score =
                        if matched_areas_mask == 0 && matched_tokens_mask == all_tokens_mask {
                            3.0
                        } else {
                            0.0
                        };

                    total_score -= house_number_score;
                    total_score -= areas_bonus;
                    total_score -= no_area_score;

                    let coords = if item.item_type == MatchItemType::HouseNumber {
                        t.house_coordinates[street_idx][item.index as usize]
                    } else {
                        t.street_pos[street_idx][item.index as usize]
                    };

                    let hn = if item.item_type == MatchItemType::HouseNumber {
                        item.index
                    } else {
                        Address::NO_HOUSE_NUMBER
                    };

                    ctx.suggestions.push(Suggestion::new(
                        str_idx,
                        SuggestionLocation::Address(Address {
                            street,
                            house_number: hn,
                        }),
                        coords,
                        *area_set_idx,
                        area_lang,
                        matched_areas_mask,
                        matched_tokens_mask,
                        total_score,
                    ));
                }
            }
        }
    }
}

/// Match places against phrases and area names.
/// C++ `match_places()`.
fn match_places(
    all_tokens_mask: TokenBitmask,
    numeric_tokens_mask: TokenBitmask,
    t: &Typeahead,
    ctx: &mut GuessContext,
    tokens: &[String],
    languages: &[LanguageIdx],
) {
    let scored = ctx.scored_place_matches.clone();
    for sm in &scored {
        let place = sm.idx;
        let place_idx = place.to_idx();
        let place_edit_dist = sm.score;
        let place_p_idx = sm.phrase_idx as usize;
        let str_idx = sm.string_idx;
        let area_set_idx = t.place_areas[place_idx];

        activate_areas(t, ctx, numeric_tokens_mask, area_set_idx, languages);

        let mut matched_tokens_mask = ctx.phrases[place_p_idx].token_bits;
        let mut matched_areas_mask = 0u32;
        let mut area_lang = [0u8; 32];
        let mut areas_edit_dist = 0.0f32;

        let area_set = &t.area_sets[area_set_idx.to_idx()];
        for (area_p_idx, area_p) in ctx.phrases.iter().enumerate() {
            if (area_p.token_bits & matched_tokens_mask) != 0 {
                continue;
            }

            let mut best_edit_dist = f32::MAX;
            let mut best_area_idx = 0usize;

            for (area_idx, &area) in area_set.iter().enumerate() {
                let match_allowed =
                    t.area_admin_level[area.to_idx()] != POSTAL_CODE_ADMIN_LEVEL
                        || (area_p.token_bits & numeric_tokens_mask) == area_p.token_bits;
                if !match_allowed {
                    continue;
                }

                let edit_dist = ctx.area_phrase_match_scores[area.to_idx()][area_p_idx];
                if best_edit_dist > edit_dist {
                    best_edit_dist = edit_dist;
                    best_area_idx = area_idx;
                }
            }

            if best_edit_dist != NO_MATCH {
                let best_area = area_set[best_area_idx];
                matched_areas_mask |= 1u32 << best_area_idx;
                area_lang[best_area_idx] =
                    ctx.area_phrase_lang[best_area.to_idx()][area_p_idx];
                areas_edit_dist += best_edit_dist;
                areas_edit_dist -= (t.area_population[best_area.to_idx()].get() as f32
                    / 10_000_000.0)
                    * 2.0;
                matched_tokens_mask |= area_p.token_bits;
            }
        }

        let mut total_score = place_edit_dist + areas_edit_dist;
        for (t_idx, token) in tokens.iter().enumerate() {
            if (matched_tokens_mask & (1u8 << t_idx)) == 0 {
                total_score += token.len() as f32 * 3.0;
            }
        }

        let category_score = get_category_score(t.place_type[place_idx]);
        let areas_score = matched_areas_mask.count_ones() as f32 * 2.0;
        let no_area_score =
            if matched_areas_mask == 0 && matched_tokens_mask == all_tokens_mask {
                2.5
            } else {
                0.0
            };
        let population = t.place_population[place_idx].get();
        let population_score =
            if t.place_type[place_idx] == crate::categories::AmenityCategory::Extra {
                (population as f32 / 2_000.0).clamp(1.2, 5.0)
            } else {
                (population as f32 / 200_000.0).clamp(0.0, 3.0)
            };
        let place_score = 5.0f32;

        // Language bonus.
        let mut lang_score = -0.1f32;
        let place_names = &t.place_names[place_idx];
        let place_name_langs = &t.place_name_lang[place_idx];
        for (&name_str, &name_lang) in place_names.iter().zip(place_name_langs.iter()) {
            if name_str != str_idx {
                continue;
            }
            if let Some(idx) = languages.iter().position(|&l| l == name_lang) {
                lang_score = lang_score.max(if idx == 0 { 0.5 } else { 0.25 });
            }
        }

        total_score -= category_score;
        total_score -= areas_score;
        total_score -= no_area_score;
        total_score -= population_score;
        total_score -= place_score;
        total_score -= lang_score;

        // Dedup: skip if same place with worse score.
        let dominated = ctx.suggestions.last().map_or(false, |last| {
            last.score < total_score
                && matches!(last.location, SuggestionLocation::Place(p) if p == place)
        });
        if !dominated {
            ctx.suggestions.push(Suggestion::new(
                str_idx,
                SuggestionLocation::Place(place),
                t.place_coordinates[place_idx],
                area_set_idx,
                area_lang,
                matched_areas_mask,
                matched_tokens_mask,
                total_score,
            ));
        }
    }
}

/// Full suggestion pipeline.
/// C++ `get_suggestions()`.
pub fn get_suggestions(
    t: &Typeahead,
    input: &str,
    n_suggestions: usize,
    languages: &[LanguageIdx],
    ctx: &mut GuessContext,
    coord: Option<(f64, f64)>,
    bias: f32,
    filter: FilterType,
    place_filter: Option<&dyn Fn(PlaceIdx) -> bool>,
) -> Vec<TokenPos> {
    ctx.suggestions.clear();
    if input.len() < 3 {
        return Vec::new();
    }

    // Tokenize input.
    let mut token_pos = Vec::new();
    let mut tokens = Vec::new();
    let mut all_tokens_mask: TokenBitmask = 0;
    let mut i = 0u32;

    for (_start, token_str) in input.split_whitespace().enumerate() {
        if token_str.is_empty() {
            continue;
        }
        let mut normalized_tok = normalize(token_str);
        erase_fillers(&mut normalized_tok);
        all_tokens_mask |= 1u8 << i;

        // Find byte offset in original string.
        let byte_start = input
            .find(token_str)
            .unwrap_or(0) as u16;
        token_pos.push(TokenPos {
            start_idx: byte_start,
            size: token_str.len() as u16,
        });
        tokens.push(normalized_tok);
        i += 1;
    }
    tokens.truncate(MAX_INPUT_TOKENS);

    ctx.phrases = get_phrases(&tokens.iter().map(|s| s.as_str()).collect::<Vec<_>>());

    // Build guess string with alt forms.
    let mut guess_str = normalize(input);
    for token in &tokens {
        if let Some(alt) = get_exact_alt(token) {
            guess_str.push_str(&alt);
        }
    }

    t.guess(&guess_str, &mut ctx.cache, &mut ctx.string_matches);

    compute_string_phrase_match_scores(ctx, t);

    // Reset area activation.
    for a in ctx.area_active.iter_mut() {
        *a = false;
    }

    let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    let numeric_tokens_mask = get_numeric_tokens_mask(&token_refs);

    get_scored_matches(t, ctx, filter, place_filter);

    match_streets(
        all_tokens_mask,
        numeric_tokens_mask,
        t,
        ctx,
        &tokens,
        languages,
    );
    match_places(
        all_tokens_mask,
        numeric_tokens_mask,
        t,
        ctx,
        &tokens,
        languages,
    );

    if ctx.suggestions.is_empty() {
        return token_pos;
    }

    // Distance bias.
    if let Some((lat, lng)) = coord {
        let query = Coordinates::from_lat_lng(lat, lng);
        for s in &mut ctx.suggestions {
            let dist = query.distance_to(&s.coordinates);
            let dist_bonus = if dist < 2_000.0 {
                2.5 * bias
            } else if dist < 10_000.0 {
                2.0 * bias
            } else if dist < 100_000.0 {
                1.0 * bias
            } else if dist < 1_000_000.0 {
                0.5 * bias
            } else {
                0.0
            };
            s.score -= dist_bonus;
        }
    }

    // Mark duplicates.
    {
        let mut sorted: Vec<u32> = (0..ctx.suggestions.len() as u32).collect();
        sorted.sort_by(|&a, &b| {
            let sa = &ctx.suggestions[a as usize];
            let sb = &ctx.suggestions[b as usize];
            sa.location
                .cmp(&sb.location)
                .then_with(|| sa.area_set.0.cmp(&sb.area_set.0))
                .then_with(|| {
                    sa.score
                        .partial_cmp(&sb.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        for i in 1..sorted.len() {
            let prev = &ctx.suggestions[sorted[i - 1] as usize];
            let curr = &ctx.suggestions[sorted[i] as usize];
            if prev.location == curr.location && prev.area_set == curr.area_set {
                ctx.suggestions[sorted[i] as usize].is_duplicate = true;
            }
        }
    }

    // Sort and truncate.
    let result_count = n_suggestions.min(ctx.suggestions.len());
    ctx.suggestions
        .select_nth_unstable_by(result_count.saturating_sub(1), |a, b| a.cmp(b));
    ctx.suggestions.truncate(result_count);
    ctx.suggestions.sort();

    // Populate areas.
    for s in &mut ctx.suggestions {
        s.populate_areas(t);
    }

    // Resolve unique areas for display.
    let len = ctx.suggestions.len();
    for i in 0..len {
        for j in (i + 1)..len {
            while ctx.suggestions[i].location == ctx.suggestions[j].location
                && ctx.suggestions[i].unique_area_idx.is_some()
                && ctx.suggestions[j].unique_area_idx.is_some()
            {
                let ui = ctx.suggestions[i].unique_area_idx.unwrap();
                let uj = ctx.suggestions[j].unique_area_idx.unwrap();
                let areas_i = &t.area_sets[ctx.suggestions[i].area_set.to_idx()];
                let areas_j = &t.area_sets[ctx.suggestions[j].area_set.to_idx()];

                if ui >= areas_i.len() || uj >= areas_j.len() {
                    break;
                }
                if areas_i[ui] != areas_j[uj] {
                    break;
                }

                ctx.suggestions[i].unique_area_idx = Some(ui + 1);
                ctx.suggestions[j].unique_area_idx = Some(uj + 1);
            }
        }
    }

    token_pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_sorted_basic() {
        let mut v = vec![1, 3, 5];
        insert_sorted(&mut v, 2, 5);
        assert_eq!(v, vec![1, 2, 3, 5]);
    }

    #[test]
    fn insert_sorted_cap() {
        let mut v = vec![1, 2, 3];
        insert_sorted(&mut v, 0, 3);
        assert_eq!(v, vec![0, 1, 2]);
    }

    #[test]
    fn get_suggestions_too_short() {
        let t = Typeahead::new();
        let mut ctx = GuessContext::new(crate::cache::Cache::new(0, 10));
        let result = get_suggestions(&t, "ab", 10, &[], &mut ctx, None, 1.0, FilterType::None, None);
        assert!(result.is_empty());
        assert!(ctx.suggestions.is_empty());
    }
}
