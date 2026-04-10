//! Suggestion types for address search results.
//!
//! Mirrors C++ `adr/guess_context.h` suggestion-related types and
//! `guess_context.cc` methods.

use crate::area_set::AreaSet;
use crate::formatter::{Formatter, FormatterAddress};
use crate::typeahead::Typeahead;
use crate::types::*;

// ---------------------------------------------------------------------------
// Address — C++ `address`
// ---------------------------------------------------------------------------

/// A street address with optional house number.
/// C++ `adr::address`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Address {
    pub street: StreetIdx,
    pub house_number: u32,
}

impl Address {
    /// Sentinel for no house number. C++ `address::kNoHouseNumber`.
    pub const NO_HOUSE_NUMBER: u32 = u32::MAX;

    pub fn has_house_number(&self) -> bool {
        self.house_number != Self::NO_HOUSE_NUMBER
    }
}

// ---------------------------------------------------------------------------
// SuggestionLocation — C++ `std::variant<place_idx_t, address>`
// ---------------------------------------------------------------------------

/// The location referenced by a suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuggestionLocation {
    Place(PlaceIdx),
    Address(Address),
}

impl PartialOrd for SuggestionLocation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SuggestionLocation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Match C++ variant ordering: place < address
        match (self, other) {
            (SuggestionLocation::Place(a), SuggestionLocation::Place(b)) => a.0.cmp(&b.0),
            (SuggestionLocation::Place(_), SuggestionLocation::Address(_)) => {
                std::cmp::Ordering::Less
            }
            (SuggestionLocation::Address(_), SuggestionLocation::Place(_)) => {
                std::cmp::Ordering::Greater
            }
            (SuggestionLocation::Address(a), SuggestionLocation::Address(b)) => {
                (a.street.0, a.house_number).cmp(&(b.street.0, b.house_number))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MatchedArea — C++ `matched_area`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct MatchedArea {
    pub area: AreaIdx,
    pub lang: LanguageIdx,
}

// ---------------------------------------------------------------------------
// Suggestion — C++ `suggestion`
// ---------------------------------------------------------------------------

/// A geocoding suggestion result.
/// C++ `adr::suggestion`.
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub str_idx: StringIdx,
    pub location: SuggestionLocation,
    pub coordinates: Coordinates,
    pub area_set: AreaSetIdx,
    pub matched_area_lang: AreaSetLang,
    pub matched_areas: u32,
    pub matched_tokens: u8,
    pub is_duplicate: bool,
    pub score: f32,

    // Populated by populate_areas()
    pub city_area_idx: Option<usize>,
    pub zip_area_idx: Option<usize>,
    pub unique_area_idx: Option<usize>,
    pub tz: TimezoneIdx,
}

impl Suggestion {
    pub fn new(
        str_idx: StringIdx,
        location: SuggestionLocation,
        coordinates: Coordinates,
        area_set: AreaSetIdx,
        matched_area_lang: AreaSetLang,
        matched_areas: u32,
        matched_tokens: u8,
        score: f32,
    ) -> Self {
        Self {
            str_idx,
            location,
            coordinates,
            area_set,
            matched_area_lang,
            matched_areas,
            matched_tokens,
            is_duplicate: false,
            score,
            city_area_idx: None,
            zip_area_idx: None,
            unique_area_idx: None,
            tz: TimezoneIdx::INVALID,
        }
    }

    /// Get country code from area set.
    /// C++ `suggestion::get_country_code()`.
    pub fn get_country_code<'a>(&self, t: &'a Typeahead) -> Option<&'a [u8; 2]> {
        let areas = &t.area_sets[self.area_set.to_idx()];
        areas
            .iter()
            .find(|&&a| t.area_country_code[a.to_idx()] != NO_COUNTRY_CODE)
            .map(|&a| &t.area_country_code[a.to_idx()])
    }

    /// Get OSM ID for place suggestions.
    /// C++ `suggestion::get_osm_id()`.
    pub fn get_osm_id(&self, t: &Typeahead) -> Option<i64> {
        match self.location {
            SuggestionLocation::Place(place) => {
                let osm_ids = &t.place_osm_ids[place.to_idx()];
                if osm_ids.len() == 1 {
                    return Some(osm_ids[0]);
                }
                let place_names = &t.place_names[place.to_idx()];
                place_names
                    .iter()
                    .position(|&s| s == self.str_idx)
                    .and_then(|pos| osm_ids.get(pos).copied())
            }
            SuggestionLocation::Address(_) => None,
        }
    }

    /// Populate area indices (city, zip, timezone).
    /// C++ `suggestion::populate_areas()`.
    pub fn populate_areas(&mut self, t: &Typeahead) {
        let area_set_idx = self.area_set.to_idx();
        if area_set_idx >= t.area_sets.len() {
            return;
        }
        let areas = &t.area_sets[area_set_idx];

        // Find zip code area.
        self.zip_area_idx = areas.iter().position(|&a| {
            a.to_idx() < t.area_admin_level.len()
                && t.area_admin_level[a.to_idx()] == POSTAL_CODE_ADMIN_LEVEL
        });

        // Find timezone.
        self.tz = t.get_tz(self.area_set);

        // Find city area (closest to admin level 8).
        const CLOSE_TO: i32 = 8;
        self.city_area_idx = if areas.is_empty() {
            None
        } else {
            Some(
                areas
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &a)| {
                        let x = t.area_admin_level[a.to_idx()].0 as i32;
                        let penalty = if x > CLOSE_TO { 10 } else { 1 };
                        penalty * (x - CLOSE_TO).abs()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0),
            )
        };
        self.unique_area_idx = self.city_area_idx;
    }

    /// Get the display name string.
    pub fn name<'a>(&self, t: &'a Typeahead) -> &'a str {
        &t.strings[self.str_idx.to_idx()]
    }

    /// Build an `AreaSet` for display.
    /// C++ `suggestion::areas()`.
    pub fn areas<'a>(
        &self,
        t: &'a Typeahead,
        languages: &'a [LanguageIdx],
    ) -> AreaSet<'a> {
        AreaSet {
            t,
            languages,
            city_area_idx: self.city_area_idx,
            area_set_idx: self.area_set,
            matched_mask: self.matched_areas,
            matched_area_lang: self.matched_area_lang,
        }
    }

    /// Format the suggestion using a country-specific formatter.
    /// C++ `suggestion::format()`.
    pub fn format(
        &self,
        t: &Typeahead,
        formatter: &Formatter,
        country_code: &str,
    ) -> String {
        let mut addr = FormatterAddress::default();
        addr.country_code = country_code.to_string();

        match self.location {
            SuggestionLocation::Place(_) => {
                addr.road = t.strings[self.str_idx.to_idx()].clone();
            }
            SuggestionLocation::Address(ref a) => {
                addr.road = t.strings[self.str_idx.to_idx()].clone();
                if a.has_house_number() {
                    let si = a.street.to_idx();
                    if si < t.house_numbers.len()
                        && (a.house_number as usize) < t.house_numbers[si].len()
                    {
                        let hn_str = t.house_numbers[si][a.house_number as usize].to_idx();
                        if hn_str < t.strings.len() {
                            addr.house_number = t.strings[hn_str].clone();
                        }
                    }
                }
            }
        }
        formatter.format_immut(&addr)
    }

    /// Build a human-readable description string.
    /// C++ `suggestion::description()`.
    pub fn description(&self, t: &Typeahead) -> String {
        let name = &t.strings[self.str_idx.to_idx()];
        let area_display = self.areas(t, &[DEFAULT_LANG]).format();

        match self.location {
            SuggestionLocation::Place(_) => {
                format!("{},{}", name, area_display)
            }
            SuggestionLocation::Address(ref a) => {
                if !a.has_house_number() {
                    format!("{},{}", name, area_display)
                } else {
                    let si = a.street.to_idx();
                    let hn_name = if si < t.house_numbers.len()
                        && (a.house_number as usize) < t.house_numbers[si].len()
                    {
                        let hn_str = t.house_numbers[si][a.house_number as usize].to_idx();
                        if hn_str < t.strings.len() {
                            &t.strings[hn_str]
                        } else {
                            ""
                        }
                    } else {
                        ""
                    };
                    format!("{} {},{}", name, hn_name, area_display)
                }
            }
        }
    }

    /// Print the suggestion to a string for debug output.
    /// C++ `suggestion::print()`.
    pub fn print(&self, t: &Typeahead, languages: &[LanguageIdx]) -> String {
        let mut out = String::new();

        match self.location {
            SuggestionLocation::Place(p) => {
                let extra = if p.to_idx() < t.place_type.len()
                    && t.place_type[p.to_idx()] == crate::categories::AmenityCategory::Extra
                {
                    " EXT"
                } else {
                    ""
                };
                out.push_str(&format!(
                    "place={} [{}{}]",
                    &t.strings[self.str_idx.to_idx()],
                    p.0,
                    extra
                ));
            }
            SuggestionLocation::Address(ref a) => {
                out.push_str(&format!(
                    "street={}[{}, {}]",
                    &t.strings[self.str_idx.to_idx()],
                    a.street.0,
                    a.house_number
                ));
                if a.has_house_number() {
                    let si = a.street.to_idx();
                    if si < t.house_numbers.len()
                        && (a.house_number as usize) < t.house_numbers[si].len()
                    {
                        let hn_str = t.house_numbers[si][a.house_number as usize].to_idx();
                        if hn_str < t.strings.len() {
                            out.push_str(&format!(", house_number={}", &t.strings[hn_str]));
                        }
                    }
                }
            }
        }

        let tz = t.get_tz(self.area_set);
        let tz_name = if tz.is_valid() && tz.to_idx() < t.timezone_names.len() {
            &t.timezone_names[tz.to_idx()]
        } else {
            ""
        };
        let area_display = self.areas(t, languages).format();
        out.push_str(&format!(
            ", pos={}, areas={}, tz={} -> score={}",
            self.coordinates, area_display, tz_name, self.score
        ));
        if self.is_duplicate {
            out.push_str(" [DUP]");
        }
        out
    }
}

impl PartialEq for Suggestion {
    fn eq(&self, other: &Self) -> bool {
        self.is_duplicate == other.is_duplicate
            && self.score == other.score
            && self.location == other.location
            && self.area_set == other.area_set
    }
}

impl Eq for Suggestion {}

impl PartialOrd for Suggestion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Suggestion {
    /// C++ `operator<`: sort by (is_duplicate, score, location, area_set).
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.is_duplicate
            .cmp(&other.is_duplicate)
            .then_with(|| {
                self.score
                    .partial_cmp(&other.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| self.location.cmp(&other.location))
            .then_with(|| self.area_set.0.cmp(&other.area_set.0))
    }
}

// ---------------------------------------------------------------------------
// GuessContext — C++ `guess_context`
// ---------------------------------------------------------------------------

use crate::normalize::Phrase as NormalizePhrase;
use crate::sift4::SiftOffset;
use crate::typeahead::{CosSimMatch, MatchItem, ScoredMatch};

/// Reusable context for the suggestion pipeline.
/// C++ `adr::guess_context`.
pub struct GuessContext {
    pub sift4_offset_arr: Vec<SiftOffset>,
    pub phrases: Vec<NormalizePhrase>,
    pub suggestions: Vec<Suggestion>,
    pub cache: Cache,

    pub string_matches: Vec<CosSimMatch>,

    pub area_active: Vec<bool>,

    pub string_phrase_match_scores: Vec<PhraseMatchScores>,
    pub area_phrase_match_scores: Vec<PhraseMatchScores>,
    pub area_phrase_lang: Vec<PhraseLang>,

    pub area_match_items: std::collections::HashMap<AreaSetIdx, Vec<MatchItem>>,
    pub item_matched_masks: std::collections::HashSet<u8>,

    pub scored_street_matches: Vec<ScoredMatch<StreetIdx>>,
    pub scored_place_matches: Vec<ScoredMatch<PlaceIdx>>,

    pub sqrt_len_vec_in: f32,
}

use crate::cache::Cache;

impl GuessContext {
    pub fn new(cache: Cache) -> Self {
        Self {
            sift4_offset_arr: Vec::new(),
            phrases: Vec::new(),
            suggestions: Vec::new(),
            cache,
            string_matches: Vec::new(),
            area_active: Vec::new(),
            string_phrase_match_scores: Vec::new(),
            area_phrase_match_scores: Vec::new(),
            area_phrase_lang: Vec::new(),
            area_match_items: std::collections::HashMap::new(),
            item_matched_masks: std::collections::HashSet::new(),
            scored_street_matches: Vec::new(),
            scored_place_matches: Vec::new(),
            sqrt_len_vec_in: 0.0,
        }
    }

    /// Resize internal buffers to match the typeahead data.
    /// C++ `guess_context::resize()`.
    pub fn resize(&mut self, t: &Typeahead) {
        let n_areas = t.area_names.len();
        self.area_phrase_lang.resize(n_areas, [0u8; MAX_INPUT_PHRASES]);
        self.area_phrase_match_scores
            .resize(n_areas, NO_MATCH_SCORES);
        self.area_active.resize(n_areas, false);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_no_house_number() {
        let addr = Address {
            street: StreetIdx(0),
            house_number: Address::NO_HOUSE_NUMBER,
        };
        assert!(!addr.has_house_number());
    }

    #[test]
    fn address_with_house_number() {
        let addr = Address {
            street: StreetIdx(0),
            house_number: 42,
        };
        assert!(addr.has_house_number());
    }

    #[test]
    fn suggestion_ordering() {
        let s1 = Suggestion::new(
            StringIdx(0),
            SuggestionLocation::Place(PlaceIdx(0)),
            Coordinates::default(),
            AreaSetIdx(0),
            [0u8; 32],
            0,
            0,
            1.0,
        );
        let s2 = Suggestion::new(
            StringIdx(1),
            SuggestionLocation::Place(PlaceIdx(1)),
            Coordinates::default(),
            AreaSetIdx(0),
            [0u8; 32],
            0,
            0,
            2.0,
        );
        assert!(s1 < s2);
    }

    #[test]
    fn suggestion_populate_areas_empty() {
        let t = Typeahead::new();
        let mut s = Suggestion::new(
            StringIdx(0),
            SuggestionLocation::Place(PlaceIdx(0)),
            Coordinates::default(),
            AreaSetIdx(0),
            [0u8; 32],
            0,
            0,
            0.0,
        );
        // Should not panic even with empty typeahead
        s.populate_areas(&t);
        assert_eq!(s.tz, TimezoneIdx::INVALID);
    }
}
