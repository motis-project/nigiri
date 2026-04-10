//! Typeahead database — core data store for address search.
//!
//! Mirrors C++ `adr/typeahead.h` + `typeahead.cc`. Holds all area, place,
//! street, and string data indexed by strong typed indices.

use std::collections::HashMap;

use crate::cache::{Cache, NgramSet};
use crate::categories::AmenityCategory;
use crate::import_context::ImportContext;
use crate::ngram::{compress_bigram, for_each_bigram, split_ngrams, N_BIGRAMS};
use crate::normalize::normalize;
use crate::types::*;

// ---------------------------------------------------------------------------
// Population re-export (C++ puts it in typeahead.h)
// ---------------------------------------------------------------------------

/// Language map: language string → LanguageIdx. C++ `lang_map_t`.
pub type LangMap = HashMap<String, LanguageIdx>;

/// Find a language in a language list. Returns index or -1.
/// C++ `find_lang()`.
pub fn find_lang(langs: &[LanguageIdx], l: LanguageIdx) -> i16 {
    if langs.is_empty() {
        return -1;
    }
    if l == DEFAULT_LANG {
        return DEFAULT_LANG_IDX as i16;
    }
    match langs.iter().position(|&x| x == l) {
        Some(pos) => pos as i16,
        None => -1,
    }
}

// ---------------------------------------------------------------------------
// Typeahead struct — the core database
// ---------------------------------------------------------------------------

/// The typeahead database containing all address data.
///
/// C++ `adr::typeahead`. All `vecvec<K, V>` → `Vec<Vec<V>>`,
/// `vector_map<K, V>` → `Vec<V>` (indexed by K).
pub struct Typeahead {
    // --- Timezone data ---
    /// Timezone names. C++ `vecvec<timezone_idx_t, char>`.
    pub timezone_names: Vec<String>,

    // --- Area data ---
    /// Area names per area (multi-language). C++ `vecvec<area_idx_t, string_idx_t>`.
    pub area_names: Vec<Vec<StringIdx>>,
    /// Language of each area name. C++ `vecvec<area_idx_t, language_idx_t>`.
    pub area_name_lang: Vec<Vec<LanguageIdx>>,
    /// Admin level per area. C++ `vector_map<area_idx_t, admin_level_t>`.
    pub area_admin_level: Vec<AdminLevel>,
    /// Population per area. C++ `vector_map<area_idx_t, population>`.
    pub area_population: Vec<Population>,
    /// Timezone per area. C++ `vector_map<area_idx_t, timezone_idx_t>`.
    pub area_timezone: Vec<TimezoneIdx>,
    /// Country code per area. C++ `vector_map<area_idx_t, country_code_t>`.
    pub area_country_code: Vec<CountryCode>,

    // --- Place data ---
    /// Place names (multi-language). C++ `vecvec<place_idx_t, string_idx_t>`.
    pub place_names: Vec<Vec<StringIdx>>,
    /// Language of each place name. C++ `vecvec<place_idx_t, language_idx_t>`.
    pub place_name_lang: Vec<Vec<LanguageIdx>>,
    /// OSM IDs per place. C++ `vecvec<place_idx_t, int64_t>`.
    pub place_osm_ids: Vec<Vec<i64>>,
    /// Place coordinates. C++ `vector_map<place_idx_t, coordinates>`.
    pub place_coordinates: Vec<Coordinates>,
    /// Area set per place. C++ `vector_map<place_idx_t, area_set_idx_t>`.
    pub place_areas: Vec<AreaSetIdx>,
    /// Population per place. C++ `vector_map<place_idx_t, population>`.
    pub place_population: Vec<Population>,
    /// Amenity category per place. C++ `vector_map<place_idx_t, amenity_category>`.
    pub place_type: Vec<AmenityCategory>,
    /// Whether the place is a way (vs node). C++ `bitvec`.
    pub place_is_way: Vec<bool>,

    // --- Street data ---
    /// Street names. C++ `vecvec<street_idx_t, string_idx_t>`.
    pub street_names: Vec<Vec<StringIdx>>,
    /// Street positions. C++ `vecvec<street_idx_t, coordinates>`.
    pub street_pos: Vec<Vec<Coordinates>>,
    /// Street area sets. C++ `vecvec<street_idx_t, area_set_idx_t>`.
    pub street_areas: Vec<Vec<AreaSetIdx>>,
    /// House numbers per street. C++ `vecvec<street_idx_t, string_idx_t>`.
    pub house_numbers: Vec<Vec<StringIdx>>,
    /// House coordinates per street. C++ `vecvec<street_idx_t, coordinates>`.
    pub house_coordinates: Vec<Vec<Coordinates>>,
    /// House area sets per street. C++ `vecvec<street_idx_t, area_set_idx_t>`.
    pub house_areas: Vec<Vec<AreaSetIdx>>,

    // --- Area sets ---
    /// Area sets (each is a list of area indices). C++ `vecvec<area_set_idx_t, area_idx_t>`.
    pub area_sets: Vec<Vec<AreaIdx>>,

    // --- String table ---
    /// All strings. C++ `vecvec<string_idx_t, char>`.
    pub strings: Vec<String>,

    // --- Bigram index ---
    /// Number of bigrams per string (denominator for cosine similarity).
    /// C++ `vector_map<string_idx_t, uint8_t>`.
    pub n_bigrams: Vec<u8>,
    /// Bigram → string indices. C++ `vecvec<ngram_t, string_idx_t, uint32_t>`.
    pub bigrams: Vec<Vec<StringIdx>>,

    // --- String→location mapping ---
    /// Location indices per string. C++ `vecvec<string_idx_t, uint32_t>`.
    pub string_to_location: Vec<Vec<u32>>,
    /// Location type per mapping. C++ `vecvec<string_idx_t, location_type_t>`.
    pub string_to_type: Vec<Vec<LocationType>>,

    // --- Language data ---
    /// Language string→index map. C++ `lang_map_t`.
    pub lang: LangMap,
    /// Language names. C++ `vecvec<language_idx_t, char, uint32_t>`.
    pub lang_names: Vec<String>,

    /// Index where "extra" places begin. C++ `ext_start_`.
    pub ext_start: u32,
}

impl Typeahead {
    pub fn new() -> Self {
        Self {
            timezone_names: Vec::new(),
            area_names: Vec::new(),
            area_name_lang: Vec::new(),
            area_admin_level: Vec::new(),
            area_population: Vec::new(),
            area_timezone: Vec::new(),
            area_country_code: Vec::new(),
            place_names: Vec::new(),
            place_name_lang: Vec::new(),
            place_osm_ids: Vec::new(),
            place_coordinates: Vec::new(),
            place_areas: Vec::new(),
            place_population: Vec::new(),
            place_type: Vec::new(),
            place_is_way: Vec::new(),
            street_names: Vec::new(),
            street_pos: Vec::new(),
            street_areas: Vec::new(),
            house_numbers: Vec::new(),
            house_coordinates: Vec::new(),
            house_areas: Vec::new(),
            area_sets: Vec::new(),
            strings: Vec::new(),
            n_bigrams: Vec::new(),
            bigrams: Vec::new(),
            string_to_location: Vec::new(),
            string_to_type: Vec::new(),
            lang: LangMap::new(),
            lang_names: Vec::new(),
            ext_start: 0,
        }
    }

    /// Build the bigram index from the string table.
    /// C++ `typeahead::build_ngram_index()`.
    pub fn build_ngram_index(&mut self) {
        let mut tmp: Vec<Vec<StringIdx>> = vec![Vec::new(); N_BIGRAMS];
        self.n_bigrams.resize(self.strings.len(), 0);

        for (i, s) in self.strings.iter().enumerate() {
            let normalized = normalize(s);
            // C++ stores min(255, normalized.size() - 1) as the denominator
            self.n_bigrams[i] =
                (normalized.len().saturating_sub(1)).min(255) as u8;
            for_each_bigram(&normalized, |bigram| {
                let compressed = compress_bigram(bigram);
                tmp[compressed as usize].push(StringIdx(i as u32));
            });
        }

        self.bigrams.clear();
        for bucket in &mut tmp {
            bucket.sort_unstable();
            bucket.dedup();
            self.bigrams.push(std::mem::take(bucket));
        }
    }

    /// Guess candidate strings by cosine similarity of bigrams.
    /// C++ `typeahead::guess()`.
    pub fn guess(
        &self,
        normalized: &str,
        cache: &mut Cache,
        string_matches: &mut Vec<CosSimMatch>,
    ) {
        string_matches.clear();

        if normalized.len() < 2 {
            return;
        }

        let _sqrt_len_vec_in = ((normalized.len() - 1) as f32).sqrt();
        let in_ngrams = split_ngrams(normalized);
        let n_in_ngrams = in_ngrams.len();

        let ngram_set: NgramSet = in_ngrams.iter().copied().collect();
        let (mut string_match_counts, missing) = cache.get_closest(&ngram_set);

        // Ensure the counts vector is large enough.
        if string_match_counts.len() < self.strings.len() {
            string_match_counts.resize(self.strings.len(), 0);
        }

        // Process missing ngrams incrementally.
        for &missing_ngram in &missing {
            if (missing_ngram as usize) < self.bigrams.len() {
                for &string_idx in &self.bigrams[missing_ngram as usize] {
                    let idx = string_idx.to_idx();
                    if idx < string_match_counts.len() {
                        string_match_counts[idx] =
                            string_match_counts[idx].saturating_add(1);
                    }
                }
            }
        }
        cache.add(ngram_set, string_match_counts.clone());

        // Compute cosine similarity.
        let min_match_count =
            2 + n_in_ngrams / (4 + n_in_ngrams / 10);
        let n_strings = self.strings.len();
        const CUTOFF: f32 = 0.17;

        for i in 0..n_strings {
            if (string_match_counts[i] as usize) < min_match_count {
                continue;
            }
            let match_count = string_match_counts[i] as f32;
            let n_bi = self.n_bigrams[i] as f32;
            if n_bi == 0.0 {
                continue;
            }
            let cos_sim =
                (match_count * match_count) / (n_bi * n_in_ngrams as f32);
            if cos_sim >= CUTOFF {
                string_matches.push(CosSimMatch {
                    idx: StringIdx(i as u32),
                    cos_sim,
                });
            }
        }

        // Sort descending by cos_sim (C++ operator< is cos_sim_ > o.cos_sim_).
        string_matches.sort_by(|a, b| {
            b.cos_sim
                .partial_cmp(&a.cos_sim)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the timezone for an area set.
    /// C++ `typeahead::get_tz()`.
    pub fn get_tz(&self, area_set: AreaSetIdx) -> TimezoneIdx {
        if !area_set.is_valid() || area_set.to_idx() >= self.area_sets.len() {
            return TimezoneIdx::INVALID;
        }
        let areas = &self.area_sets[area_set.to_idx()];
        areas
            .iter()
            .filter(|&&a| {
                let idx = a.to_idx();
                idx < self.area_timezone.len()
                    && self.area_timezone[idx].is_valid()
            })
            .min_by_key(|&&a| {
                // Prefer areas with timezone AND lowest admin level
                let has_tz = !self.area_timezone[a.to_idx()].is_valid();
                let neg_level = -(self.area_admin_level[a.to_idx()].0 as i16);
                (has_tz, neg_level)
            })
            .map(|&a| self.area_timezone[a.to_idx()])
            .unwrap_or(TimezoneIdx::INVALID)
    }

    /// Resolve a language string to its index.
    /// C++ `typeahead::resolve_language()`.
    pub fn resolve_language(&self, s: &str) -> LanguageIdx {
        self.lang
            .get(s)
            .copied()
            .unwrap_or(LanguageIdx::INVALID)
    }

    /// Verify consistency of string→location mappings.
    /// C++ `typeahead::verify()`.
    pub fn verify(&self) -> bool {
        for (i, (locations, types)) in self
            .string_to_location
            .iter()
            .zip(self.string_to_type.iter())
            .enumerate()
        {
            for (&loc, &loc_type) in locations.iter().zip(types.iter()) {
                match loc_type {
                    LocationType::Street => {
                        let street_idx = loc as usize;
                        if street_idx >= self.street_names.len()
                            || !self.street_names[street_idx]
                                .contains(&StringIdx(i as u32))
                        {
                            return false;
                        }
                    }
                    LocationType::Place => {
                        let place_idx = loc as usize;
                        if place_idx >= self.place_names.len()
                            || !self.place_names[place_idx]
                                .contains(&StringIdx(i as u32))
                        {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Import functions — C++ `typeahead.cc`
    // -----------------------------------------------------------------------

    /// Get or create a string index. C++ `typeahead::get_or_create_string()`.
    pub fn get_or_create_string(&mut self, ctx: &mut ImportContext, s: &str) -> StringIdx {
        if let Some(&idx) = ctx.string_lookup.get(s) {
            return idx;
        }
        let idx = StringIdx(self.strings.len() as u32);
        self.strings.push(s.to_string());
        ctx.string_lookup.insert(s.to_string(), idx);
        idx
    }

    /// Get or create a language index. C++ `typeahead::get_or_create_lang_idx()`.
    pub fn get_or_create_lang_idx(&mut self, s: &str) -> LanguageIdx {
        if let Some(&idx) = self.lang.get(s) {
            return idx;
        }
        // +1 skips zero as 0 == default language
        let idx = LanguageIdx((self.lang.len() + 1) as u16);
        self.lang_names.push(s.to_string());
        self.lang.insert(s.to_string(), idx);
        idx
    }

    /// Get or create a timezone index. C++ `typeahead::get_or_create_timezone()`.
    pub fn get_or_create_timezone(
        &mut self,
        ctx: &mut ImportContext,
        tz: &str,
    ) -> TimezoneIdx {
        if tz.is_empty() {
            return TimezoneIdx::INVALID;
        }
        if let Some(&idx) = ctx.tz_lookup.get(tz) {
            return idx;
        }
        let idx = TimezoneIdx(self.timezone_names.len() as u16);
        self.timezone_names.push(tz.to_string());
        ctx.tz_lookup.insert(tz.to_string(), idx);
        idx
    }

    /// Get or create a street index. C++ `typeahead::get_or_create_street()`.
    pub fn get_or_create_street(
        &mut self,
        ctx: &mut ImportContext,
        street_name: &str,
    ) -> StreetIdx {
        let string_idx = self.get_or_create_string(ctx, street_name);
        if let Some(&idx) = ctx.street_lookup.get(&string_idx) {
            return idx;
        }
        let idx = StreetIdx(self.street_names.len() as u32);

        // Register string→location mapping.
        let si = string_idx.to_idx();
        while self.string_to_location.len() <= si {
            self.string_to_location.push(Vec::new());
            self.string_to_type.push(Vec::new());
        }
        self.string_to_location[si].push(idx.0);
        self.string_to_type[si].push(LocationType::Street);

        self.street_names.push(vec![string_idx]);
        ctx.street_lookup.insert(string_idx, idx);
        idx
    }

    /// Get or create an area set. C++ `typeahead::get_or_create_area_set()`.
    pub fn get_or_create_area_set(
        &mut self,
        ctx: &mut ImportContext,
        areas: &[AreaIdx],
    ) -> AreaSetIdx {
        if let Some(&idx) = ctx.area_set_lookup.get(areas) {
            return idx;
        }
        let idx = AreaSetIdx(self.area_sets.len() as u32);
        self.area_sets.push(areas.to_vec());
        ctx.area_set_lookup.insert(areas.to_vec(), idx);
        idx
    }

    /// Add a postal code area. C++ `typeahead::add_postal_code_area()`.
    /// Returns the new area index, or `AreaIdx::INVALID` if postal_code is empty.
    pub fn add_postal_code_area(
        &mut self,
        ctx: &mut ImportContext,
        postal_code: &str,
    ) -> AreaIdx {
        if postal_code.is_empty() {
            return AreaIdx::INVALID;
        }
        let str_idx = self.get_or_create_string(ctx, postal_code);
        self.area_population.push(Population::default());
        self.area_names.push(vec![str_idx]);
        self.area_name_lang.push(vec![DEFAULT_LANG]);
        self.area_timezone.push(TimezoneIdx::INVALID);
        self.area_country_code.push(NO_COUNTRY_CODE);
        let idx = AreaIdx(self.area_admin_level.len() as u32);
        self.area_admin_level.push(POSTAL_CODE_ADMIN_LEVEL);
        idx
    }

    /// Add a timezone area. C++ `typeahead::add_timezone_area()`.
    pub fn add_timezone_area(
        &mut self,
        ctx: &mut ImportContext,
        timezone: &str,
    ) -> AreaIdx {
        if timezone.is_empty() {
            return AreaIdx::INVALID;
        }
        let str_idx = self.get_or_create_string(ctx, timezone);
        let tz_idx = self.get_or_create_timezone(ctx, timezone);
        self.area_population.push(Population::default());
        self.area_names.push(vec![str_idx]);
        self.area_name_lang.push(Vec::new());
        self.area_timezone.push(tz_idx);
        self.area_country_code.push(NO_COUNTRY_CODE);
        let idx = AreaIdx(self.area_admin_level.len() as u32);
        self.area_admin_level.push(TIMEZONE_ADMIN_LEVEL);
        idx
    }

    /// Add an admin area. C++ `typeahead::add_admin_area()`.
    ///
    /// `admin_level`: numeric admin level (only 2–11 accepted).
    /// `name`: primary name.
    /// `names_langs`: additional (name, language) pairs from `for_each_name`.
    /// `population_raw`: raw population value (will be compressed).
    /// `timezone`: optional timezone string.
    /// `country_code`: optional ISO 3166-1 two-letter code.
    pub fn add_admin_area(
        &mut self,
        ctx: &mut ImportContext,
        admin_level: u8,
        name: &str,
        names_langs: &[(&str, LanguageIdx)],
        population_raw: u32,
        timezone: Option<&str>,
        country_code: Option<&[u8; 2]>,
    ) -> AreaIdx {
        if !(2..=11).contains(&admin_level) || name.is_empty() {
            return AreaIdx::INVALID;
        }

        let mut names = Vec::new();
        let mut langs = Vec::new();

        // Add primary name with default language.
        let primary_str = self.get_or_create_string(ctx, name);
        names.push(primary_str);
        langs.push(DEFAULT_LANG);

        // Add additional name/lang pairs.
        for &(alt_name, lang) in names_langs {
            let str_idx = self.get_or_create_string(ctx, alt_name);
            names.push(str_idx);
            langs.push(lang);
        }

        self.area_names.push(names);
        self.area_name_lang.push(langs);
        self.area_population.push(Population::from_raw(population_raw));

        let tz_idx = match timezone {
            Some(tz) => self.get_or_create_timezone(ctx, tz),
            None => TimezoneIdx::INVALID,
        };
        self.area_timezone.push(tz_idx);

        let cc = match country_code {
            Some(c) => *c,
            None => NO_COUNTRY_CODE,
        };
        self.area_country_code.push(cc);

        let idx = AreaIdx(self.area_admin_level.len() as u32);
        self.area_admin_level.push(AdminLevel(admin_level));
        idx
    }

    /// Add an address (house number on a street).
    /// C++ `typeahead::add_address()`.
    pub fn add_address(
        &mut self,
        ctx: &mut ImportContext,
        street_name: &str,
        house_number: &str,
        coord: Coordinates,
        area_set: AreaSetIdx,
    ) {
        if house_number.is_empty() || street_name.is_empty() {
            return;
        }
        let street_idx = self.get_or_create_street(ctx, street_name);
        let hn_str_idx = self.get_or_create_string(ctx, house_number);
        let si = street_idx.to_idx();

        // Grow parallel vectors if needed.
        while self.house_numbers.len() <= si {
            self.house_numbers.push(Vec::new());
            self.house_coordinates.push(Vec::new());
            self.house_areas.push(Vec::new());
        }
        self.house_numbers[si].push(hn_str_idx);
        self.house_coordinates[si].push(coord);
        self.house_areas[si].push(area_set);
    }

    /// Add a street. Deduplicates by haversine distance < 1500m.
    /// C++ `typeahead::add_street()`.
    pub fn add_street(
        &mut self,
        ctx: &mut ImportContext,
        name: &str,
        coord: Coordinates,
        area_set: AreaSetIdx,
    ) -> StreetIdx {
        if name.is_empty() {
            return StreetIdx::INVALID;
        }
        let street_idx = self.get_or_create_street(ctx, name);
        let si = street_idx.to_idx();

        // Grow parallel vectors if needed.
        while self.street_pos.len() <= si {
            self.street_pos.push(Vec::new());
            self.street_areas.push(Vec::new());
        }

        // Check dedup by distance.
        for &existing_pos in &self.street_pos[si] {
            if coord.distance_to(&existing_pos) < 1500.0 {
                return street_idx;
            }
        }
        self.street_pos[si].push(coord);
        self.street_areas[si].push(area_set);
        street_idx
    }

    /// Add a place. C++ `typeahead::add_place()`.
    pub fn add_place(
        &mut self,
        ctx: &mut ImportContext,
        osm_id: i64,
        is_way: bool,
        coord: Coordinates,
        area_set: AreaSetIdx,
        category: AmenityCategory,
        population_raw: u32,
        names_langs: &[(&str, LanguageIdx)],
    ) -> PlaceIdx {
        let idx = PlaceIdx(self.place_names.len() as u32);

        let mut names = Vec::new();
        let mut langs = Vec::new();
        for &(name, lang) in names_langs {
            let str_idx = self.get_or_create_string(ctx, name);

            // Register string→location mapping.
            let si = str_idx.to_idx();
            while self.string_to_location.len() <= si {
                self.string_to_location.push(Vec::new());
                self.string_to_type.push(Vec::new());
            }
            self.string_to_location[si].push(idx.0);
            self.string_to_type[si].push(LocationType::Place);

            names.push(str_idx);
            langs.push(lang);
        }

        self.place_names.push(names);
        self.place_name_lang.push(langs);
        self.place_osm_ids.push(vec![osm_id]);
        self.place_coordinates.push(coord);
        self.place_areas.push(area_set);
        self.place_population.push(Population::from_raw(population_raw));
        self.place_type.push(category);
        self.place_is_way.push(is_way);

        idx
    }
}

/// Iterate over OSM-style name tags, calling `f` for each (name, language).
/// C++ `for_each_name()`.
///
/// Parses tags like `name`, `old_name`, `alt_name`, `short_name`,
/// `official_name`, and prefixed variants like `name:de`, `alt_name:en`.
/// Semi-colon-separated values are split into individual names.
pub fn for_each_name<F>(
    t: &mut Typeahead,
    tags: &[(&str, &str)],
    mut f: F,
) where
    F: FnMut(&str, LanguageIdx),
{
    let call_fn = |name: &str, lang: LanguageIdx, f: &mut F| {
        for token in name.split(';') {
            let token = token.trim();
            if !token.is_empty() {
                f(token, lang);
            }
        }
    };

    // Process simple name tags with default language.
    for prefix in &["name", "old_name", "alt_name", "short_name", "official_name"] {
        for &(key, value) in tags {
            if key == *prefix {
                call_fn(value, DEFAULT_LANG, &mut f);
            }
        }
    }

    // Process language-prefixed name tags (e.g., "name:de", "alt_name:en").
    let lang_prefixes = ["name:", "short_name:", "alt_name:", "official_name:"];
    for prefix in &lang_prefixes {
        for &(key, value) in tags {
            if let Some(lang_str) = key.strip_prefix(prefix) {
                let lang = t.get_or_create_lang_idx(lang_str);
                call_fn(value, lang, &mut f);
            }
        }
    }
}

impl Default for Typeahead {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity match — C++ `cos_sim_match`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct CosSimMatch {
    pub idx: StringIdx,
    pub cos_sim: f32,
}

impl CosSimMatch {
    /// C++ operator< sorts by cos_sim descending.
    pub fn cmp_desc(&self, other: &Self) -> std::cmp::Ordering {
        other
            .cos_sim
            .partial_cmp(&self.cos_sim)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Scored match — C++ `scored_match<T>`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct ScoredMatch<T: Copy> {
    pub score: Score,
    pub phrase_idx: PhraseIdx,
    pub string_idx: StringIdx,
    pub idx: T,
}

impl<T: Copy + PartialEq> PartialEq for ScoredMatch<T> {
    fn eq(&self, _: &Self) -> bool {
        false // C++ operator== always returns false
    }
}

impl<T: Copy + PartialOrd> PartialOrd for ScoredMatch<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl<T: Copy + PartialOrd> Ord for ScoredMatch<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<T: Copy + PartialOrd> Eq for ScoredMatch<T> {}

// ---------------------------------------------------------------------------
// Match item — C++ `match_item`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchItemType {
    Street,
    HouseNumber,
    Place,
}

#[derive(Debug, Clone, Copy)]
pub struct MatchItem {
    pub item_type: MatchItemType,
    pub score: Score,
    pub index: u32,
    pub house_number_p_idx: PhraseIdx,
    pub matched_mask: u8,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_lang_default() {
        let langs = vec![LanguageIdx(0), LanguageIdx(5), LanguageIdx(10)];
        assert_eq!(find_lang(&langs, DEFAULT_LANG), DEFAULT_LANG_IDX as i16);
    }

    #[test]
    fn find_lang_found() {
        let langs = vec![LanguageIdx(0), LanguageIdx(5), LanguageIdx(10)];
        assert_eq!(find_lang(&langs, LanguageIdx(10)), 2);
    }

    #[test]
    fn find_lang_not_found() {
        let langs = vec![LanguageIdx(0), LanguageIdx(5)];
        assert_eq!(find_lang(&langs, LanguageIdx(99)), -1);
    }

    #[test]
    fn find_lang_empty() {
        assert_eq!(find_lang(&[], LanguageIdx(0)), -1);
    }

    #[test]
    fn typeahead_build_ngram_index() {
        let mut t = Typeahead::new();
        t.strings.push("darmstadt".to_string());
        t.strings.push("frankfurt".to_string());
        t.build_ngram_index();
        assert_eq!(t.n_bigrams.len(), 2);
        assert_eq!(t.n_bigrams[0], 8); // "darmstadt".len() - 1
        assert_eq!(t.n_bigrams[1], 8); // "frankfurt".len() - 1
        assert!(!t.bigrams.is_empty());
    }

    #[test]
    fn typeahead_guess_basic() {
        let mut t = Typeahead::new();
        t.strings.push("darmstadt".to_string());
        t.strings.push("frankfurt".to_string());
        t.strings.push("aschaffenburg".to_string());
        t.build_ngram_index();

        let mut cache = Cache::new(t.strings.len(), 10);
        let mut matches = Vec::new();
        t.guess("darmstadt", &mut cache, &mut matches);
        // Should find darmstadt as a strong match
        assert!(!matches.is_empty());
        assert_eq!(matches[0].idx, StringIdx(0));
    }

    #[test]
    fn typeahead_verify_empty() {
        let t = Typeahead::new();
        assert!(t.verify());
    }

    // --- Import function tests ---

    #[test]
    fn get_or_create_string_dedup() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        let idx1 = t.get_or_create_string(&mut ctx, "darmstadt");
        let idx2 = t.get_or_create_string(&mut ctx, "darmstadt");
        let idx3 = t.get_or_create_string(&mut ctx, "frankfurt");
        assert_eq!(idx1, idx2);
        assert_ne!(idx1, idx3);
        assert_eq!(t.strings.len(), 2);
    }

    #[test]
    fn get_or_create_lang_idx_dedup() {
        let mut t = Typeahead::new();
        let de1 = t.get_or_create_lang_idx("de");
        let de2 = t.get_or_create_lang_idx("de");
        let en = t.get_or_create_lang_idx("en");
        assert_eq!(de1, de2);
        assert_ne!(de1, en);
        // Index starts at 1 (0 = default)
        assert!(de1.0 >= 1);
    }

    #[test]
    fn get_or_create_timezone_dedup() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        let tz1 = t.get_or_create_timezone(&mut ctx, "Europe/Berlin");
        let tz2 = t.get_or_create_timezone(&mut ctx, "Europe/Berlin");
        let tz3 = t.get_or_create_timezone(&mut ctx, "America/New_York");
        assert_eq!(tz1, tz2);
        assert_ne!(tz1, tz3);
        assert_eq!(t.timezone_names.len(), 2);
    }

    #[test]
    fn get_or_create_timezone_empty() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        assert_eq!(t.get_or_create_timezone(&mut ctx, ""), TimezoneIdx::INVALID);
    }

    #[test]
    fn add_postal_code_area_basic() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        let idx = t.add_postal_code_area(&mut ctx, "64283");
        assert!(idx.is_valid());
        assert_eq!(t.area_admin_level[idx.to_idx()], POSTAL_CODE_ADMIN_LEVEL);
        assert_eq!(t.area_names[idx.to_idx()].len(), 1);
    }

    #[test]
    fn add_postal_code_area_empty() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        assert_eq!(t.add_postal_code_area(&mut ctx, ""), AreaIdx::INVALID);
    }

    #[test]
    fn add_timezone_area_basic() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        let idx = t.add_timezone_area(&mut ctx, "Europe/Berlin");
        assert!(idx.is_valid());
        assert_eq!(t.area_admin_level[idx.to_idx()], TIMEZONE_ADMIN_LEVEL);
        assert!(t.area_timezone[idx.to_idx()].is_valid());
    }

    #[test]
    fn add_admin_area_basic() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        let idx = t.add_admin_area(
            &mut ctx,
            8, // municipality
            "Darmstadt",
            &[],
            300_000,
            Some("Europe/Berlin"),
            Some(b"DE"),
        );
        assert!(idx.is_valid());
        assert_eq!(t.area_admin_level[idx.to_idx()].0, 8);
        assert_eq!(t.area_population[idx.to_idx()].get(), 300_000);
        assert_eq!(t.area_country_code[idx.to_idx()], *b"DE");
    }

    #[test]
    fn add_admin_area_rejects_invalid_level() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        assert_eq!(
            t.add_admin_area(&mut ctx, 1, "Bad", &[], 0, None, None),
            AreaIdx::INVALID
        );
        assert_eq!(
            t.add_admin_area(&mut ctx, 12, "Bad", &[], 0, None, None),
            AreaIdx::INVALID
        );
    }

    #[test]
    fn add_street_dedup_by_distance() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        let area = t.get_or_create_area_set(&mut ctx, &[]);
        let c1 = Coordinates::from_lat_lng(49.87, 8.65);
        let c2 = Coordinates::from_lat_lng(49.8701, 8.6501); // ~10m away
        let c3 = Coordinates::from_lat_lng(51.0, 10.0); // far away

        let s1 = t.add_street(&mut ctx, "Hauptstraße", c1, area);
        let s2 = t.add_street(&mut ctx, "Hauptstraße", c2, area); // dedup'd
        let s3 = t.add_street(&mut ctx, "Hauptstraße", c3, area); // new position

        assert_eq!(s1, s2);
        assert_eq!(s1, s3); // same street idx
        // But s3 added a new position entry
        assert_eq!(t.street_pos[s1.to_idx()].len(), 2);
    }

    #[test]
    fn add_address_basic() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        let area = t.get_or_create_area_set(&mut ctx, &[]);
        let coord = Coordinates::from_lat_lng(49.87, 8.65);
        t.add_address(&mut ctx, "Hauptstraße", "42", coord, area);
        let street_idx = ctx.street_lookup[&ctx.string_lookup["Hauptstraße"]];
        assert_eq!(t.house_numbers[street_idx.to_idx()].len(), 1);
        assert_eq!(t.house_coordinates[street_idx.to_idx()].len(), 1);
    }

    #[test]
    fn add_place_basic() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        let area = t.get_or_create_area_set(&mut ctx, &[]);
        let coord = Coordinates::from_lat_lng(49.87, 8.65);
        let idx = t.add_place(
            &mut ctx,
            12345,
            false,
            coord,
            area,
            AmenityCategory::City,
            300_000,
            &[("Darmstadt", DEFAULT_LANG)],
        );
        assert!(idx.is_valid());
        assert_eq!(t.place_coordinates[idx.to_idx()], coord);
        assert_eq!(t.place_type[idx.to_idx()], AmenityCategory::City);
        assert!(!t.place_is_way[idx.to_idx()]);
    }

    #[test]
    fn for_each_name_basic() {
        let mut t = Typeahead::new();
        let tags = vec![
            ("name", "Darmstadt"),
            ("name:de", "Darmstadt"),
            ("name:en", "Darmstadt"),
            ("alt_name", "DA"),
        ];
        let mut names = Vec::new();
        for_each_name(&mut t, &tags, |name, lang| {
            names.push((name.to_string(), lang));
        });
        // Should have: "Darmstadt" (default), "DA" (default), "Darmstadt" (de), "Darmstadt" (en)
        assert!(names.len() >= 3);
        assert!(names.iter().any(|(n, _)| n == "DA"));
    }

    #[test]
    fn for_each_name_semicolon_split() {
        let mut t = Typeahead::new();
        let tags = vec![("name", "A;B;C")];
        let mut names = Vec::new();
        for_each_name(&mut t, &tags, |name, _| {
            names.push(name.to_string());
        });
        assert_eq!(names, vec!["A", "B", "C"]);
    }

    #[test]
    fn import_and_verify() {
        let mut t = Typeahead::new();
        let mut ctx = ImportContext::new();
        let area = t.get_or_create_area_set(&mut ctx, &[]);
        let coord = Coordinates::from_lat_lng(49.87, 8.65);

        // Add a place and a street.
        t.add_place(
            &mut ctx,
            1,
            false,
            coord,
            area,
            AmenityCategory::City,
            0,
            &[("Darmstadt", DEFAULT_LANG)],
        );
        t.add_street(&mut ctx, "Hauptstraße", coord, area);

        assert!(t.verify());
    }
}
