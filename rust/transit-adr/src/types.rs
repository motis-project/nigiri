//! Core types for the ADR module.
//!
//! Provides strongly-typed index wrappers, score types, and shared constants
//! that mirror the C++ `adr::types.h` definitions.

use std::fmt;

// ---------------------------------------------------------------------------
// Strong index macro
// ---------------------------------------------------------------------------

macro_rules! strong_idx {
    ($name:ident, $inner:ty) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        impl $name {
            pub const INVALID: Self = Self(<$inner>::MAX);

            #[inline]
            pub fn to_idx(self) -> usize {
                self.0 as usize
            }

            #[inline]
            pub fn is_valid(self) -> bool {
                self != Self::INVALID
            }
        }

        impl From<usize> for $name {
            #[inline]
            fn from(v: usize) -> Self {
                Self(v as $inner)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Strong typed indices — C++ `cista::strong<T, tag>`
// ---------------------------------------------------------------------------

strong_idx!(TimezoneIdx, u16);
strong_idx!(AreaSetIdx, u32);
strong_idx!(AreaIdx, u32);
strong_idx!(AdminLevel, u8);
strong_idx!(StringIdx, u32);
strong_idx!(StreetIdx, u32);
strong_idx!(PlaceIdx, u32);
strong_idx!(LanguageIdx, u16);

// ---------------------------------------------------------------------------
// Score / edit-distance types
// ---------------------------------------------------------------------------

/// Edit distance type (matching C++ `edit_dist_t = uint8_t`).
pub type EditDist = u8;

/// Maximum edit distance sentinel value.
pub const MAX_EDIT_DIST: EditDist = u8::MAX;

/// Score type for match ranking (matching C++ `score_t = float`).
pub type Score = f32;

/// Sentinel value indicating no match was found.
pub const NO_MATCH: Score = f32::MAX;

/// N-gram type — a compressed bigram stored as `u16`.
pub type Ngram = u16;

/// Token bitmask tracking which input tokens participate in a phrase.
pub type TokenBitmask = u8;

/// Maximum number of input tokens supported (8 bits in bitmask).
pub const MAX_TOKENS: usize = std::mem::size_of::<TokenBitmask>() * 8;

/// Maximum number of generated phrases from input tokens.
pub const MAX_INPUT_PHRASES: usize = 32;

/// C++ `kMaxInputTokens`
pub const MAX_INPUT_TOKENS: usize = 8;

/// Phrase index type. C++ `phrase_idx_t = uint8_t`.
pub type PhraseIdx = u8;

/// Per-phrase match scores array. C++ `phrase_match_scores_t`.
pub type PhraseMatchScores = [Score; MAX_INPUT_PHRASES];

/// Per-phrase language index array. C++ `phrase_lang_t`.
pub type PhraseLang = [u8; MAX_INPUT_PHRASES];

/// Sentinel: all scores = NO_MATCH.
pub const NO_MATCH_SCORES: PhraseMatchScores = [NO_MATCH; MAX_INPUT_PHRASES];

// ---------------------------------------------------------------------------
// Country code — C++ `country_code_t = std::array<char, 2>`
// ---------------------------------------------------------------------------

pub type CountryCode = [u8; 2];
pub const NO_COUNTRY_CODE: CountryCode = [0, 0];

// ---------------------------------------------------------------------------
// Language defaults
// ---------------------------------------------------------------------------

pub const DEFAULT_LANG_IDX: usize = 0;
pub const DEFAULT_LANG: LanguageIdx = LanguageIdx(0);

// ---------------------------------------------------------------------------
// Admin level constants
// ---------------------------------------------------------------------------

pub const POSTAL_CODE_ADMIN_LEVEL: AdminLevel = AdminLevel(12);
pub const TIMEZONE_ADMIN_LEVEL: AdminLevel = AdminLevel(13);

pub const ADMIN_STRINGS: [&str; 12] = [
    "0", "1", "2", "region", "state", "district", "county", "municipality",
    "town", "subtownship", "neighbourhood", "zip",
];

pub fn admin_level_to_str(x: AdminLevel) -> &'static str {
    if (x.0 as usize) < ADMIN_STRINGS.len() {
        ADMIN_STRINGS[x.0 as usize]
    } else {
        ""
    }
}

// ---------------------------------------------------------------------------
// Area set language mapping — C++ `area_set_lang_t = std::array<uint8_t, 32>`
// ---------------------------------------------------------------------------

pub type AreaSetLang = [u8; 32];

// ---------------------------------------------------------------------------
// Location type — C++ `location_type_t`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LocationType {
    Place = 0,
    Street = 1,
}

// ---------------------------------------------------------------------------
// Filter type — C++ `filter_type`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FilterType {
    None = 0,
    Address = 1,
    Place = 2,
    Extra = 3,
}

impl Default for FilterType {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// Coordinates — C++ `coordinates` (i32 lat/lng, osmium fixed-point)
// ---------------------------------------------------------------------------

/// Coordinates stored as osmium fixed-point i32 values.
/// Conversion: degrees = value / 10_000_000.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Coordinates {
    pub lat: i32,
    pub lng: i32,
}

impl Coordinates {
    pub const SCALE: f64 = 10_000_000.0;

    pub fn from_lat_lng(lat: f64, lng: f64) -> Self {
        Self {
            lat: (lat * Self::SCALE) as i32,
            lng: (lng * Self::SCALE) as i32,
        }
    }

    pub fn lat_deg(&self) -> f64 {
        self.lat as f64 / Self::SCALE
    }

    pub fn lng_deg(&self) -> f64 {
        self.lng as f64 / Self::SCALE
    }

    /// Haversine distance in meters to another coordinate.
    pub fn distance_to(&self, other: &Coordinates) -> f64 {
        let lat1 = self.lat_deg().to_radians();
        let lat2 = other.lat_deg().to_radians();
        let dlat = (other.lat_deg() - self.lat_deg()).to_radians();
        let dlng = (other.lng_deg() - self.lng_deg()).to_radians();
        let a = (dlat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (dlng / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();
        6_371_000.0 * c
    }
}

impl fmt::Display for Coordinates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.lat_deg(), self.lng_deg())
    }
}

// ---------------------------------------------------------------------------
// Population — C++ `population` (compressed u16, factor 200)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Population {
    pub value: u16,
}

impl Population {
    pub const COMPRESSION_FACTOR: u32 = 200;

    pub fn get(&self) -> u32 {
        self.value as u32 * Self::COMPRESSION_FACTOR
    }

    pub fn from_raw(pop: u32) -> Self {
        Self {
            value: (pop / Self::COMPRESSION_FACTOR) as u16,
        }
    }
}

// ---------------------------------------------------------------------------
// Token position — C++ `token` in adr.h
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct TokenPos {
    pub start_idx: u16,
    pub size: u16,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_cpp() {
        assert_eq!(MAX_TOKENS, 8);
        assert_eq!(MAX_INPUT_PHRASES, 32);
        assert_eq!(MAX_EDIT_DIST, 255);
        assert_eq!(NO_MATCH, f32::MAX);
    }

    #[test]
    fn strong_idx_invalid() {
        assert_eq!(TimezoneIdx::INVALID, TimezoneIdx(u16::MAX));
        assert_eq!(AreaSetIdx::INVALID, AreaSetIdx(u32::MAX));
        assert_eq!(StringIdx::INVALID, StringIdx(u32::MAX));
        assert!(!TimezoneIdx::INVALID.is_valid());
        assert!(TimezoneIdx(0).is_valid());
    }

    #[test]
    fn coordinates_conversion() {
        let c = Coordinates::from_lat_lng(51.5074, -0.1278);
        assert!((c.lat_deg() - 51.5074).abs() < 1e-6);
        assert!((c.lng_deg() - (-0.1278)).abs() < 1e-6);
    }

    #[test]
    fn population_compression() {
        let p = Population::from_raw(1_000_000);
        assert_eq!(p.value, 5000);
        assert_eq!(p.get(), 1_000_000);
    }

    #[test]
    fn admin_level_constants() {
        assert_eq!(POSTAL_CODE_ADMIN_LEVEL.0, 12);
        assert_eq!(TIMEZONE_ADMIN_LEVEL.0, 13);
        assert_eq!(admin_level_to_str(AdminLevel(3)), "region");
        assert_eq!(admin_level_to_str(AdminLevel(11)), "zip");
        assert_eq!(admin_level_to_str(AdminLevel(99)), "");
    }

    #[test]
    fn no_match_scores_sentinel() {
        for &s in &NO_MATCH_SCORES {
            assert_eq!(s, NO_MATCH);
        }
    }
}
