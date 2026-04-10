//! Amenity categories for places.
//!
//! Ported from C++ `adr/categories.h`. Only the variants used in scoring logic
//! are named explicitly; the hundreds of POI types use `Other(u16)`.

/// Amenity category for a place. C++ `amenity_category` enum.
///
/// The C++ enum has ~308 variants (auto-generated from OSM tag mappings).
/// We name only the variants used in `get_category_score()` and place scoring;
/// all others are represented as `Other(u16)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmenityCategory {
    None,
    Place6,
    PlaceCapital8,
    Country,
    State,
    Region,
    Province,
    District,
    County,
    Subdistrict,
    Municipality,
    City,
    Borough,
    Suburb,
    Quarter,
    Neighbourhood,
    CityBlock,
    Plot,
    Town,
    Village,
    Hamlet,
    IsolatedDwelling,
    Farm,
    Allotments,
    Continent,
    Archipelago,
    Island,
    Islet,
    Square,
    Locality,
    Polder,
    Sea,
    Ocean,
    Extra,
    /// Catch-all for the ~280 POI-level categories not used in scoring.
    Other(u16),
}

impl Default for AmenityCategory {
    fn default() -> Self {
        Self::None
    }
}

/// Get the category bonus score for place ranking.
///
/// C++ `get_category_score()` in `get_suggestions.cc`.
pub fn get_category_score(cat: AmenityCategory) -> f32 {
    match cat {
        AmenityCategory::Country
        | AmenityCategory::State
        | AmenityCategory::Region => 1.0,
        AmenityCategory::Municipality => 1.0,
        AmenityCategory::City => 3.0,
        AmenityCategory::Town => 2.0,
        AmenityCategory::Village => 1.0,
        AmenityCategory::Island => 1.0,
        AmenityCategory::Place6 => 1.0,
        AmenityCategory::PlaceCapital8 => 3.0,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_scores_match_cpp() {
        assert_eq!(get_category_score(AmenityCategory::Country), 1.0);
        assert_eq!(get_category_score(AmenityCategory::State), 1.0);
        assert_eq!(get_category_score(AmenityCategory::Region), 1.0);
        assert_eq!(get_category_score(AmenityCategory::Municipality), 1.0);
        assert_eq!(get_category_score(AmenityCategory::City), 3.0);
        assert_eq!(get_category_score(AmenityCategory::Town), 2.0);
        assert_eq!(get_category_score(AmenityCategory::Village), 1.0);
        assert_eq!(get_category_score(AmenityCategory::Island), 1.0);
        assert_eq!(get_category_score(AmenityCategory::Place6), 1.0);
        assert_eq!(get_category_score(AmenityCategory::PlaceCapital8), 3.0);
        assert_eq!(get_category_score(AmenityCategory::None), 0.0);
        assert_eq!(get_category_score(AmenityCategory::Extra), 0.0);
        assert_eq!(get_category_score(AmenityCategory::Other(42)), 0.0);
    }

    #[test]
    fn default_is_none() {
        assert_eq!(AmenityCategory::default(), AmenityCategory::None);
    }
}
