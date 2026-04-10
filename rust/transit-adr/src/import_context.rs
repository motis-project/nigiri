//! Import context for building the typeahead database.
//!
//! Mirrors C++ `adr/import_context.h`. Holds lookup tables and mutable
//! state used during OSM data import.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::types::*;

/// Import context holding lookup tables and mutable state.
/// C++ `adr::import_context`.
pub struct ImportContext {
    /// Area index list → area set index. C++ `area_set_lookup_`.
    pub area_set_lookup: HashMap<Vec<AreaIdx>, AreaSetIdx>,
    /// String → string index. C++ `string_lookup_`.
    pub string_lookup: HashMap<String, StringIdx>,
    /// Timezone string → timezone index. C++ `tz_lookup_`.
    pub tz_lookup: HashMap<String, TimezoneIdx>,
    /// String index → street index. C++ `street_lookup_`.
    pub street_lookup: HashMap<StringIdx, StreetIdx>,
    /// Street names (import-time). C++ `street_names_`.
    pub street_names: Vec<StringIdx>,

    /// Street positions (import-time). C++ `street_pos_`.
    pub street_pos: Vec<Vec<Coordinates>>,
    /// House numbers per street (import-time). C++ `house_numbers_`.
    pub house_numbers: Vec<Vec<StringIdx>>,
    /// House coordinates per street (import-time). C++ `house_coordinates_`.
    pub house_coordinates: Vec<Vec<Coordinates>>,

    /// Place statistics. C++ `place_stats_`.
    pub place_stats: HashMap<String, u32>,

    /// String → (location_idx, location_type) pairs. C++ `string_to_location_`.
    pub string_to_location: Vec<Vec<(u32, LocationType)>>,

    /// Street segments for reverse geocoding. C++ `street_segments_`.
    pub street_segments: Vec<Vec<Vec<Coordinates>>>,

    /// Mutex for thread safety (matches C++ `mutex_`).
    pub mutex: Mutex<()>,
    /// Reverse geocoding mutex. C++ `reverse_mutex_`.
    pub reverse_mutex: Mutex<()>,
}

impl ImportContext {
    pub fn new() -> Self {
        Self {
            area_set_lookup: HashMap::new(),
            string_lookup: HashMap::new(),
            tz_lookup: HashMap::new(),
            street_lookup: HashMap::new(),
            street_names: Vec::new(),
            street_pos: Vec::new(),
            house_numbers: Vec::new(),
            house_coordinates: Vec::new(),
            place_stats: HashMap::new(),
            string_to_location: Vec::new(),
            street_segments: Vec::new(),
            mutex: Mutex::new(()),
            reverse_mutex: Mutex::new(()),
        }
    }
}

impl Default for ImportContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_context_new() {
        let ctx = ImportContext::new();
        assert!(ctx.string_lookup.is_empty());
        assert!(ctx.area_set_lookup.is_empty());
    }
}
