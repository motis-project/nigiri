//! Reverse geocoding via spatial queries.
//!
//! Mirrors C++ `adr/reverse.h` + `reverse.cc`. Uses an R-tree for spatial
//! indexing of streets, places, and house numbers.

use crate::suggestion::{Address, Suggestion, SuggestionLocation};
use crate::typeahead::Typeahead;
use crate::types::*;

// ---------------------------------------------------------------------------
// Entity types for R-tree storage — C++ `entity_type` + `rtree_entity`
// ---------------------------------------------------------------------------

/// Type of entity stored in the spatial index.
/// C++ `entity_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntityType {
    HouseNumber = 0,
    Street = 1,
    Place = 2,
}

/// A house number entity in the R-tree.
#[derive(Debug, Clone, Copy)]
pub struct HouseNumberEntity {
    pub idx: u16,
    pub street: StreetIdx,
}

/// A place entity in the R-tree.
#[derive(Debug, Clone, Copy)]
pub struct PlaceEntity {
    pub place: PlaceIdx,
}

/// A street segment entity in the R-tree.
#[derive(Debug, Clone, Copy)]
pub struct StreetSegmentEntity {
    pub segment: u16,
    pub street: StreetIdx,
}

/// Union-like enum for R-tree entities.
/// C++ `rtree_entity` union.
#[derive(Debug, Clone, Copy)]
pub enum RtreeEntity {
    HouseNumber(HouseNumberEntity),
    Place(PlaceEntity),
    StreetSegment(StreetSegmentEntity),
}

impl RtreeEntity {
    pub fn entity_type(&self) -> EntityType {
        match self {
            RtreeEntity::HouseNumber(_) => EntityType::HouseNumber,
            RtreeEntity::Place(_) => EntityType::Place,
            RtreeEntity::StreetSegment(_) => EntityType::Street,
        }
    }
}

// ---------------------------------------------------------------------------
// Reverse geocoder — C++ `adr::reverse`
// ---------------------------------------------------------------------------

/// Reverse geocoder using spatial indexing.
/// C++ `adr::reverse`.
///
/// The C++ uses `cista::mm_rtree` with memory-mapped storage.
/// This Rust version uses in-memory storage with a simple brute-force
/// approach. For production use, integrate an R-tree crate (e.g., `rstar`).
pub struct Reverse {
    /// All entities with their bounding boxes (min_lat, min_lng, max_lat, max_lng).
    entities: Vec<(Coordinates, Coordinates, RtreeEntity)>,
    /// Street segments for distance computation.
    pub street_segments: Vec<Vec<Vec<Coordinates>>>,
}

impl Reverse {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            street_segments: Vec::new(),
        }
    }

    /// Add street way geometry segments.
    /// C++ `reverse::add_street()`.
    ///
    /// The C++ version takes an `osmium::Way`; this Rust version takes the
    /// pre-extracted node coordinates directly.
    pub fn add_street(&mut self, street: StreetIdx, way_nodes: &[Coordinates]) {
        let si = street.to_idx();
        // Grow to fit.
        if si >= self.street_segments.len() {
            self.street_segments.resize_with(si + 1, Vec::new);
        }
        self.street_segments[si].push(way_nodes.to_vec());
    }

    /// Write import-context street segments into the reverse index.
    /// C++ `reverse::write(import_context&)`.
    pub fn write_segments(&mut self, segments: &[Vec<Vec<Coordinates>>]) {
        for street_segs in segments {
            self.street_segments.push(street_segs.clone());
        }
    }

    /// Lookup nearby suggestions for a given coordinate.
    /// C++ `reverse::lookup()`.
    pub fn lookup(
        &self,
        t: &Typeahead,
        lat: f64,
        lng: f64,
        n_guesses: usize,
        filter: FilterType,
    ) -> Vec<Suggestion> {
        let query = Coordinates::from_lat_lng(lat, lng);
        let radius = 500.0; // meters

        let mut suggestions = Vec::new();

        for (min, max, entity) in &self.entities {
            // Simple bounding box check.
            let entity_lat = (min.lat + max.lat) as f64 / 2.0 / Coordinates::SCALE;
            let entity_lng = (min.lng + max.lng) as f64 / 2.0 / Coordinates::SCALE;
            let entity_coord = Coordinates::from_lat_lng(entity_lat, entity_lng);
            let dist = query.distance_to(&entity_coord);

            if dist > radius {
                continue;
            }

            match entity {
                RtreeEntity::HouseNumber(hn) => {
                    if filter == FilterType::Place || filter == FilterType::Extra {
                        continue;
                    }
                    let c = t.house_coordinates[hn.street.to_idx()][hn.idx as usize];
                    let hn_dist = query.distance_to(&c) as f32;
                    suggestions.push(Suggestion::new(
                        t.street_names[hn.street.to_idx()][DEFAULT_LANG_IDX],
                        SuggestionLocation::Address(Address {
                            street: hn.street,
                            house_number: hn.idx as u32,
                        }),
                        c,
                        t.house_areas[hn.street.to_idx()][hn.idx as usize],
                        [DEFAULT_LANG_IDX as u8; 32],
                        u32::MAX,
                        0,
                        hn_dist - 10.0,
                    ));
                }
                RtreeEntity::Place(p) => {
                    if filter == FilterType::Address {
                        continue;
                    }
                    if filter == FilterType::Extra
                        && t.place_type[p.place.to_idx()]
                            != crate::categories::AmenityCategory::Extra
                    {
                        continue;
                    }
                    let c = t.place_coordinates[p.place.to_idx()];
                    let p_dist = query.distance_to(&c) as f32;
                    suggestions.push(Suggestion::new(
                        t.place_names[p.place.to_idx()][DEFAULT_LANG_IDX],
                        SuggestionLocation::Place(p.place),
                        c,
                        t.place_areas[p.place.to_idx()],
                        [DEFAULT_LANG_IDX as u8; 32],
                        u32::MAX,
                        0,
                        p_dist - 10.0,
                    ));
                }
                RtreeEntity::StreetSegment(s) => {
                    if filter == FilterType::Place || filter == FilterType::Extra {
                        continue;
                    }
                    // Simplified: use midpoint of segment.
                    let street_idx = s.street.to_idx();
                    if street_idx < t.street_pos.len() && !t.street_pos[street_idx].is_empty()
                    {
                        let c = t.street_pos[street_idx][0];
                        let s_dist = query.distance_to(&c) as f32;
                        let area_set = if !t.street_areas[street_idx].is_empty() {
                            t.street_areas[street_idx][0]
                        } else {
                            AreaSetIdx(0)
                        };
                        suggestions.push(Suggestion::new(
                            t.street_names[street_idx][DEFAULT_LANG_IDX],
                            SuggestionLocation::Address(Address {
                                street: s.street,
                                house_number: Address::NO_HOUSE_NUMBER,
                            }),
                            c,
                            area_set,
                            [DEFAULT_LANG_IDX as u8; 32],
                            u32::MAX,
                            0,
                            s_dist,
                        ));
                    }
                }
            }
        }

        // Sort and truncate.
        if suggestions.len() > n_guesses {
            suggestions.select_nth_unstable_by(n_guesses - 1, |a, b| a.cmp(b));
            suggestions.truncate(n_guesses);
        }
        suggestions.sort();

        for s in &mut suggestions {
            s.populate_areas(t);
        }

        suggestions
    }

    /// Build the R-tree from typeahead data.
    /// C++ `reverse::build_rtree()`.
    pub fn build_rtree(&mut self, t: &Typeahead) {
        self.entities.clear();

        // Add street segments.
        for (street_idx, segments) in self.street_segments.iter().enumerate() {
            for (seg_idx, segment) in segments.iter().enumerate() {
                if segment.is_empty() {
                    continue;
                }
                let mut min = segment[0];
                let mut max = segment[0];
                for c in segment {
                    min.lat = min.lat.min(c.lat);
                    min.lng = min.lng.min(c.lng);
                    max.lat = max.lat.max(c.lat);
                    max.lng = max.lng.max(c.lng);
                }
                self.entities.push((
                    min,
                    max,
                    RtreeEntity::StreetSegment(StreetSegmentEntity {
                        segment: seg_idx as u16,
                        street: StreetIdx(street_idx as u32),
                    }),
                ));
            }
        }

        // Add places.
        for (i, &c) in t.place_coordinates.iter().enumerate() {
            self.entities.push((
                c,
                c,
                RtreeEntity::Place(PlaceEntity {
                    place: PlaceIdx(i as u32),
                }),
            ));
        }

        // Add house numbers.
        for (street_idx, house_coords) in t.house_coordinates.iter().enumerate() {
            for (hn_idx, &c) in house_coords.iter().enumerate() {
                self.entities.push((
                    c,
                    c,
                    RtreeEntity::HouseNumber(HouseNumberEntity {
                        idx: hn_idx as u16,
                        street: StreetIdx(street_idx as u32),
                    }),
                ));
            }
        }
    }
}

impl Default for Reverse {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_type_variants() {
        let hn = RtreeEntity::HouseNumber(HouseNumberEntity {
            idx: 0,
            street: StreetIdx(0),
        });
        assert_eq!(hn.entity_type(), EntityType::HouseNumber);

        let p = RtreeEntity::Place(PlaceEntity {
            place: PlaceIdx(0),
        });
        assert_eq!(p.entity_type(), EntityType::Place);

        let s = RtreeEntity::StreetSegment(StreetSegmentEntity {
            segment: 0,
            street: StreetIdx(0),
        });
        assert_eq!(s.entity_type(), EntityType::Street);
    }

    #[test]
    fn reverse_empty() {
        let r = Reverse::new();
        let t = Typeahead::new();
        let results = r.lookup(&t, 51.5, -0.1, 10, FilterType::None);
        assert!(results.is_empty());
    }
}
