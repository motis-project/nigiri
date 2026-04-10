//! Spatial area database for polygon containment queries.
//!
//! Mirrors C++ `adr/area_database.h`. Provides lookup of which administrative
//! areas contain a given coordinate.

use crate::types::*;

/// Spatial area database for polygon containment.
/// C++ `adr::area_database`.
///
/// The C++ uses an R-tree with polygon geometry for containment testing.
/// This Rust version provides the interface; full polygon support requires
/// a geometry crate (e.g., `geo`).
pub struct AreaDatabase {
    /// Areas stored as (area_idx, polygon_points).
    areas: Vec<(AreaIdx, Vec<Coordinates>)>,
}

impl AreaDatabase {
    pub fn new() -> Self {
        Self {
            areas: Vec::new(),
        }
    }

    /// Add an area polygon.
    /// C++ `area_database::add_area()`.
    pub fn add_area(&mut self, area_idx: AreaIdx, polygon: Vec<Coordinates>) {
        self.areas.push((area_idx, polygon));
    }

    /// Lookup which areas contain the given coordinate.
    /// C++ `area_database::lookup()`.
    pub fn lookup(&self, coord: Coordinates) -> Vec<AreaIdx> {
        let mut result = Vec::new();
        for (area_idx, polygon) in &self.areas {
            if point_in_polygon(coord, polygon) {
                result.push(*area_idx);
            }
        }
        result
    }

    /// Check if a coordinate is within a specific area.
    /// C++ `area_database::is_within()`.
    pub fn is_within(&self, coord: Coordinates, area_idx: AreaIdx) -> bool {
        self.areas
            .iter()
            .find(|(idx, _)| *idx == area_idx)
            .map(|(_, polygon)| point_in_polygon(coord, polygon))
            .unwrap_or(false)
    }
}

impl Default for AreaDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Ray-casting point-in-polygon test.
fn point_in_polygon(point: Coordinates, polygon: &[Coordinates]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let px = point.lat as f64;
    let py = point.lng as f64;
    let mut inside = false;
    let n = polygon.len();

    let mut j = n - 1;
    for i in 0..n {
        let xi = polygon[i].lat as f64;
        let yi = polygon[i].lng as f64;
        let xj = polygon[j].lat as f64;
        let yj = polygon[j].lng as f64;

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
        {
            inside = !inside;
        }
        j = i;
    }

    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_in_simple_polygon() {
        // Square polygon: (0,0), (10,0), (10,10), (0,10)
        let polygon = vec![
            Coordinates { lat: 0, lng: 0 },
            Coordinates { lat: 10, lng: 0 },
            Coordinates { lat: 10, lng: 10 },
            Coordinates { lat: 0, lng: 10 },
        ];
        assert!(point_in_polygon(Coordinates { lat: 5, lng: 5 }, &polygon));
        assert!(!point_in_polygon(
            Coordinates { lat: 15, lng: 15 },
            &polygon
        ));
    }

    #[test]
    fn area_database_lookup() {
        let mut db = AreaDatabase::new();
        db.add_area(
            AreaIdx(0),
            vec![
                Coordinates { lat: 0, lng: 0 },
                Coordinates { lat: 100, lng: 0 },
                Coordinates { lat: 100, lng: 100 },
                Coordinates { lat: 0, lng: 100 },
            ],
        );

        let result = db.lookup(Coordinates { lat: 50, lng: 50 });
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], AreaIdx(0));

        assert!(db.is_within(Coordinates { lat: 50, lng: 50 }, AreaIdx(0)));
        assert!(!db.is_within(Coordinates { lat: 200, lng: 200 }, AreaIdx(0)));
    }

    #[test]
    fn area_database_empty() {
        let db = AreaDatabase::new();
        let result = db.lookup(Coordinates { lat: 0, lng: 0 });
        assert!(result.is_empty());
    }
}
