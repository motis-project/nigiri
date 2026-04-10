use rstar::{PointDistance, RTree, RTreeObject, AABB};
use transit_core::LocationIdx;

/// An entry in the spatial index, mapping a location to its coordinates.
#[derive(Debug, Clone)]
pub struct LocationEntry {
    pub idx: LocationIdx,
    pub lat: f64,
    pub lon: f64,
}

impl RTreeObject for LocationEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.lon, self.lat])
    }
}

impl PointDistance for LocationEntry {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let dx = self.lon - point[0];
        let dy = self.lat - point[1];
        dx * dx + dy * dy
    }
}

/// Build an R-tree spatial index over all timetable locations.
pub fn build_location_rtree(tt: &nigiri::Timetable) -> RTree<LocationEntry> {
    let count = tt.location_count();
    let mut entries = Vec::with_capacity(count as usize);
    for i in 0..count {
        if let Ok(loc) = tt.get_location(i) {
            let lat = loc.lat();
            let lon = loc.lon();
            // Skip locations with invalid/zero coordinates
            if lat != 0.0 || lon != 0.0 {
                entries.push(LocationEntry {
                    idx: LocationIdx(i),
                    lat,
                    lon,
                });
            }
        }
    }
    tracing::info!("Built location R-tree with {} entries", entries.len());
    RTree::bulk_load(entries)
}

/// Find all locations within a bounding box.
pub fn locations_in_bbox(
    rtree: &RTree<LocationEntry>,
    min_lat: f64,
    min_lon: f64,
    max_lat: f64,
    max_lon: f64,
) -> Vec<&LocationEntry> {
    let envelope = AABB::from_corners([min_lon, min_lat], [max_lon, max_lat]);
    rtree.locate_in_envelope(&envelope).collect()
}

/// Find all locations within a radius (in degrees) of a point.
pub fn locations_in_radius(
    rtree: &RTree<LocationEntry>,
    lat: f64,
    lon: f64,
    radius_degrees: f64,
) -> Vec<&LocationEntry> {
    rtree
        .locate_within_distance([lon, lat], radius_degrees * radius_degrees)
        .collect()
}

/// Approximate meters to degrees at a given latitude.
/// This is a rough approximation suitable for nearby-stop queries.
pub fn meters_to_degrees(meters: f64, lat: f64) -> f64 {
    // 1 degree latitude ≈ 111,320 meters
    // 1 degree longitude ≈ 111,320 * cos(lat) meters
    // Use the average for a rough approximation
    let lat_rad = lat.to_radians();
    let avg_meters_per_degree = 111_320.0 * ((1.0 + lat_rad.cos()) / 2.0);
    meters / avg_meters_per_degree
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtree_basic_operations() {
        let entries = vec![
            LocationEntry {
                idx: LocationIdx(0),
                lat: 48.0,
                lon: 11.0,
            },
            LocationEntry {
                idx: LocationIdx(1),
                lat: 48.1,
                lon: 11.1,
            },
            LocationEntry {
                idx: LocationIdx(2),
                lat: 52.5,
                lon: 13.4,
            },
        ];
        let rtree = RTree::bulk_load(entries);

        // Bounding box query around Munich
        let results = locations_in_bbox(&rtree, 47.5, 10.5, 48.5, 11.5);
        assert_eq!(results.len(), 2);

        // Radius query around Berlin
        let results = locations_in_radius(&rtree, 52.5, 13.4, 0.1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].idx, LocationIdx(2));
    }

    #[test]
    fn meters_to_degrees_reasonable() {
        let d = meters_to_degrees(500.0, 48.0);
        // 500m should be roughly 0.004-0.006 degrees
        assert!(d > 0.003 && d < 0.01);
    }
}
