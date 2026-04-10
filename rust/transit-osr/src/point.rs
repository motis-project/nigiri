//! Translation of osr/include/osr/point.h
//!
//! Geographic point representation with fixed-point coordinates.
//! Uses the same encoding as osmium::Location (i32 fixed-point).
//!
//! This is a 1:1 translation maintaining compatibility with OSM PBF coordinate encoding.

use std::fmt;

use rkyv::{Archive, Deserialize, Serialize};

/// Geographic point with fixed-point coordinates
///
/// Coordinates are stored as i32 values in the same format as osmium::Location:
/// - Multiply lat/lon by 10^7 to get fixed-point representation
/// - Range: approximately ±214 degrees (covers entire world)
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
)]
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
pub struct Point {
    /// Latitude in fixed-point format (lat * 10^7)
    pub lat_: i32,
    /// Longitude in fixed-point format (lon * 10^7)
    pub lng_: i32,
}

/// Constant for coordinate conversion
const COORDINATE_PRECISION: f64 = 10_000_000.0;

impl Point {
    /// Create point from latitude and longitude in degrees
    #[inline]
    pub fn from_latlng(lat: f64, lon: f64) -> Self {
        Self {
            lng_: (lon * COORDINATE_PRECISION) as i32,
            lat_: (lat * COORDINATE_PRECISION) as i32,
        }
    }

    /// Create point from fixed-point coordinates (osmium::Location format)
    #[inline]
    pub const fn from_fixed(lat: i32, lon: i32) -> Self {
        Self {
            lat_: lat,
            lng_: lon,
        }
    }

    /// Get latitude in degrees
    #[inline]
    pub fn lat(&self) -> f64 {
        (self.lat_ as f64) / COORDINATE_PRECISION
    }

    /// Get longitude in degrees
    #[inline]
    pub fn lng(&self) -> f64 {
        (self.lng_ as f64) / COORDINATE_PRECISION
    }

    /// Get as (lat, lon) tuple in degrees
    #[inline]
    pub fn as_latlng(&self) -> (f64, f64) {
        (self.lat(), self.lng())
    }

    /// Check if point is valid (not default/zero)
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.lat_ != 0 || self.lng_ != 0
    }

    /// Calculate distance to another point using Haversine formula
    /// Returns distance in meters
    pub fn distance_to(&self, other: &Point) -> f64 {
        haversine_distance(self.lat(), self.lng(), other.lat(), other.lng())
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.lat(), self.lng())
    }
}

impl From<(f64, f64)> for Point {
    #[inline]
    fn from((lat, lon): (f64, f64)) -> Self {
        Self::from_latlng(lat, lon)
    }
}

impl From<Point> for (f64, f64) {
    #[inline]
    fn from(p: Point) -> Self {
        p.as_latlng()
    }
}

// ============================================================================
// Distance Calculations
// ============================================================================

/// Earth radius in meters
const EARTH_RADIUS_M: f64 = 6371000.0;

/// Calculate distance between two points using Haversine formula
/// Returns distance in meters
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();

    let a =
        (d_lat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (d_lon / 2.0).sin().powi(2);

    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_M * c
}

/// Calculate approximate distance using equirectangular projection
/// Faster but less accurate than Haversine for large distances
/// Returns distance in meters
pub fn equirectangular_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let d_lat = (lat2 - lat1).to_radians();

    let x = d_lon * ((lat1_rad + lat2_rad) / 2.0).cos();
    let y = d_lat;

    EARTH_RADIUS_M * (x * x + y * y).sqrt()
}

// ============================================================================
// Bounding Box
// ============================================================================

/// Geographic bounding box
///
/// Represents a rectangular region defined by minimum and maximum lat/lng coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub min: Point,
    pub max: Point,
}

impl BBox {
    /// Create a new bounding box from two points
    pub fn new(min: Point, max: Point) -> Self {
        Self { min, max }
    }

    /// Create a bounding box from lat/lng coordinates
    pub fn from_latlng(min_lat: f64, min_lng: f64, max_lat: f64, max_lng: f64) -> Self {
        Self {
            min: Point::from_latlng(min_lat, min_lng),
            max: Point::from_latlng(max_lat, max_lng),
        }
    }

    /// Check if the bounding box contains a point
    pub fn contains(&self, point: &Point) -> bool {
        point.lat() >= self.min.lat()
            && point.lat() <= self.max.lat()
            && point.lng() >= self.min.lng()
            && point.lng() <= self.max.lng()
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> Point {
        Point::from_latlng(
            (self.min.lat() + self.max.lat()) / 2.0,
            (self.min.lng() + self.max.lng()) / 2.0,
        )
    }

    /// Get width in degrees (longitude range)
    pub fn width(&self) -> f64 {
        self.max.lng() - self.min.lng()
    }

    /// Get height in degrees (latitude range)
    pub fn height(&self) -> f64 {
        self.max.lat() - self.min.lat()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point::from_latlng(52.52, 13.405);
        assert!((p.lat() - 52.52).abs() < 0.0001);
        assert!((p.lng() - 13.405).abs() < 0.0001);
    }

    #[test]
    fn test_point_roundtrip() {
        let original = (48.8566, 2.3522); // Paris
        let p = Point::from_latlng(original.0, original.1);
        let (lat, lon) = p.as_latlng();

        assert!((lat - original.0).abs() < 0.0001);
        assert!((lon - original.1).abs() < 0.0001);
    }

    #[test]
    fn test_point_fixed_format() {
        // Test osmium::Location compatibility
        let p = Point::from_fixed(525200000, 134050000);
        assert!((p.lat() - 52.52).abs() < 0.001);
        assert!((p.lng() - 13.405).abs() < 0.001);
    }

    #[test]
    fn test_haversine_distance() {
        // Berlin to Paris: ~878 km
        let berlin = Point::from_latlng(52.52, 13.405);
        let paris = Point::from_latlng(48.8566, 2.3522);

        let dist = berlin.distance_to(&paris);
        assert!((dist - 878_000.0).abs() < 5000.0); // Within 5km tolerance
    }

    #[test]
    fn test_equirectangular_distance() {
        // Short distance: should be very close to Haversine
        let _p1 = Point::from_latlng(52.52, 13.405);
        let _p2 = Point::from_latlng(52.53, 13.415);

        let dist_h = haversine_distance(52.52, 13.405, 52.53, 13.415);
        let dist_e = equirectangular_distance(52.52, 13.405, 52.53, 13.415);

        assert!((dist_h - dist_e).abs() < 10.0); // Within 10m for short distances
    }

    #[test]
    fn test_point_display() {
        let p = Point::from_latlng(52.52, 13.405);
        let s = format!("{}", p);
        assert!(s.contains("52.52"));
        assert!(s.contains("13.405"));
    }

    #[test]
    fn test_point_validity() {
        let invalid = Point::from_fixed(0, 0);
        assert!(!invalid.is_valid());

        let valid = Point::from_latlng(52.52, 13.405);
        assert!(valid.is_valid());
    }

    #[test]
    fn test_bbox_creation() {
        let bbox = BBox::from_latlng(50.0, 10.0, 55.0, 15.0);
        assert!((bbox.min.lat() - 50.0).abs() < 0.001);
        assert!((bbox.min.lng() - 10.0).abs() < 0.001);
        assert!((bbox.max.lat() - 55.0).abs() < 0.001);
        assert!((bbox.max.lng() - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_bbox_contains() {
        let bbox = BBox::from_latlng(50.0, 10.0, 55.0, 15.0);

        // Inside
        assert!(bbox.contains(&Point::from_latlng(52.5, 12.5)));

        // On edge
        assert!(bbox.contains(&Point::from_latlng(50.0, 10.0)));
        assert!(bbox.contains(&Point::from_latlng(55.0, 15.0)));

        // Outside
        assert!(!bbox.contains(&Point::from_latlng(49.0, 12.0)));
        assert!(!bbox.contains(&Point::from_latlng(52.0, 16.0)));
    }

    #[test]
    fn test_bbox_center() {
        let bbox = BBox::from_latlng(50.0, 10.0, 54.0, 14.0);
        let center = bbox.center();

        assert!((center.lat() - 52.0).abs() < 0.001);
        assert!((center.lng() - 12.0).abs() < 0.001);
    }

    #[test]
    fn test_bbox_dimensions() {
        let bbox = BBox::from_latlng(50.0, 10.0, 55.0, 15.0);

        assert!((bbox.width() - 5.0).abs() < 0.001);
        assert!((bbox.height() - 5.0).abs() < 0.001);
    }
}
