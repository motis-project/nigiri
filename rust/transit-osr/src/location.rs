//! Translation of osr/include/osr/location.h
//!
//! Location combines a geographic point with a level for multi-level routing.
//! Used for positioning within buildings, stations with multiple floors, etc.
//!
//! This is a 1:1 translation of the C++ location struct.

use std::fmt;

use rkyv::{Archive, Deserialize, Serialize};

use crate::{Level, Point};

/// Location with position and level
///
/// Combines a geographic point with a vertical level for multi-level routing.
/// This allows routing through buildings, stations with multiple floors, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
pub struct Location {
    /// Geographic position
    pub pos_: Point,
    /// Vertical level
    pub lvl_: Level,
}

impl Location {
    /// Create a new location
    #[inline]
    pub const fn new(pos: Point, lvl: Level) -> Self {
        Self {
            pos_: pos,
            lvl_: lvl,
        }
    }

    /// Create location from lat/lon and level
    #[inline]
    pub fn from_latlng(lat: f64, lon: f64, lvl: Level) -> Self {
        Self {
            pos_: Point::from_latlng(lat, lon),
            lvl_: lvl,
        }
    }

    /// Create location from lat/lon (default level)
    #[inline]
    pub fn from_latlng_no_level(lat: f64, lon: f64) -> Self {
        Self {
            pos_: Point::from_latlng(lat, lon),
            lvl_: Level::default(),
        }
    }

    /// Get position
    #[inline]
    pub const fn pos(&self) -> Point {
        self.pos_
    }

    /// Get level
    #[inline]
    pub const fn lvl(&self) -> Level {
        self.lvl_
    }

    /// Get latitude in degrees
    #[inline]
    pub fn lat(&self) -> f64 {
        self.pos_.lat()
    }

    /// Get longitude in degrees
    #[inline]
    pub fn lng(&self) -> f64 {
        self.pos_.lng()
    }

    /// Create location from Point with optional level
    #[inline]
    pub fn from_point(point: Point, level: Option<f32>) -> Self {
        Self {
            pos_: point,
            lvl_: level.map(Level::from_float).unwrap_or(Level::default()),
        }
    }

    /// Calculate distance to another location (ignoring level)
    /// Returns distance in meters
    #[inline]
    pub fn distance_to(&self, other: &Location) -> f64 {
        self.pos_.distance_to(&other.pos_)
    }

    /// Check if location is valid
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.pos_.is_valid()
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ pos={}, lvl={} }}", self.pos_, self.lvl_)
    }
}

impl From<Point> for Location {
    #[inline]
    fn from(pos: Point) -> Self {
        Self {
            pos_: pos,
            lvl_: Level::default(),
        }
    }
}

impl From<(f64, f64)> for Location {
    #[inline]
    fn from((lat, lon): (f64, f64)) -> Self {
        Self::from_latlng_no_level(lat, lon)
    }
}

impl From<(f64, f64, f32)> for Location {
    #[inline]
    fn from((lat, lon, level): (f64, f64, f32)) -> Self {
        Self::from_latlng(lat, lon, Level::from_float(level))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_creation() {
        let loc = Location::from_latlng(52.52, 13.405, Level::from_float(1.0));
        assert!((loc.lat() - 52.52).abs() < 0.0001);
        assert!((loc.lng() - 13.405).abs() < 0.0001);
        assert_eq!(loc.lvl().to_float(), 1.0);
    }

    #[test]
    fn test_location_no_level() {
        let loc = Location::from_latlng_no_level(48.8566, 2.3522);
        assert!(!loc.lvl().is_valid());
    }

    #[test]
    fn test_location_from_point() {
        let point = Point::from_latlng(52.52, 13.405);
        let loc = Location::from(point);

        assert_eq!(loc.pos(), point);
        assert!(!loc.lvl().is_valid());
    }

    #[test]
    fn test_location_from_tuple() {
        let loc: Location = (52.52, 13.405).into();
        assert!((loc.lat() - 52.52).abs() < 0.0001);
        assert!((loc.lng() - 13.405).abs() < 0.0001);
    }

    #[test]
    fn test_location_from_tuple_with_level() {
        let loc: Location = (52.52, 13.405, 2.5).into();
        assert!((loc.lat() - 52.52).abs() < 0.0001);
        assert!((loc.lvl().to_float() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_location_distance() {
        let berlin = Location::from_latlng(52.52, 13.405, Level::from_float(0.0));
        let paris = Location::from_latlng(48.8566, 2.3522, Level::from_float(1.0));

        let dist = berlin.distance_to(&paris);
        assert!((dist - 878_000.0).abs() < 5000.0); // Within 5km tolerance
    }

    #[test]
    fn test_location_display() {
        let loc = Location::from_latlng(52.52, 13.405, Level::from_float(1.5));
        let s = format!("{}", loc);

        assert!(s.contains("pos="));
        assert!(s.contains("lvl="));
    }

    #[test]
    fn test_location_equality() {
        let loc1 = Location::from_latlng(52.52, 13.405, Level::from_float(1.0));
        let loc2 = Location::from_latlng(52.52, 13.405, Level::from_float(1.0));
        let loc3 = Location::from_latlng(52.52, 13.405, Level::from_float(2.0));

        assert_eq!(loc1, loc2);
        assert_ne!(loc1, loc3);
    }
}
