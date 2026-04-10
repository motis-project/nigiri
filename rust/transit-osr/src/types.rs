//! Translation of osr/include/osr/types.h
//!
//! Core type definitions for the OSR routing engine, including:
//! - Strongly-typed indices (node_idx_t, way_idx_t, osm_node_idx_t, osm_way_idx_t, platform_idx_t)
//! - Level system for multi-level routing (level_t)
//! - Direction enum (Forward/Backward)
//! - Speed limit system
//! - Distance and cost types
//!
//! This is a 1:1 translation of the C++ types.h file, maintaining full functional parity.

use packed_struct::prelude::*;
use std::fmt;
use std::hash::Hash;

use rkyv::{Archive, Deserialize, Serialize};

// ============================================================================
// Core Constants
// ============================================================================

/// Maximum cost value (infeasible route)
pub const K_INFEASIBLE: u16 = u16::MAX;

/// Minimum level for multi-level routing
pub const K_MIN_LEVEL: f32 = -8.0;

/// Maximum level for multi-level routing
pub const K_MAX_LEVEL: f32 = 7.5;

// ============================================================================
// Strongly-Typed Indices
// ============================================================================

macro_rules! define_idx {
    ($name:ident, $inner:ty, $doc:expr) => {
        #[doc = $doc]
        #[derive(
            Debug,
            Clone,
            Copy,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            Archive,
            Serialize,
            Deserialize,
            serde::Serialize,
            serde::Deserialize,
        )]
        #[repr(transparent)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable)]
        pub struct $name(pub $inner);

        impl $name {
            pub const INVALID: Self = Self(<$inner>::MAX);

            #[inline]
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            #[inline]
            pub const fn value(self) -> $inner {
                self.0
            }

            #[inline]
            pub const fn is_valid(self) -> bool {
                self.0 != <$inner>::MAX
            }
        }

        impl From<$inner> for $name {
            #[inline]
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $inner {
            #[inline]
            fn from(idx: $name) -> Self {
                idx.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

// String index
define_idx!(StringIdx, u32, "Index into string pool");

// OSM indices (64-bit for OSM IDs)
define_idx!(OsmNodeIdx, u64, "OSM node ID");
define_idx!(OsmWayIdx, u64, "OSM way ID");

// Internal routing indices (32-bit for efficiency)
define_idx!(WayIdx, u32, "Index into ways array");
define_idx!(NodeIdx, u32, "Index into multi-nodes array");
define_idx!(ComponentIdx, u32, "Connected component index");
define_idx!(PlatformIdx, u32, "Index into platforms array");
define_idx!(
    MultiLevelElevatorIdx,
    u32,
    "Index into multi-level elevators array"
);

// ============================================================================
// Basic Types
// ============================================================================

/// Distance in meters (max ~65km)
pub type Distance = u16;

/// Monotonic elevation value (0-65535 encoded)
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Archive, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct ElevationMonotonic(pub u16);

/// Position along a way (0-255)
pub type WayPos = u8;

/// Routing cost (lower is better, u16::MAX = infeasible)
pub type Cost = u16;

// ============================================================================
// Direction
// ============================================================================

/// Routing direction (forward or backward)
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
#[repr(u8)]
pub enum Direction {
    Forward = 0,
    Backward = 1,
}

impl Direction {
    /// Get the opposite direction
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Direction::Forward => Direction::Backward,
            Direction::Backward => Direction::Forward,
        }
    }

    /// Flip direction based on template parameter
    #[inline]
    pub const fn flip<const DIR: bool>(self) -> Self {
        if DIR {
            self
        } else {
            self.opposite()
        }
    }

    /// Flip direction based on runtime search direction
    #[inline]
    pub const fn flip_runtime(self, search_dir: Direction) -> Self {
        match search_dir {
            Direction::Forward => self,
            Direction::Backward => self.opposite(),
        }
    }

    /// Convert to string representation
    pub const fn as_str(self) -> &'static str {
        match self {
            Direction::Forward => "forward",
            Direction::Backward => "backward",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "forward" => Some(Direction::Forward),
            "backward" => Some(Direction::Backward),
            _ => None,
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::Forward => write!(f, "fwd"),
            Direction::Backward => write!(f, "bwd"),
        }
    }
}

// ============================================================================
// Level (Multi-Level Routing)
// ============================================================================

/// Level for multi-level routing (e.g., floor in a building)
/// Stored as u8, representing levels from -8.0 to 7.5 in 0.25 increments
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
)]
#[repr(transparent)]
pub struct Level {
    v: u8,
}

impl Level {
    /// No level specified
    pub const NO_LEVEL: u8 = 0;

    /// Number of bits needed to store level
    pub const LEVEL_BITS: u32 = 6;

    /// Create level from encoded u8 value
    #[inline]
    pub const fn from_idx(v: u8) -> Self {
        Self { v }
    }

    /// Create level from float value (e.g., 0.0, 0.5, 1.0, -1.5)
    #[inline]
    pub const fn from_float(f: f32) -> Self {
        Self {
            v: ((f - K_MIN_LEVEL) / 0.25 + 1.0) as u8,
        }
    }

    /// Get encoded u8 value
    #[inline]
    pub const fn to_idx(self) -> u8 {
        self.v
    }

    /// Convert to float representation
    #[inline]
    pub const fn to_float(self) -> f32 {
        if self.v == Self::NO_LEVEL {
            0.0
        } else {
            K_MIN_LEVEL + ((self.v - 1) as f32 / 4.0)
        }
    }

    /// Check if level is specified
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.v != Self::NO_LEVEL
    }

    /// Hash value for use in hash maps
    #[inline]
    pub const fn hash_value(self) -> u64 {
        self.v as u64
    }
}

impl Default for Level {
    #[inline]
    fn default() -> Self {
        Self { v: Self::NO_LEVEL }
    }
}

impl fmt::Debug for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.v == Self::NO_LEVEL {
            write!(f, "Level(-)")
        } else {
            write!(f, "Level({})", self.to_float())
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.v == Self::NO_LEVEL {
            write!(f, "-")
        } else {
            write!(f, "{}", self.to_float())
        }
    }
}

/// Level bits type for bitfield operations
pub type LevelBits = u64;

/// Extract level range from bitfield
/// Returns (from_level, to_level, has_multiple_levels)
pub fn get_levels(has_level: bool, levels: LevelBits) -> (Level, Level, bool) {
    if !has_level {
        return (Level::default(), Level::default(), false);
    }

    let mut from = Level::default();
    let mut to = Level::default();
    let mut count = 0;

    // Iterate over set bits
    for bit in 0..64 {
        if (levels & (1 << bit)) != 0 {
            if !from.is_valid() {
                from = Level::from_idx(bit as u8);
            } else {
                to = Level::from_idx(bit as u8);
            }
            count += 1;
        }
    }

    let to = if to.is_valid() { to } else { from };
    let has_multiple = count > 2;

    (from, to, has_multiple)
}

// ============================================================================
// Speed Limit
// ============================================================================

/// Speed limit categories
#[derive(
    PrimitiveEnum_u8,
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
pub enum SpeedLimit {
    Kmh10 = 0,
    Kmh30 = 1,
    Kmh50 = 2,
    Kmh60 = 3,
    Kmh70 = 4,
    Kmh80 = 5,
    Kmh100 = 6,
    Kmh120 = 7,
}

impl SpeedLimit {
    /// Convert unsigned km/h value to speed limit category
    pub const fn from_kmh(kmh: u32) -> Self {
        if kmh >= 120 {
            SpeedLimit::Kmh120
        } else if kmh >= 100 {
            SpeedLimit::Kmh100
        } else if kmh >= 80 {
            SpeedLimit::Kmh80
        } else if kmh >= 70 {
            SpeedLimit::Kmh70
        } else if kmh >= 60 {
            SpeedLimit::Kmh60
        } else if kmh >= 50 {
            SpeedLimit::Kmh50
        } else if kmh >= 30 {
            SpeedLimit::Kmh30
        } else {
            SpeedLimit::Kmh10
        }
    }

    /// Convert to km/h value
    pub const fn to_kmh(self) -> u16 {
        match self {
            SpeedLimit::Kmh10 => 10,
            SpeedLimit::Kmh30 => 30,
            SpeedLimit::Kmh50 => 50,
            SpeedLimit::Kmh60 => 60,
            SpeedLimit::Kmh70 => 70,
            SpeedLimit::Kmh80 => 80,
            SpeedLimit::Kmh100 => 100,
            SpeedLimit::Kmh120 => 120,
        }
    }

    /// Convert to meters per second
    pub const fn to_meters_per_second(self) -> u16 {
        (self.to_kmh() as f32 / 3.6) as u16
    }

    /// Convert to 3-bit representation for bit-packing
    pub const fn to_bits(self) -> u8 {
        self as u8
    }

    /// Convert from 3-bit representation
    pub const fn from_bits(bits: u8) -> Self {
        match bits & 0b111 {
            0 => SpeedLimit::Kmh10,
            1 => SpeedLimit::Kmh30,
            2 => SpeedLimit::Kmh50,
            3 => SpeedLimit::Kmh60,
            4 => SpeedLimit::Kmh70,
            5 => SpeedLimit::Kmh80,
            6 => SpeedLimit::Kmh100,
            _ => SpeedLimit::Kmh120,
        }
    }
}

impl Default for SpeedLimit {
    fn default() -> Self {
        SpeedLimit::Kmh10
    }
}

impl fmt::Display for SpeedLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}km/h", self.to_kmh())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_opposite() {
        assert_eq!(Direction::Forward.opposite(), Direction::Backward);
        assert_eq!(Direction::Backward.opposite(), Direction::Forward);
    }

    #[test]
    fn test_level_encoding() {
        let l0 = Level::from_float(0.0);
        assert_eq!(l0.to_float(), 0.0);

        let l1 = Level::from_float(1.0);
        assert_eq!(l1.to_float(), 1.0);

        let l_neg = Level::from_float(-1.5);
        assert!((l_neg.to_float() - (-1.5)).abs() < 0.01);
    }

    #[test]
    fn test_level_no_level() {
        let l = Level::default();
        assert!(!l.is_valid());
        assert_eq!(l.to_idx(), Level::NO_LEVEL);
    }

    #[test]
    fn test_speed_limit() {
        assert_eq!(SpeedLimit::from_kmh(25), SpeedLimit::Kmh10);
        assert_eq!(SpeedLimit::from_kmh(55), SpeedLimit::Kmh50);
        assert_eq!(SpeedLimit::from_kmh(150), SpeedLimit::Kmh120);

        assert_eq!(SpeedLimit::Kmh50.to_kmh(), 50);
        assert_eq!(SpeedLimit::Kmh100.to_kmh(), 100);
    }

    #[test]
    fn test_idx_invalid() {
        assert!(!WayIdx::INVALID.is_valid());
        assert!(!NodeIdx::INVALID.is_valid());
        assert!(WayIdx::new(0).is_valid());
        assert!(NodeIdx::new(12345).is_valid());
    }

    #[test]
    fn test_get_levels() {
        // No level
        let (from, to, multi) = get_levels(false, 0);
        assert!(!from.is_valid());
        assert!(!to.is_valid());
        assert!(!multi);

        // Single level (bit 10 set)
        let (from, to, multi) = get_levels(true, 1 << 10);
        assert_eq!(from.to_idx(), 10);
        assert_eq!(to.to_idx(), 10);
        assert!(!multi);

        // Two levels (bits 5 and 10 set)
        let (from, to, multi) = get_levels(true, (1 << 5) | (1 << 10));
        assert_eq!(from.to_idx(), 5);
        assert_eq!(to.to_idx(), 10);
        assert!(!multi);

        // Three levels (bits 5, 10, 15 set)
        let (from, to, multi) = get_levels(true, (1 << 5) | (1 << 10) | (1 << 15));
        assert_eq!(from.to_idx(), 5);
        assert_eq!(to.to_idx(), 15);
        assert!(multi);
    }
}
