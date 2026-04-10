//! Translation of osr/include/osr/preprocessing/elevation/shared.h
//!
//! Shared types for elevation preprocessing.

use super::resolution::Resolution;
use crate::Point;

/// Elevation in meters (signed 16-bit)
///
/// Can represent elevations from -32,767 to +32,767 meters.
/// Uses -32,768 as invalid/void value (matching SRTM spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElevationMeters(pub i16);

impl ElevationMeters {
    /// Invalid/void elevation marker
    pub const INVALID: i16 = i16::MIN; // -32768

    /// Create a new elevation value
    pub fn new(meters: i16) -> Self {
        Self(meters)
    }

    /// Create an invalid elevation
    pub fn invalid() -> Self {
        Self(Self::INVALID)
    }

    /// Check if elevation is valid
    pub fn is_valid(&self) -> bool {
        self.0 != Self::INVALID
    }

    /// Get the raw value in meters
    pub fn meters(&self) -> Option<i16> {
        if self.is_valid() {
            Some(self.0)
        } else {
            None
        }
    }
}

impl Default for ElevationMeters {
    fn default() -> Self {
        Self::invalid()
    }
}

/// Tile index with bit-packed components
///
/// Packs three indices into a 32-bit value:
/// - driver_idx: 4 bits (0-15) - which elevation driver
/// - tile_idx: 16 bits (0-65535) - which tile within driver
/// - sub_tile_idx: 12 bits (0-4095) - position within tile
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileIdx {
    data: u32,
}

impl TileIdx {
    pub const DRIVER_IDX_BITS: u32 = 4;
    pub const TILE_IDX_BITS: u32 = 16;
    pub const SUB_TILE_IDX_BITS: u32 = 12; // 32 - 4 - 16 = 12

    const DRIVER_MASK: u32 = (1 << Self::DRIVER_IDX_BITS) - 1;
    const TILE_MASK: u32 = (1 << Self::TILE_IDX_BITS) - 1;
    const SUB_TILE_MASK: u32 = (1 << Self::SUB_TILE_IDX_BITS) - 1;

    const DRIVER_SHIFT: u32 = Self::TILE_IDX_BITS + Self::SUB_TILE_IDX_BITS;
    const TILE_SHIFT: u32 = Self::SUB_TILE_IDX_BITS;

    /// Create invalid tile index
    pub fn invalid() -> Self {
        Self {
            data: u32::MAX, // All bits set
        }
    }

    /// Create from sub-tile index only
    pub fn from_sub_tile(sub_tile: u32) -> Self {
        let mut idx = Self::invalid();
        idx.set_sub_tile_idx(sub_tile);
        idx
    }

    /// Create from all components
    pub fn new(driver: u32, tile: u32, sub_tile: u32) -> Self {
        let mut idx = Self { data: 0 };
        idx.set_driver_idx(driver);
        idx.set_tile_idx(tile);
        idx.set_sub_tile_idx(sub_tile);
        idx
    }

    /// Check if tile index is valid
    pub fn is_valid(&self) -> bool {
        *self != Self::invalid()
    }

    /// Get driver index (4 bits)
    pub fn driver_idx(&self) -> u32 {
        (self.data >> Self::DRIVER_SHIFT) & Self::DRIVER_MASK
    }

    /// Get tile index (16 bits)
    pub fn tile_idx(&self) -> u32 {
        (self.data >> Self::TILE_SHIFT) & Self::TILE_MASK
    }

    /// Get sub-tile index (12 bits)
    pub fn sub_tile_idx(&self) -> u32 {
        self.data & Self::SUB_TILE_MASK
    }

    /// Set driver index
    pub fn set_driver_idx(&mut self, idx: u32) {
        let masked = idx & Self::DRIVER_MASK;
        self.data = (self.data & !(Self::DRIVER_MASK << Self::DRIVER_SHIFT))
            | (masked << Self::DRIVER_SHIFT);
    }

    /// Set tile index
    pub fn set_tile_idx(&mut self, idx: u32) {
        let masked = idx & Self::TILE_MASK;
        self.data =
            (self.data & !(Self::TILE_MASK << Self::TILE_SHIFT)) | (masked << Self::TILE_SHIFT);
    }

    /// Set sub-tile index
    pub fn set_sub_tile_idx(&mut self, idx: u32) {
        let masked = idx & Self::SUB_TILE_MASK;
        self.data = (self.data & !(Self::SUB_TILE_MASK)) | masked;
    }
}

/// Trait for elevation providers
///
/// Provides elevation data and tile indexing functionality.
pub trait ElevationProvider {
    /// Get elevation at a geographic position
    fn get(&self, pos: &Point) -> ElevationMeters;

    /// Get tile index for a geographic position
    fn tile_idx(&self, pos: &Point) -> TileIdx;

    /// Get maximum resolution of this provider
    fn max_resolution(&self) -> Resolution;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elevation_meters_creation() {
        let elev = ElevationMeters::new(100);
        assert_eq!(elev.0, 100);
        assert!(elev.is_valid());
        assert_eq!(elev.meters(), Some(100));
    }

    #[test]
    fn test_elevation_meters_invalid() {
        let elev = ElevationMeters::invalid();
        assert_eq!(elev.0, ElevationMeters::INVALID);
        assert!(!elev.is_valid());
        assert_eq!(elev.meters(), None);
    }

    #[test]
    fn test_elevation_meters_default() {
        let elev = ElevationMeters::default();
        assert!(!elev.is_valid());
    }

    #[test]
    fn test_tile_idx_creation() {
        let idx = TileIdx::new(5, 1000, 200);
        assert_eq!(idx.driver_idx(), 5);
        assert_eq!(idx.tile_idx(), 1000);
        assert_eq!(idx.sub_tile_idx(), 200);
        assert!(idx.is_valid());
    }

    #[test]
    fn test_tile_idx_invalid() {
        let idx = TileIdx::invalid();
        assert!(!idx.is_valid());
        assert_eq!(idx.driver_idx(), (1 << TileIdx::DRIVER_IDX_BITS) - 1);
        assert_eq!(idx.tile_idx(), (1 << TileIdx::TILE_IDX_BITS) - 1);
        assert_eq!(idx.sub_tile_idx(), (1 << TileIdx::SUB_TILE_IDX_BITS) - 1);
    }

    #[test]
    fn test_tile_idx_from_sub_tile() {
        let idx = TileIdx::from_sub_tile(42);
        assert_eq!(idx.sub_tile_idx(), 42);
    }

    #[test]
    fn test_tile_idx_bit_packing() {
        let mut idx = TileIdx::new(0, 0, 0);

        idx.set_driver_idx(15); // Max 4 bits
        idx.set_tile_idx(65535); // Max 16 bits
        idx.set_sub_tile_idx(4095); // Max 12 bits

        assert_eq!(idx.driver_idx(), 15);
        assert_eq!(idx.tile_idx(), 65535);
        assert_eq!(idx.sub_tile_idx(), 4095);
    }

    #[test]
    fn test_tile_idx_ordering() {
        let idx1 = TileIdx::new(1, 100, 50);
        let idx2 = TileIdx::new(1, 100, 51);
        let idx3 = TileIdx::new(1, 101, 50);
        let idx4 = TileIdx::new(2, 100, 50);

        assert!(idx1 < idx2);
        assert!(idx2 < idx3);
        assert!(idx3 < idx4);
    }

    #[test]
    fn test_tile_idx_masking() {
        // Test that overflow bits are masked
        let idx = TileIdx::new(31, 100000, 10000);
        assert_eq!(idx.driver_idx(), 15); // 31 & 0xF = 15
        assert_eq!(idx.tile_idx(), 34464); // 100000 & 0xFFFF
        assert_eq!(idx.sub_tile_idx(), 1808); // 10000 & 0xFFF
    }
}
