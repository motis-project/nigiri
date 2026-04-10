//! Translation of osr/include/osr/preprocessing/elevation/hgt_driver.h + src/hgt_driver.cc
//!
//! HGT driver managing multiple HGT tiles with spatial indexing.

use std::fs;
use std::path::Path;

use rstar::{RTree, RTreeObject};

use super::hgt_tile::{Hgt1, Hgt3};
use super::resolution::Resolution;
use super::shared::{ElevationMeters, ElevationProvider, TileIdx};
use crate::point::BBox;
use crate::Point;

/// HGT tile variant (either SRTM1 or SRTM3)
pub enum HgtTileVariant {
    /// SRTM1: 1 arc-second resolution (~30m)
    Srtm1(Hgt1),
    /// SRTM3: 3 arc-second resolution (~90m)
    Srtm3(Hgt3),
}

impl ElevationProvider for HgtTileVariant {
    fn get(&self, pos: &Point) -> ElevationMeters {
        match self {
            HgtTileVariant::Srtm1(tile) => tile.get(pos),
            HgtTileVariant::Srtm3(tile) => tile.get(pos),
        }
    }

    fn tile_idx(&self, pos: &Point) -> TileIdx {
        match self {
            HgtTileVariant::Srtm1(tile) => tile.tile_idx(pos),
            HgtTileVariant::Srtm3(tile) => tile.tile_idx(pos),
        }
    }

    fn max_resolution(&self) -> Resolution {
        match self {
            HgtTileVariant::Srtm1(tile) => tile.max_resolution(),
            HgtTileVariant::Srtm3(tile) => tile.max_resolution(),
        }
    }
}

impl HgtTileVariant {
    /// Get bounding box for this tile
    pub fn get_box(&self) -> BBox {
        match self {
            HgtTileVariant::Srtm1(tile) => tile.get_box(),
            HgtTileVariant::Srtm3(tile) => tile.get_box(),
        }
    }
}

/// R-tree entry for spatial indexing
#[derive(Debug, Clone)]
struct RTreeEntry {
    bbox: BBox,
    tile_idx: usize,
}

impl rstar::RTreeObject for RTreeEntry {
    type Envelope = rstar::AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_corners(
            [self.bbox.min.lng(), self.bbox.min.lat()],
            [self.bbox.max.lng(), self.bbox.max.lat()],
        )
    }
}

impl rstar::PointDistance for RTreeEntry {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        self.envelope().distance_2(point)
    }
}

/// HGT driver managing multiple HGT tiles
///
/// Provides efficient spatial lookup using an R-tree index.
pub struct HgtDriver {
    /// R-tree for spatial indexing of tiles
    rtree: RTree<RTreeEntry>,
    /// Collection of HGT tiles
    tiles: Vec<HgtTileVariant>,
}

impl HgtDriver {
    /// Create a new empty HGT driver
    pub fn new() -> Self {
        Self {
            rtree: RTree::new(),
            tiles: Vec::new(),
        }
    }

    /// Add a tile from a file path
    ///
    /// Returns true if the tile was successfully added.
    /// Automatically detects SRTM1 vs SRTM3 based on file size.
    pub fn add_tile(&mut self, path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();

        // Check extension
        if let Some(ext) = path.extension() {
            if ext != "hgt" {
                return false;
            }
        } else {
            return false;
        }

        // Try to open the tile
        if let Ok(tile) = Self::open(path) {
            let bbox = tile.get_box();
            let tile_idx = self.tiles.len();

            // Add to R-tree
            self.rtree.insert(RTreeEntry { bbox, tile_idx });

            // Add to tiles collection
            self.tiles.push(tile);
            true
        } else {
            false
        }
    }

    /// Open an HGT file, detecting SRTM1 vs SRTM3 by file size
    fn open(path: &Path) -> Result<HgtTileVariant, String> {
        // Parse filename to get SW corner coordinates
        let filename = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| "Invalid filename".to_string())?;

        let (lat, lng) = Self::parse_grid_point(filename)?;

        // Detect tile type by file size
        let file_size = fs::metadata(path)
            .map_err(|e| format!("Failed to read file metadata: {}", e))?
            .len() as usize;

        match file_size {
            size if size == Hgt1::file_size() => {
                let tile = Hgt1::new(path, lat, lng)?;
                Ok(HgtTileVariant::Srtm1(tile))
            }
            size if size == Hgt3::file_size() => {
                let tile = Hgt3::new(path, lat, lng)?;
                Ok(HgtTileVariant::Srtm3(tile))
            }
            _ => Err(format!(
                "Unexpected file size {} (expected {} for SRTM1 or {} for SRTM3)",
                file_size,
                Hgt1::file_size(),
                Hgt3::file_size()
            )),
        }
    }

    /// Parse HGT filename to extract SW corner coordinates
    ///
    /// Format: N37W105 or S12E034, etc.
    fn parse_grid_point(filename: &str) -> Result<(i8, i16), String> {
        if filename.len() < 7 {
            return Err(format!("Filename too short: {}", filename));
        }

        let chars: Vec<char> = filename.chars().collect();

        // Parse latitude direction and value
        let lat_dir = chars[0];
        if lat_dir != 'N' && lat_dir != 'S' {
            return Err(format!("Invalid latitude direction: {}", lat_dir));
        }

        let lat_str: String = chars[1..3].iter().collect();
        let lat_val: i32 = lat_str
            .parse()
            .map_err(|_| format!("Invalid latitude: {}", lat_str))?;

        if lat_val < 0 || lat_val >= 90 {
            return Err(format!("Latitude out of range: {}", lat_val));
        }

        let lat = if lat_dir == 'N' {
            lat_val as i8
        } else {
            -(lat_val as i8)
        };

        // Parse longitude direction and value
        let lng_dir = chars[3];
        if lng_dir != 'E' && lng_dir != 'W' {
            return Err(format!("Invalid longitude direction: {}", lng_dir));
        }

        let lng_str: String = chars[4..7].iter().collect();
        let lng_val: i32 = lng_str
            .parse()
            .map_err(|_| format!("Invalid longitude: {}", lng_str))?;

        if lng_val < 0 || lng_val >= 180 {
            return Err(format!("Longitude out of range: {}", lng_val));
        }

        let lng = if lng_dir == 'E' {
            lng_val as i16
        } else {
            -(lng_val as i16)
        };

        Ok((lat, lng))
    }

    /// Get number of tiles
    pub fn n_tiles(&self) -> usize {
        self.tiles.len()
    }
}

impl ElevationProvider for HgtDriver {
    fn get(&self, pos: &Point) -> ElevationMeters {
        let point = [pos.lng(), pos.lat()];

        // Search R-tree for tiles containing this point
        for entry in self.rtree.locate_all_at_point(&point) {
            let elevation = self.tiles[entry.tile_idx].get(pos);
            if elevation.is_valid() {
                return elevation;
            }
        }

        ElevationMeters::invalid()
    }

    fn tile_idx(&self, pos: &Point) -> TileIdx {
        let point = [pos.lng(), pos.lat()];

        // Search R-tree for tiles containing this point
        for entry in self.rtree.locate_all_at_point(&point) {
            let mut idx = self.tiles[entry.tile_idx].tile_idx(pos);

            if idx.is_valid() {
                // Set tile index within this driver
                idx.set_tile_idx(entry.tile_idx as u32);
                return idx;
            }
        }

        TileIdx::invalid()
    }

    fn max_resolution(&self) -> Resolution {
        let mut res = Resolution::new();

        for tile in &self.tiles {
            res.update(&tile.max_resolution());
        }

        res
    }
}

impl Default for HgtDriver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_grid_point_north_west() {
        let (lat, lng) = HgtDriver::parse_grid_point("N37W105").unwrap();
        assert_eq!(lat, 37);
        assert_eq!(lng, -105);
    }

    #[test]
    fn test_parse_grid_point_south_east() {
        let (lat, lng) = HgtDriver::parse_grid_point("S12E034").unwrap();
        assert_eq!(lat, -12);
        assert_eq!(lng, 34);
    }

    #[test]
    fn test_parse_grid_point_north_east() {
        let (lat, lng) = HgtDriver::parse_grid_point("N45E123").unwrap();
        assert_eq!(lat, 45);
        assert_eq!(lng, 123);
    }

    #[test]
    fn test_parse_grid_point_south_west() {
        let (lat, lng) = HgtDriver::parse_grid_point("S89W179").unwrap();
        assert_eq!(lat, -89);
        assert_eq!(lng, -179);
    }

    #[test]
    fn test_parse_grid_point_invalid_lat_dir() {
        assert!(HgtDriver::parse_grid_point("X37W105").is_err());
    }

    #[test]
    fn test_parse_grid_point_invalid_lng_dir() {
        assert!(HgtDriver::parse_grid_point("N37X105").is_err());
    }

    #[test]
    fn test_parse_grid_point_too_short() {
        assert!(HgtDriver::parse_grid_point("N37W10").is_err());
    }

    #[test]
    fn test_hgt_driver_creation() {
        let driver = HgtDriver::new();
        assert_eq!(driver.n_tiles(), 0);
    }

    #[test]
    fn test_hgt_driver_max_resolution_empty() {
        let driver = HgtDriver::new();
        let res = driver.max_resolution();
        assert!(!res.is_valid());
    }
}
