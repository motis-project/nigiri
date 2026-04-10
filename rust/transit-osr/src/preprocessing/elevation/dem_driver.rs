use std::path::Path;

use rstar::{RTree, RTreeObject};

use super::dem_tile::DemTile;
use super::resolution::Resolution;
use super::shared::{ElevationMeters, ElevationProvider, TileIdx};
use crate::point::BBox;
use crate::Point;

/// R-tree entry for spatial indexing of DEM tiles
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

/// DEM file driver with R-tree spatial indexing
///
/// Manages collection of DEM tiles (.hdr/.bil format)
pub struct DemDriver {
    rtree: RTree<RTreeEntry>,
    tiles: Vec<DemTile>,
}

impl DemDriver {
    /// Create empty driver
    pub fn new() -> Self {
        Self {
            rtree: RTree::new(),
            tiles: Vec::new(),
        }
    }

    /// Add tile from .hdr file path
    ///
    /// Returns true if tile was added successfully
    pub fn add_tile(&mut self, path: &Path) -> bool {
        // Check for .hdr extension
        if path.extension().and_then(|s| s.to_str()) != Some("hdr") {
            return false;
        }

        // Try to load tile
        let tile = match DemTile::new(path) {
            Ok(t) => t,
            Err(_) => return false,
        };

        // Add to collection and R-tree
        let idx = self.tiles.len();
        let bbox = tile.get_box();
        self.tiles.push(tile);

        let entry = RTreeEntry {
            bbox,
            tile_idx: idx,
        };
        self.rtree.insert(entry);

        true
    }

    /// Get number of tiles
    pub fn n_tiles(&self) -> usize {
        self.tiles.len()
    }
}

impl Default for DemDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl ElevationProvider for DemDriver {
    fn get(&self, pos: &Point) -> ElevationMeters {
        let point = [pos.lng(), pos.lat()];

        for entry in self.rtree.locate_all_at_point(&point) {
            let meters = self.tiles[entry.tile_idx].get(pos);
            if meters.is_valid() {
                return meters;
            }
        }

        ElevationMeters::invalid()
    }

    fn tile_idx(&self, pos: &Point) -> TileIdx {
        let point = [pos.lng(), pos.lat()];

        for entry in self.rtree.locate_all_at_point(&point) {
            let mut idx = self.tiles[entry.tile_idx].tile_idx(pos);
            if idx != TileIdx::new(0, 0, 0) {
                idx.set_tile_idx(entry.tile_idx as u32);
                return idx;
            }
        }

        TileIdx::new(0, 0, 0)
    }

    fn max_resolution(&self) -> Resolution {
        let mut res = Resolution::new();
        for tile in &self.tiles {
            res.update(&tile.max_resolution());
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_dem_driver_creation() {
        let driver = DemDriver::new();
        assert_eq!(driver.n_tiles(), 0);
    }

    #[test]
    fn test_dem_driver_default() {
        let driver = DemDriver::default();
        assert_eq!(driver.n_tiles(), 0);
    }

    #[test]
    fn test_dem_driver_add_tile_wrong_extension() {
        let mut driver = DemDriver::new();
        let path = PathBuf::from("test.txt");
        assert!(!driver.add_tile(&path));
        assert_eq!(driver.n_tiles(), 0);
    }

    #[test]
    fn test_dem_driver_add_tile_missing_file() {
        let mut driver = DemDriver::new();
        let path = PathBuf::from("nonexistent.hdr");
        assert!(!driver.add_tile(&path));
        assert_eq!(driver.n_tiles(), 0);
    }

    #[test]
    fn test_dem_driver_max_resolution_empty() {
        let driver = DemDriver::new();
        let res = driver.max_resolution();
        assert!(!res.is_valid());
    }

    #[test]
    fn test_dem_driver_get_empty() {
        let driver = DemDriver::new();
        let pos = Point::from_latlng(37.7749, -122.4194);
        let elevation = driver.get(&pos);
        assert!(!elevation.is_valid());
    }

    #[test]
    fn test_dem_driver_tile_idx_empty() {
        let driver = DemDriver::new();
        let pos = Point::from_latlng(37.7749, -122.4194);
        let idx = driver.tile_idx(&pos);
        assert_eq!(idx, TileIdx::new(0, 0, 0));
    }

    #[test]
    fn test_rtree_entry_envelope() {
        let bbox = BBox::from_latlng(37.0, -122.0, 38.0, -121.0);
        let entry = RTreeEntry { bbox, tile_idx: 0 };

        let envelope = entry.envelope();
        // AABB should contain the corners
        assert_eq!(envelope.lower(), [-122.0, 37.0]);
        assert_eq!(envelope.upper(), [-121.0, 38.0]);
    }
}
