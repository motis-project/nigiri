//! Translation of osr/include/osr/preprocessing/elevation/hgt_tile.h + hgt_tile_def.h
//!
//! HGT (Height) tile format for SRTM elevation data.
//!
//! SRTM HGT File Format:
//! - Binary 16-bit signed integer raster data
//! - Row-major order (north to south, west to east)
//! - No header or trailer bytes
//! - Filename indicates SW corner (e.g., N37W105.hgt)
//! - Void value: -32768 (version 1.0 and 2.1)
//! - Reference: https://lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf

use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::resolution::Resolution;
use super::shared::{ElevationMeters, ElevationProvider, TileIdx};
use crate::point::BBox;
use crate::Point;

/// Void elevation value in SRTM HGT files (versions 1.0 and 2.1)
const VOID_VALUE: i16 = -32768;

/// HGT tile with fixed raster size
///
/// Generic over RASTER_SIZE to support different SRTM resolutions:
/// - 1201x1201 (SRTM3, 3 arc-second, ~90m resolution)
/// - 3601x3601 (SRTM1, 1 arc-second, ~30m resolution)
pub struct HgtTile<const RASTER_SIZE: usize> {
    /// Memory-mapped file data
    data: Vec<u8>,
    /// Southwest corner latitude
    sw_lat: i8,
    /// Southwest corner longitude
    sw_lng: i16,
}

impl<const RASTER_SIZE: usize> HgtTile<RASTER_SIZE> {
    /// Bytes per pixel (16-bit signed integer)
    pub const BYTES_PER_PIXEL: usize = 2;

    /// Expected file size in bytes
    pub const fn file_size() -> usize {
        RASTER_SIZE * RASTER_SIZE * Self::BYTES_PER_PIXEL
    }

    /// Step width in degrees (distance between pixels)
    const STEP_WIDTH: f64 = 1.0 / (RASTER_SIZE - 1) as f64;

    /// Center offset (half a pixel)
    const CENTER_OFFSET: f64 = Self::STEP_WIDTH / 2.0;

    /// Load HGT tile from file
    ///
    /// # Arguments
    /// * `path` - Path to .hgt file
    /// * `lat` - Southwest corner latitude
    /// * `lng` - Southwest corner longitude
    pub fn new(path: impl AsRef<Path>, lat: i8, lng: i16) -> Result<Self, String> {
        let path = path.as_ref();
        let mut file =
            File::open(path).map_err(|e| format!("Failed to open HGT file {:?}: {}", path, e))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Failed to read HGT file {:?}: {}", path, e))?;

        let expected_size = Self::file_size();
        if data.len() != expected_size {
            return Err(format!(
                "HGT tile {:?} ({}x{}) has incorrect file size ({} != {})",
                path,
                RASTER_SIZE,
                RASTER_SIZE,
                data.len(),
                expected_size
            ));
        }

        Ok(Self {
            data,
            sw_lat: lat,
            sw_lng: lng,
        })
    }

    /// Get bounding box covered by this tile
    pub fn get_box(&self) -> BBox {
        BBox::from_latlng(
            self.sw_lat as f64 - Self::CENTER_OFFSET,
            self.sw_lng as f64 - Self::CENTER_OFFSET,
            self.sw_lat as f64 + 1.0 + Self::CENTER_OFFSET,
            self.sw_lng as f64 + 1.0 + Self::CENTER_OFFSET,
        )
    }

    /// Get raster offset for a position
    ///
    /// Returns offset in the raster grid, or usize::MAX if out of bounds.
    fn get_offset<const UPPER_BOUND: usize>(&self, pos: &Point) -> usize {
        let bbox = self.get_box();

        if !bbox.contains(pos) {
            return usize::MAX;
        }

        let lat = pos.lat();
        let lng = pos.lng();

        // Column: Left to right (west to east)
        let column = ((lng - bbox.min.lng()) * (RASTER_SIZE - 1) as f64 / RASTER_SIZE as f64
            * UPPER_BOUND as f64)
            .floor() as usize;
        let column = column.min(UPPER_BOUND - 1);

        // Row: Top to bottom (north to south)
        let row = ((bbox.max.lat() - lat) * (RASTER_SIZE - 1) as f64 / RASTER_SIZE as f64
            * UPPER_BOUND as f64)
            .floor() as usize;
        let row = row.min(UPPER_BOUND - 1);

        // Data in row-major order
        UPPER_BOUND * row + column
    }

    /// Get elevation at byte offset
    fn get_at_offset(&self, offset: usize) -> ElevationMeters {
        if offset + 1 >= self.data.len() {
            return ElevationMeters::invalid();
        }

        // Read big-endian 16-bit signed integer
        let bytes = [self.data[offset], self.data[offset + 1]];
        let raw_value = i16::from_be_bytes(bytes);

        if raw_value == VOID_VALUE {
            ElevationMeters::invalid()
        } else {
            ElevationMeters::new(raw_value)
        }
    }
}

impl<const RASTER_SIZE: usize> ElevationProvider for HgtTile<RASTER_SIZE> {
    fn get(&self, pos: &Point) -> ElevationMeters {
        let offset = self.get_offset::<RASTER_SIZE>(pos);

        if offset == usize::MAX {
            return ElevationMeters::invalid();
        }

        self.get_at_offset(Self::BYTES_PER_PIXEL * offset)
    }

    fn tile_idx(&self, pos: &Point) -> TileIdx {
        // Segments: sqrt of max sub-tile index
        const SEGMENTS: usize = 1 << (TileIdx::SUB_TILE_IDX_BITS / 2);

        let offset = self.get_offset::<SEGMENTS>(pos);

        if offset == usize::MAX {
            TileIdx::invalid()
        } else {
            TileIdx::from_sub_tile(offset as u32)
        }
    }

    fn max_resolution(&self) -> Resolution {
        Resolution {
            x: Self::STEP_WIDTH,
            y: Self::STEP_WIDTH,
        }
    }
}

/// SRTM3 tile (3 arc-second, ~90m resolution)
pub type Hgt3 = HgtTile<1201>;

/// SRTM1 tile (1 arc-second, ~30m resolution)
pub type Hgt1 = HgtTile<3601>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_size_srtm3() {
        assert_eq!(Hgt3::file_size(), 1201 * 1201 * 2);
    }

    #[test]
    fn test_file_size_srtm1() {
        assert_eq!(Hgt1::file_size(), 3601 * 3601 * 2);
    }

    #[test]
    fn test_step_width() {
        let step_3 = Hgt3::STEP_WIDTH;
        assert!((step_3 - 1.0 / 1200.0).abs() < 0.0001);

        let step_1 = Hgt1::STEP_WIDTH;
        assert!((step_1 - 1.0 / 3600.0).abs() < 0.0001);
    }

    #[test]
    fn test_bbox_calculation() {
        let data = vec![0u8; Hgt3::file_size()];
        let tile = HgtTile::<1201> {
            data,
            sw_lat: 37,
            sw_lng: -105,
        };

        let bbox = tile.get_box();

        // Southwest corner should be slightly offset
        assert!(bbox.min.lat() < 37.0);
        assert!(bbox.min.lng() < -105.0);

        // Northeast corner should extend 1 degree plus offset
        assert!(bbox.max.lat() > 38.0);
        assert!(bbox.max.lng() > -104.0);
    }

    #[test]
    fn test_resolution() {
        let data = vec![0u8; Hgt3::file_size()];
        let tile = HgtTile::<1201> {
            data,
            sw_lat: 37,
            sw_lng: -105,
        };

        let res = tile.max_resolution();
        assert!((res.x - Hgt3::STEP_WIDTH).abs() < 0.000001);
        assert!((res.y - Hgt3::STEP_WIDTH).abs() < 0.000001);
    }

    #[test]
    fn test_get_elevation_void() {
        // Create tile with void values
        let mut data = vec![0u8; Hgt3::file_size()];

        // Write void value at first pixel (northwest corner, row 0 col 0)
        data[0] = 0x80;
        data[1] = 0x00;

        let tile = HgtTile::<1201> {
            data,
            sw_lat: 37,
            sw_lng: -105,
        };

        // Point in northwest corner of tile (top-left)
        // SW is at 37,-105, so NW is at 38,-105
        let pos = Point::from_latlng(38.0, -105.0);
        let elev = tile.get(&pos);

        assert!(!elev.is_valid());
    }

    #[test]
    fn test_get_elevation_valid() {
        // Create tile with valid elevation
        let mut data = vec![0u8; Hgt3::file_size()];

        // Write elevation value 1234m at first pixel (northwest corner)
        data[0] = (1234i16 >> 8) as u8;
        data[1] = (1234i16 & 0xFF) as u8;

        let tile = HgtTile::<1201> {
            data,
            sw_lat: 37,
            sw_lng: -105,
        };

        // Point in northwest corner of tile
        let pos = Point::from_latlng(38.0, -105.0);
        let elev = tile.get(&pos);

        assert!(elev.is_valid());
        assert_eq!(elev.meters(), Some(1234));
    }

    #[test]
    fn test_out_of_bounds() {
        let data = vec![0u8; Hgt3::file_size()];
        let tile = HgtTile::<1201> {
            data,
            sw_lat: 37,
            sw_lng: -105,
        };

        // Point outside tile
        let pos = Point::from_latlng(36.0, -105.5);
        let elev = tile.get(&pos);

        assert!(!elev.is_valid());
    }
}
