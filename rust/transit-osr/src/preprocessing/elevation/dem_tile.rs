use ahash::AHashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use memmap2::Mmap;

use super::resolution::Resolution;
use super::shared::{ElevationMeters, ElevationProvider, TileIdx};
use crate::point::BBox;
use crate::Point;

/// DEM/BIL pixel type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelType {
    Int16,
    Float32,
}

/// Pixel value union (int16 or float32)
#[derive(Clone, Copy)]
pub union PixelValue {
    pub int16: i16,
    pub float32: f32,
}

impl std::fmt::Debug for PixelValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PixelValue{{ union }}")
    }
}

/// BIL header information
#[derive(Debug)]
struct BilHeader {
    rows: u32,
    cols: u32,
    ulx: f64,        // upper left longitude
    uly: f64,        // upper left latitude
    brx: f64,        // bottom right longitude
    bry: f64,        // bottom right latitude
    xdim: f64,       // x pixel dimension (degrees)
    ydim: f64,       // y pixel dimension (degrees)
    pixel_size: u32, // bytes per pixel
    row_size: u32,   // bytes per row
    pixel_type: PixelType,
    nodata: PixelValue,
}

impl BilHeader {
    /// Parse .hdr file and create header
    fn from_hdr_file(hdr_path: &Path) -> Result<Self, String> {
        let file = File::open(hdr_path)
            .map_err(|e| format!("Failed to open HDR file {:?}: {}", hdr_path, e))?;

        let reader = BufReader::new(file);
        let mut map = AHashMap::new();

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Error reading HDR file: {}", e))?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let key = parts[0].to_uppercase();
                let value = parts[1].to_uppercase();
                map.insert(key, value);
            }
        }

        Self::from_map(&map)
    }

    /// Build header from parsed key-value map
    fn from_map(map: &AHashMap<String, String>) -> Result<Self, String> {
        let rows = Self::get_uint(map, "NROWS", 0);
        let cols = Self::get_uint(map, "NCOLS", 0);

        if rows == 0 || cols == 0 {
            return Err("Missing or invalid NROWS/NCOLS".to_string());
        }

        if Self::get_uint(map, "NBANDS", 1) != 1 {
            return Err("Unsupported nbands value (must be 1)".to_string());
        }

        if Self::get_string(map, "BYTEORDER", "I") != "I" {
            return Err("Unsupported byte order (must be I = Intel/little-endian)".to_string());
        }

        if Self::get_uint(map, "SKIPBYTES", 0) != 0 {
            return Err("Unsupported skipbytes (must be 0)".to_string());
        }

        let nbits = Self::get_uint(map, "NBITS", 8);
        let pixeltype = Self::get_string(map, "PIXELTYPE", "UNSIGNEDINT");

        let (pixel_type, pixel_size) = if nbits == 16 && pixeltype.starts_with('S') {
            (PixelType::Int16, 2)
        } else if nbits == 32 && pixeltype.starts_with('F') {
            (PixelType::Float32, 4)
        } else {
            return Err(format!(
                "Unsupported pixel type: nbits={}, pixeltype={}",
                nbits, pixeltype
            ));
        };

        let row_size = pixel_size * cols;

        let ulx = Self::get_double(map, "ULXMAP", f64::MIN);
        let uly = Self::get_double(map, "ULYMAP", f64::MIN);

        if ulx == f64::MIN || uly == f64::MIN {
            return Err("Missing ULXMAP/ULYMAP".to_string());
        }

        let xdim = Self::get_double(map, "XDIM", 0.0);
        let ydim = Self::get_double(map, "YDIM", 0.0);

        if xdim == 0.0 || ydim == 0.0 {
            return Err("Missing XDIM/YDIM".to_string());
        }

        let brx = ulx + (cols as f64) * xdim;
        let bry = uly - (rows as f64) * ydim;

        let nodata = match pixel_type {
            PixelType::Int16 => {
                let val = Self::get_int(map, "NODATA", 0);
                PixelValue { int16: val as i16 }
            }
            PixelType::Float32 => {
                let val = Self::get_double(map, "NODATA", 0.0);
                PixelValue {
                    float32: val as f32,
                }
            }
        };

        let bandrowbytes = Self::get_uint(map, "BANDROWBYTES", row_size);
        let totalrowbytes = Self::get_uint(map, "TOTALROWBYTES", row_size);

        if bandrowbytes != row_size || totalrowbytes != row_size {
            return Err("Unsupported bandrowbytes/totalrowbytes".to_string());
        }

        Ok(Self {
            rows,
            cols,
            ulx,
            uly,
            brx,
            bry,
            xdim,
            ydim,
            pixel_size,
            row_size,
            pixel_type,
            nodata,
        })
    }

    fn get_string(map: &AHashMap<String, String>, key: &str, default: &str) -> String {
        map.get(key)
            .map(|s| s.clone())
            .unwrap_or_else(|| default.to_string())
    }

    fn get_int(map: &AHashMap<String, String>, key: &str, default: i32) -> i32 {
        map.get(key).and_then(|s| s.parse().ok()).unwrap_or(default)
    }

    fn get_uint(map: &AHashMap<String, String>, key: &str, default: u32) -> u32 {
        map.get(key).and_then(|s| s.parse().ok()).unwrap_or(default)
    }

    fn get_double(map: &AHashMap<String, String>, key: &str, default: f64) -> f64 {
        map.get(key).and_then(|s| s.parse().ok()).unwrap_or(default)
    }
}

/// DEM tile (EHdr/BIL format)
///
/// Supports ESRI BIL format with .hdr header and .bil data file
/// http://www.gdal.org/frmt_various.html#EHdr
pub struct DemTile {
    hdr: BilHeader,
    _file: File,
    mmap: Mmap,
}

impl DemTile {
    const NODATA: i16 = -32768;

    /// Load DEM tile from .hdr file path
    ///
    /// The .bil data file must exist in the same directory
    pub fn new(hdr_path: &Path) -> Result<Self, String> {
        // Parse header
        let hdr = BilHeader::from_hdr_file(hdr_path)?;

        // Find .bil file
        let bil_path = hdr_path.with_extension("bil");
        if !bil_path.exists() {
            return Err(format!("Missing BIL file: {:?}", bil_path));
        }

        // Memory-map the .bil file
        let file = File::open(&bil_path)
            .map_err(|e| format!("Failed to open BIL file {:?}: {}", bil_path, e))?;

        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("Failed to mmap BIL file: {}", e))?;

        // Verify file size
        let expected_size = (hdr.row_size * hdr.rows) as usize;
        if mmap.len() != expected_size {
            return Err(format!(
                "BIL file size mismatch: expected {}, got {}",
                expected_size,
                mmap.len()
            ));
        }

        Ok(Self {
            hdr,
            _file: file,
            mmap,
        })
    }

    /// Get geographic bounding box
    pub fn get_box(&self) -> BBox {
        BBox::from_latlng(self.hdr.bry, self.hdr.ulx, self.hdr.uly, self.hdr.brx)
    }

    /// Get raw pixel value (int16 or float32)
    pub fn get_raw(&self, pos: &Point) -> PixelValue {
        if !self.get_box().contains(pos) {
            return self.hdr.nodata;
        }

        let pix_x = ((pos.lng() - self.hdr.ulx) / self.hdr.xdim) as u32;
        let pix_x = pix_x.min(self.hdr.cols - 1);

        let pix_y = ((self.hdr.uly - pos.lat()) / self.hdr.ydim) as u32;
        let pix_y = pix_y.min(self.hdr.rows - 1);

        let byte_pos = (self.hdr.row_size * pix_y + self.hdr.pixel_size * pix_x) as usize;

        match self.hdr.pixel_type {
            PixelType::Int16 => {
                let bytes = &self.mmap[byte_pos..byte_pos + 2];
                let val = i16::from_le_bytes([bytes[0], bytes[1]]);
                PixelValue { int16: val }
            }
            PixelType::Float32 => {
                let bytes = &self.mmap[byte_pos..byte_pos + 4];
                let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                PixelValue { float32: val }
            }
        }
    }

    /// Get pixel type
    pub fn get_pixel_type(&self) -> PixelType {
        self.hdr.pixel_type
    }

    /// Get pixel coordinates for position
    fn get_pixel<const PIXEL_SIZE: u32>(&self, pos: &Point) -> (u32, u32) {
        let pix_x = ((pos.lng() - self.hdr.ulx) * PIXEL_SIZE as f64) as u32;
        let pix_x = pix_x.min(PIXEL_SIZE - 1);

        let pix_y = ((self.hdr.uly - pos.lat()) * PIXEL_SIZE as f64) as u32;
        let pix_y = pix_y.min(PIXEL_SIZE - 1);

        (pix_x, pix_y)
    }
}

impl ElevationProvider for DemTile {
    fn get(&self, pos: &Point) -> ElevationMeters {
        let val = self.get_raw(pos);

        match self.hdr.pixel_type {
            PixelType::Int16 => unsafe {
                if val.int16 == self.hdr.nodata.int16 {
                    ElevationMeters::invalid()
                } else if val.int16 == Self::NODATA {
                    ElevationMeters::invalid()
                } else {
                    ElevationMeters(val.int16)
                }
            },
            PixelType::Float32 => unsafe {
                if val.float32 == self.hdr.nodata.float32 {
                    ElevationMeters::invalid()
                } else {
                    let rounded = val.float32.round() as i16;
                    if rounded == Self::NODATA {
                        ElevationMeters::invalid()
                    } else {
                        ElevationMeters(rounded)
                    }
                }
            },
        }
    }

    fn tile_idx(&self, pos: &Point) -> TileIdx {
        const PIXEL_SIZE: u32 = 1 << (TileIdx::SUB_TILE_IDX_BITS / 2);

        if self.get_box().contains(pos) {
            let (pix_x, pix_y) = self.get_pixel::<PIXEL_SIZE>(pos);
            TileIdx::new(0, 0, pix_x * PIXEL_SIZE + pix_y)
        } else {
            TileIdx::new(0, 0, 0)
        }
    }

    fn max_resolution(&self) -> Resolution {
        Resolution {
            x: self.hdr.xdim,
            y: self.hdr.ydim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_type_ordering() {
        assert_eq!(PixelType::Int16, PixelType::Int16);
        assert_eq!(PixelType::Float32, PixelType::Float32);
        assert_ne!(PixelType::Int16, PixelType::Float32);
    }

    #[test]
    fn test_bil_header_missing_rows() {
        let map = AHashMap::from([("NCOLS".to_string(), "100".to_string())]);

        let result = BilHeader::from_map(&map);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("NROWS"));
    }

    #[test]
    fn test_bil_header_missing_cols() {
        let map = AHashMap::from([("NROWS".to_string(), "100".to_string())]);

        let result = BilHeader::from_map(&map);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("NCOLS"));
    }

    #[test]
    fn test_bil_header_unsupported_nbands() {
        let map = AHashMap::from([
            ("NROWS".to_string(), "100".to_string()),
            ("NCOLS".to_string(), "100".to_string()),
            ("NBANDS".to_string(), "3".to_string()),
        ]);

        let result = BilHeader::from_map(&map);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nbands"));
    }

    #[test]
    fn test_bil_header_unsupported_byteorder() {
        let map = AHashMap::from([
            ("NROWS".to_string(), "100".to_string()),
            ("NCOLS".to_string(), "100".to_string()),
            ("BYTEORDER".to_string(), "M".to_string()),
        ]);

        let result = BilHeader::from_map(&map);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("byte order"));
    }

    #[test]
    fn test_bil_header_int16() {
        let map = AHashMap::from([
            ("NROWS".to_string(), "100".to_string()),
            ("NCOLS".to_string(), "200".to_string()),
            ("NBITS".to_string(), "16".to_string()),
            ("PIXELTYPE".to_string(), "SIGNEDINT".to_string()),
            ("ULXMAP".to_string(), "-122.5".to_string()),
            ("ULYMAP".to_string(), "37.8".to_string()),
            ("XDIM".to_string(), "0.001".to_string()),
            ("YDIM".to_string(), "0.001".to_string()),
            ("NODATA".to_string(), "-9999".to_string()),
        ]);

        let hdr = BilHeader::from_map(&map).expect("Valid header");

        assert_eq!(hdr.rows, 100);
        assert_eq!(hdr.cols, 200);
        assert_eq!(hdr.pixel_type, PixelType::Int16);
        assert_eq!(hdr.pixel_size, 2);
        assert_eq!(hdr.row_size, 400);
        assert_eq!(hdr.ulx, -122.5);
        assert_eq!(hdr.uly, 37.8);
        assert_eq!(hdr.xdim, 0.001);
        assert_eq!(hdr.ydim, 0.001);
        assert_eq!(hdr.brx, -122.5 + 200.0 * 0.001);
        assert_eq!(hdr.bry, 37.8 - 100.0 * 0.001);

        unsafe {
            assert_eq!(hdr.nodata.int16, -9999);
        }
    }

    #[test]
    fn test_bil_header_float32() {
        let map = AHashMap::from([
            ("NROWS".to_string(), "50".to_string()),
            ("NCOLS".to_string(), "75".to_string()),
            ("NBITS".to_string(), "32".to_string()),
            ("PIXELTYPE".to_string(), "FLOAT".to_string()),
            ("ULXMAP".to_string(), "10.0".to_string()),
            ("ULYMAP".to_string(), "50.0".to_string()),
            ("XDIM".to_string(), "0.01".to_string()),
            ("YDIM".to_string(), "0.01".to_string()),
            ("NODATA".to_string(), "-999.0".to_string()),
        ]);

        let hdr = BilHeader::from_map(&map).expect("Valid header");

        assert_eq!(hdr.rows, 50);
        assert_eq!(hdr.cols, 75);
        assert_eq!(hdr.pixel_type, PixelType::Float32);
        assert_eq!(hdr.pixel_size, 4);
        assert_eq!(hdr.row_size, 300);

        unsafe {
            assert_eq!(hdr.nodata.float32, -999.0);
        }
    }

    #[test]
    fn test_dem_tile_nodata_constant() {
        assert_eq!(DemTile::NODATA, -32768);
    }
}
