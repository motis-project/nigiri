use std::fs;
use std::path::Path;

use super::dem_driver::DemDriver;
use super::hgt_driver::HgtDriver;
use super::resolution::Resolution;
use super::shared::{ElevationMeters, ElevationProvider, TileIdx};
use crate::Point;

/// Elevation provider variant (DEM or HGT)
enum ProviderVariant {
    Dem(DemDriver),
    Hgt(HgtDriver),
}

impl ElevationProvider for ProviderVariant {
    fn get(&self, pos: &Point) -> ElevationMeters {
        match self {
            ProviderVariant::Dem(d) => d.get(pos),
            ProviderVariant::Hgt(h) => h.get(pos),
        }
    }

    fn tile_idx(&self, pos: &Point) -> TileIdx {
        match self {
            ProviderVariant::Dem(d) => d.tile_idx(pos),
            ProviderVariant::Hgt(h) => h.tile_idx(pos),
        }
    }

    fn max_resolution(&self) -> Resolution {
        match self {
            ProviderVariant::Dem(d) => d.max_resolution(),
            ProviderVariant::Hgt(h) => h.max_resolution(),
        }
    }
}

/// Main elevation provider
///
/// Scans directory for DEM (.hdr) and HGT (.hgt) files
/// Maintains separate drivers for each format
pub struct Provider {
    drivers: Vec<ProviderVariant>,
}

impl Provider {
    /// Create provider from directory path
    ///
    /// Recursively scans for .hdr and .hgt files
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();

        if !path.is_dir() {
            return Err(format!("Path is not a directory: {:?}", path));
        }

        let mut dem = DemDriver::new();
        let mut hgt = HgtDriver::new();

        Self::scan_directory(path, &mut dem, &mut hgt)?;

        // Add drivers that have tiles
        let mut drivers = Vec::new();
        if dem.n_tiles() > 0 {
            drivers.push(ProviderVariant::Dem(dem));
        }
        if hgt.n_tiles() > 0 {
            drivers.push(ProviderVariant::Hgt(hgt));
        }

        Ok(Self { drivers })
    }

    /// Recursively scan directory for elevation files
    fn scan_directory(path: &Path, dem: &mut DemDriver, hgt: &mut HgtDriver) -> Result<(), String> {
        let entries = fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory {:?}: {}", path, e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();

            if path.is_dir() {
                // Recurse into subdirectories
                Self::scan_directory(&path, dem, hgt)?;
            } else if path.is_file() {
                // Try to add as DEM or HGT
                dem.add_tile(&path);
                hgt.add_tile(&path);
            }
        }

        Ok(())
    }

    /// Get number of drivers
    pub fn driver_count(&self) -> usize {
        self.drivers.len()
    }
}

impl ElevationProvider for Provider {
    fn get(&self, pos: &Point) -> ElevationMeters {
        for driver in &self.drivers {
            let meters = driver.get(pos);
            if meters.is_valid() {
                return meters;
            }
        }
        ElevationMeters::invalid()
    }

    fn tile_idx(&self, pos: &Point) -> TileIdx {
        for (driver_idx, driver) in self.drivers.iter().enumerate() {
            let mut idx = driver.tile_idx(pos);
            if idx != TileIdx::new(0, 0, 0) {
                idx.set_driver_idx(driver_idx as u32);
                return idx;
            }
        }
        TileIdx::new(0, 0, 0)
    }

    fn max_resolution(&self) -> Resolution {
        let mut res = Resolution::new();
        for driver in &self.drivers {
            res.update(&driver.max_resolution());
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_provider_creation_missing_path() {
        let path = PathBuf::from("nonexistent_directory_xyz");
        let result = Provider::new(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_provider_creation_not_directory() {
        // Try with a file that doesn't exist
        let path = PathBuf::from("not_a_directory.txt");
        let result = Provider::new(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_provider_variant_dem() {
        let driver = DemDriver::new();
        let variant = ProviderVariant::Dem(driver);
        let pos = Point::from_latlng(37.0, -122.0);

        // Should return invalid for empty driver
        let elevation = variant.get(&pos);
        assert!(!elevation.is_valid());
    }

    #[test]
    fn test_provider_variant_hgt() {
        let driver = HgtDriver::new();
        let variant = ProviderVariant::Hgt(driver);
        let pos = Point::from_latlng(37.0, -122.0);

        // Should return invalid for empty driver
        let elevation = variant.get(&pos);
        assert!(!elevation.is_valid());
    }

    #[test]
    fn test_provider_empty_drivers() {
        let provider = Provider {
            drivers: Vec::new(),
        };

        assert_eq!(provider.driver_count(), 0);

        let pos = Point::from_latlng(37.0, -122.0);
        let elevation = provider.get(&pos);
        assert!(!elevation.is_valid());

        let idx = provider.tile_idx(&pos);
        assert_eq!(idx, TileIdx::new(0, 0, 0));

        let res = provider.max_resolution();
        assert!(!res.is_valid());
    }
}
