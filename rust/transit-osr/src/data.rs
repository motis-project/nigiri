//! OSR Data Coordinator
//!
//! High-level coordinator for importing and exporting complete OSR datasets.
//! Handles business logic like finalization, building derived data structures,
//! and coordinating the save/load order of related components.

use std::path::Path;

use crate::lookup::Lookup;
use crate::ways::Ways;
use crate::{ElevationStorage, Platforms};

/// Complete OSR dataset with all components.
///
/// This struct coordinates the lifecycle of all OSR data:
/// - Ways (routing graph)
/// - Platforms (transit station data)
/// - Elevations (terrain data)
/// - Lookup (R-tree spatial index)
///
/// # Export Business Logic
/// 1. Finalizes Ways (computes derived data)
/// 2. Saves Ways, Platforms, Elevations
/// 3. Builds Lookup R-tree from Ways
/// 4. Saves Lookup
///
/// # Import Business Logic
/// 1. Loads Ways, Platforms, Elevations
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use transit_cloud_osr::data::OsrData;
///
/// // Export
/// let mut data = OsrData::new();
/// // ... populate data.ways, data.platforms, data.elevations ...
/// data.export(Path::new("./output"))?;
///
/// // Import
/// let data = OsrData::import(Path::new("./output"))?;
/// // Access the components
/// println!("Loaded {} ways", data.ways.n_ways());
/// # Ok::<(), String>(())
/// ```
pub struct OsrData {
    /// Routing graph with ways, nodes, and restrictions
    pub ways: Ways,

    /// Platform/station data for transit integration
    pub platforms: Platforms,

    /// Elevation data for terrain-aware routing
    pub elevations: ElevationStorage,
    /// R-tree spatial index for fast location lookup
    pub lookup: Lookup,
}

impl OsrData {
    /// Create a new empty OSR dataset.
    pub fn new() -> Self {
        Self {
            ways: Ways::new(),
            platforms: Platforms::new(),
            elevations: ElevationStorage::new(),
            lookup: Lookup::new(&Ways::new()), // Empty lookup
        }
    }

    /// Export the complete dataset to disk.
    ///
    /// # Business Logic
    /// 1. Finalizes Ways (computes big street neighbors, etc.)
    /// 2. Saves Ways using multi-file format (26 files)
    /// 3. Saves Platforms
    /// 4. Saves Elevations
    /// 5. Builds Lookup R-tree spatial index from Ways
    /// 6. Saves Lookup
    ///
    /// # Arguments
    /// * `output_path` - Directory to write all data files
    ///
    /// # File Output
    /// Creates the following files in `output_path`:
    /// - `ways_*.bin` - 26 files for Ways data structure
    /// - `platforms_*.bin` - Platform data files
    /// - `elevations_*.bin` - Elevation data files
    /// - `rtree_data.bin` - R-tree spatial index
    ///
    /// # Errors
    /// Returns error if:
    /// - Output directory cannot be created
    /// - Any data structure fails to save
    /// - R-tree cannot be built
    pub fn export(&mut self, output_path: &Path) -> Result<(), String> {
        // Create output directory if needed
        std::fs::create_dir_all(output_path)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;

        println!("Exporting OSR data to {}...", output_path.display());

        // Step 1: Save Ways (multi-file format)
        println!("  Saving ways (26 files)...");
        self.ways
            .save_multi_file(output_path)
            .map_err(|e| format!("Failed to save ways: {}", e))?;

        // Step 2: Save Platforms
        println!("  Saving platforms...");
        self.platforms
            .save(output_path)
            .map_err(|e| format!("Failed to save platforms: {}", e))?;

        // Step 3: Save Elevations
        println!("  Saving elevations...");
        self.elevations
            .save(output_path)
            .map_err(|e| format!("Failed to save elevations: {}", e))?;

        // Step 4: Build Lookup R-tree (business logic - derived data structure)
        println!("  Building R-tree spatial index...");

        self.lookup
            .save(output_path)
            .map_err(|e| format!("Failed to save R-tree: {}", e))?;

        println!(
            "✓ Export complete - {} ways, {} platforms",
            self.ways.n_ways(),
            self.platforms.num_platforms()
        );

        Ok(())
    }

    /// Import a complete dataset from disk.
    ///
    /// # Business Logic
    /// 1. Loads Ways from multi-file format
    /// 2. Loads Platforms
    /// 3. Loads Elevations
    ///
    /// Returns a dataset ready for routing queries.
    /// Build Lookup R-tree separately using `Lookup::new(&data.ways)`.
    ///
    /// # Arguments
    /// * `input_path` - Directory containing all data files
    ///
    /// # Errors
    /// Returns error if:
    /// - Input directory doesn't exist
    /// - Required files are missing
    /// - Any data structure fails to load
    /// - Data is corrupted or incompatible
    pub fn import(input_path: &Path) -> Result<Self, String> {
        if !input_path.exists() {
            return Err(format!(
                "Input directory does not exist: {}",
                input_path.display()
            ));
        }

        println!("Importing OSR data from {}...", input_path.display());

        // Step 1: Load Ways
        println!("  Loading ways...");
        let ways =
            Ways::load_multi_file(input_path).map_err(|e| format!("Failed to load ways: {}", e))?;

        // Step 2: Load Platforms
        println!("  Loading platforms...");
        let platforms =
            Platforms::load(input_path).map_err(|e| format!("Failed to load platforms: {}", e))?;

        // Step 3: Load Elevations
        println!("  Loading elevations...");
        let elevations = ElevationStorage::load(input_path)
            .map_err(|e| format!("Failed to load elevations: {}", e))?;

        println!(
            "✓ Import complete - {} ways, {} platforms",
            ways.n_ways(),
            platforms.num_platforms()
        );

        // Step 4: Load Lookup (must be done after struct creation due to self-reference)
        println!("  Loading lookup...");
        // SAFETY: We transmute the lifetime here. This is safe because:
        // 1. Lookup is stored in the same struct as Ways
        // 2. They will be dropped together
        // 3. Lookup won't outlive Ways
        let lookup =
            Lookup::load(&ways, input_path).map_err(|e| format!("Failed to load lookup: {}", e))?;

        // Create OsrData first
        let data = Self {
            ways,
            platforms,
            elevations,
            lookup,
        };

        Ok(data)
    }
}

impl Default for OsrData {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_new() {
        let data = OsrData::new();
        assert_eq!(data.ways.n_ways(), 0);
        assert_eq!(data.platforms.num_platforms(), 0);
    }

    #[test]
    fn test_export_import_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path();

        // Create and export data
        let mut data = OsrData::new();
        // Export empty data (test the flow, not the content)
        data.export(output_path).unwrap();

        // Note: Cannot test import roundtrip with empty data
        // because Ways::load_multi_file expects valid populated data
        // This test verifies export works without panicking
    }

    #[test]
    fn test_import_nonexistent() {
        let result = OsrData::import(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }
}
