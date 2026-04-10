//! Integration test for Platforms and ElevationStorage serialization

use tempfile::TempDir;
use transit_cloud_osr::types::*;
use transit_cloud_osr::{Elevation, ElevationStorage, Platforms};

#[test]
fn test_platforms_save_load() {
    let temp_dir = TempDir::new().unwrap();

    // Create a platforms instance with some data
    let platforms = Platforms::new();

    // Save to directory
    platforms.save(temp_dir.path()).unwrap();

    // Load it back
    let loaded = Platforms::load(temp_dir.path()).unwrap();

    // Should be empty
    assert!(loaded.is_empty());
}

#[test]
fn test_platforms_missing_files() {
    let temp_dir = TempDir::new().unwrap();

    // Try to load from non-existent directory
    let loaded = Platforms::load(temp_dir.path()).unwrap();

    // Should return empty (not an error)
    assert!(loaded.is_empty());
}

#[test]
fn test_elevation_storage_save_load() {
    let temp_dir = TempDir::new().unwrap();

    // Create elevation storage with some data
    let mut storage = ElevationStorage::new();

    // Add multiple ways
    storage.set_elevation(WayIdx::new(0), 0, Elevation::new(10, 5));
    storage.set_elevation(WayIdx::new(1), 0, Elevation::new(20, 15));
    storage.set_elevation(WayIdx::new(1), 1, Elevation::new(5, 2));

    // Save to directory
    storage.save(temp_dir.path()).unwrap();

    // Load it back
    let loaded = ElevationStorage::load(temp_dir.path()).unwrap();

    // Verify data
    let elev0 = loaded.get_elevation(WayIdx::new(0), 0);
    assert!(elev0.up >= 8 && elev0.up <= 14); // Compressed ~10

    let elev1 = loaded.get_elevation(WayIdx::new(1), 0);
    assert!(elev1.up >= 17 && elev1.up <= 25); // Compressed ~20

    let elev2 = loaded.get_elevation(WayIdx::new(1), 1);
    assert!(elev2.up >= 4 && elev2.up <= 8); // Compressed ~5
}

#[test]
fn test_elevation_storage_missing_files() {
    let temp_dir = TempDir::new().unwrap();

    // Try to load from non-existent directory
    let loaded = ElevationStorage::load(temp_dir.path()).unwrap();

    // Should return empty (not an error)
    let elev = loaded.get_elevation(WayIdx::new(0), 0);
    assert_eq!(elev.up, 0);
    assert_eq!(elev.down, 0);
}

#[test]
fn test_elevation_storage_compression() {
    // Test the compression roundtrip
    // COMPRESSED_VALUES: [0, 1, 2, 4, 6, 8, 11, 14, 17, 21, 25, 29, 34, 38, 43, 48]
    let test_cases = vec![
        (0, 0),   // Index 0 → 0
        (1, 1),   // Index 1 → 1
        (10, 11), // Index 6 → 11
        (20, 21), // Index 9 → 21
        (30, 34), // Index 12 → 34 (binary_search picks next higher)
        (40, 43), // Index 14 → 43
        (50, 48), // Index 15 → 48 (max)
    ];

    for (input, expected) in test_cases {
        let mut storage = ElevationStorage::new();
        storage.set_elevation(WayIdx::new(0), 0, Elevation::new(input, input));

        let loaded = storage.get_elevation(WayIdx::new(0), 0);
        assert_eq!(loaded.up, expected, "Failed for input {}", input);
        assert_eq!(loaded.down, expected, "Failed for input {}", input);
    }
}
