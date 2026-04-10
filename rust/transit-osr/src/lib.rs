//! Open Street Router - Memory-efficient multi-profile routing on OpenStreetMap
//!
//! This is a 1:1 Rust translation of OSR (https://github.com/motis-project/osr)
//! maintaining 100% functional parity with the C++ implementation.
//!
//! ## Module Structure (matches C++ exactly)
//!
//! - `types` - Strong-typed indices (types.h)
//! - `point` - Geographic point (point.h)
//! - `location` - Location with level (location.h)
//! - `ways` - Street network storage (ways.h/cc)
//! - `platforms` - Transit platform features (platforms.h)
//! - `lookup` - OSM ID mapping (lookup.h/cc)
//! - `elevation_storage` - Elevation data (elevation_storage.h/cc)
//! - `geojson` - GeoJSON output (geojson.h)
//! - `extract` - OSM extraction pipeline (extract/)
//! - `preprocessing` - Data preprocessing (preprocessing/)
//! - `routing` - Routing algorithms (routing/)
//! - `util` - Utility functions (util/)

// Core types and data structures
pub mod elevation_storage;
pub mod geojson;
pub mod location;
pub mod lookup;
pub mod platforms;
pub mod point;
pub mod types;
pub mod ways;

// Extraction and preprocessing
pub mod extract;
pub mod preprocessing;

// Routing
pub mod routing;

// Utilities
pub mod util;

// Import/Export for binary serialization
pub mod data;

// Re-exports for convenience
pub use data::OsrData;
pub use elevation_storage::{Elevation, ElevationStorage};
pub use location::Location;
pub use lookup::Lookup;
pub use platforms::Platforms;
pub use point::Point;
pub use routing::profile::SearchProfile;
pub use types::*;
pub use ways::Ways;

/// OSR result type
pub type Result<T> = std::result::Result<T, Error>;

/// OSR error types (matching C++ exceptions)
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("OSM parsing error: {0}")]
    OsmParse(String),

    #[error("Routing error: {0}")]
    Routing(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Elevation error: {0}")]
    Elevation(String),
}
