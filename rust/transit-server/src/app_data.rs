use std::path::PathBuf;

use rstar::RTree;
use transit_core::{Config, TagLookup};
use transit_import::LocationEntry;

/// Central application state shared across all GraphQL resolvers.
pub struct AppData {
    pub config: Config,
    pub data_path: PathBuf,
    pub timetable: Option<nigiri::Timetable>,
    pub location_rtree: Option<RTree<LocationEntry>>,
    pub tags: Option<TagLookup>,
    // Phase 4: pub osr: Option<OsrHandle>,
    // Phase 6: pub rt: Arc<RwLock<Option<RtState>>>,
    // Phase 7: pub gbfs: Arc<RwLock<Option<GbfsData>>>,
}

impl AppData {
    /// Create AppData with no timetable loaded (health-check only).
    pub fn empty(config: Config, data_path: PathBuf) -> Self {
        Self {
            config,
            data_path,
            timetable: None,
            location_rtree: None,
            tags: None,
        }
    }

    /// Create AppData from import result.
    pub fn from_import(
        config: Config,
        data_path: PathBuf,
        import: transit_import::ImportResult,
    ) -> Self {
        Self {
            config,
            data_path,
            timetable: Some(import.timetable),
            location_rtree: Some(import.location_rtree),
            tags: Some(import.tag_lookup),
        }
    }
}
