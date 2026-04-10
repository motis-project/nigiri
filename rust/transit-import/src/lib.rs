pub mod hashes;
pub mod osr_import;
pub mod rtree;

use std::path::{Path, PathBuf};

use transit_core::config::TimetableConfig;
use transit_core::{Config, TagLookup, TransitError, TransitResult};

pub use rtree::{build_location_rtree, LocationEntry};

/// Result of the import pipeline.
pub struct ImportResult {
    pub timetable: nigiri::Timetable,
    pub location_rtree: rstar::RTree<LocationEntry>,
    pub tag_lookup: TagLookup,
}

/// Run the import pipeline: load timetable, build spatial index.
pub fn run_import(config: &Config, data_path: &Path) -> TransitResult<ImportResult> {
    let tt_config = config
        .timetable
        .as_ref()
        .ok_or_else(|| TransitError::Config("timetable configuration is required".to_string()))?;

    // Check for incremental rebuild
    let mut import_hashes = hashes::ImportHashes::load(data_path);

    // Resolve dataset paths for hash checking
    let dataset_paths: Vec<PathBuf> = tt_config
        .datasets
        .values()
        .map(|ds| PathBuf::from(&ds.path))
        .collect();
    let path_refs: Vec<&Path> = dataset_paths.iter().map(|p| p.as_path()).collect();

    let needs_rebuild = import_hashes.needs_rebuild("timetable", &path_refs);

    if needs_rebuild {
        tracing::info!("Timetable inputs changed, rebuilding...");
    } else {
        tracing::info!(
            "Timetable inputs unchanged, but reloading (serialization not yet supported)"
        );
    }

    // Load timetable
    let tt = load_timetable(tt_config)?;

    // Build spatial index
    let location_rtree = build_location_rtree(&tt);

    // Build tag lookup
    let tag_lookup = build_tag_lookup(tt_config);

    // Record hashes for next time
    import_hashes.mark_built("timetable", &path_refs);
    if let Err(e) = import_hashes.save() {
        tracing::warn!("Failed to save import hashes: {e}");
    }

    Ok(ImportResult {
        timetable: tt,
        location_rtree,
        tag_lookup,
    })
}

/// Load timetable from the first configured dataset.
fn load_timetable(config: &TimetableConfig) -> TransitResult<nigiri::Timetable> {
    if config.datasets.is_empty() {
        return Err(TransitError::Config(
            "no datasets configured in timetable section".to_string(),
        ));
    }

    // Resolve date range
    let (from_ts, to_ts) = resolve_date_range(&config.first_day, config.num_days)?;

    // For now, load only the first dataset (multi-dataset requires nigiri_load_multi)
    let (tag, dataset) = config.datasets.iter().next().unwrap();
    tracing::info!("Loading dataset '{}' from {}", tag, dataset.path);

    let tt = nigiri::Timetable::load_linking_stops(
        &dataset.path,
        from_ts,
        to_ts,
        config.link_stop_distance,
    )
    .map_err(|e| TransitError::Nigiri(format!("failed to load timetable: {e}")))?;

    tracing::info!(
        "Loaded timetable: {} locations, {} routes, {} transports",
        tt.location_count(),
        tt.route_count(),
        tt.transport_count()
    );

    Ok(tt)
}

/// Build a TagLookup from the configured dataset tags.
fn build_tag_lookup(config: &TimetableConfig) -> TagLookup {
    let mut lookup = TagLookup::new();
    for tag in config.datasets.keys() {
        lookup.register_tag(tag);
    }
    tracing::info!("Built tag lookup with {} tags", lookup.tag_count());
    lookup
}

/// Resolve the date range from config first_day + num_days to Unix timestamps.
fn resolve_date_range(first_day: &str, num_days: u16) -> TransitResult<(i64, i64)> {
    use chrono::{NaiveDate, Utc};

    let start_date = if first_day == "TODAY" {
        Utc::now().date_naive()
    } else {
        NaiveDate::parse_from_str(first_day, "%Y-%m-%d")
            .map_err(|e| TransitError::Config(format!("invalid first_day '{first_day}': {e}")))?
    };

    let end_date = start_date + chrono::Duration::days(i64::from(num_days));

    let from_ts = start_date
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    let to_ts = end_date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();

    Ok((from_ts, to_ts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_date_range_today() {
        let (from, to) = resolve_date_range("TODAY", 30).unwrap();
        assert!(to > from);
        assert!((to - from) >= 30 * 86400);
    }

    #[test]
    fn resolve_date_range_specific() {
        let (from, to) = resolve_date_range("2023-08-09", 3).unwrap();
        assert_eq!(from, 1691539200);
        assert_eq!(to, 1691798400);
    }

    #[test]
    fn resolve_date_range_invalid() {
        assert!(resolve_date_range("not-a-date", 30).is_err());
    }
}
