use std::path::Path;

use transit_core::TransitResult;

/// Load OSR street routing data from a pre-extracted directory.
///
/// The OSR data must have been previously extracted from an OSM PBF file
/// using `transit_osr::extract::extract()`.
pub fn load_osr_data(osr_data_path: &Path) -> TransitResult<transit_osr::data::OsrData> {
    tracing::info!("Loading OSR data from {}", osr_data_path.display());
    let data = transit_osr::data::OsrData::import(osr_data_path)
        .map_err(|e| transit_core::TransitError::Osr(format!("failed to load OSR data: {e}")))?;
    tracing::info!("OSR data loaded successfully");
    Ok(data)
}
