use transit_core::{Position, TransitError, TransitResult};
use transit_osr::data::OsrData;
use transit_osr::routing::{route, Path, RoutingAlgorithm, SearchProfile};
use transit_osr::{Level, Location as OsrLocation};

/// Default maximum walking cost in seconds (15 minutes).
pub const DEFAULT_MAX_WALK_COST: u32 = 900;

/// Compute walking offsets from a coordinate to nearby transit stops.
///
/// Returns a list of (location_idx, duration_minutes) pairs for stops
/// reachable within `max_walk_seconds`.
pub fn compute_walk_offsets(
    osr: &OsrData,
    tt: &nigiri::Timetable,
    from: Position,
    max_walk_seconds: u32,
) -> TransitResult<Vec<(u32, i16)>> {
    let from_loc = OsrLocation::from_latlng(from.lat, from.lon, Level::default());

    let mut offsets = Vec::new();
    let loc_count = tt.location_count();

    for i in 0..loc_count {
        if let Ok(detail) = tt.get_location_detail(i) {
            if detail.lat == 0.0 && detail.lon == 0.0 {
                continue;
            }

            // Quick beeline distance filter (rough: 1 degree ≈ 111km)
            let dlat = (from.lat - detail.lat).abs();
            let dlon = (from.lon - detail.lon).abs();
            if dlat > 0.02 || dlon > 0.02 {
                // More than ~2km away, skip
                continue;
            }

            let to_loc = OsrLocation::from_latlng(detail.lat, detail.lon, Level::default());

            if let Some(path) = route(
                &osr.ways,
                &osr.lookup,
                Some(&osr.elevations),
                SearchProfile::Foot,
                from_loc.clone(),
                to_loc,
                max_walk_seconds,
                RoutingAlgorithm::Dijkstra,
            ) {
                let duration_minutes = (path.cost / 60).max(1) as i16;
                if duration_minutes <= (max_walk_seconds / 60) as i16 {
                    offsets.push((i, duration_minutes));
                }
            }
        }
    }

    Ok(offsets)
}

/// Route directly between two coordinates using a street profile.
pub fn route_direct(
    osr: &OsrData,
    from: Position,
    to: Position,
    profile: SearchProfile,
    max_cost: u32,
) -> Option<Path> {
    let from_loc = OsrLocation::from_latlng(from.lat, from.lon, Level::default());
    let to_loc = OsrLocation::from_latlng(to.lat, to.lon, Level::default());

    route(
        &osr.ways,
        &osr.lookup,
        Some(&osr.elevations),
        profile,
        from_loc,
        to_loc,
        max_cost,
        RoutingAlgorithm::Dijkstra,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_walk_cost_is_15_minutes() {
        assert_eq!(DEFAULT_MAX_WALK_COST, 900);
    }
}
