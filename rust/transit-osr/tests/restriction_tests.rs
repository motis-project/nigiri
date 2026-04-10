//! Turn restriction tests - Rust port of C++ restriction_test.cc
//!
//! Tests turn restrictions and elevation-aware routing.
//!
//! C++ reference: osr/test/restriction_test.cc (TEST(extract, wa))
//!
//! Expected C++ values:
//! - Restriction: rhoenring → arheilger at node 528944 is restricted
//! - Shortest distance (bike, no elevation): ≈163.0m
//! - No-elevation route: elevation up=5, down=6
//! - High-elevation route: elevation up=1, down=4, distance > 165m

use std::path::Path;

use transit_cloud_osr::{
    elevation_storage::ElevationStorage,
    extract::extract,
    routing::profile::SearchProfile,
    routing::{route::route, RoutingAlgorithm},
    ways::{NodeProperties, WayProperties},
    Level, Location, OsrData, OsmNodeIdx, OsmWayIdx,
};

// ─── integration test (requires external file) ──────────────────────────────

/// C++ expected values from restriction_test.cc
const K_SHORTEST_DISTANCE: f64 = 163.0;
const K_DISTANCE_TOLERANCE: f64 = 2.0;
const K_MAX_COST: u16 = 3600;

#[test]
#[ignore] // Requires test data files: test/map.osm + test/restriction_test_elevation/
fn test_turn_restriction_and_elevation() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path();

    // Extract with elevation data (matching C++ test paths)
    let osm_path = Path::new("test/map.osm");
    let elevation_path = Some(Path::new("test/restriction_test_elevation/"));

    if !osm_path.exists() {
        println!("Skipping test - test data not found at {:?}", osm_path);
        return;
    }

    extract(false, osm_path, output_path, elevation_path)
        .map_err(|e| format!("{}", e))
        .unwrap();

    let data = OsrData::import(output_path)
        .map_err(|e| format!("{}", e))
        .unwrap();

    // Verify extraction succeeded
    assert!(data.ways.n_ways() > 0, "No ways extracted");
    assert!(data.ways.n_nodes() > 0, "No nodes extracted");

    // ── OSM ID lookups (C++ lines 40-44) ──────────────────────────────────
    let from_osm_node = 528944_u64;
    let to_osm_node = 586157_u64;
    let rhoenring_way = 120682496_u64;
    let arheilger_way = 1068971150_u64;

    let n = data
        .ways
        .find_node_by_osm_id(from_osm_node)
        .expect("Node 528944 (Rhönring) must exist");
    let n_dst = data
        .ways
        .find_node_by_osm_id(to_osm_node)
        .expect("Node 586157 (Eckhardt/Barkhaus) must exist");
    let rw = data
        .ways
        .find_way_by_osm_id(rhoenring_way)
        .expect("Way 120682496 (Rhönring) must exist");
    let aw = data
        .ways
        .find_way_by_osm_id(arheilger_way)
        .expect("Way 1068971150 (Arheilger) must exist");

    // ── Turn restriction check (C++ lines 65-68) ─────────────────────────
    let rw_pos = data.ways.get_way_pos(n, rw);
    let aw_pos = data.ways.get_way_pos(n, aw);
    let is_restricted = data.ways.is_restricted(n, rw_pos, aw_pos);
    assert!(
        is_restricted,
        "Rhönring → Arheilger at node 528944 must be restricted"
    );

    // ── Route with node-pinned locations (C++ lines 48-49) ───────────────
    let from = Location::new(data.ways.get_node_pos(n), Level::default());
    let to = Location::new(data.ways.get_node_pos(n_dst), Level::default());

    // Load elevation data
    let elevations = ElevationStorage::load(output_path).ok();

    // ── Route 1: Bike with no elevation cost (C++ kParamsNoCosts) ────────
    let lookup = lookup_for_test(&data, output_path);
    let route_no_costs = route(
        &data.ways,
        &lookup,
        elevations.as_ref(),
        SearchProfile::Bike,
        from,
        to,
        K_MAX_COST,
        RoutingAlgorithm::Dijkstra,
    );

    let route_no_costs =
        route_no_costs.expect("Bike route (no elevation cost) must find a path");

    // C++ assertion: |route_no_costs.dist_ - kShortestDistance| < 2.0
    assert!(
        (route_no_costs.dist - K_SHORTEST_DISTANCE).abs() < K_DISTANCE_TOLERANCE,
        "No-cost route distance should be ≈{:.1}m, got {:.1}m",
        K_SHORTEST_DISTANCE,
        route_no_costs.dist
    );

    // C++ assertion: elevation up=5, down=6
    assert_eq!(
        route_no_costs.elevation.up, 5,
        "No-cost route elevation up must be 5, got {}",
        route_no_costs.elevation.up
    );
    assert_eq!(
        route_no_costs.elevation.down, 6,
        "No-cost route elevation down must be 6, got {}",
        route_no_costs.elevation.down
    );

    // ── Route 2: Bike with high elevation cost (C++ kParamsHighCosts) ────
    let route_high_costs = route(
        &data.ways,
        &lookup,
        elevations.as_ref(),
        SearchProfile::BikeElevationHigh,
        from,
        to,
        K_MAX_COST,
        RoutingAlgorithm::Dijkstra,
    );

    let route_high_costs =
        route_high_costs.expect("Bike route (high elevation cost) must find a path");

    // C++ assertion: route_high_costs.dist_ - kShortestDistance > 2.0
    assert!(
        route_high_costs.dist - K_SHORTEST_DISTANCE > K_DISTANCE_TOLERANCE,
        "High-cost route must be longer than {:.1}m + {:.1}m, got {:.1}m",
        K_SHORTEST_DISTANCE,
        K_DISTANCE_TOLERANCE,
        route_high_costs.dist
    );

    // C++ assertion: elevation up=1, down=4
    assert_eq!(
        route_high_costs.elevation.up, 1,
        "High-cost route elevation up must be 1, got {}",
        route_high_costs.elevation.up
    );
    assert_eq!(
        route_high_costs.elevation.down, 4,
        "High-cost route elevation down must be 4, got {}",
        route_high_costs.elevation.down
    );
}

/// Helper to create a Lookup for test use.
/// Uses unsafe pointer aliasing like the C++ memory-mapped approach.
#[allow(non_snake_case)]
fn lookup_for_test(data: &OsrData, output_path: &Path) -> transit_cloud_osr::Lookup {
    unsafe {
        let ways_ptr = &data.ways as *const _;
        transit_cloud_osr::Lookup::load(&*ways_ptr, output_path)
    }
    .expect("Lookup load must succeed for test data")
}

// ─── unit tests (no external file required) ──────────────────────────────────

/// TODO #6 – Minimal `no_*` turn restriction test.
///
/// Topology (T-junction):
///
/// ```text
///         north (3)
///           │
/// west(0)──[center(1)]──east(2)
/// ```
///
/// Ways:
/// - `ew_way` (OSM 100): west → center → east
/// - `ns_way` (OSM 101): north → center
///
/// Restriction: **no left turn** from `ew_way` → `ns_way` at `center`.
#[test]
fn test_basic_turn_restriction() {
    use transit_cloud_osr::{ways::Ways, Point};

    let mut ways = Ways::new();

    // Add nodes ---------------------------------------------------------------
    let west = ways.add_node(
        OsmNodeIdx(10),
        Point::from_latlng(0.0, -0.001),
        NodeProperties::default(),
    );
    let center = ways.add_node(
        OsmNodeIdx(11),
        Point::from_latlng(0.0, 0.0),
        NodeProperties::default(),
    );
    let east = ways.add_node(
        OsmNodeIdx(12),
        Point::from_latlng(0.0, 0.001),
        NodeProperties::default(),
    );
    let north = ways.add_node(
        OsmNodeIdx(13),
        Point::from_latlng(0.001, 0.0),
        NodeProperties::default(),
    );

    // Add ways ----------------------------------------------------------------
    let ew_way = ways.add_way(
        OsmWayIdx(100),
        vec![west, center, east],
        WayProperties::default(),
    );
    let ns_way = ways.add_way(
        OsmWayIdx(101),
        vec![north, center],
        WayProperties::default(),
    );

    // Build reverse node→way mappings (manual mode)
    ways.connect_ways();

    // TODO #3 – verify OSM ID lookups -------------------------------------------
    assert_eq!(
        ways.find_node_by_osm_id(11),
        Some(center),
        "OSM node 11 must resolve to center"
    );
    assert_eq!(
        ways.find_way_by_osm_id(100),
        Some(ew_way),
        "OSM way 100 must resolve to ew_way"
    );
    assert_eq!(
        ways.find_way_by_osm_id(101),
        Some(ns_way),
        "OSM way 101 must resolve to ns_way"
    );

    // Determine way-positions at `center` in its node_ways list ----------------
    let center_ways = ways.get_node_ways(center);
    assert!(
        center_ways.len() >= 2,
        "center must be connected to at least 2 ways; got {}",
        center_ways.len()
    );
    let ew_pos = ways.get_way_pos(center, ew_way);
    let ns_pos = ways.get_way_pos(center, ns_way);

    // -- Before restriction: all turns are allowed ----------------------------
    assert!(
        !ways.is_restricted(center, ew_pos, ns_pos),
        "no restriction should exist before add_restriction"
    );
    assert!(
        !ways.is_restricted(center, ns_pos, ew_pos),
        "reverse turn also unrestricted before add_restriction"
    );

    // Add "no left turn" restriction: ew_way → ns_way at center (is_only=false)
    ways.add_restriction(ew_way, ns_way, center, false);

    // -- After restriction: ew → ns is prohibited -----------------------------
    assert!(
        ways.is_restricted(center, ew_pos, ns_pos),
        "ew_way → ns_way at center must now be restricted"
    );

    // The reverse direction (ns → ew) must remain unrestricted.
    assert!(
        !ways.is_restricted(center, ns_pos, ew_pos),
        "ns_way → ew_way must remain unrestricted"
    );

    // Continuing straight along ew_way (ew → ew) must also be unrestricted.
    assert!(
        !ways.is_restricted(center, ew_pos, ew_pos),
        "straight-ahead along ew_way must remain unrestricted"
    );
}

/// TODO #7 – `only_*` turn restriction test.
///
/// Topology (cross-junction):
///
/// ```text
///         north (4)
///           │
/// west(0)──[center(1)]──east(2)
///           │
///         south (3)
/// ```
///
/// Ways:
/// - `ew_way` (OSM 200): west → center → east
/// - `ns_way` (OSM 201): south → center → north
///
/// Restriction: **only straight-on** when following `ns_way` through `center`.
/// The prohibited side-turns (ns→ew) are stored via `add_restriction(ns_way, ew_way, …, is_only=true)`.
#[test]
fn test_only_restriction() {
    use transit_cloud_osr::{ways::Ways, Point};

    let mut ways = Ways::new();

    let west = ways.add_node(
        OsmNodeIdx(20),
        Point::from_latlng(0.0, -0.001),
        NodeProperties::default(),
    );
    let center = ways.add_node(
        OsmNodeIdx(21),
        Point::from_latlng(0.0, 0.0),
        NodeProperties::default(),
    );
    let east = ways.add_node(
        OsmNodeIdx(22),
        Point::from_latlng(0.0, 0.001),
        NodeProperties::default(),
    );
    let south = ways.add_node(
        OsmNodeIdx(23),
        Point::from_latlng(-0.001, 0.0),
        NodeProperties::default(),
    );
    let north = ways.add_node(
        OsmNodeIdx(24),
        Point::from_latlng(0.001, 0.0),
        NodeProperties::default(),
    );

    let ew_way = ways.add_way(
        OsmWayIdx(200),
        vec![west, center, east],
        WayProperties::default(),
    );
    let ns_way = ways.add_way(
        OsmWayIdx(201),
        vec![south, center, north],
        WayProperties::default(),
    );

    ways.connect_ways();

    // Way positions at center
    let ns_pos = ways.get_way_pos(center, ns_way);
    let ew_pos = ways.get_way_pos(center, ew_way);

    // "only_straight_on" : the only permitted move on ns_way is to continue on
    // ns_way.  The prohibited turn (ns → ew) is stored explicitly.
    // `is_only = true` records the RestrictionType::Only tag; the Dijkstra
    // currently treats any stored restriction as prohibited (same check path).
    ways.add_restriction(ns_way, ew_way, center, true);

    // Straight-on (ns → ns) must NOT appear as a stored restriction
    assert!(
        !ways.is_restricted(center, ns_pos, ns_pos),
        "straight-on (ns → ns) must NOT be restricted for only_straight_on"
    );

    // The prohibited side-turn (ns → ew) must be blocked
    assert!(
        ways.is_restricted(center, ns_pos, ew_pos),
        "ns_way → ew_way must be prohibited by only_straight_on"
    );

    // Travelling along ew_way is unaffected (different incoming way)
    assert!(
        !ways.is_restricted(center, ew_pos, ns_pos),
        "ew_way → ns_way must remain unrestricted (different incoming way)"
    );
}

/// TODO #5 – Elevation cost profiles (compile-time smoke test).
///
/// The routing engine exposes elevation sensitivity via `SearchProfile`:
/// - `Bike`               → no elevation penalty
/// - `BikeElevationLow`   → mild climb penalty
/// - `BikeElevationHigh`  → strong climb penalty
///
/// A full integration test (requiring real OSM + DEM data) would route
/// between two hilly points and assert that the high-elevation profile
/// produces a longer but flatter path than the low-elevation profile.
#[test]
fn test_elevation_profiles_exist() {
    use transit_cloud_osr::routing::mode::Mode;

    // All three sensitivity levels must be distinct enum variants
    assert_ne!(SearchProfile::Bike, SearchProfile::BikeElevationLow);
    assert_ne!(SearchProfile::BikeElevationLow, SearchProfile::BikeElevationHigh);
    assert_ne!(SearchProfile::Bike, SearchProfile::BikeElevationHigh);

    // All elevation variants must use the Bike mode
    assert_eq!(SearchProfile::Bike.mode(), Mode::Bike);
    assert_eq!(SearchProfile::BikeElevationLow.mode(), Mode::Bike);    assert_eq!(SearchProfile::BikeElevationHigh.mode(), Mode::Bike);
}