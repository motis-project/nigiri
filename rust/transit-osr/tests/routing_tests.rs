//! Routing tests - Rust port of C++ routing_tests.cc
//!
//! Tests basic routing functionality with real OSM data.
//! Validates GeoJSON output against expected C++ values from the nigiri-based
//! test suite (same coordinates, profiles, and expected costs/distances).
//!
//! C++ reference: osr/test/routing_tests.cc
//! - Uses foot profile with max_cost=900, max_match_distance=250
//! - Compares full GeoJSON FeatureCollection output

use std::path::Path;

use serde_json::Value;
use transit_cloud_osr::{
    extract::extract,
    geojson::path_to_geojson,
    routing::{profile::SearchProfile, route::route, Path as RoutePath, RoutingAlgorithm},
    Level, Location, Lookup, OsrData,
};

/// C++ test parameters
const MAX_COST: u16 = 900; // C++ uses 900s max cost

/// Extract OSM data and route between two points, returning the Path.
fn extract_and_route(
    osm_path: &str,
    from: Location,
    to: Location,
    profile: SearchProfile,
) -> Option<(OsrData, RoutePath)> {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let output_path = temp_dir.path();

    extract(false, Path::new(osm_path), output_path, None)
        .map_err(|e| eprintln!("Extract failed: {}", e))
        .ok()?;

    let data = OsrData::import(output_path)
        .map_err(|e| eprintln!("Import failed: {}", e))
        .ok()?;

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, output_path)
    }
    .map_err(|e| eprintln!("Lookup load failed: {}", e))
    .ok()?;

    let result = route(
        &data.ways,
        &lookup,
        None,
        profile,
        from,
        to,
        MAX_COST,
        RoutingAlgorithm::Dijkstra,
    )?;

    Some((data, result))
}

/// Compare routing result against expected C++ GeoJSON output.
///
/// Parses the expected C++ GeoJSON string and compares:
/// - Number of segments (features)
/// - Per-segment: cost, distance, level, osm_way_id
/// - Coordinate count per segment
fn assert_route_matches_cpp(
    data: &OsrData,
    path: &RoutePath,
    expected_geojson: &str,
    test_name: &str,
) {
    let actual_collection = path_to_geojson(&data.ways, path, false);
    let actual_json = actual_collection.to_value();

    let expected: Value =
        serde_json::from_str(expected_geojson).expect("Invalid expected GeoJSON");

    let expected_features = expected["features"].as_array().expect("features array");
    let actual_features = actual_json["features"].as_array().expect("features array");

    // Compare segment count
    assert_eq!(
        actual_features.len(),
        expected_features.len(),
        "{}: segment count mismatch (actual={}, expected={})",
        test_name,
        actual_features.len(),
        expected_features.len()
    );

    // Compare per-segment properties
    let mut total_cost_expected: i64 = 0;
    let mut total_dist_expected: i64 = 0;
    let mut total_cost_actual: i64 = 0;
    let mut total_dist_actual: i64 = 0;

    for (i, (actual, expected)) in actual_features
        .iter()
        .zip(expected_features.iter())
        .enumerate()
    {
        let a_props = &actual["properties"];
        let e_props = &expected["properties"];

        let a_cost = a_props["cost"].as_i64().unwrap_or(-1);
        let e_cost = e_props["cost"].as_i64().unwrap_or(-1);
        let a_dist = a_props["distance"].as_i64().unwrap_or(-1);
        let e_dist = e_props["distance"].as_i64().unwrap_or(-1);

        total_cost_actual += a_cost;
        total_dist_actual += a_dist;
        total_cost_expected += e_cost;
        total_dist_expected += e_dist;

        // Check level matches
        let a_level = a_props["level"].as_f64().unwrap_or(f64::NAN);
        let e_level = e_props["level"].as_f64().unwrap_or(f64::NAN);
        assert!(
            (a_level - e_level).abs() < 0.1,
            "{}: segment {} level mismatch (actual={}, expected={})",
            test_name,
            i,
            a_level,
            e_level
        );

        // Check osm_way_id matches
        let a_way = a_props["osm_way_id"].as_u64().unwrap_or(0);
        let e_way = e_props["osm_way_id"].as_u64().unwrap_or(0);
        assert_eq!(
            a_way, e_way,
            "{}: segment {} osm_way_id mismatch (actual={}, expected={})",
            test_name, i, a_way, e_way
        );

        // Check cost and distance (allow small tolerance)
        assert!(
            (a_cost - e_cost).abs() <= 1,
            "{}: segment {} cost mismatch (actual={}, expected={})",
            test_name,
            i,
            a_cost,
            e_cost
        );
        assert!(
            (a_dist - e_dist).abs() <= 1,
            "{}: segment {} distance mismatch (actual={}, expected={})",
            test_name,
            i,
            a_dist,
            e_dist
        );

        // Check coordinate count matches
        let a_coords = actual["geometry"]["coordinates"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        let e_coords = expected["geometry"]["coordinates"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        assert_eq!(
            a_coords, e_coords,
            "{}: segment {} coordinate count mismatch (actual={}, expected={})",
            test_name, i, a_coords, e_coords
        );
    }

    // Verify total cost and distance
    assert!(
        (total_cost_actual - total_cost_expected).abs() <= actual_features.len() as i64,
        "{}: total cost mismatch (actual={}, expected={})",
        test_name,
        total_cost_actual,
        total_cost_expected
    );
    assert!(
        (total_dist_actual - total_dist_expected).abs() <= actual_features.len() as i64,
        "{}: total distance mismatch (actual={}, expected={})",
        test_name,
        total_dist_actual,
        total_dist_expected
    );
}

// ============================================================================
// C++ expected GeoJSON outputs (from routing_tests.cc)
// ============================================================================

/// Expected GeoJSON from C++ `TEST(routing, island)` — Luisenplatz, Darmstadt
const EXPECTED_ISLAND: &str = r#"{"type":"FeatureCollection","metadata":{},"features":[{"type":"Feature","properties":{"level":0,"osm_way_id":0,"cost":3,"distance":3},"geometry":{"type":"LineString","coordinates":[[8.651514431787469,49.87274386418564],[8.6515174,49.8727447]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1201551426,"cost":3,"distance":3},"geometry":{"type":"LineString","coordinates":[[8.6515174,49.8727447],[8.6515563,49.8727556]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":22937760,"cost":12,"distance":12},"geometry":{"type":"LineString","coordinates":[[8.6515563,49.8727556],[8.6515406,49.8727706],[8.6514528,49.8728367]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":847710844,"cost":6,"distance":7},"geometry":{"type":"LineString","coordinates":[[8.6514528,49.8728367],[8.6514754,49.8728959]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":23511479,"cost":10,"distance":10},"geometry":{"type":"LineString","coordinates":[[8.6514754,49.8728959],[8.651539,49.8729497],[8.6515596,49.8729618]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":0,"cost":8,"distance":9},"geometry":{"type":"LineString","coordinates":[[8.6515596,49.8729618],[8.651493334251755,49.87300334757231]]}}]}"#;

/// Expected GeoJSON from C++ `TEST(routing, ferry)` — Ajaccio ferry
const EXPECTED_FERRY: &str = r#"{"type":"FeatureCollection","metadata":{},"features":[{"type":"Feature","properties":{"level":0,"osm_way_id":0,"cost":84,"distance":101},"geometry":{"type":"LineString","coordinates":[[8.7415295,41.9221369],[8.7415321,41.9222147]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":0,"cost":147,"distance":177},"geometry":{"type":"LineString","coordinates":[[8.7415321,41.9222147],[8.7408991,41.9222267],[8.7400905,41.9218375]]}}]}"#;

/// Expected GeoJSON from C++ `TEST(routing, corridor)` — London corridor
const EXPECTED_CORRIDOR: &str = r#"{"type":"FeatureCollection","metadata":{},"features":[{"type":"Feature","properties":{"level":0,"osm_way_id":0,"cost":1,"distance":1},"geometry":{"type":"LineString","coordinates":[[-0.056227579350404025,51.54663151628954],[-0.0562413,51.5466308]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1057782512,"cost":6,"distance":6},"geometry":{"type":"LineString","coordinates":[[-0.0562413,51.5466308],[-0.0562459,51.5466862]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":881103497,"cost":3,"distance":4},"geometry":{"type":"LineString","coordinates":[[-0.0562459,51.5466862],[-0.0562474,51.5467056],[-0.0562483,51.5467228]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":881103497,"cost":6,"distance":7},"geometry":{"type":"LineString","coordinates":[[-0.0562483,51.5467228],[-0.0562524,51.5467815]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1079378708,"cost":5,"distance":6},"geometry":{"type":"LineString","coordinates":[[-0.0562524,51.5467815],[-0.0562531,51.5468061],[-0.0562538,51.5468316]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1079378709,"cost":3,"distance":4},"geometry":{"type":"LineString","coordinates":[[-0.0562538,51.5468316],[-0.0562546,51.5468484],[-0.0562621,51.5468663]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":881103498,"cost":3,"distance":4},"geometry":{"type":"LineString","coordinates":[[-0.0562621,51.5468663],[-0.0562628,51.5469052]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1063762150,"cost":1,"distance":1},"geometry":{"type":"LineString","coordinates":[[-0.0562628,51.5469052],[-0.056263,51.5469163]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1063762149,"cost":2,"distance":3},"geometry":{"type":"LineString","coordinates":[[-0.056263,51.5469163],[-0.0562635,51.546946]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":881103496,"cost":7,"distance":9},"geometry":{"type":"LineString","coordinates":[[-0.0562635,51.546946],[-0.0562657,51.5470109],[-0.0562908,51.5470107]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":0,"cost":41,"distance":49},"geometry":{"type":"LineString","coordinates":[[-0.0562908,51.5470107],[-0.0562911,51.547031],[-0.0566186,51.5470287],[-0.0569331,51.5470277],[-0.0569440194997824,51.547027764967275]]}}]}"#;

/// Expected GeoJSON from C++ `TEST(routing, stop_area)` — Paris station multi-level
const EXPECTED_STOP_AREA: &str = r#"{"type":"FeatureCollection","metadata":{},"features":[{"type":"Feature","properties":{"level":0,"osm_way_id":0,"cost":13,"distance":13},"geometry":{"type":"LineString","coordinates":[[2.261314500627132,48.72526636283443],[2.2613838,48.7253219]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1290289280,"cost":6,"distance":6},"geometry":{"type":"LineString","coordinates":[[2.2613838,48.7253219],[2.2613206,48.7253586]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1290289280,"cost":5,"distance":5},"geometry":{"type":"LineString","coordinates":[[2.2613206,48.7253586],[2.2612721,48.7253867]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1290289280,"cost":3,"distance":3},"geometry":{"type":"LineString","coordinates":[[2.2612721,48.7253867],[2.2612448,48.7254022]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":892871341,"cost":6,"distance":7},"geometry":{"type":"LineString","coordinates":[[2.2612448,48.7254022],[2.2611731,48.7254412]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1419149027,"cost":7,"distance":9},"geometry":{"type":"LineString","coordinates":[[2.2611731,48.7254412],[2.2610764,48.7254927]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1419149026,"cost":16,"distance":19},"geometry":{"type":"LineString","coordinates":[[2.2610764,48.7254927],[2.260814,48.7254962]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1359929257,"cost":20,"distance":24},"geometry":{"type":"LineString","coordinates":[[2.260814,48.7254962],[2.2605996,48.7253278]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1429030767,"cost":10,"distance":12},"geometry":{"type":"LineString","coordinates":[[2.2605996,48.7253278],[2.2604973,48.7252419]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1429030769,"cost":2,"distance":2},"geometry":{"type":"LineString","coordinates":[[2.2604973,48.7252419],[2.2604761,48.7252248]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1429030768,"cost":7,"distance":8},"geometry":{"type":"LineString","coordinates":[[2.2604761,48.7252248],[2.2604036,48.7251661]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1359929257,"cost":22,"distance":26},"geometry":{"type":"LineString","coordinates":[[2.2604036,48.7251661],[2.2602208,48.7250223],[2.2601787,48.7249893]]}},{"type":"Feature","properties":{"level":1,"osm_way_id":222040526,"cost":25,"distance":30},"geometry":{"type":"LineString","coordinates":[[2.2601787,48.7249893],[2.2600672,48.7250507],[2.2600272,48.7250721],[2.2598576,48.7251608]]}},{"type":"Feature","properties":{"level":1,"osm_way_id":222040526,"cost":11,"distance":13},"geometry":{"type":"LineString","coordinates":[[2.2598576,48.7251608],[2.2597231,48.7252337]]}},{"type":"Feature","properties":{"level":1,"osm_way_id":222040526,"cost":4,"distance":5},"geometry":{"type":"LineString","coordinates":[[2.2597231,48.7252337],[2.2596673,48.7252638]]}},{"type":"Feature","properties":{"level":1,"osm_way_id":222040526,"cost":93,"distance":112},"geometry":{"type":"LineString","coordinates":[[2.2596673,48.7252638],[2.2596292,48.725284],[2.2584789,48.7258964]]}},{"type":"Feature","properties":{"level":1,"osm_way_id":222040526,"cost":13,"distance":16},"geometry":{"type":"LineString","coordinates":[[2.2584789,48.7258964],[2.2583131,48.7259901]]}},{"type":"Feature","properties":{"level":1,"osm_way_id":642263621,"cost":8,"distance":10},"geometry":{"type":"LineString","coordinates":[[2.2583131,48.7259901],[2.2582647,48.7259491],[2.2582257,48.725916]]}},{"type":"Feature","properties":{"level":1,"osm_way_id":642578765,"cost":2,"distance":3},"geometry":{"type":"LineString","coordinates":[[2.2582257,48.725916],[2.2581937,48.7259329]]}},{"type":"Feature","properties":{"level":1,"osm_way_id":642263625,"cost":6,"distance":7},"geometry":{"type":"LineString","coordinates":[[2.2581937,48.7259329],[2.2581458,48.7258947],[2.2581625,48.7258856]]}},{"type":"Feature","properties":{"level":1,"osm_way_id":642578757,"cost":7,"distance":8},"geometry":{"type":"LineString","coordinates":[[2.2581625,48.7258856],[2.2582443,48.7258411]]}},{"type":"Feature","properties":{"level":0.5,"osm_way_id":642263631,"cost":12,"distance":14},"geometry":{"type":"LineString","coordinates":[[2.2582443,48.7258411],[2.2581285,48.7257422]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1192127892,"cost":1,"distance":1},"geometry":{"type":"LineString","coordinates":[[2.2581285,48.7257422],[2.2581167,48.7257321]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":1192127892,"cost":2,"distance":2},"geometry":{"type":"LineString","coordinates":[[2.2581167,48.7257321],[2.2581393,48.7257206]]}},{"type":"Feature","properties":{"level":0,"osm_way_id":0,"cost":55,"distance":66},"geometry":{"type":"LineString","coordinates":[[2.2581393,48.7257206],[2.2582250818254357,48.72579309702946]]}}]}"#;

// ============================================================================
// Test cases - direct port of C++ routing_tests.cc
// ============================================================================

#[test]
#[ignore] // Requires test data file: luisenplatz-darmstadt.osm.pbf
fn test_island() {
    let from = Location::from_latlng(49.872715, 8.651534, Level::default());
    let to = Location::from_latlng(49.873023, 8.651523, Level::default());

    let (data, path) = extract_and_route(
        "test/luisenplatz-darmstadt.osm.pbf",
        from,
        to,
        SearchProfile::Foot,
    )
    .expect("Island routing must succeed");

    // C++ expected: 6 segments, total cost=42, total distance=44
    assert_eq!(path.segments.len(), 6, "island: expected 6 segments");
    assert_route_matches_cpp(&data, &path, EXPECTED_ISLAND, "island");
}

#[test]
#[ignore] // Requires test data file: ajaccio-ferry.osm.pbf
fn test_ferry() {
    let from = Location::from_latlng(41.921472, 8.742216, Level::default());
    let to = Location::from_latlng(41.921436, 8.740166, Level::default());

    let (data, path) = extract_and_route(
        "test/ajaccio-ferry.osm.pbf",
        from,
        to,
        SearchProfile::Foot,
    )
    .expect("Ferry routing must succeed");

    // C++ expected: 2 segments, total cost=231, total distance=278
    assert_eq!(path.segments.len(), 2, "ferry: expected 2 segments");
    assert_route_matches_cpp(&data, &path, EXPECTED_FERRY, "ferry");
}

#[test]
#[ignore] // Requires test data file: london-corridor.osm.pbf
fn test_corridor() {
    let from = Location::from_latlng(51.54663831994142, -0.05622849779558692, Level::default());
    let to = Location::from_latlng(51.547004658329, -0.05694437499428773, Level::default());

    let (data, path) = extract_and_route(
        "test/london-corridor.osm.pbf",
        from,
        to,
        SearchProfile::Foot,
    )
    .expect("Corridor routing must succeed");

    // C++ expected: 11 segments, total cost=78, total distance=94
    assert_eq!(path.segments.len(), 11, "corridor: expected 11 segments");
    assert_route_matches_cpp(&data, &path, EXPECTED_CORRIDOR, "corridor");
}

#[test]
#[ignore] // Requires test data file: station-border.osm.pbf
fn test_stop_area() {
    let from = Location::from_latlng(48.725296645530705, 2.2612587304760723, Level::default());
    let to = Location::from_latlng(48.725480463902784, 2.2588322597458728, Level::default());

    let (data, path) = extract_and_route(
        "test/station-border.osm.pbf",
        from,
        to,
        SearchProfile::Foot,
    )
    .expect("Stop area routing must succeed");

    // C++ expected: 25 segments with multi-level routing (levels 0, 1, 0.5)
    assert_eq!(path.segments.len(), 25, "stop_area: expected 25 segments");
    assert_route_matches_cpp(&data, &path, EXPECTED_STOP_AREA, "stop_area");

    // Verify multi-level routing: segments traverse levels 0 → 1 → 0.5 → 0
    let levels: Vec<f32> = path.segments.iter().map(|s| s.from_level.to_float()).collect();
    assert!(
        levels.contains(&1.0),
        "stop_area: route must traverse level 1"
    );
    assert!(
        levels.contains(&0.5),
        "stop_area: route must traverse level 0.5"
    );
}
