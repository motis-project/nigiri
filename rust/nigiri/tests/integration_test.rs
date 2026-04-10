use nigiri::Timetable;
use std::path::PathBuf;

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gtfs")
}

/// Aug 9 2023 00:00:00 UTC
const FROM_TS: i64 = 1691539200;
/// Aug 12 2023 00:00:00 UTC  
const TO_TS: i64 = 1691798400;

#[test]
fn load_and_query_counts() {
    let path = fixtures_path();
    let tt =
        Timetable::load(path.to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load timetable");

    assert_eq!(tt.route_count(), 1, "expected 1 route");
    assert_eq!(tt.transport_count(), 1, "expected 1 transport");
    // 28 stops + special stations (START, END, etc.) + parent grouping
    assert!(tt.location_count() > 27, "expected at least 27 locations");
    assert!(tt.day_count() > 0, "expected at least 1 day");
}

#[test]
fn get_transport_metadata() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let transport = tt.get_transport(0).expect("transport 0 exists");
    assert_eq!(transport.route_idx(), 0);

    // 27 stops → 26 segments → 52 event mams (dep/arr pairs)
    assert_eq!(transport.event_mams().len(), 52);

    // Verify the name is non-empty
    let name = transport.name();
    assert!(!name.is_empty(), "transport should have a name");
}

#[test]
fn get_transport_name_fast() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let name = tt.transport_name(0);
    assert!(!name.is_empty());

    // Should match the one from get_transport
    let transport = tt.get_transport(0).unwrap();
    assert_eq!(name, transport.name());
}

#[test]
fn get_route_metadata() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let route = tt.get_route(0).expect("route 0 exists");
    let stops = route.stops();
    assert_eq!(stops.len(), 27, "route has 27 stops");

    // First stop should allow boarding
    assert!(stops[0].out_allowed, "first stop allows boarding");
    // Last stop — verify we can read bitfield without panicking
    let _last = stops[stops.len() - 1];
}

#[test]
fn get_location_by_index() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    // Find "Block Line Station" (stop_id 2351)
    let loc_count = tt.location_count();
    let mut found = false;
    for idx in 0..loc_count {
        let loc = tt.get_location(idx).expect("location exists");
        if loc.id() == "2351" {
            assert_eq!(loc.name(), "Block Line Station");
            assert!((loc.lat() - 43.422095).abs() < 0.001);
            assert!((loc.lon() - (-80.462740)).abs() < 0.001);
            found = true;
            break;
        }
    }
    assert!(found, "should find stop 2351");
}

#[test]
fn find_location_by_id() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let idx = tt.find_location("2351");
    assert!(idx.is_some(), "should find stop 2351");

    let loc = tt.get_location(idx.unwrap()).unwrap();
    assert_eq!(loc.name(), "Block Line Station");

    // Non-existent ID
    assert!(tt.find_location("NONEXISTENT").is_none());
}

#[test]
fn external_interval() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let start = tt.external_interval_start();
    let end = tt.external_interval_end();
    assert!(start > 0);
    assert!(end > start);
    assert_eq!(start, FROM_TS);
    assert_eq!(end, TO_TS);
}

#[test]
fn transport_route_mapping() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let route_idx = tt.transport_route(0);
    assert_eq!(route_idx, 0);
}

#[test]
fn route_stop_count() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let count = tt.route_stop_count(0);
    assert_eq!(count, 27);
}

#[test]
fn transport_active_check() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    // At least one day in the range should have the transport active
    let days = tt.day_count();
    let mut any_active = false;
    for d in 0..days {
        if tt.is_transport_active(0, d) {
            any_active = true;
            break;
        }
    }
    assert!(any_active, "transport should be active on at least one day");
}

#[test]
fn footpaths() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    // Find a station with parent_station set (1916 or 1918 have parent 4073)
    // These should generate footpaths between siblings
    if let Some(idx) = tt.find_location("1918") {
        let loc = tt.get_location_with_footpaths(idx, false).unwrap();
        // Should have footpaths to sibling stations under parent 4073
        let fps = loc.footpaths();
        assert!(
            fps.len() > 0,
            "station 1918 should have footpaths to siblings"
        );
    }
}

#[test]
fn source_count() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");
    assert_eq!(tt.source_count(), 1, "single-source fixture");
}

#[test]
fn location_detail() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    // Use find_location to get a known stop, then get its detail
    let idx = tt.find_location("2351").expect("stop 2351 should exist");
    let detail = tt.get_location_detail(idx).expect("detail should work");
    assert!(!detail.name.is_empty(), "location should have a name");
    assert!(
        detail.lat != 0.0 || detail.lon != 0.0,
        "should have coordinates"
    );
    assert_eq!(detail.src_idx, 0, "single source fixture");

    // Out of bounds should fail
    assert!(tt.get_location_detail(u32::MAX).is_err());
}

#[test]
fn route_detail() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let detail = tt.get_route_detail(0).expect("route 0 should exist");
    // clasz should be a valid value
    assert!(detail.clasz < 20, "clasz should be reasonable");

    // Out of bounds should fail
    assert!(tt.get_route_detail(u32::MAX).is_err());
}

#[test]
fn transport_trip_id() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let trip_id = tt.transport_trip_id(0);
    assert!(trip_id.is_some(), "transport 0 should have a trip_id");
    assert!(!trip_id.unwrap().is_empty());
}

#[test]
fn transport_source_and_first_dep() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let src = tt.transport_source(0);
    assert_eq!(src, Some(0), "single source fixture");

    let mam = tt.transport_first_dep_mam(0);
    assert!(mam.is_some(), "should have first departure");
    assert!(mam.unwrap() >= 0, "departure should be non-negative");
}

#[test]
fn route_gtfs_id() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let id = tt.route_gtfs_id(0);
    assert!(id.is_some(), "route 0 should have a gtfs_id");
    assert!(!id.unwrap().is_empty());
}

#[test]
fn day_to_date_str() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let date = tt.day_to_date_str(0);
    assert!(date.is_some());
    let date_str = date.unwrap();
    assert_eq!(date_str.len(), 8, "YYYYMMDD format");
    assert!(date_str.chars().all(|c| c.is_ascii_digit()));
}

#[test]
fn location_routes_and_stop_times() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    // Find stop 2351 and check it has routes
    let idx = tt.find_location("2351").expect("stop 2351 exists");
    let routes = tt.location_routes(idx);
    assert!(
        !routes.is_empty(),
        "stop should be served by at least one route"
    );

    // Check stop_idx_in_route
    let route = routes[0];
    let stop_idx = tt.stop_idx_in_route(route, idx);
    assert!(stop_idx.is_some(), "stop should be in the route");

    // Check transport range
    let (from, to) = tt
        .route_transport_range(route)
        .expect("route has transports");
    assert!(to > from, "route should have at least one transport");

    // Check event MAM
    let mam = tt.event_mam(from, stop_idx.unwrap(), false);
    assert!(mam.is_some(), "should have departure MAM");
}

#[test]
fn transport_stop_times_full() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let stop_times = tt
        .transport_stop_times(0)
        .expect("transport 0 should have stops");
    assert_eq!(stop_times.len(), 27, "27 stops in fixture route");

    // First stop should have departure but no arrival
    assert!(stop_times[0].departure_mam.is_some());
    assert!(stop_times[0].arrival_mam.is_none());

    // Last stop should have arrival but no departure
    let last = &stop_times[stop_times.len() - 1];
    assert!(last.departure_mam.is_none());
    assert!(last.arrival_mam.is_some());
}

#[test]
fn transport_display_name() {
    let tt =
        Timetable::load(fixtures_path().to_str().unwrap(), FROM_TS, TO_TS).expect("failed to load");

    let name = tt.transport_display_name(0);
    // May or may not be set in the fixture, but should not crash
    let _ = name;
}
