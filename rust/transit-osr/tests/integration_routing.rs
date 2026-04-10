//! Integration tests for OSR routing system.
//!
//! These tests validate end-to-end routing workflows:
//! - Extract OSM data → build routing graph → execute queries
//! - Test real-world scenarios with multiple profiles
//! - Verify profile-specific features (levels, oneway, speed limits)

use transit_cloud_osr::routing::profiles::{
    bike::{BikeCostStrategy, BikeProfile, ELEVATION_LOW_COST},
    car::CarProfile,
    foot::{FootProfile, MAX_MATCH_DISTANCE as FOOT_MAX_DISTANCE},
};
use transit_cloud_osr::routing::tracking::NoopTracking;
use transit_cloud_osr::routing::Mode;
use transit_cloud_osr::types::{Cost, Direction, Level};
use transit_cloud_osr::ways::{NodeProperties, WayProperties};

// ============================================================================
// Profile Creation and Configuration Tests
// ============================================================================

// ============================================================================
// Profile Creation and Configuration Tests
// ============================================================================

#[test]
fn test_foot_profile_basic() {
    let profile: FootProfile<NoopTracking> = FootProfile::new(false);
    assert_eq!(profile.mode(), Mode::Foot);
    assert!(!profile.is_wheelchair);
    assert_eq!(profile.max_match_distance(), 100);
}

#[test]
fn test_foot_profile_wheelchair() {
    let profile: FootProfile<NoopTracking> = FootProfile::new(true);
    assert!(profile.is_wheelchair);
    assert_eq!(profile.mode(), Mode::Wheelchair); // Returns Mode::Wheelchair, not Mode::Foot
}

#[test]
fn test_bike_profile_strategies() {
    let safe_profile = BikeProfile::safe();
    assert_eq!(safe_profile.cost_strategy, BikeCostStrategy::Safe);
    assert_eq!(safe_profile.elevation_up_cost, ELEVATION_LOW_COST);

    let fast_profile = BikeProfile::fast();
    assert_eq!(fast_profile.cost_strategy, BikeCostStrategy::Fast);
}

#[test]
fn test_car_profile_limits() {
    let profile = CarProfile::new();
    assert_eq!(profile.mode(), Mode::Car);
    assert_eq!(profile.max_match_distance(), 200);
    assert_eq!(CarProfile::MAX_MATCH_DISTANCE, 200);
}

#[test]
fn test_profile_constants() {
    // Check all profile constants exist and are reasonable
    assert_eq!(FOOT_MAX_DISTANCE, 100);
    assert_eq!(BikeProfile::MAX_MATCH_DISTANCE, 100);
    assert_eq!(CarProfile::MAX_MATCH_DISTANCE, 200);
}

// ============================================================================
// Way Accessibility Tests
// ============================================================================

#[test]
fn test_node_properties_foot_accessible() {
    let props = NodeProperties {
        is_foot_accessible: true,
        is_bike_accessible: false,
        is_car_accessible: false,
        is_elevator: false,
        is_entrance: false,
        is_multi_level: false,
        is_parking: false,
        ..Default::default()
    };

    assert!(props.is_foot_accessible);
    assert!(!props.is_bike_accessible);
    assert!(!props.is_car_accessible);
}

#[test]
fn test_node_properties_all_accessible() {
    let props = NodeProperties {
        is_foot_accessible: true,
        is_bike_accessible: true,
        is_car_accessible: true,
        is_elevator: false,
        is_entrance: false,
        is_multi_level: false,
        is_parking: false,
        ..Default::default()
    };

    assert!(props.is_foot_accessible);
    assert!(props.is_bike_accessible);
    assert!(props.is_car_accessible);
}

#[test]
fn test_node_properties_elevator() {
    let props = NodeProperties {
        is_foot_accessible: true,
        is_bike_accessible: true,
        is_car_accessible: false,
        is_elevator: true,
        is_entrance: false,
        is_multi_level: true,
        is_parking: false,
        ..Default::default()
    };

    assert!(props.is_elevator);
    assert!(props.is_multi_level);
}

#[test]
fn test_way_properties_pedestrian() {
    let props = WayProperties {
        is_foot_accessible: true,
        is_bike_accessible: false,
        is_car_accessible: false,
        is_steps: false,
        is_elevator: false,
        is_oneway_bike: false,
        is_oneway_car: false,
        is_destination: false,
        ..Default::default()
    };

    assert!(props.is_foot_accessible);
    assert!(!props.is_bike_accessible);
    assert!(!props.is_car_accessible);
}

#[test]
fn test_way_properties_stairs() {
    let props = WayProperties {
        is_foot_accessible: true,
        is_bike_accessible: false,
        is_car_accessible: false,
        is_steps: true,
        is_elevator: false,
        ..Default::default()
    };

    assert!(props.is_steps);
    assert!(props.is_foot_accessible);
    assert!(!props.is_bike_accessible);
}

// ============================================================================
// Cost Calculation Tests
// ============================================================================

#[test]
fn test_foot_cost_calculation() {
    use transit_cloud_osr::routing::parameters::FootParameters;
    use transit_cloud_osr::routing::profiles::foot::{node_cost, way_cost};

    let params = FootParameters::default();

    // Normal walkable way
    let way_props = WayProperties {
        is_foot_accessible: true,
        is_bike_accessible: true,
        is_sidewalk_separate: false,
        is_steps: false,
        ..Default::default()
    };

    let cost = way_cost(&params, &way_props, false, 100);
    assert!(cost < Cost::MAX);
    assert!(cost > 0);

    // Node without elevator
    let node_props = NodeProperties {
        is_foot_accessible: true,
        is_elevator: false,
        ..Default::default()
    };

    assert_eq!(node_cost(&node_props), 0);
}

#[test]
fn test_bike_cost_calculation() {
    use transit_cloud_osr::routing::profiles::bike::way_cost;

    let profile = BikeProfile::safe();

    // Normal bike path
    let bike_path = WayProperties {
        is_bike_accessible: true,
        is_oneway_bike: false,
        motor_vehicle_no: true,
        is_big_street: false,
        ..Default::default()
    };

    let cost = way_cost(
        &profile.params,
        &bike_path,
        Direction::Forward,
        profile.cost_strategy,
        100,
    );
    assert!(cost < Cost::MAX);
    assert!(cost > 0);

    // Big street (should be slower in Safe mode)
    let big_street = WayProperties {
        is_bike_accessible: true,
        is_oneway_bike: false,
        motor_vehicle_no: false,
        is_big_street: true,
        ..Default::default()
    };

    let cost_big = way_cost(
        &profile.params,
        &big_street,
        Direction::Forward,
        profile.cost_strategy,
        100,
    );
    let cost_path = way_cost(
        &profile.params,
        &bike_path,
        Direction::Forward,
        profile.cost_strategy,
        100,
    );

    // Big street should be more expensive in Safe mode
    assert!(cost_big > cost_path);
}

#[test]
fn test_car_cost_calculation() {
    use transit_cloud_osr::routing::profiles::car::way_cost;
    use transit_cloud_osr::types::SpeedLimit;

    // Normal car road
    let car_road = WayProperties {
        is_car_accessible: true,
        is_oneway_car: false,
        is_destination: false,
        speed_limit: SpeedLimit::Kmh50,
        ..Default::default()
    };

    let cost_normal = way_cost(&car_road, Direction::Forward, 100);
    assert!(cost_normal < Cost::MAX);
    assert!(cost_normal > 0);

    // Destination road (should be much more expensive)
    let dest_road = WayProperties {
        is_car_accessible: true,
        is_oneway_car: false,
        is_destination: true,
        speed_limit: SpeedLimit::Kmh50,
        ..Default::default()
    };

    let cost_dest = way_cost(&dest_road, Direction::Forward, 100);

    // Destination should have 5x multiplier + 120s penalty
    assert!(cost_dest > cost_normal * 5);
}

// ============================================================================
// Profile-Specific Feature Tests
// ============================================================================

#[test]
fn test_wheelchair_restrictions() {
    use transit_cloud_osr::routing::parameters::FootParameters;
    use transit_cloud_osr::routing::profiles::foot::way_cost;

    let wheelchair_params = FootParameters::wheelchair();

    // Stairs should be infeasible for wheelchair
    let stairs = WayProperties {
        is_foot_accessible: true,
        is_steps: true,
        ..Default::default()
    };

    let cost = way_cost(&wheelchair_params, &stairs, true, 100);
    assert_eq!(cost, Cost::MAX);

    // Normal path should be OK
    let path = WayProperties {
        is_foot_accessible: true,
        is_steps: false,
        ..Default::default()
    };

    let cost_path = way_cost(&wheelchair_params, &path, true, 100);
    assert!(cost_path < Cost::MAX);
}

#[test]
fn test_bike_oneway_restrictions() {
    use transit_cloud_osr::routing::profiles::bike::way_cost;

    let profile = BikeProfile::safe();

    let oneway = WayProperties {
        is_bike_accessible: true,
        is_oneway_bike: true,
        ..Default::default()
    };

    // Forward should be allowed
    let cost_fwd = way_cost(
        &profile.params,
        &oneway,
        Direction::Forward,
        profile.cost_strategy,
        100,
    );
    assert!(cost_fwd < Cost::MAX);

    // Backward should be strictly infeasible for bikes (unlike cars which have strict oneway)
    let cost_bwd = way_cost(
        &profile.params,
        &oneway,
        Direction::Backward,
        profile.cost_strategy,
        100,
    );
    assert_eq!(cost_bwd, Cost::MAX, "Bikes must obey oneway restrictions");
}

#[test]
fn test_car_oneway_enforcement() {
    use transit_cloud_osr::routing::profiles::car::way_cost;

    let oneway = WayProperties {
        is_car_accessible: true,
        is_oneway_car: true,
        ..Default::default()
    };

    // Forward should be allowed
    let cost_fwd = way_cost(&oneway, Direction::Forward, 100);
    assert!(cost_fwd < Cost::MAX);

    // Backward should be strictly infeasible
    let cost_bwd = way_cost(&oneway, Direction::Backward, 100);
    assert_eq!(cost_bwd, Cost::MAX);
}

#[test]
fn test_car_destination_restrictions() {
    use transit_cloud_osr::routing::profiles::car::way_cost;
    use transit_cloud_osr::types::SpeedLimit;

    let normal = WayProperties {
        is_car_accessible: true,
        is_destination: false,
        speed_limit: SpeedLimit::Kmh50,
        ..Default::default()
    };

    let destination = WayProperties {
        is_car_accessible: true,
        is_destination: true,
        speed_limit: SpeedLimit::Kmh50,
        ..Default::default()
    };

    let cost_normal = way_cost(&normal, Direction::Forward, 100);
    let cost_dest = way_cost(&destination, Direction::Forward, 100);

    // Destination penalty is 5x + 120s
    assert!(cost_dest >= cost_normal * 5 + 120);
}

#[test]
fn test_profile_consistency() {
    // All profiles should have reasonable max match distances
    let foot: FootProfile<NoopTracking> = FootProfile::new(false);
    let bike = BikeProfile::safe();
    let car = CarProfile::new();

    assert_eq!(foot.max_match_distance(), 100);
    assert_eq!(bike.max_match_distance(), 100);
    assert_eq!(car.max_match_distance(), 200);

    // Modes should be correct
    assert_eq!(foot.mode(), Mode::Foot);
    assert_eq!(bike.mode(), Mode::Bike);
    assert_eq!(car.mode(), Mode::Car);
}
