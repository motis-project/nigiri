//! Test to validate multi-way node filtering (C++ architecture compatibility)
//!
//! This test verifies that only nodes shared by multiple ways become routing nodes,
//! matching the C++ OSR behavior.

use transit_cloud_osr::{
    types::{OsmNodeIdx, OsmWayIdx},
    ways::{NodeProperties, WayProperties},
    Point, Ways,
};

#[test]
fn test_multi_way_node_filtering() {
    /*
     * Create street network:
     *
     *   N1 ---[W10]--- N2 ---[W20]--- N3
     *                   |
     *                 [W30]
     *                   |
     *                  N4
     *
     * N2 is in 3 ways (W10, W20, W30) → should become routing node
     * N1, N3, N4 are endpoints → NOT multi-way → should NOT become routing nodes
     *
     * Expected routing graph:
     *   - 1 routing node (N2 only)
     *   - 3 ways, each with empty or single-node routing sequences
     */

    let mut ways = Ways::new();

    // Create 4 OSM nodes
    let n1 = OsmNodeIdx(1);
    let n2 = OsmNodeIdx(2); // Multi-way intersection
    let n3 = OsmNodeIdx(3);
    let n4 = OsmNodeIdx(4);

    let p1 = Point::from_latlng(49.8727, 8.6515);
    let p2 = Point::from_latlng(49.8730, 8.6515);
    let p3 = Point::from_latlng(49.8733, 8.6515);
    let p4 = Point::from_latlng(49.8730, 8.6520);

    let mut props = WayProperties::default();
    props.is_foot_accessible = true;

    // W10: N1 → N2 (2 OSM nodes, but N1 is endpoint)
    ways.add_way_osm_only(OsmWayIdx(10), vec![n1, n2], props, vec![p1, p2]);
    ways.increment_node_way_counter(n1);
    ways.increment_node_way_counter(n2);

    // W20: N2 → N3 (2 OSM nodes, but N3 is endpoint)
    ways.add_way_osm_only(OsmWayIdx(20), vec![n2, n3], props, vec![p2, p3]);
    ways.increment_node_way_counter(n2);
    ways.increment_node_way_counter(n3);

    // W30: N2 → N4 (2 OSM nodes, but N4 is endpoint)
    ways.add_way_osm_only(OsmWayIdx(30), vec![n2, n4], props, vec![p2, p4]);
    ways.increment_node_way_counter(n2);
    ways.increment_node_way_counter(n4);

    // Before connect_ways()
    assert_eq!(
        ways.n_nodes(),
        0,
        "Should have 0 routing nodes before connect_ways"
    );

    // Call connect_ways() to build routing graph
    ways.connect_ways();

    // After connect_ways()
    assert_eq!(ways.n_nodes(), 1, "Should have exactly 1 routing node (N2)");

    // Verify the routing node is N2
    assert_eq!(
        ways.get_node_osm_id(transit_cloud_osr::types::NodeIdx(0)),
        Some(n2),
        "The single routing node should be N2"
    );

    // Verify node counters
    assert!(ways.is_multi_node(n2), "N2 should be multi");
    assert!(!ways.is_multi_node(n1), "N1 should not be multi");
    assert!(!ways.is_multi_node(n3), "N3 should not be multi");
    assert!(!ways.is_multi_node(n4), "N4 should not be multi");

    println!("✓ Multi-way node filtering works correctly");
    println!("  Total OSM nodes: 4");
    println!("  Multi-way nodes: 1 (N2)");
    println!("  Routing nodes created: {}", ways.n_nodes());
}

#[test]
fn test_no_multi_way_nodes() {
    /*
     * Create disconnected ways:
     *
     *   N1 ---[W10]--- N2
     *
     *   N3 ---[W20]--- N4
     *
     * No node is in multiple ways → should create 0 routing nodes
     */

    let mut ways = Ways::new();

    let n1 = OsmNodeIdx(1);
    let n2 = OsmNodeIdx(2);
    let n3 = OsmNodeIdx(3);
    let n4 = OsmNodeIdx(4);

    let p1 = Point::from_latlng(49.8727, 8.6515);
    let p2 = Point::from_latlng(49.8730, 8.6515);
    let p3 = Point::from_latlng(49.8733, 8.6515);
    let p4 = Point::from_latlng(49.8736, 8.6515);

    let props = WayProperties::default();

    ways.add_way_osm_only(OsmWayIdx(10), vec![n1, n2], props, vec![p1, p2]);
    ways.increment_node_way_counter(n1);
    ways.increment_node_way_counter(n2);

    ways.add_way_osm_only(OsmWayIdx(20), vec![n3, n4], props, vec![p3, p4]);
    ways.increment_node_way_counter(n3);
    ways.increment_node_way_counter(n4);

    ways.connect_ways();

    assert_eq!(
        ways.n_nodes(),
        0,
        "Should have 0 routing nodes (no multi-way nodes)"
    );

    println!("✓ No multi-way nodes detected correctly");
}

#[test]
fn test_all_nodes_multi() {
    /*
     * Create triangle where all nodes are multi:
     *
     *       N1
     *      /  \
     *   [W10] [W30]
     *    /      \
     *  N2--[W20]--N3
     *
     * All 3 nodes are in 2 ways → all should become routing nodes
     */

    let mut ways = Ways::new();

    let n1 = OsmNodeIdx(1);
    let n2 = OsmNodeIdx(2);
    let n3 = OsmNodeIdx(3);

    let p1 = Point::from_latlng(49.8730, 8.6520);
    let p2 = Point::from_latlng(49.8727, 8.6515);
    let p3 = Point::from_latlng(49.8733, 8.6515);

    let props = WayProperties::default();

    // W10: N1 → N2
    ways.add_way_osm_only(OsmWayIdx(10), vec![n1, n2], props, vec![p1, p2]);
    ways.increment_node_way_counter(n1);
    ways.increment_node_way_counter(n2);

    // W20: N2 → N3
    ways.add_way_osm_only(OsmWayIdx(20), vec![n2, n3], props, vec![p2, p3]);
    ways.increment_node_way_counter(n2);
    ways.increment_node_way_counter(n3);

    // W30: N3 → N1
    ways.add_way_osm_only(OsmWayIdx(30), vec![n3, n1], props, vec![p3, p1]);
    ways.increment_node_way_counter(n3);
    ways.increment_node_way_counter(n1);

    ways.connect_ways();

    assert_eq!(ways.n_nodes(), 3, "Should have 3 routing nodes (all multi)");

    // All should be multi
    assert!(ways.is_multi_node(n1), "N1 should be multi");
    assert!(ways.is_multi_node(n2), "N2 should be multi");
    assert!(ways.is_multi_node(n3), "N3 should be multi");

    println!("✓ All multi-way nodes detected correctly");
}
