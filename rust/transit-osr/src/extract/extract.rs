//! OSM data extraction.
//!
//! Complete translation of osr/src/extract.cc
//!
//! This module extracts routing-relevant data from OpenStreetMap PBF files.
//! It processes nodes, ways, and relations to build a routing graph.
//!
//! # Process Overview
//! 1. **First pass (nodes & relations)**: Collect node coordinates, mark inaccessible nodes, extract relation ways
//! 2. **Second pass (ways)**: Extract way geometry and properties, add to Ways structure
//! 3. **Third pass (nodes & relations)**: Extract node properties and turn restrictions
//! 4. **Post-processing**: Connect ways, build components, add elevation, build R-tree
//!
//! # Architecture
//!
//! Unlike the C++ version which uses osmium's NodeLocationsForWays handler to automatically
//! provide node coordinates during way processing, the Rust osmpbf crate requires us to
//! explicitly store node coordinates in the first pass and retrieve them in the second pass.
//!
//! This implementation uses a HashMap<OsmNodeId, Point> to store coordinates for nodes that
//! are referenced by ways, built during the first pass and consumed during the second pass.
//!
//! # C++ Equivalent
//! ```cpp
//! void extract(bool const with_platforms,
//!              fs::path const& in,
//!              fs::path const& out,
//!              fs::path const& elevation_dir);
//! ```

use ahash::AHashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use osmpbf::{Element, ElementReader};

use crate::extract::tags::Tags;
use crate::types::{LevelBits, NodeIdx, OsmNodeIdx, OsmWayIdx, SpeedLimit, WayIdx};
use crate::ways::{NodeProperties, WayProperties, Ways};
use crate::{ElevationStorage, Level, Platforms, Point};

/// OSM node ID (64-bit signed integer) - type alias for clarity
pub type OsmNodeId = i64;

/// OSM way ID (64-bit signed integer) - type alias for clarity
pub type OsmWayId = i64;

/// Way information from relation processing
#[derive(Debug, Clone)]
struct RelationWay {
    properties: WayProperties,
}

/// Turn restriction extracted from OSM relations
#[derive(Debug, Clone)]
pub struct TurnRestriction {
    pub from_way: WayIdx,
    pub to_way: WayIdx,
    pub via_node: NodeIdx,
    pub is_only: bool, // false = "no_*" restriction, true = "only_*" restriction
}

/// Extract routing data from an OSM PBF file.
///
/// # Arguments
/// * `with_platforms` - Whether to extract platform/station data
/// * `input_path` - Path to OSM PBF file
/// * `output_path` - Path for output data
/// * `elevation_dir` - Optional directory for elevation data
///
/// # Three-Pass Architecture
///
/// **Pass 1 (Nodes & Relations)**:
/// - Collect node coordinates for all nodes that will be referenced by ways
/// - Mark inaccessible nodes (barriers, etc.)
/// - Extract relation way properties
///
/// **Pass 2 (Ways)**:
/// - Filter routable ways based on accessibility
/// - Extract way geometry using stored node coordinates
/// - Add ways to routing graph with properties
///
/// **Pass 3 (Nodes & Relations)**:
/// - Set node properties (elevators, entrances, etc.)
/// - Extract turn restrictions from relations
///
/// **Post-Processing**:
/// - Connect ways at intersections
/// - Build routing graph components
/// - Add elevation data (optional)
/// - Build R-tree spatial index
///
/// # C++ Equivalent
/// ```cpp
/// void extract(bool const with_platforms, fs::path const& in,
///              fs::path const& out, fs::path const& elevation_dir);
/// ```
pub fn extract(
    with_platforms: bool,
    input_path: &Path,
    output_path: &Path,
    elevation_dir: Option<&Path>,
) -> Result<(), String> {
    if !input_path.exists() {
        return Err(format!(
            "Input file does not exist: {}",
            input_path.display()
        ));
    }

    // Create output directory
    std::fs::create_dir_all(output_path)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    // Initialize data structures
    let mut ways = Ways::new();
    let platforms = Platforms::new();
    let mut elevations = ElevationStorage::new();

    // Node coordinate storage for pass 2
    let node_coordinates: Arc<RwLock<AHashMap<OsmNodeId, Point>>> =
        Arc::new(RwLock::new(AHashMap::new()));

    // Relation way properties
    let relation_ways: Arc<RwLock<AHashMap<OsmWayId, RelationWay>>> =
        Arc::new(RwLock::new(AHashMap::new()));

    // Turn restrictions (collected in pass 3)
    let restrictions: Arc<Mutex<Vec<TurnRestriction>>> = Arc::new(Mutex::new(Vec::new()));

    println!("Pass 1: Collecting node coordinates and relation ways...");
    collect_node_coordinates_and_relations(
        input_path,
        with_platforms,
        Arc::clone(&node_coordinates),
        Arc::clone(&relation_ways),
    )?;
    let node_count = node_coordinates.read().unwrap().len();
    let relation_way_count = relation_ways.read().unwrap().len();
    println!(
        "  Stored {} node coordinates, {} relation ways",
        node_count, relation_way_count
    );

    println!("Pass 2: Extracting ways...");
    let elevator_nodes = extract_ways_with_geometry(
        input_path,
        &mut ways,
        Arc::clone(&node_coordinates),
        Arc::clone(&relation_ways),
    )?;
    println!(
        "  Extracted {} ways, {} elevator nodes from way loops",
        ways.n_ways(),
        elevator_nodes.len()
    );

    println!("Pass 3: Processing node properties and restrictions...");
    extract_node_properties_and_restrictions(
        input_path,
        &mut ways,
        Arc::clone(&restrictions),
        &elevator_nodes,
    )?;
    let restriction_count = restrictions.lock().unwrap().len();
    println!("  Found {} turn restrictions", restriction_count);

    // Sort multi_level_elevators for binary search (matches C++: utl::sort)
    ways.sort_multi_level_elevators();
    println!(
        "  {} multi-level elevators",
        ways.multi_level_elevators().len()
    );

    // Add restrictions to Ways
    for restriction in restrictions.lock().unwrap().iter() {
        ways.add_restriction(
            restriction.from_way,
            restriction.to_way,
            restriction.via_node,
            restriction.is_only,
        );
    }

    println!("Post-processing: Building routing graph...");
    ways.connect_ways();
    ways.build_components();

    // Optionally add elevation data
    if let Some(elev_dir) = elevation_dir {
        if elev_dir.exists() {
            println!("Adding elevation data from {}...", elev_dir.display());

            match crate::preprocessing::elevation::provider::Provider::new(elev_dir) {
                Ok(provider) => {
                    println!("  Found {} elevation driver(s)", provider.driver_count());

                    match elevations.populate_from_ways(&ways, &provider) {
                        Ok(count) => {
                            println!("  Populated elevations for {} ways", count);
                        }
                        Err(e) => {
                            println!("  Warning: Failed to populate elevations: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("  Warning: Failed to load elevation provider: {}", e);
                }
            }
        } else {
            println!(
                "Warning: Elevation directory does not exist: {}",
                elev_dir.display()
            );
        }
    }

    // Compute big street neighbors for routing optimization
    println!("Computing big street neighbors...");
    ways.compute_big_street_neighbors();

    println!("Finalize ways...");
    ways.finalize();

    println!("Building lookups...");
    let lookup = crate::lookup::Lookup::new(&ways);

    // Use OsrData coordinator to handle export business logic
    println!("Exporting dataset...");

    let mut osr_data = crate::data::OsrData {
        ways,
        platforms,
        elevations,
        lookup,
    };

    osr_data.export(output_path)?;

    println!("\nExtract complete!");
    print_statistics(
        &osr_data.ways,
        &osr_data.platforms,
        &osr_data.elevations,
        restriction_count,
    );

    Ok(())
}

/*
/// Statistics from analyzing OSM file
struct OsmStats {
    nodes: usize,
    ways: usize,
    relations: usize,
}
*/

/// First pass: Collect node coordinates and relation way properties.
///
/// This pass:
/// - Stores coordinates for all nodes (we'll filter later to only keep referenced nodes)
/// - Marks inaccessible nodes (barriers, etc.)
/// - Extracts way properties from relations (e.g., multipolygon areas)
///
/// # C++ Equivalent
/// ```cpp
/// // Collect node coordinates using hybrid_node_idx_builder
/// auto inaccessible_handler = mark_inaccessible_handler{pl != nullptr, w};
/// auto rel_ways_h = rel_ways_handler{pl.get(), rel_ways};
/// osm::apply(buffer, node_idx_builder, inaccessible_handler, rel_ways_h);
/// ```
fn collect_node_coordinates_and_relations(
    input_path: &Path,
    with_platforms: bool,
    node_coordinates: Arc<RwLock<AHashMap<OsmNodeId, Point>>>,
    relation_ways: Arc<RwLock<AHashMap<OsmWayId, RelationWay>>>,
) -> Result<(), String> {
    let reader = ElementReader::from_path(input_path)
        .map_err(|e| format!("Failed to open OSM file: {}", e))?;

    let nodes = Arc::clone(&node_coordinates);
    let rel_ways = Arc::clone(&relation_ways);

    reader
        .for_each(|element| {
            match element {
                Element::Node(node) => {
                    // Store all node coordinates
                    // We could optimize by only storing nodes that are referenced by ways,
                    // but that would require a two-pass approach or pre-scanning
                    let point = Point::from_latlng(node.lat(), node.lon());
                    nodes.write().unwrap().insert(node.id(), point);

                    // Mark inaccessible nodes (barriers, etc.) so they become routing nodes
                    let osm_tags: Vec<(&str, &str)> = node.tags().map(|(k, v)| (k, v)).collect();
                    let tags = Tags::from_osm_tags(osm_tags);

                    let accessible = tags.is_accessible_foot()
                        && tags.is_accessible_bike()
                        && tags.is_accessible_car();

                    if !accessible || tags.is_elevator || (with_platforms && tags.is_platform()) {
                        // These nodes should become routing nodes even if not at intersections
                        // TODO: Track these nodes for special handling
                    }
                }
                Element::DenseNode(node) => {
                    // DenseNodes don't have tags, so just store coordinates
                    let point = Point::from_latlng(node.lat(), node.lon());
                    nodes.write().unwrap().insert(node.id(), point);
                }
                Element::Relation(relation) => {
                    // Extract way properties from relations (e.g., multipolygon areas)
                    let osm_tags: Vec<(&str, &str)> =
                        relation.tags().map(|(k, v)| (k, v)).collect();
                    let tags = Tags::from_osm_tags(osm_tags);
                    let props = get_way_properties(&tags);

                    if props.is_accessible() {
                        // Store properties for all ways in this relation
                        for member in relation.members() {
                            if member.member_type == osmpbf::RelMemberType::Way {
                                rel_ways
                                    .write()
                                    .unwrap()
                                    .insert(member.member_id, RelationWay { properties: props });
                            }
                        }
                    }
                }
                _ => {}
            }
        })
        .map_err(|e| format!("Error reading OSM file: {}", e))?;

    Ok(())
}

/// Second pass: Extract ways with geometry.
///
/// This pass:
/// - Collects elevator_nodes for way-based elevators (loops)
/// - Filters ways based on accessibility (highway tags, etc.)
/// - Extracts way geometry by looking up node coordinates
/// - Adds ways to the Ways structure with properties
/// - Creates NodeIdx for each unique OSM node
///
/// Returns elevator_nodes map: OSM node ID → level_bits for elevator ways.
///
/// # C++ Equivalent
/// ```cpp
/// auto h = way_handler{w, pl.get(), rel_ways, elevator_nodes};
/// // ... parallel pipeline with update_locations and apply
/// ```
fn extract_ways_with_geometry(
    input_path: &Path,
    ways: &mut Ways,
    node_coordinates: Arc<RwLock<AHashMap<OsmNodeId, Point>>>,
    relation_ways: Arc<RwLock<AHashMap<OsmWayId, RelationWay>>>,
) -> Result<AHashMap<OsmNodeIdx, LevelBits>, String> {
    let reader = ElementReader::from_path(input_path)
        .map_err(|e| format!("Failed to open OSM file: {}", e))?;

    let coords = node_coordinates.read().unwrap();
    let rel_ways = relation_ways.read().unwrap();

    // Collect elevator nodes from way-based elevators (loops)
    // C++: hash_map<osm_node_idx_t, level_bits_t> elevator_nodes;
    let mut elevator_nodes: AHashMap<OsmNodeIdx, LevelBits> = AHashMap::new();

    reader
        .for_each(|element| {
            if let Element::Way(way) = element {
                let osm_way_id = way.id();

                // Convert tags
                let osm_tags: Vec<(&str, &str)> = way.tags().map(|(k, v)| (k, v)).collect();
                let tags = Tags::from_osm_tags(osm_tags);

                // C++: Elevator way handling (before should_include_way check)
                // Way elevators must form loops (first node == last node).
                // Non-loop elevator ways are skipped entirely.
                // Loop elevator ways record their first node + level_bits.
                if tags.is_elevator {
                    let node_refs: Vec<i64> = way.refs().collect();
                    if node_refs.first() != node_refs.last() {
                        return; // way elevators have to be loops
                    }
                    if let Some(&first_id) = node_refs.first() {
                        let first_node = OsmNodeIdx(first_id as u64);
                        elevator_nodes.insert(first_node, tags.level_bits);
                    }
                }

                // Check if this way should be included
                if !should_include_way(&tags) {
                    return;
                }

                // Get way properties (from tags or from relation)
                let mut props = get_way_properties(&tags);

                // Override with relation properties if present
                if let Some(rel_way) = rel_ways.get(&osm_way_id) {
                    // Merge relation properties (relation takes precedence for some fields)
                    if rel_way.properties.is_platform {
                        props.is_platform = true;
                    }
                }

                // Skip if not accessible
                if !props.is_accessible() {
                    return;
                }

                // Extract geometry: convert node refs to coordinates
                let node_refs: Vec<i64> = way.refs().collect();
                let mut geometry: Vec<Point> = Vec::with_capacity(node_refs.len());
                let mut osm_node_ids: Vec<OsmNodeIdx> = Vec::with_capacity(node_refs.len());

                for node_id in node_refs {
                    let osm_node_idx = OsmNodeIdx(node_id as u64);

                    if let Some(&point) = coords.get(&node_id) {
                        // Increment counter for this OSM node (used to identify multi-way nodes)
                        ways.increment_node_way_counter(osm_node_idx);

                        geometry.push(point);
                        osm_node_ids.push(osm_node_idx);
                    } else {
                        // Node coordinate not found - this shouldn't happen if pass 1 worked
                        eprintln!(
                            "Warning: Node {} referenced by way {} not found",
                            node_id, osm_way_id
                        );
                    }
                }

                // Skip ways with insufficient geometry
                if geometry.len() < 2 {
                    return;
                }

                // Add way with OSM node sequence and geometry (routing nodes created later in connect_ways)
                let way_idx = ways.add_way_osm_only(
                    OsmWayIdx(osm_way_id as u64),
                    osm_node_ids,
                    props,
                    geometry,
                );

                // Set way name if present
                if !tags.name.is_empty() {
                    ways.set_way_name(way_idx, tags.name.clone());
                }
            }
        })
        .map_err(|e| format!("Error reading OSM file: {}", e))?;

    Ok(elevator_nodes)
}

/// Third pass: Extract node properties and turn restrictions.
///
/// This pass:
/// - Sets node properties (elevators, entrances, levels, etc.)
/// - Populates multi_level_elevators from node tags and way-based elevator_nodes
/// - Extracts turn restrictions from relations
///
/// # C++ Equivalent
/// ```cpp
/// auto h = node_handler{w, pl.get(), r, elevator_nodes};
/// osm::apply(b, h);
/// ```
fn extract_node_properties_and_restrictions(
    input_path: &Path,
    ways: &mut Ways,
    restrictions: Arc<Mutex<Vec<TurnRestriction>>>,
    elevator_nodes: &AHashMap<OsmNodeIdx, LevelBits>,
) -> Result<(), String> {
    let reader = ElementReader::from_path(input_path)
        .map_err(|e| format!("Failed to open OSM file: {}", e))?;

    reader
        .for_each(|element| {
            match element {
                Element::Node(node) => {
                    let osm_node_id = OsmNodeIdx(node.id() as u64);

                    // Check if this node exists in our routing graph
                    if let Some(node_idx) = ways.find_node_idx(osm_node_id) {
                        // Extract tags
                        let osm_tags: Vec<(&str, &str)> =
                            node.tags().map(|(k, v)| (k, v)).collect();
                        let tags = Tags::from_osm_tags(osm_tags);

                        // Get node properties (includes elevator detection)
                        let (props, level_bits) = get_node_properties(&tags);

                        // Set properties in routing graph
                        ways.set_node_properties(node_idx, props);

                        // C++: Multi-level elevator handling
                        // Path A: Node's own tags indicate elevator + multi-level
                        if props.is_elevator() && props.is_multi_level() {
                            ways.add_multi_level_elevator(node_idx, level_bits);
                        } else if let Some(&elevator_level_bits) =
                            elevator_nodes.get(&osm_node_id)
                        {
                            // Path B: Node is referenced by an elevator way (from pass 2)
                            // Update node properties with elevator info from the way
                            let (from, to, is_multi) =
                                get_levels_from_bits(true, elevator_level_bits);
                            let updated_props = NodeProperties {
                                is_elevator: true,
                                from_level: from.into(),
                                to_level: to.into(),
                                is_multi_level: is_multi,
                                ..props
                            };
                            ways.set_node_properties(node_idx, updated_props);
                            if is_multi {
                                ways.add_multi_level_elevator(node_idx, elevator_level_bits);
                            }
                        }
                    }
                }
                Element::Relation(relation) => {
                    // Extract turn restrictions
                    let osm_tags: Vec<(&str, &str)> =
                        relation.tags().map(|(k, v)| (k, v)).collect();
                    let tags_map: AHashMap<String, String> = osm_tags
                        .iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect();

                    if let Some(rel_type) = tags_map.get("type") {
                        if rel_type == "restriction" {
                            if let Some(restriction_type) = tags_map.get("restriction") {
                                extract_turn_restriction(
                                    &relation,
                                    restriction_type,
                                    ways,
                                    Arc::clone(&restrictions),
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
        })
        .map_err(|e| format!("Error reading OSM file: {}", e))?;

    Ok(())
}

/// Extract a single turn restriction from a relation.
///
/// # C++ Equivalent
/// ```cpp
/// void relation_handler::relation(osm::Relation const& r) {
///   // Extract from/to/via members and create restriction
/// }
/// ```
fn extract_turn_restriction(
    relation: &osmpbf::Relation,
    restriction_type: &str,
    ways: &Ways,
    restrictions: Arc<Mutex<Vec<TurnRestriction>>>,
) {
    let is_only = restriction_type.starts_with("only_");
    let is_no = restriction_type.starts_with("no_");

    if !is_only && !is_no {
        return;
    }

    let mut from_ways: Vec<WayIdx> = Vec::new();
    let mut to_ways: Vec<WayIdx> = Vec::new();
    let mut via_node: Option<NodeIdx> = None;

    // Extract from/to/via members
    for member in relation.members() {
        let role = member.role().unwrap_or("");

        match role {
            "from" => {
                if member.member_type == osmpbf::RelMemberType::Way {
                    let osm_way_id = OsmWayIdx(member.member_id as u64);
                    if let Some(way_idx) = ways.find_way(osm_way_id) {
                        from_ways.push(way_idx);
                    }
                }
            }
            "to" => {
                if member.member_type == osmpbf::RelMemberType::Way {
                    let osm_way_id = OsmWayIdx(member.member_id as u64);
                    if let Some(way_idx) = ways.find_way(osm_way_id) {
                        to_ways.push(way_idx);
                    }
                }
            }
            "via" => {
                if member.member_type == osmpbf::RelMemberType::Node {
                    let osm_node_id = OsmNodeIdx(member.member_id as u64);
                    via_node = ways.find_node_idx(osm_node_id);
                }
            }
            _ => {}
        }
    }

    // Create restrictions for all from/to combinations
    if let Some(via) = via_node {
        if !from_ways.is_empty() && !to_ways.is_empty() {
            let mut restr = restrictions.lock().unwrap();
            for from in &from_ways {
                for to in &to_ways {
                    restr.push(TurnRestriction {
                        from_way: *from,
                        to_way: *to,
                        via_node: via,
                        is_only,
                    });
                }
            }
        }
    }
}

/// Check if a way should be included in the routing graph
fn should_include_way(tags: &Tags) -> bool {
    !tags.highway.is_empty() || tags.is_platform() || tags.is_parking
}

/// Get way properties from tags.
///
/// # C++ Equivalent
/// ```cpp
/// way_properties get_way_properties(tags const& t);
/// ```
fn get_way_properties(tags: &Tags) -> WayProperties {
    let (from_level, to_level, _is_multi) = get_levels(tags);

    WayProperties {
        is_foot_accessible: tags.is_accessible_foot(),
        is_bike_accessible: tags.is_accessible_bike(),
        is_car_accessible: tags.is_accessible_car(),
        is_destination: tags.is_destination,
        is_oneway_car: tags.oneway,
        is_oneway_bike: tags.oneway && !tags.not_oneway_bike,
        is_elevator: tags.is_elevator,
        is_steps: tags.highway == "steps",
        is_parking: tags.is_parking,
        speed_limit: get_speed_limit(tags),
        from_level: from_level.into(),
        to_level: to_level.into(),
        is_incline_down: tags.is_incline_down,
        is_platform: tags.is_platform(),
        is_ramp: tags.is_ramp,
        is_sidewalk_separate: tags.sidewalk_separate,
        motor_vehicle_no: tags.motor_vehicle == "no"
            || tags.vehicle == crate::extract::tags::Override::Blacklist,
        has_toll: tags.toll,
        is_big_street: is_big_street(&tags.highway),
    }
}

/// Decode level range and multi-level flag from level bits.
///
/// Iterates set bits to find first (from) and last (to) levels.
/// `is_multi` is true when 3+ levels are present (popcount > 2).
///
/// # C++ Equivalent
/// ```cpp
/// constexpr std::tuple<level_t, level_t, bool> get_levels(
///     bool has_level, level_bits_t levels) noexcept;
/// ```
fn get_levels_from_bits(has_level: bool, levels: LevelBits) -> (u8, u8, bool) {
    if !has_level {
        return (Level::NO_LEVEL, Level::NO_LEVEL, false);
    }
    let mut from: u8 = Level::NO_LEVEL;
    let mut to: u8 = Level::NO_LEVEL;
    let mut bits = levels;
    while bits != 0 {
        let bit = bits.trailing_zeros() as u8;
        bits &= bits - 1; // clear lowest set bit
        if from == Level::NO_LEVEL {
            from = bit;
        } else {
            to = bit;
        }
    }
    let final_to = if to == Level::NO_LEVEL { from } else { to };
    (from, final_to, levels.count_ones() > 2)
}

/// Extract from/to levels from tags.
///
/// Returns (from_level, to_level, is_multi) tuple.
///
/// # C++ Equivalent
/// ```cpp
/// std::tuple<level_t, level_t, bool> get_levels(tags const& t);
/// ```
fn get_levels(tags: &Tags) -> (u8, u8, bool) {
    get_levels_from_bits(tags.has_level, tags.level_bits)
}

/// Get node properties from tags.
///
/// Returns (NodeProperties, level_bits) tuple matching C++ signature.
///
/// # C++ Equivalent
/// ```cpp
/// std::pair<node_properties, level_bits_t> get_node_properties(tags const& t);
/// ```
fn get_node_properties(tags: &Tags) -> (NodeProperties, LevelBits) {
    let (from_level, to_level, is_multi) = get_levels(tags);

    let props = NodeProperties {
        from_level: from_level.into(),
        is_foot_accessible: tags.is_accessible_foot(),
        is_bike_accessible: tags.is_accessible_bike(),
        is_car_accessible: tags.is_accessible_car(),
        is_elevator: tags.is_elevator,
        is_entrance: tags.is_entrance,
        is_multi_level: is_multi,
        is_parking: tags.is_parking,
        to_level: to_level.into(),
    };
    (props, tags.level_bits)
}

/// Determine if a highway tag represents a major street.
pub fn is_big_street(highway: &str) -> bool {
    matches!(
        highway,
        "motorway"
            | "motorway_link"
            | "trunk"
            | "trunk_link"
            | "primary"
            | "primary_link"
            | "secondary"
            | "secondary_link"
            | "tertiary"
            | "tertiary_link"
            | "unclassified"
    )
}

/// Get default speed limit for a highway type.
pub fn get_speed_limit(tags: &Tags) -> SpeedLimit {
    // Try to parse max_speed tag
    if !tags.max_speed.is_empty() {
        if let Ok(kmh) = tags.max_speed.parse::<u32>() {
            return SpeedLimit::from_kmh(kmh);
        }
    }

    // Default based on highway type
    match tags.highway.as_str() {
        "motorway" => SpeedLimit::from_kmh(90),
        "motorway_link" => SpeedLimit::from_kmh(45),
        "trunk" => SpeedLimit::from_kmh(85),
        "trunk_link" => SpeedLimit::from_kmh(40),
        "primary" => {
            if tags.name.is_empty() {
                SpeedLimit::from_kmh(80)
            } else {
                SpeedLimit::from_kmh(40)
            }
        }
        "primary_link" => SpeedLimit::from_kmh(30),
        "secondary" => {
            if tags.name.is_empty() {
                SpeedLimit::from_kmh(80)
            } else {
                SpeedLimit::from_kmh(60)
            }
        }
        "secondary_link" => SpeedLimit::from_kmh(25),
        "tertiary" => {
            if tags.name.is_empty() {
                SpeedLimit::from_kmh(70)
            } else {
                SpeedLimit::from_kmh(40)
            }
        }
        "tertiary_link" => SpeedLimit::from_kmh(20),
        "unclassified" => SpeedLimit::from_kmh(40),
        "residential" => SpeedLimit::from_kmh(30),
        "living_street" => SpeedLimit::from_kmh(10),
        "service" => SpeedLimit::from_kmh(15),
        "track" => SpeedLimit::from_kmh(12),
        "path" => SpeedLimit::from_kmh(13),
        _ => SpeedLimit::Kmh10,
    }
}

/// Print comprehensive statistics about extracted data
fn print_statistics(
    ways: &Ways,
    platforms: &Platforms,
    elevations: &ElevationStorage,
    restriction_count: usize,
) {
    let n_nodes = ways.n_nodes() as usize;
    let n_ways = ways.n_ways() as usize;
    let n_platforms = platforms.num_platforms();

    println!("\n════════════════════════════════════════════════════");
    println!("                EXTRACTION SUMMARY");
    println!("════════════════════════════════════════════════════");
    println!("\n📊 Basic Counts:");
    println!("   Nodes:        {:>10}", n_nodes);
    println!("   Ways:         {:>10}", n_ways);
    println!("   Platforms:    {:>10}", n_platforms);
    println!("   Elevations:   {:>10}", elevations.n_ways());
    println!("   Restrictions: {:>10}", restriction_count);

    // Analyze way accessibility
    let mut foot_count = 0;
    let mut bike_count = 0;
    let mut car_count = 0;
    let mut foot_only = 0;
    let mut bike_only = 0;
    let mut car_only = 0;
    let mut multi_modal = 0;

    // Speed limit distribution
    let mut speed_dist = [0usize; 8];

    // Way lengths
    let mut total_nodes_in_ways = 0usize;
    let mut max_nodes = 0usize;
    let mut loops = 0;
    let mut big_streets = 0;
    let mut one_way_car = 0;
    let mut one_way_bike = 0;
    let mut elevators = 0;
    let mut steps = 0;
    let mut platforms = 0;

    for i in 0..n_ways {
        let way_idx = crate::types::WayIdx(i as u32);
        if let Some(props) = ways.get_way_properties(way_idx) {
            let foot = props.is_foot_accessible();
            let bike = props.is_bike_accessible();
            let car = props.is_car_accessible();

            if foot {
                foot_count += 1;
            }
            if bike {
                bike_count += 1;
            }
            if car {
                car_count += 1;
            }

            let mode_count = [foot, bike, car].iter().filter(|&&x| x).count();
            match mode_count {
                1 => {
                    if foot {
                        foot_only += 1;
                    }
                    if bike {
                        bike_only += 1;
                    }
                    if car {
                        car_only += 1;
                    }
                }
                2 | 3 => multi_modal += 1,
                _ => {}
            }

            speed_dist[props.speed_limit as usize] += 1;

            if props.is_big_street() {
                big_streets += 1;
            }
            if props.is_oneway_car() {
                one_way_car += 1;
            }
            if props.is_oneway_bike() {
                one_way_bike += 1;
            }
            if props.is_elevator {
                elevators += 1;
            }
            if props.is_steps {
                steps += 1;
            }
            if props.is_platform {
                platforms += 1;
            }
        }

        let nodes = ways.get_way_nodes(way_idx);
        total_nodes_in_ways += nodes.len();
        max_nodes = max_nodes.max(nodes.len());

        if ways.is_loop(way_idx) {
            loops += 1;
        }
    }

    println!("\n🚶 Accessibility:");
    println!(
        "   Foot:         {:>10}  ({:>5.1}%)",
        foot_count,
        100.0 * foot_count as f64 / n_ways as f64
    );
    println!(
        "   Bike:         {:>10}  ({:>5.1}%)",
        bike_count,
        100.0 * bike_count as f64 / n_ways as f64
    );
    println!(
        "   Car:          {:>10}  ({:>5.1}%)",
        car_count,
        100.0 * car_count as f64 / n_ways as f64
    );
    println!(
        "   Foot only:    {:>10}  ({:>5.1}%)",
        foot_only,
        100.0 * foot_only as f64 / n_ways as f64
    );
    println!(
        "   Bike only:    {:>10}  ({:>5.1}%)",
        bike_only,
        100.0 * bike_only as f64 / n_ways as f64
    );
    println!(
        "   Car only:     {:>10}  ({:>5.1}%)",
        car_only,
        100.0 * car_only as f64 / n_ways as f64
    );
    println!(
        "   Multi-modal:  {:>10}  ({:>5.1}%)",
        multi_modal,
        100.0 * multi_modal as f64 / n_ways as f64
    );

    println!("\n🚗 Speed Limits:");
    let speed_labels = [
        "10 km/h", "20 km/h", "30 km/h", "40 km/h", "50 km/h", "70 km/h", "90 km/h", "120 km/h",
    ];
    for (i, &count) in speed_dist.iter().enumerate() {
        if count > 0 {
            println!(
                "   {:<10} {:>10}  ({:>5.1}%)",
                speed_labels[i],
                count,
                100.0 * count as f64 / n_ways as f64
            );
        }
    }

    println!("\n🛣️  Way Properties:");
    println!(
        "   Big streets:  {:>10}  ({:>5.1}%)",
        big_streets,
        100.0 * big_streets as f64 / n_ways as f64
    );
    println!(
        "   One-way car:  {:>10}  ({:>5.1}%)",
        one_way_car,
        100.0 * one_way_car as f64 / n_ways as f64
    );
    println!(
        "   One-way bike: {:>10}  ({:>5.1}%)",
        one_way_bike,
        100.0 * one_way_bike as f64 / n_ways as f64
    );
    println!(
        "   Loops:        {:>10}  ({:>5.1}%)",
        loops,
        100.0 * loops as f64 / n_ways as f64
    );
    if elevators > 0 {
        println!("   Elevators:    {:>10}", elevators);
    }
    if steps > 0 {
        println!("   Steps:        {:>10}", steps);
    }
    if platforms > 0 {
        println!("   Platforms:    {:>10}", platforms);
    }

    // Node connectivity analysis
    let mut node_degree_dist = vec![0usize; 11]; // 0-9 ways, 10+
    let mut isolated_nodes = 0;
    let mut max_degree = 0;

    for i in 0..n_nodes {
        let node_idx = crate::types::NodeIdx(i as u32);
        let degree = ways.get_node_ways(node_idx).len();

        if degree == 0 {
            isolated_nodes += 1;
        } else {
            let bucket = degree.min(10);
            node_degree_dist[bucket] += 1;
        }

        max_degree = max_degree.max(degree);
    }

    println!("\n🔗 Node Connectivity:");
    let avg_nodes_per_way = if n_ways > 0 {
        total_nodes_in_ways as f64 / n_ways as f64
    } else {
        0.0
    };
    println!("   Avg nodes/way:  {:>7.1}", avg_nodes_per_way);
    println!("   Max nodes/way:  {:>10}", max_nodes);
    println!(
        "   Isolated nodes: {:>10}  ({:>5.1}%)",
        isolated_nodes,
        100.0 * isolated_nodes as f64 / n_nodes as f64
    );
    println!("   Max node degree:{:>10}", max_degree);

    println!("\n   Node Degree Distribution:");
    for (degree, &count) in node_degree_dist.iter().enumerate() {
        if count > 0 {
            let label = if degree == 10 {
                "10+ ways"
            } else {
                &format!("{} way{}", degree, if degree == 1 { "" } else { "s" })
            };
            println!(
                "     {:<10} {:>10}  ({:>5.1}%)",
                label,
                count,
                100.0 * count as f64 / n_nodes as f64
            );
        }
    }

    // Component analysis
    let mut component_sizes: std::collections::HashMap<crate::types::ComponentIdx, usize> =
        std::collections::HashMap::new();
    for i in 0..n_ways {
        let way_idx = crate::types::WayIdx(i as u32);
        if let Some(comp) = ways.get_component(way_idx) {
            *component_sizes.entry(comp).or_insert(0) += 1;
        }
    }

    let n_components = component_sizes.len();
    let largest_component = component_sizes.values().max().copied().unwrap_or(0);
    let smallest_component = component_sizes.values().min().copied().unwrap_or(0);

    println!("\n🗺️  Connected Components:");
    println!("   Components:   {:>10}", n_components);
    if n_components > 0 {
        println!(
            "   Largest:      {:>10} ways ({:>5.1}%)",
            largest_component,
            100.0 * largest_component as f64 / n_ways as f64
        );
        println!("   Smallest:     {:>10} ways", smallest_component);

        let avg_component_size = n_ways as f64 / n_components as f64;
        println!("   Avg size:     {:>10.1} ways", avg_component_size);

        // Show distribution of component sizes
        let mut size_buckets = vec![0usize; 6]; // 1, 2-10, 11-100, 101-1000, 1001-10000, 10001+
        for &size in component_sizes.values() {
            let bucket = match size {
                1 => 0,
                2..=10 => 1,
                11..=100 => 2,
                101..=1000 => 3,
                1001..=10000 => 4,
                _ => 5,
            };
            size_buckets[bucket] += 1;
        }

        println!("\n   Component Size Distribution:");
        let bucket_labels = [
            "1 way",
            "2-10 ways",
            "11-100 ways",
            "101-1K ways",
            "1K-10K ways",
            "10K+ ways",
        ];
        for (i, &count) in size_buckets.iter().enumerate() {
            if count > 0 {
                println!("     {:<12} {:>7} components", bucket_labels[i], count);
            }
        }
    }

    // Turn restrictions analysis
    if restriction_count > 0 {
        let mut restricted_nodes = 0;
        for i in 0..n_nodes {
            let node_idx = crate::types::NodeIdx(i as u32);
            let node_ways = ways.get_node_ways(node_idx);
            if node_ways.len() >= 2 {
                // Check if any turn is restricted
                let mut has_restriction = false;
                for j in 0..node_ways.len() {
                    for k in 0..node_ways.len() {
                        if j != k && ways.is_restricted(node_idx, j as u8, k as u8) {
                            has_restriction = true;
                            break;
                        }
                    }
                    if has_restriction {
                        break;
                    }
                }
                if has_restriction {
                    restricted_nodes += 1;
                }
            }
        }

        println!("\n🚦 Turn Restrictions:");
        println!("   Total restrictions: {:>10}", restriction_count);
        println!(
            "   Restricted nodes:   {:>10}  ({:>5.1}%)",
            restricted_nodes,
            100.0 * restricted_nodes as f64 / n_nodes as f64
        );
        let avg_restrictions_per_node = if restricted_nodes > 0 {
            restriction_count as f64 / restricted_nodes as f64
        } else {
            0.0
        };
        println!("   Avg per node:       {:>10.1}", avg_restrictions_per_node);
    }

    println!("\n════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_big_street() {
        assert!(is_big_street("motorway"));
        assert!(is_big_street("trunk"));
        assert!(is_big_street("primary"));
        assert!(is_big_street("secondary"));
        assert!(is_big_street("tertiary"));
        assert!(is_big_street("unclassified"));
        assert!(!is_big_street("residential"));
        assert!(!is_big_street("service"));
        assert!(!is_big_street("path"));
    }

    #[test]
    fn test_speed_limit_from_kmh() {
        let limit = SpeedLimit::from_kmh(100);
        assert_eq!(limit, SpeedLimit::Kmh100);
    }

    #[test]
    fn test_speed_limit_defaults() {
        let mut tags = Tags::default();

        tags.highway = "motorway".to_string();
        assert_eq!(get_speed_limit(&tags), SpeedLimit::Kmh80); // 90 -> Kmh80

        tags.highway = "residential".to_string();
        assert_eq!(get_speed_limit(&tags), SpeedLimit::Kmh30);

        tags.highway = "living_street".to_string();
        assert_eq!(get_speed_limit(&tags), SpeedLimit::Kmh10);
    }

    #[test]
    fn test_speed_limit_with_max_speed_tag() {
        let mut tags = Tags::default();
        tags.highway = "motorway".to_string();
        tags.max_speed = "120".to_string();

        let limit = get_speed_limit(&tags);
        assert_eq!(limit, SpeedLimit::Kmh120);
    }

    #[test]
    fn test_should_include_way() {
        let mut tags = Tags::default();

        // Highway ways should be included
        tags.highway = "residential".to_string();
        assert!(should_include_way(&tags));

        // Platform should be included
        tags.highway = String::new();
        tags.is_platform = true;
        assert!(should_include_way(&tags));

        // Parking should be included
        tags.is_platform = false;
        tags.is_parking = true;
        assert!(should_include_way(&tags));

        // Empty tags should not be included
        tags.is_parking = false;
        assert!(!should_include_way(&tags));
    }

    #[test]
    fn test_get_way_properties_accessible() {
        let mut tags = Tags::default();
        tags.highway = "residential".to_string();
        tags.foot = "yes".to_string();
        tags.bicycle = "yes".to_string();

        let props = get_way_properties(&tags);
        assert!(props.is_foot_accessible);
        assert!(props.is_bike_accessible);
        assert_eq!(props.speed_limit, SpeedLimit::Kmh30);
        assert!(!props.is_big_street);
    }

    #[test]
    fn test_get_way_properties_big_street() {
        let mut tags = Tags::default();
        tags.highway = "motorway".to_string();

        let props = get_way_properties(&tags);
        assert!(props.is_big_street);
        assert_eq!(props.speed_limit, SpeedLimit::Kmh80); // 90 -> Kmh80
    }

    #[test]
    fn test_get_way_properties_oneway() {
        let mut tags = Tags::default();
        tags.highway = "residential".to_string();
        tags.oneway = true;

        let props = get_way_properties(&tags);
        assert!(props.is_oneway_car);
        assert!(props.is_oneway_bike);

        // Bikes can go against oneway if explicitly allowed
        tags.not_oneway_bike = true;
        let props2 = get_way_properties(&tags);
        assert!(props2.is_oneway_car);
        assert!(!props2.is_oneway_bike);
    }

    #[test]
    fn test_get_way_properties_elevator() {
        let mut tags = Tags::default();
        tags.is_elevator = true;

        let props = get_way_properties(&tags);
        assert!(props.is_elevator);
    }

    #[test]
    fn test_get_way_properties_steps() {
        let mut tags = Tags::default();
        tags.highway = "steps".to_string();

        let props = get_way_properties(&tags);
        assert!(props.is_steps);
    }

    #[test]
    fn test_get_node_properties() {
        let mut tags = Tags::default();
        tags.is_elevator = true;
        tags.is_entrance = true;
        tags.is_parking = true;

        let (props, _level_bits) = get_node_properties(&tags);
        assert!(props.is_elevator);
        assert!(props.is_entrance);
        assert!(props.is_parking);
        assert!(!props.is_multi_level); // no levels set → not multi-level
    }

    #[test]
    fn test_get_levels() {
        let tags = Tags::default();
        let (from, to, is_multi) = get_levels(&tags);
        assert_eq!(from, Level::NO_LEVEL);
        assert_eq!(to, Level::NO_LEVEL);
        assert!(!is_multi);

        // Test with level bits set (simulate level=0;1;2 → 3 levels → is_multi=true)
        let mut tags_multi = Tags::default();
        tags_multi.has_level = true;
        // Set bits for 3 levels
        let lvl0 = Level::from_float(0.0).to_idx();
        let lvl1 = Level::from_float(1.0).to_idx();
        let lvl2 = Level::from_float(2.0).to_idx();
        tags_multi.level_bits = (1u64 << lvl0) | (1u64 << lvl1) | (1u64 << lvl2);
        let (from_m, to_m, is_multi_m) = get_levels(&tags_multi);
        assert_eq!(from_m, lvl0);
        assert_eq!(to_m, lvl2);
        assert!(is_multi_m); // popcount(3) > 2

        // Test with 2 levels → not multi
        let mut tags_two = Tags::default();
        tags_two.has_level = true;
        tags_two.level_bits = (1u64 << lvl0) | (1u64 << lvl1);
        let (_from_t, _to_t, is_multi_t) = get_levels(&tags_two);
        assert!(!is_multi_t); // popcount(2) <= 2
    }

    // Integration tests require actual OSM PBF files
    // These would test:
    // - extract() with a small test OSM file
    // - Node coordinate collection
    // - Way geometry extraction
    // - Turn restriction extraction
    // - OSM ID to internal index mapping
    //
    // Example (requires test data):
    // #[test]
    // fn test_extract_integration() {
    //     let input = Path::new("test_data/small_area.osm.pbf");
    //     let output = Path::new("test_output");
    //     let result = extract(false, input, output, None);
    //     assert!(result.is_ok());
    // }
}
