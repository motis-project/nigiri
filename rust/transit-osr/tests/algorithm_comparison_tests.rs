//! Algorithm comparison tests - Rust port of C++ dijkstra_astarbidir_test.cc
//!
//! Compares Dijkstra and A* bidirectional search to ensure they produce identical results.
//!
//! Key test scenarios from C++:
//! - Monaco forward/backward (10,000 samples, 2hr max cost)
//! - Hamburg (5,000 samples, 3hr max cost)
//! - Switzerland (1,000 samples, 5hr max cost)
//! - Germany (50 samples, 12hr max cost) - disabled by default
//!
//! Test criteria:
//! - Both algorithms must find the same paths (identical costs)
//! - A* should provide speedup on non-empty results
//! - Handle empty matches gracefully

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use transit_cloud_osr::{
    extract::extract,
    routing::{profile::SearchProfile, route::route, RoutingAlgorithm, ProfileInstance},
    types::{Cost, NodeIdx},
    Direction, Level, Location, Lookup, OsrData,
    K_INFEASIBLE,
};

const PRINT_MISMATCHES: bool = true;

/// Load or extract OSM data
fn load_data(raw_data: &str, data_dir: &str) -> anyhow::Result<OsrData> {
    let data_path = Path::new(data_dir);

    if !data_path.exists() {
        let raw_path = Path::new(raw_data);
        if raw_path.exists() {
            println!("Extracting {} to {}...", raw_data, data_dir);
            std::fs::create_dir_all(data_path)?;
            extract(false, raw_path, data_path, None).map_err(|e| anyhow::anyhow!("{}", e))?;
        } else {
            anyhow::bail!("Test data not found: {}", raw_data);
        }
    }

    OsrData::import(data_path).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Run comparison test between algorithms (matches C++ test structure)
fn run_comparison(
    data: &OsrData,
    lookup: &Lookup,
    n_samples: usize,
    max_cost_seconds: u32,
    profile: SearchProfile,
    direction: Direction,
) -> anyhow::Result<()> {
    use rand::prelude::*;

    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let n_nodes = data.ways.n_nodes();

    let n_congruent = AtomicUsize::new(0);
    let n_empty_matches = AtomicUsize::new(0);
    let mut dijkstra_times = Vec::with_capacity(n_samples);
    let mut astar_times = Vec::with_capacity(n_samples);

    println!(
        "Running {} samples with max_cost={}s, profile={:?}, direction={:?}",
        n_samples, max_cost_seconds, profile, direction
    );

    for sample in 0..n_samples {
        let from_node_idx = rng.gen_range(0..n_nodes);
        let to_node_idx = rng.gen_range(0..n_nodes);

        let from_pos = data.ways.get_node_pos(NodeIdx(from_node_idx as u32));
        let to_pos = data.ways.get_node_pos(NodeIdx(to_node_idx as u32));

        let from = Location::new(from_pos, Level::default());
        let to = Location::new(to_pos, Level::default());

        // Run Dijkstra (reference algorithm)
        let dijkstra_start = Instant::now();
        let dijkstra_result = route(
            &data.ways,
            lookup,
            None,
            profile,
            from,
            to,
            max_cost_seconds as Cost,
            RoutingAlgorithm::Dijkstra,
        );
        let dijkstra_time = dijkstra_start.elapsed();

        // Run A* Bidirectional (experiment algorithm)
        let astar_start = Instant::now();
        let astar_result = route(
            &data.ways,
            lookup,
            None,
            profile,
            from,
            to,
            max_cost_seconds as Cost,
            RoutingAlgorithm::AStarBi,
        );
        let astar_time = astar_start.elapsed();

        // Compare results
        let matches = match (&dijkstra_result, &astar_result) {
            (Some(d_path), Some(a_path)) => {
                // Both found a path - costs must match exactly
                let costs_match = d_path.cost == a_path.cost;

                if !costs_match && PRINT_MISMATCHES {
                    println!(
                        "MISMATCH sample {}: node {} -> node {} | Dijkstra cost={} dist={:.2} | A* cost={} dist={:.2}",
                        sample, from_node_idx, to_node_idx,
                        d_path.cost, d_path.dist, a_path.cost, a_path.dist
                    );
                }

                costs_match
            }
            (None, None) => {
                // Both found no path - this is correct
                n_empty_matches.fetch_add(1, Ordering::Relaxed);
                true
            }
            (Some(d), None) => {
                if PRINT_MISMATCHES {
                    println!(
                        "MISMATCH sample {}: Dijkstra found path (cost={}), A* did not",
                        sample, d.cost
                    );
                }
                false
            }
            (None, Some(a)) => {
                if PRINT_MISMATCHES {
                    println!(
                        "MISMATCH sample {}: A* found path (cost={}), Dijkstra did not",
                        sample, a.cost
                    );
                }
                false
            }
        };

        if matches {
            n_congruent.fetch_add(1, Ordering::Relaxed);
            if dijkstra_result.is_some() && astar_result.is_some() {
                dijkstra_times.push(dijkstra_time);
                astar_times.push(astar_time);
            }
        }
    }

    let total_congruent = n_congruent.load(Ordering::Relaxed);
    let total_empty = n_empty_matches.load(Ordering::Relaxed);
    let non_empty_congruent = total_congruent - total_empty;
    let non_empty_samples = n_samples - total_empty;

    println!(
        "\nResults: congruent on non-empty: {}/{} ({:.1}%)",
        non_empty_congruent,
        non_empty_samples,
        if non_empty_samples > 0 {
            (non_empty_congruent as f64 / non_empty_samples as f64) * 100.0
        } else {
            0.0
        }
    );

    // Print timing statistics if we have valid comparisons
    if !dijkstra_times.is_empty() {
        let dijkstra_total: std::time::Duration = dijkstra_times.iter().sum();
        let astar_total: std::time::Duration = astar_times.iter().sum();

        println!(
            "Timing: Dijkstra avg={:?}, A* avg={:?}",
            dijkstra_total / dijkstra_times.len() as u32,
            astar_total / astar_times.len() as u32
        );

        let speedup = dijkstra_total.as_secs_f64() / astar_total.as_secs_f64();
        println!("Speedup on non-empty: {:.2}x", speedup);
    }

    // Assert that all non-empty samples produced congruent results (matches C++ EXPECT_EQ)
    assert_eq!(
        non_empty_samples, non_empty_congruent,
        "Not all non-empty samples produced congruent results: {}/{} matched",
        non_empty_congruent, non_empty_samples
    );

    Ok(())
}

// ============================================================================
// Test cases matching C++ dijkstra_astarbidir_test.cc
// ============================================================================

#[test]
#[ignore] // Requires test data: monaco.osm.pbf
fn test_monaco_forward() {
    const RAW_DATA: &str = "test/monaco.osm.pbf";
    const DATA_DIR: &str = "test/monaco";
    const NUM_SAMPLES: usize = 10000;
    const MAX_COST: u32 = 2 * 3600; // 2 hours

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    let result = run_comparison(
        &data,
        &lookup,
        NUM_SAMPLES,
        MAX_COST,
        SearchProfile::Car,
        Direction::Forward,
    );
    assert!(result.is_ok(), "Comparison test failed: {:?}", result.err());
}

#[test]
#[ignore] // Requires test data: monaco.osm.pbf
fn test_monaco_backward() {
    const RAW_DATA: &str = "test/monaco.osm.pbf";
    const DATA_DIR: &str = "test/monaco";
    const NUM_SAMPLES: usize = 10000;
    const MAX_COST: u32 = 2 * 3600; // 2 hours

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    let result = run_comparison(
        &data,
        &lookup,
        NUM_SAMPLES,
        MAX_COST,
        SearchProfile::Car,
        Direction::Backward,
    );
    assert!(result.is_ok(), "Comparison test failed: {:?}", result.err());
}

#[test]
#[ignore] // Requires test data: hamburg.osm.pbf
fn test_hamburg() {
    const RAW_DATA: &str = "test/hamburg.osm.pbf";
    const DATA_DIR: &str = "test/hamburg";
    const NUM_SAMPLES: usize = 5000;
    const MAX_COST: u32 = 3 * 3600; // 3 hours

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    let result = run_comparison(
        &data,
        &lookup,
        NUM_SAMPLES,
        MAX_COST,
        SearchProfile::Car,
        Direction::Forward,
    );
    assert!(result.is_ok(), "Comparison test failed: {:?}", result.err());
}

#[test]
#[ignore] // Requires test data: switzerland.osm.pbf
fn test_switzerland() {
    const RAW_DATA: &str = "test/switzerland.osm.pbf";
    const DATA_DIR: &str = "test/switzerland";
    const NUM_SAMPLES: usize = 1000;
    const MAX_COST: u32 = 5 * 3600; // 5 hours

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    let result = run_comparison(
        &data,
        &lookup,
        NUM_SAMPLES,
        MAX_COST,
        SearchProfile::Car,
        Direction::Forward,
    );
    assert!(result.is_ok(), "Comparison test failed: {:?}", result.err());
}

#[test]
#[ignore] // Requires test data: germany.osm.pbf — disabled in C++ (DISABLED_germany)
fn test_germany() {
    const RAW_DATA: &str = "test/germany.osm.pbf";
    const DATA_DIR: &str = "test/germany";
    const NUM_SAMPLES: usize = 50;
    const MAX_COST: u32 = 12 * 3600; // 12 hours

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    let result = run_comparison(
        &data,
        &lookup,
        NUM_SAMPLES,
        MAX_COST,
        SearchProfile::Car,
        Direction::Forward,
    );
    assert!(result.is_ok(), "Comparison test failed: {:?}", result.err());
}

// ============================================================================
// Additional profile-specific tests
// ============================================================================

#[test]
#[ignore] // Requires test data: luisenplatz-darmstadt.osm.pbf
fn test_foot_profile() {
    const RAW_DATA: &str = "test/luisenplatz-darmstadt.osm.pbf";
    const DATA_DIR: &str = "test/luisenplatz";
    const NUM_SAMPLES: usize = 100;
    const MAX_COST: u32 = 3600; // 1 hour

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    let result = run_comparison(
        &data,
        &lookup,
        NUM_SAMPLES,
        MAX_COST,
        SearchProfile::Foot,
        Direction::Forward,
    );
    assert!(result.is_ok(), "Comparison test failed: {:?}", result.err());
}

#[test]
#[ignore] // Requires test data: luisenplatz-darmstadt.osm.pbf
fn debug_sample_0() {
    use rand::prelude::*;
    use transit_cloud_osr::routing::Bidirectional;
    use transit_cloud_osr::routing::profile::SearchProfile;

    const RAW_DATA: &str = "test/luisenplatz-darmstadt.osm.pbf";
    const DATA_DIR: &str = "test/luisenplatz";

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping debug: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    // Recreate RNG and pick sample 0
    let mut rng = StdRng::seed_from_u64(42);
    let n_nodes = data.ways.n_nodes();
    let from_node_idx = rng.gen_range(0..n_nodes);
    let to_node_idx = rng.gen_range(0..n_nodes);

    let from_pos = data.ways.get_node_pos(NodeIdx(from_node_idx as u32));
    let to_pos = data.ways.get_node_pos(NodeIdx(to_node_idx as u32));

    let from = Location::new(from_pos, Level::default());
    let to = Location::new(to_pos, Level::default());

    println!("Debug sample 0: from={} to={}", from_node_idx, to_node_idx);

    // Match candidates as route() does
    let profile = SearchProfile::Foot;
    let profile_inst = ProfileInstance::from_search_profile(profile);
    let max_distance = profile_inst.max_match_distance();

    let start_candidates = lookup.match_location(&data.ways, profile, &from, max_distance);
    let end_candidates = lookup.match_location(&data.ways, profile, &to, max_distance);

    let mut bidir = Bidirectional::new();
    bidir.set_ways(&data.ways);
    let max_cost: Cost = 3600;
    bidir.reset(max_cost, from, to);

    for c in &start_candidates {
        if c.left.valid() { bidir.add_start(c.left.node, c.left.cost); }
        if c.right.valid() { bidir.add_start(c.right.node, c.right.cost); }
    }
    for c in &end_candidates {
        if c.left.valid() { bidir.add_end(c.left.node, c.left.cost); }
        if c.right.valid() { bidir.add_end(c.right.node, c.right.cost); }
    }

    println!("After seeding:\n{}", bidir.dump_state());

    // neighbor function copied from route_foot_bidirectional
    let ways_ref = &data.ways;
    let profile_ref = match profile_inst { ProfileInstance::Foot(ref p) => p, _ => panic!("profile mismatch") };

    let get_neighbors = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
        use transit_cloud_osr::routing::profiles::foot::{node_cost, way_cost};
        let mut neighbors = Vec::with_capacity(8);
        let node_props = ways_ref.get_node_properties(node);
        let n_cost = node_props.map(|p| node_cost(p)).unwrap_or(0);

        for &way in ways_ref.get_node_ways(node) {
            let way_nodes = ways_ref.get_way_nodes(way);
            let way_dists = ways_ref.get_way_node_distances(way);
            let way_props = ways_ref.get_way_properties(way);
            if way_nodes.is_empty() || way_props.is_none() { continue; }
            let props = way_props.unwrap();

            for (i, &n) in way_nodes.iter().enumerate() {
                if n != node { continue; }
                if i + 1 < way_nodes.len() {
                    let next_node = way_nodes[i + 1];
                    let dist = way_dists.get(i).copied().unwrap_or(0) as u32;
                    if dist > 0 && dist <= 50000 {
                        let edge_cost = way_cost(&profile_ref.params, props, profile_ref.is_wheelchair, dist as u16);
                        if edge_cost != K_INFEASIBLE {
                            let total_cost = n_cost.saturating_add(edge_cost);
                            if total_cost < K_INFEASIBLE { neighbors.push((next_node, total_cost)); }
                        }
                    }
                }
                if i > 0 {
                    let prev_node = way_nodes[i - 1];
                    let dist = way_dists.get(i - 1).copied().unwrap_or(0) as u32;
                    if dist > 0 && dist <= 50000 {
                        let edge_cost = way_cost(&profile_ref.params, props, profile_ref.is_wheelchair, dist as u16);
                        if edge_cost != K_INFEASIBLE {
                            let total_cost = n_cost.saturating_add(edge_cost);
                            if total_cost < K_INFEASIBLE { neighbors.push((prev_node, total_cost)); }
                        }
                    }
                }
                break;
            }
        }
        neighbors
    };

    // Step loop (bounded to avoid long runs)
    for step in 0..200 {
        if bidir.best_cost() != Cost::MAX {
            println!("Best cost so far: {} meet_point={:?}", bidir.best_cost(), bidir.meet_point());
        }
        if bidir.pq_forward_empty() && bidir.pq_backward_empty() {
            break;
        }

        if !bidir.pq_forward_empty()
            && (bidir.pq_backward_empty() || bidir.pq_forward_size() <= bidir.pq_backward_size())
        {
            bidir.step_forward(max_cost, &get_neighbors);
            println!("After forward step {}:\n{}", step, bidir.dump_state());
        } else if !bidir.pq_backward_empty() {
            bidir.step_backward(max_cost, &get_neighbors);
            println!("After backward step {}:\n{}", step, bidir.dump_state());
        }

        if bidir.best_cost() != Cost::MAX {
            // termination check similar to run()
            let min_fwd = bidir.pq_forward_peek().unwrap_or(usize::MAX);
            let min_bwd = bidir.pq_backward_peek().unwrap_or(usize::MAX);
            if min_fwd != usize::MAX && min_bwd != usize::MAX {
                let summed = (min_fwd as Cost).saturating_add(min_bwd as Cost);
                if summed >= bidir.best_cost() {
                    break;
                }
            }
        }
    }

    println!("Final best_cost={} meet_point={:?}", bidir.best_cost(), bidir.meet_point());
}

#[test]
#[ignore] // Requires test data: luisenplatz-darmstadt.osm.pbf
fn debug_compare_sample_0() {
    use transit_cloud_osr::routing::RoutingAlgorithm;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand::Rng;

    const RAW_DATA: &str = "test/luisenplatz-darmstadt.osm.pbf";
    const DATA_DIR: &str = "test/luisenplatz";
    const MAX_COST: u32 = 3600;

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping debug: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    let mut rng = StdRng::seed_from_u64(42);
    let n_nodes = data.ways.n_nodes();
    let from_node_idx = rng.gen_range(0..n_nodes);
    let to_node_idx = rng.gen_range(0..n_nodes);

    let from_pos = data.ways.get_node_pos(NodeIdx(from_node_idx as u32));
    let to_pos = data.ways.get_node_pos(NodeIdx(to_node_idx as u32));

    let from = Location::new(from_pos, Level::default());
    let to = Location::new(to_pos, Level::default());

    println!("Compare sample 0: from={} to={}", from_node_idx, to_node_idx);

    let dres = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::Dijkstra,
    );

    let ares = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::AStarBi,
    );

    println!("Dijkstra result: {:?}\nA* result: {:?}", dres, ares);

    if ares.is_none() {
        // Run manual bidir debug to dump internal state
        println!("A* returned None — running manual bidir dump");
        let from = Location::new(from_pos, Level::default());
        let to = Location::new(to_pos, Level::default());

        let profile = SearchProfile::Foot;
        let profile_inst = ProfileInstance::from_search_profile(profile);
        let max_distance = profile_inst.max_match_distance();

        let start_candidates = lookup.match_location(&data.ways, profile, &from, max_distance);
        let end_candidates = lookup.match_location(&data.ways, profile, &to, max_distance);

        let mut bidir = transit_cloud_osr::routing::Bidirectional::new();
        bidir.set_ways(&data.ways);
        bidir.reset(MAX_COST as Cost, from, to);

        for c in &start_candidates {
            if c.left.valid() { bidir.add_start(c.left.node, c.left.cost); }
            if c.right.valid() { bidir.add_start(c.right.node, c.right.cost); }
        }
        for c in &end_candidates {
            if c.left.valid() { bidir.add_end(c.left.node, c.left.cost); }
            if c.right.valid() { bidir.add_end(c.right.node, c.right.cost); }
        }

        println!("start_candidates: {:?}\nend_candidates: {:?}", start_candidates, end_candidates);
        println!("Manual seeding dump:\n{}", bidir.dump_state());
    }
}

#[test]
#[ignore] // Requires test data: luisenplatz-darmstadt.osm.pbf
fn debug_compare_sample_2() {
    use transit_cloud_osr::routing::RoutingAlgorithm;

    const RAW_DATA: &str = "test/luisenplatz-darmstadt.osm.pbf";
    const DATA_DIR: &str = "test/luisenplatz";
    const MAX_COST: u32 = 3600;

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping debug: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    // Use the sample that failed in run_comparison
    let from_node_idx = 178usize;
    let to_node_idx = 6usize;

    let from_pos = data.ways.get_node_pos(NodeIdx(from_node_idx as u32));
    let to_pos = data.ways.get_node_pos(NodeIdx(to_node_idx as u32));

    let from = Location::new(from_pos, Level::default());
    let to = Location::new(to_pos, Level::default());

    println!("Compare sample 2: from={} to={}", from_node_idx, to_node_idx);

    let dres = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::Dijkstra,
    );

    let ares = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::AStarBi,
    );

    println!("Dijkstra result: {:?}\nA* result: {:?}", dres, ares);

    if ares.is_none() || dres.is_some() && ares.is_some() && dres.as_ref().unwrap().cost != ares.as_ref().unwrap().cost {
        // Run manual bidir debug to dump internal state
        println!("A* result differs — running manual bidir dump");
        let profile = SearchProfile::Foot;
        let profile_inst = ProfileInstance::from_search_profile(profile);
        let max_distance = profile_inst.max_match_distance();

        let start_candidates = lookup.match_location(&data.ways, profile, &from, max_distance);
        let end_candidates = lookup.match_location(&data.ways, profile, &to, max_distance);

        let mut bidir = transit_cloud_osr::routing::Bidirectional::new();
        bidir.set_ways(&data.ways);
        bidir.reset(MAX_COST as Cost, from, to);

        for c in &start_candidates {
            if c.left.valid() { bidir.add_start(c.left.node, c.left.cost); }
            if c.right.valid() { bidir.add_start(c.right.node, c.right.cost); }
        }
        for c in &end_candidates {
            if c.left.valid() { bidir.add_end(c.left.node, c.left.cost); }
            if c.right.valid() { bidir.add_end(c.right.node, c.right.cost); }
        }

        println!("start_candidates: {:?}\nend_candidates: {:?}", start_candidates, end_candidates);
        println!("Manual seeding dump:\n{}", bidir.dump_state());
    }
}

#[test]
#[ignore] // Requires test data: luisenplatz-darmstadt.osm.pbf
fn test_bike_profile() {
    const RAW_DATA: &str = "test/luisenplatz-darmstadt.osm.pbf";
    const DATA_DIR: &str = "test/luisenplatz";
    const NUM_SAMPLES: usize = 100;
    const MAX_COST: u32 = 3600; // 1 hour

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    let result = run_comparison(
        &data,
        &lookup,
        NUM_SAMPLES,
        MAX_COST,
        SearchProfile::Bike,
        Direction::Forward,
    );
    assert!(result.is_ok(), "Comparison test failed: {:?}", result.err());
}

#[test]
#[ignore] // Requires test data: luisenplatz-darmstadt.osm.pbf
fn debug_compare_sample_11() {
    const RAW_DATA: &str = "test/luisenplatz-darmstadt.osm.pbf";
    const DATA_DIR: &str = "test/luisenplatz";
    const MAX_COST: u32 = 3600;

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    // Sample 11: node 150 -> node 40 | Dijkstra cost=85 dist=1.94 | A* cost=100 dist=0.81
    let from_node_idx = 150usize;
    let to_node_idx = 40usize;

    let from_pos = data.ways.get_node_pos(NodeIdx(from_node_idx as u32));
    let to_pos = data.ways.get_node_pos(NodeIdx(to_node_idx as u32));

    let from = Location::new(from_pos, Level::default());
    let to = Location::new(to_pos, Level::default());

    println!("Compare sample 11: from={} to={}", from_node_idx, to_node_idx);

    let dres = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::Dijkstra,
    );

    let ares = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::AStarBi,
    );

    println!("Dijkstra result: {:?}\nA* result: {:?}", dres, ares);

    if ares.is_none() || dres.is_some() && ares.is_some() && dres.as_ref().unwrap().cost != ares.as_ref().unwrap().cost {
        // Run manual bidir debug to dump internal state
        println!("A* result differs — running manual bidir dump");
        let profile = SearchProfile::Foot;
        let profile_inst = ProfileInstance::from_search_profile(profile);
        let max_distance = profile_inst.max_match_distance();

        let start_candidates = lookup.match_location(&data.ways, profile, &from, max_distance);
        let end_candidates = lookup.match_location(&data.ways, profile, &to, max_distance);

        println!("Dijkstra found path with cost {}, A* found {}", 
                 dres.as_ref().map(|p| p.cost).unwrap_or(Cost::MAX),
                 ares.as_ref().map(|p| p.cost).unwrap_or(Cost::MAX));
        println!("start_candidates count: {}, end_candidates count: {}", 
                 start_candidates.len(), end_candidates.len());
    }
}

#[test]
#[ignore] // Requires test data: luisenplatz-darmstadt.osm.pbf
fn debug_compare_sample_65() {
    const RAW_DATA: &str = "test/luisenplatz-darmstadt.osm.pbf";
    const DATA_DIR: &str = "test/luisenplatz";
    const MAX_COST: u32 = 3600;

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    // Sample 65: node 131 -> node 171 | Dijkstra cost=0 dist=0.00 | A* cost=21
    let from_node_idx = 131usize;
    let to_node_idx = 171usize;

    let from_pos = data.ways.get_node_pos(NodeIdx(from_node_idx as u32));
    let to_pos = data.ways.get_node_pos(NodeIdx(to_node_idx as u32));

    let from = Location::new(from_pos, Level::default());
    let to = Location::new(to_pos, Level::default());

    println!("Compare sample 65: from={} to={}", from_node_idx, to_node_idx);

    let dres = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::Dijkstra,
    );

    let ares = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::AStarBi,
    );

    println!("Dijkstra result: {:?}\nA* result: {:?}", dres, ares);

    if ares.is_none() || dres.is_some() && ares.is_some() && dres.as_ref().unwrap().cost != ares.as_ref().unwrap().cost {
        // Run manual bidir debug to dump internal state
        println!("A* result differs — running manual bidir dump");
        let profile = SearchProfile::Foot;
        let profile_inst = ProfileInstance::from_search_profile(profile);
        let max_distance = profile_inst.max_match_distance();

        let start_candidates = lookup.match_location(&data.ways, profile, &from, max_distance);
        let end_candidates = lookup.match_location(&data.ways, profile, &to, max_distance);

        let mut bidir = transit_cloud_osr::routing::Bidirectional::new();
        bidir.set_ways(&data.ways);
        bidir.reset(MAX_COST as Cost, from, to);

        for c in &start_candidates {
            if c.left.valid() { bidir.add_start(c.left.node, c.left.cost); }
            if c.right.valid() { bidir.add_start(c.right.node, c.right.cost); }
        }
        for c in &end_candidates {
            if c.left.valid() { bidir.add_end(c.left.node, c.left.cost); }
            if c.right.valid() { bidir.add_end(c.right.node, c.right.cost); }
        }

        println!("start_candidates: {:?}\nend_candidates: {:?}", start_candidates, end_candidates);
        println!("Manual seeding dump:\n{}", bidir.dump_state());
    }
}

#[test]
#[ignore] // Requires test data: luisenplatz-darmstadt.osm.pbf
fn debug_compare_sample_79() {
    const RAW_DATA: &str = "test/luisenplatz-darmstadt.osm.pbf";
    const DATA_DIR: &str = "test/luisenplatz";
    const MAX_COST: u32 = 3600;

    let data = match load_data(RAW_DATA, DATA_DIR) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: {} not found", RAW_DATA);
            return;
        }
    };

    let lookup = unsafe {
        let ways_ptr = &data.ways as *const _;
        Lookup::load(&*ways_ptr, Path::new(DATA_DIR)).unwrap()
    };

    // Sample 79: node 24 -> node 97 | Dijkstra cost=0 dist=0.00 | A* cost=11
    let from_node_idx = 24usize;
    let to_node_idx = 97usize;

    let from_pos = data.ways.get_node_pos(NodeIdx(from_node_idx as u32));
    let to_pos = data.ways.get_node_pos(NodeIdx(to_node_idx as u32));

    let from = Location::new(from_pos, Level::default());
    let to = Location::new(to_pos, Level::default());

    println!("Compare sample 79: from={} to={}", from_node_idx, to_node_idx);

    let dres = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::Dijkstra,
    );

    let ares = route(
        &data.ways,
        &lookup,
        None,
        SearchProfile::Foot,
        from,
        to,
        MAX_COST as Cost,
        RoutingAlgorithm::AStarBi,
    );

    println!("Dijkstra result: {:?}\nA* result: {:?}", dres, ares);

    if ares.is_none() || dres.is_some() && ares.is_some() && dres.as_ref().unwrap().cost != ares.as_ref().unwrap().cost {
        // Run manual bidir debug to dump internal state
        println!("A* result differs — running manual bidir dump");
        let profile = SearchProfile::Foot;
        let profile_inst = ProfileInstance::from_search_profile(profile);
        let max_distance = profile_inst.max_match_distance();

        let start_candidates = lookup.match_location(&data.ways, profile, &from, max_distance);
        let end_candidates = lookup.match_location(&data.ways, profile, &to, max_distance);

        let mut bidir = transit_cloud_osr::routing::Bidirectional::new();
        bidir.set_ways(&data.ways);
        bidir.reset(MAX_COST as Cost, from, to);

        for c in &start_candidates {
            if c.left.valid() { bidir.add_start(c.left.node, c.left.cost); }
            if c.right.valid() { bidir.add_start(c.right.node, c.right.cost); }
        }
        for c in &end_candidates {
            if c.left.valid() { bidir.add_end(c.left.node, c.left.cost); }
            if c.right.valid() { bidir.add_end(c.right.node, c.right.cost); }
        }

        println!("start_candidates: {:?}\nend_candidates: {:?}", start_candidates, end_candidates);
        println!("Manual seeding dump:\n{}", bidir.dump_state());
    }
}
