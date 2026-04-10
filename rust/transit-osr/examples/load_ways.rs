//! Example: Load extracted ways data and display comprehensive statistics
//!
//! # Usage
//! ```bash
//! cargo run --example load_ways -- extracted_test/ways.bin
//! cargo run --example load_ways -- extracted_test  # Auto-detects format
//! ```

use std::env;
use std::path::Path;
use transit_cloud_osr::types::{NodeIdx, WayIdx};
use transit_cloud_osr::ways::Ways;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <ways.bin|directory>", args[0]);
        eprintln!("Example: {} extracted_test/ways.bin", args[0]);
        eprintln!("Example: {} extracted_test", args[0]);
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);

    // Get file/directory size
    let file_size = if path.is_dir() {
        // For directory, sum all .bin files
        std::fs::read_dir(path)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("bin"))
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0)
    } else {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    };

    println!("Loading ways from {}...", path.display());
    println!("Size: {:.2} MB\n", file_size as f64 / 1024.0 / 1024.0);

    let start = std::time::Instant::now();

    match Ways::load(path) {
        Ok(ways) => {
            let elapsed = start.elapsed();
            let throughput = file_size as f64 / 1024.0 / 1024.0 / elapsed.as_secs_f64();

            println!(
                "✓ Loaded successfully in {:.2}s ({:.1} MB/s)\n",
                elapsed.as_secs_f64(),
                throughput
            );

            print_statistics(&ways);
            print_samples(&ways);

            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("✗ Failed to load: {}", e);
            std::process::exit(1);
        }
    }
}

fn print_statistics(ways: &Ways) {
    let n_nodes = ways.n_nodes() as usize;
    let n_ways = ways.n_ways() as usize;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                DATA STATISTICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\n📊 Basic Counts:");
    println!("   Nodes:  {:>12}", n_nodes);
    println!("   Ways:   {:>12}", n_ways);

    // Analyze ways
    let mut foot_ways = 0;
    let mut bike_ways = 0;
    let mut car_ways = 0;
    let mut total_way_length = 0usize;
    let mut named_ways = 0;
    let mut big_streets = 0;
    let mut speed_dist = [0usize; 8];

    for i in 0..n_ways {
        let way_idx = WayIdx(i as u32);
        if let Some(props) = ways.get_way_properties(way_idx) {
            if props.is_foot_accessible() {
                foot_ways += 1;
            }
            if props.is_bike_accessible() {
                bike_ways += 1;
            }
            if props.is_car_accessible() {
                car_ways += 1;
            }
            if props.is_big_street() {
                big_streets += 1;
            }
            speed_dist[props.speed_limit as usize] += 1;
        }

        let nodes = ways.get_way_nodes(way_idx);
        total_way_length += nodes.len();

        if ways.get_way_name(way_idx).is_some() {
            named_ways += 1;
        }
    }

    println!("\n🚶 Way Accessibility:");
    println!(
        "   Foot accessible: {:>8}  ({:>5.1}%)",
        foot_ways,
        100.0 * foot_ways as f64 / n_ways as f64
    );
    println!(
        "   Bike accessible: {:>8}  ({:>5.1}%)",
        bike_ways,
        100.0 * bike_ways as f64 / n_ways as f64
    );
    println!(
        "   Car accessible:  {:>8}  ({:>5.1}%)",
        car_ways,
        100.0 * car_ways as f64 / n_ways as f64
    );

    println!("\n🛣️  Way Properties:");
    println!(
        "   Named ways:      {:>8}  ({:>5.1}%)",
        named_ways,
        100.0 * named_ways as f64 / n_ways as f64
    );
    println!(
        "   Big streets:     {:>8}  ({:>5.1}%)",
        big_streets,
        100.0 * big_streets as f64 / n_ways as f64
    );
    println!(
        "   Avg nodes/way:   {:>12.1}",
        total_way_length as f64 / n_ways as f64
    );

    println!("\n🚗 Speed Limit Distribution:");
    let speed_labels = [
        "10 km/h", "20 km/h", "30 km/h", "40 km/h", "50 km/h", "70 km/h", "90 km/h", "120 km/h",
    ];
    for (i, &count) in speed_dist.iter().enumerate() {
        if count > 0 {
            println!(
                "   {:<10} {:>8}  ({:>5.1}%)",
                speed_labels[i],
                count,
                100.0 * count as f64 / n_ways as f64
            );
        }
    }

    // Node analysis
    let mut total_degree = 0usize;
    let mut max_degree = 0usize;
    let mut isolated = 0;

    for i in 0..n_nodes {
        let node_idx = NodeIdx(i as u32);
        let degree = ways.get_node_ways(node_idx).len();
        total_degree += degree;
        max_degree = max_degree.max(degree);
        if degree == 0 {
            isolated += 1;
        }
    }

    println!("\n🔗 Node Connectivity:");
    let avg_degree = total_degree as f64 / n_nodes as f64;
    println!("   Avg degree:      {:>12.2}", avg_degree);
    println!("   Max degree:      {:>12}", max_degree);
    println!(
        "   Isolated nodes:  {:>12}  ({:>5.1}%)",
        isolated,
        100.0 * isolated as f64 / n_nodes as f64
    );

    // Component analysis
    let mut component_map: std::collections::HashMap<
        transit_cloud_osr::types::ComponentIdx,
        usize,
    > = std::collections::HashMap::new();
    for i in 0..n_ways {
        if let Some(comp) = ways.get_component(WayIdx(i as u32)) {
            *component_map.entry(comp).or_insert(0) += 1;
        }
    }

    let n_components = component_map.len();
    if n_components > 0 {
        let largest = component_map.values().max().copied().unwrap_or(0);
        println!("\n🗺️  Connected Components:");
        println!("   Total components: {:>11}", n_components);
        println!(
            "   Largest component:{:>11} ways ({:>5.1}%)",
            largest,
            100.0 * largest as f64 / n_ways as f64
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

fn print_samples(ways: &Ways) {
    let n_ways = ways.n_ways();

    if n_ways == 0 {
        return;
    }

    println!("Sample Ways (first 5):\n");

    for i in 0..5.min(n_ways) {
        let way_idx = WayIdx(i);
        let nodes = ways.get_way_nodes(way_idx);
        let props = ways.get_way_properties(way_idx);
        let name = ways.get_way_name(way_idx);
        let osm_id = ways.get_way_osm_id(way_idx);

        println!("Way {} (OSM: {:?}):", i, osm_id);
        if let Some(n) = name {
            println!("  Name: {}", n);
        }
        println!("  Nodes: {}", nodes.len());

        if let Some(p) = props {
            println!(
                "  Access: Foot={}, Bike={}, Car={}",
                p.is_foot_accessible(),
                p.is_bike_accessible(),
                p.is_car_accessible()
            );
            println!("  Speed: {} km/h", p.max_speed_km_per_h());

            if p.is_oneway_car() {
                println!("  One-way (car): yes");
            }
            if p.is_oneway_bike() {
                println!("  One-way (bike): yes");
            }
            if p.is_big_street() {
                println!("  Big street: yes");
            }
        }

        // Show first and last nodes
        if nodes.len() > 0 {
            let first_pos = ways.get_node_pos(nodes[0]);
            println!("  Start: ({:.6}, {:.6})", first_pos.lat(), first_pos.lng());

            if nodes.len() > 1 {
                let last_pos = ways.get_node_pos(nodes[nodes.len() - 1]);
                println!("  End:   ({:.6}, {:.6})", last_pos.lat(), last_pos.lng());
            }
        }

        println!();
    }
}
