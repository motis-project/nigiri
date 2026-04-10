//! OSM extraction binary
//!
//! Extracts routing data from OpenStreetMap PBF files.
//!
//! # Usage
//! ```bash
//! cargo run --bin extract -- input.osm.pbf output_dir
//! cargo run --bin extract -- --with-platforms input.osm.pbf output_dir
//! cargo run --bin extract -- --with-platforms --elevation /path/to/dem input.osm.pbf output_dir
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

/// OSM extraction tool
#[derive(Parser, Debug)]
#[command(name = "osr-extract")]
#[command(about = "Extract routing data from OpenStreetMap PBF files", long_about = None)]
struct Args {
    /// Input OSM PBF file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output directory for extracted data
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,

    /// Extract platform/station data for transit integration
    #[arg(long)]
    with_platforms: bool,

    /// Optional elevation data directory (DEM/HGT tiles)
    #[arg(long, value_name = "DIR")]
    elevation: Option<PathBuf>,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    println!("OSR Extract Tool");
    println!("================");
    println!("Input:  {}", args.input.display());
    println!("Output: {}", args.output.display());
    if args.with_platforms {
        println!("Including platforms/stations");
    }
    if let Some(ref elev_dir) = args.elevation {
        println!("Elevation: {}", elev_dir.display());
    }
    println!();

    let start = Instant::now();

    match transit_cloud_osr::extract::extract(
        args.with_platforms,
        &args.input,
        &args.output,
        args.elevation.as_deref(),
    ) {
        Ok(()) => {
            let elapsed = start.elapsed();
            println!();
            println!("✓ Extraction complete in {:.2}s", elapsed.as_secs_f64());
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!();
            eprintln!("✗ Extraction failed: {}", e);
            std::process::exit(1);
        }
    }
}
