//! OSR HTTP Server
//!
//! A lightweight HTTP API for OSR routing queries.
//!
//! ## Usage
//! ```bash
//! cargo run --release --bin osr-server --features http-server -- --data ./extracted_local_with_elevation
//! ```
//!
//! ## API Endpoints
//! - `POST /api/route` - Calculate route between two points
//! - `GET /health` - Health check

use anyhow::{Context, Result};
use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::Deserialize;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};
use transit_cloud_osr::{
    elevation_storage::ElevationStorage,
    geojson::path_to_geojson,
    lookup::Lookup,
    routing::{route, RoutingAlgorithm, SearchProfile},
    ways::Ways,
    Level, Location,
};

/// Server configuration from command line
#[derive(clap::Parser, Debug)]
#[command(name = "osr-server")]
#[command(about = "OSR HTTP routing server", long_about = None)]
struct Config {
    /// Path to extracted OSR data directory
    #[arg(short, long, default_value = "./extracted")]
    data: PathBuf,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind to
    #[arg(short, long, default_value = "8080")]
    port: u16,
}

/// Shared application state
struct AppState {
    ways: &'static Ways,
    lookup: &'static Lookup,
    elevations: Option<ElevationStorage>,
}

/// Location with lat/lng coordinates
#[derive(Debug, Deserialize)]
struct LocationDto {
    lat: f64,
    #[serde(alias = "lng", alias = "lon")]
    lng: f64,
    #[serde(default)]
    level: Option<f32>,
}

impl From<LocationDto> for Location {
    fn from(loc: LocationDto) -> Self {
        Location::from_latlng(
            loc.lat,
            loc.lng,
            loc.level.map(Level::from_float).unwrap_or_default(),
        )
    }
}

/// Routing profile enum
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ProfileType {
    Foot,
    Wheelchair,
    Bike,
    BikeSafe,
    BikeFast,
    BikeElevationLow,
    BikeElevationHigh,
    Car,
}

/// Route request query parameters
#[derive(Debug, Deserialize)]
struct RouteQuery {
    /// Start coordinates as "lat,lng" or "lat,lng,level"
    start: String,
    /// Destination coordinates as "lat,lng" or "lat,lng,level"
    dest: String,
    #[serde(default = "default_profile")]
    profile: ProfileType,
    #[serde(default = "default_max_cost")]
    max: u16,
}

/// Parse comma-separated coordinates "lat,lng" or "lat,lng,level"
fn parse_coords(s: &str) -> Result<(f64, f64, Option<f32>), String> {
    let parts: Vec<&str> = s.split(',').collect();
    match parts.len() {
        2 => {
            let lat = parts[0]
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid latitude: {}", parts[0]))?;
            let lng = parts[1]
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid longitude: {}", parts[1]))?;
            Ok((lat, lng, None))
        }
        3 => {
            let lat = parts[0]
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid latitude: {}", parts[0]))?;
            let lng = parts[1]
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid longitude: {}", parts[1]))?;
            let level = parts[2]
                .trim()
                .parse::<f32>()
                .map_err(|_| format!("Invalid level: {}", parts[2]))?;
            Ok((lat, lng, Some(level)))
        }
        _ => Err(format!(
            "Expected 2 or 3 comma-separated values, got {}",
            parts.len()
        )),
    }
}

fn default_profile() -> ProfileType {
    ProfileType::Foot
}

fn default_max_cost() -> u16 {
    3600 // 1 hour default
}

/// Error response type
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let message = format!("{{\"error\": \"{}\"}}", self.0);
        (StatusCode::INTERNAL_SERVER_ERROR, message).into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

/// Health check endpoint
async fn health() -> &'static str {
    "OK"
}

/// Route endpoint handler
async fn handle_route(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RouteQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Parse start coordinates
    let (start_lat, start_lng, start_level) = parse_coords(&params.start)
        .map_err(|e| AppError(anyhow::anyhow!("Invalid start coordinates: {}", e)))?;

    // Parse destination coordinates
    let (dest_lat, dest_lng, dest_level) = parse_coords(&params.dest)
        .map_err(|e| AppError(anyhow::anyhow!("Invalid dest coordinates: {}", e)))?;

    info!(
        "Route request: ({}, {}) -> ({}, {}) (profile: {:?})",
        start_lat, start_lng, dest_lat, dest_lng, params.profile
    );

    let start = Location::from_latlng(
        start_lat,
        start_lng,
        start_level.map(Level::from_float).unwrap_or_default(),
    );
    let dest = Location::from_latlng(
        dest_lat,
        dest_lng,
        dest_level.map(Level::from_float).unwrap_or_default(),
    );
    let max_cost = params.max;

    info!("Start location: {:?}", start);
    info!("Dest location: {:?}", dest);

    // Convert to SearchProfile
    let search_profile = match params.profile {
        ProfileType::Foot => SearchProfile::Foot,
        ProfileType::Wheelchair => SearchProfile::Wheelchair,
        ProfileType::Bike | ProfileType::BikeSafe => SearchProfile::Bike,
        ProfileType::BikeFast => SearchProfile::BikeFast,
        ProfileType::BikeElevationLow => SearchProfile::BikeElevationLow,
        ProfileType::BikeElevationHigh => SearchProfile::BikeElevationHigh,
        ProfileType::Car => SearchProfile::Car,
    };

    // Check lookup
    let start_matches = state
        .lookup
        .match_location(state.ways, search_profile, &start, 200.0);
    let dest_matches = state
        .lookup
        .match_location(state.ways, search_profile, &dest, 200.0);
    info!(
        "Start candidates: {}, Dest candidates: {}",
        start_matches.len(),
        dest_matches.len()
    );

    // Route with Dijkstra algorithm
    let path = route(
        state.ways,
        &state.lookup,
        state.elevations.as_ref(),
        search_profile,
        start,
        dest,
        max_cost,
        RoutingAlgorithm::Dijkstra,
    );

    match path {
        Some(p) => {
            info!(
                "Found route: cost={}, segments={}",
                p.cost,
                p.segments.len()
            );
            Ok(Json(path_to_geojson(state.ways, &p, true).to_value()))
        }
        None => {
            warn!("No route found");
            Err(AppError(anyhow::anyhow!("No route found")))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    use clap::Parser;

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "osr_server=info,tower_http=debug".into()),
        )
        .init();

    let config = Config::parse();

    info!("Loading OSR data from: {:?}", config.data);

    // Load ways (Box it to get a stable address)
    let ways = Box::new(Ways::load(&config.data).context("Failed to load ways")?);
    info!("Loaded ways data");

    // Get a 'static reference by leaking the Box (acceptable for long-lived server)
    let ways_ref: &'static Ways = Box::leak(ways);

    // Load lookup with static reference
    let lookup = Lookup::load(ways_ref, &config.data).context("Failed to load lookup")?;
    info!("Loaded lookup index");
    lookup.check_rtree();
    let lookup_ref: &'static Lookup = Box::leak(Box::new(lookup));

    // Try to load elevations
    let elevations = ElevationStorage::load(&config.data).ok();
    if elevations.is_some() {
        info!("Loaded elevation data");
    } else {
        warn!("No elevation data found - routing without elevation costs");
    }

    // Create shared state
    let state = Arc::new(AppState {
        ways: ways_ref,
        lookup: lookup_ref,
        elevations,
    });

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/api/route", get(handle_route))
        .layer(cors)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state.clone());

    // Bind and serve
    let addr = SocketAddr::from((config.host.parse::<std::net::IpAddr>()?, config.port));

    info!("Starting server on http://{}", addr);
    info!("Health check: http://{}/health", addr);
    info!(
        "Route API: http://{}/api/route?start=-27.466644,153.035175&dest=-27.465877,153.027467",
        addr
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // // Test route call
    // let test_result = handle_route(
    //     State(state),
    //     Json(RouteRequest {
    //         start: LocationDto {
    //             lat: -27.466644,
    //             lng: 153.035175,
    //             level: None,
    //         },
    //         destination: LocationDto {
    //             lat: -27.465877,
    //             lng: 153.027467,
    //             level: None,
    //         },
    //         profile: ProfileType::Foot,
    //         max: 36000,
    //     })
    // ).await;

    // match test_result {
    //     Ok(response) => info!("Test route succeeded: {} features", response.0.features.len()),
    //     Err(e) => warn!("Test route failed: {}", e.0),
    // }

    axum::serve(listener, app).await?;

    Ok(())
}
