# Phase 0 — Foundation: Types, Config & GraphQL Scaffold

**Priority**: First  
**Type**: Pure Rust  
**Depends on**: Nothing  
**Crate**: `transit-core`, `transit-server`  
**Estimated effort**: Medium

## Objective

Establish the shared type system, configuration model, and GraphQL server skeleton that every subsequent phase builds on.

## What motis Does

### Config (`include/motis/config.h`)

motis uses a deeply nested struct with `reflect-cpp` for YAML deserialization. Key sections:

```
config
├── server: { host, port, web_folder, n_threads, data_attribution_link, lbs }
├── osm: PathBuf (for street routing)
├── tiles: { profile, coastline, db_size, flush_threshold }
├── timetable
│   ├── first_day: "TODAY" or YYYY-MM-DD
│   ├── num_days: 365
│   ├── tb, railviz, with_shapes, adjust_footpaths: bool
│   ├── merge_dupes_intra_src, merge_dupes_inter_src: bool
│   ├── link_stop_distance: 100m
│   ├── update_interval: 60s
│   ├── http_timeout: 30s
│   ├── max_footpath_length: 15min
│   ├── max_matching_distance: 25.0m
│   ├── datasets: Map<String, Dataset>
│   │   └── Dataset: { path, script?, default_bikes/cars_allowed, rt: [{ url, headers, protocol }] }
│   └── route_shapes: { mode, cache settings, debug options }
├── gbfs: { feeds: Map<id, Feed>, groups, restrictions, ttl, proxy, cache_size }
├── prima: { url, bounds, ride_sharing_bounds }
├── elevators: { url, init, osm_mapping, http_timeout, headers }
├── street_routing: { elevation_data_dir }
├── limits: { stoptimes_max_results, plan_max_results, routing_max_timeout_seconds, ... }
└── logging: { log_level }
```

### Data Layer (`include/motis/data.h`)

Central struct holding all loaded state. For Rust we model this as `AppData`:

- **Immutable after load**: timetable, OSR ways/lookup, tags, shapes, flex areas, matches
- **Mutable via atomic swap**: `rt` (rt_timetable + elevators + railviz)
- **Shared**: `gbfs_data` behind `Arc<RwLock<>>`

### Types (`include/motis/fwd.h`, `include/motis/types.h`)

Strong index types (newtypes over `u32`):
- `location_idx_t`, `route_idx_t`, `transport_idx_t`, `trip_idx_t`, `source_idx_t`
- `day_idx_t`, `minutes_after_midnight_t`
- `transit_osr::node_idx_t`, `transit_osr::way_idx_t`, `transit_osr::platform_idx_t`

## Implementation Plan

### 1. `transit-core` Crate

**Rust strong index types** — newtype wrappers with arithmetic, Display, serde:

```rust
/// Macro for generating index newtypes
macro_rules! define_idx {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
        // impl Display, From<u32>, Into<u32>, serde::Serialize/Deserialize
    };
}

define_idx!(LocationIdx);
define_idx!(RouteIdx);
define_idx!(TransportIdx);
define_idx!(TripIdx);
define_idx!(SourceIdx);
define_idx!(DayIdx);
```

**Configuration model** — serde-based, maps 1:1 to motis config but drops `server` (we have our own) and `tiles`:

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub osm: Option<PathBuf>,
    pub timetable: Option<TimetableConfig>,
    pub gbfs: Option<GbfsConfig>,
    pub prima: Option<PrimaConfig>,
    pub elevators: Option<ElevatorConfig>,
    pub street_routing: Option<StreetRoutingConfig>,
    pub limits: Option<LimitsConfig>,
    pub logging: Option<LoggingConfig>,
    // Our own GraphQL server config
    pub server: ServerConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,           // default "0.0.0.0"
    pub port: u16,              // default 8080
    pub n_threads: Option<usize>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TimetableConfig {
    pub first_day: String,      // "TODAY" or YYYY-MM-DD
    pub num_days: u16,          // default 365
    pub with_shapes: bool,
    pub adjust_footpaths: bool,
    pub merge_dupes_intra_src: bool,
    pub merge_dupes_inter_src: bool,
    pub link_stop_distance: u32,
    pub update_interval: u32,   // seconds
    pub max_footpath_length: u16,
    pub max_matching_distance: f64,
    pub datasets: HashMap<String, DatasetConfig>,
    pub route_shapes: Option<RouteShapesConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatasetConfig {
    pub path: String,
    pub script: Option<String>,
    pub default_bikes_allowed: bool,
    pub default_cars_allowed: bool,
    pub extend_calendar: bool,
    pub rt: Option<Vec<RtFeedConfig>>,
    pub default_timezone: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RtFeedConfig {
    pub url: String,
    pub headers: Option<HashMap<String, String>>,
    pub protocol: RtProtocol,  // gtfsrt | auser | siri | siri_json
}

#[derive(Debug, Deserialize, Clone)]
pub enum RtProtocol {
    #[serde(rename = "gtfsrt")] GtfsRt,
    #[serde(rename = "auser")]  Auser,
    #[serde(rename = "siri")]   Siri,
    #[serde(rename = "siri_json")] SiriJson,
}
```

**Limits config:**

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct LimitsConfig {
    pub stoptimes_max_results: u32,     // 256
    pub plan_max_results: u32,          // 256
    pub plan_max_search_window_minutes: u32, // 5760
    pub stops_max_results: u32,         // 2048
    pub onetomany_max_many: u32,        // 128
    pub onetoall_max_results: u32,      // 65535
    pub onetoall_max_travel_minutes: u32, // 90
    pub routing_max_timeout_seconds: u32, // 90
}
```

**Geo types:**

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Position {
    pub lat: f64,
    pub lon: f64,
}

pub struct BoundingBox {
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
}
```

**Error type:**

```rust
#[derive(Debug, thiserror::Error)]
pub enum TransitError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid ID format: {0}")]
    InvalidId(String),
    #[error("nigiri error: {0}")]
    Nigiri(String),
    #[error("osr error: {0}")]
    Osr(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
```

### 2. `transit-server` Crate (Skeleton)

GraphQL server using `async-graphql` + `axum`:

```rust
use async_graphql::{Schema, EmptyMutation, EmptySubscription};

pub struct QueryRoot;

#[async_graphql::Object]
impl QueryRoot {
    // Phase 0: just a health check
    async fn health(&self) -> bool { true }
    
    // Stubs for all queries from schema.generated.graphql
    // Filled in by subsequent phases
}

pub type AppSchema = Schema<QueryRoot, EmptyMutation, EmptySubscription>;
```

Server wiring:

```rust
// transit-app/src/main.rs
#[tokio::main]
async fn main() {
    let config = Config::load("config.yaml")?;
    
    // Phase 1+ will populate AppData
    let data = AppData::new(config).await?;
    
    let schema = Schema::build(QueryRoot, EmptyMutation, EmptySubscription)
        .data(data)
        .finish();
    
    let app = Router::new()
        .route("/graphql", post(graphql_handler))
        .route("/graphql", get(graphql_playground));
    
    axum::serve(listener, app).await?;
}
```

### 3. `AppData` Struct (Skeleton)

```rust
pub struct AppData {
    pub config: Config,
    pub data_path: PathBuf,
    
    // Phase 1: Timetable
    pub tt: Option<nigiri::Timetable>,
    
    // Phase 2: Tags
    pub tags: Option<TagLookup>,
    
    // Phase 4: OSR
    pub osr: Option<OsrHandle>,
    
    // Phase 6: RT
    pub rt: Arc<RwLock<Option<RtState>>>,
    
    // Phase 7: GBFS
    pub gbfs: Arc<RwLock<Option<GbfsData>>>,
    
    // Phase 8: Flex
    pub flex_areas: Option<FlexAreas>,
    
    // Phase 10: Elevators
    pub elevators: Option<Elevators>,
    
    // Phase 11: Geocoder
    pub geocoder: Option<Box<dyn Geocoder>>,
    
    // Phase 12: Shapes
    pub shapes: Option<ShapesStorage>,
    
    // Spatial index
    pub location_rtree: Option<RTree<LocationIdx>>,
}
```

## GraphQL Schema Types (from `schema.generated.graphql`)

Map these to `async_graphql` Rust types:

| GraphQL Type | Rust Struct | Phase Implemented |
|---|---|---|
| `Position` | `Position { lat, lon }` | 0 |
| `Color` | `RouteColor { background, text }` | 0 |
| `PageInfo` | `PageInfo { page, page_size, total_items, total_pages }` | 0 |
| `Stop` | `GqlStop` | 1 |
| `Station` | `GqlStation` | 1 |
| `Route` | `GqlRoute` | 1 |
| `RouteReference` | `GqlRouteRef` | 1 |
| `StopReference` | `GqlStopRef` | 1 |
| `Trip` | `GqlTrip` | 3 |
| `Journey` | `GqlJourney` | 3/13 |
| `TransitLeg` / `WalkLeg` | `GqlLeg` (union) | 3/13 |
| `Departure` | `GqlDeparture` | 3 |
| `ServiceAlert` | `GqlServiceAlert` | 6 |
| `Vehicle` | `GqlVehicle` | 6 |
| `Itinerary` | `GqlItinerary` | 3 |
| `Address` | `GqlAddress` | 11 |
| `Poi` | `GqlPoi` | 11 |

## Rust Crate Dependencies (Phase 0)

```toml
# transit-core/Cargo.toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
thiserror = "2"
chrono = { version = "0.4", features = ["serde"] }
chrono-tz = "0.10"

# transit-server/Cargo.toml
[dependencies]
async-graphql = "7"
async-graphql-axum = "7"
axum = "0.8"
tokio = { version = "1", features = ["full"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "trace"] }
tracing = "0.1"
tracing-subscriber = "0.3"
```

## Acceptance Criteria

1. `cargo build` succeeds for `transit-core` and `transit-server`
2. Config loads from YAML file with all timetable/gbfs/limits/server fields
3. GraphQL playground accessible at `http://localhost:8080/graphql`
4. `{ health }` query returns `true`
5. All shared types (Position, Color, PageInfo, enums) compile and serialize

## Files Created

| File | Purpose |
|---|---|
| `rust/transit-core/src/lib.rs` | Re-exports modules |
| `rust/transit-core/src/types.rs` | Index newtypes, geo types |
| `rust/transit-core/src/config.rs` | Config structs with serde |
| `rust/transit-core/src/error.rs` | Error enum |
| `rust/transit-server/src/main.rs` | Server entry point |
| `rust/transit-server/src/schema.rs` | GraphQL type definitions |
| `rust/transit-server/src/app_data.rs` | AppData struct |

## motis Source Reference

| File | Lines | Relevance |
|---|---|---|
| `include/motis/config.h` | ~600 | Full config schema — port field-by-field |
| `include/motis/data.h` | ~300 | Data struct layout — mirror in AppData |
| `include/motis/fwd.h` | ~80 | Type universe — all strong index types |
| `include/motis/types.h` | ~50 | Type aliases, hash maps, vector maps |
