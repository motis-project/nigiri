# Phase 4 — Street Routing (OSR Integration)

**Priority**: Fifth  
**Type**: Pure Rust (using existing `transit-osr` crate)  
**Depends on**: Phase 0 (types)  
**Crate**: `transit-osr` (exists at `rust/transit-osr/`), integration in `transit-routing`  
**Estimated effort**: Small–Medium (crate already exists, only integration work needed)

## Status: `transit-osr` Crate Already Implemented

The `transit-osr` crate at `rust/transit-osr/` is a **complete pure-Rust** street routing library.
No C++ FFI is needed — this phase focuses solely on integrating the existing crate
with transit routing.

### Existing Capabilities

| Feature | Status | Details |
|---|---|---|
| OSM PBF extraction | **Done** | 3-pass: nodes → ways → restrictions → connected components |
| 13 routing profiles | **Done** | Foot, Wheelchair, Bike (×3), Car (×5), BikeSharing, CarSharing |
| Dijkstra + Bidirectional A* | **Done** | Two algorithms, configurable via `RoutingAlgorithm` |
| Multi-level routing | **Done** | Floor-aware (levels -8.0 to 7.5, elevators) |
| Elevation support | **Done** | `ElevationStorage` with 4-bit compression, DEM/HGT loading |
| Turn restrictions | **Done** | No-turn / only-turn relation support |
| Platform extraction | **Done** | Transit platform matching to street graph |
| GBFS sharing data | **Done** | `SharingData` with start/end/through bitvecs + additional edges |
| Serialization | **Done** | rkyv zero-copy + multi-file format (26 files for ways) |
| HTTP server | **Done** | Optional `http-server` feature with axum |

### Key Types (Already Exist)

```rust
// Main data container — rust/transit-osr/src/data.rs
pub struct OsrData {
    pub ways: Ways,
    pub platforms: Platforms,
    pub elevations: ElevationStorage,
    pub lookup: Lookup,
}

// Routing — rust/transit-osr/src/routing/route.rs
pub fn route(
    ways: &Ways,
    lookup: &Lookup,
    elevations: Option<&ElevationStorage>,
    profile: SearchProfile,
    from: Location,
    to: Location,
    max_cost: Cost,              // seconds
    algorithm: RoutingAlgorithm, // Dijkstra or AStarBi
) -> Option<Path>

// Result — rust/transit-osr/src/routing/path.rs
pub struct Path {
    pub cost: Cost,               // Total seconds
    pub dist: f64,                // Total meters
    pub elevation: Elevation,
    pub segments: Vec<Segment>,
    pub uses_elevator: bool,
}

pub struct Segment {
    pub polyline: Vec<Point>,
    pub from_level: Level,
    pub to_level: Level,
    pub from: NodeIdx,
    pub to: NodeIdx,
    pub way: WayIdx,
    pub cost: Cost,               // seconds
    pub dist: Distance,           // meters
    pub elevation: Elevation,
    pub mode: Mode,
}

// 13 routing profiles — rust/transit-osr/src/profiles.rs
pub enum SearchProfile {
    Foot, Wheelchair,
    Bike, BikeFast, BikeElevationLow, BikeElevationHigh,
    Car, CarDropOff, CarDropOffWheelchair,
    CarParking, CarParkingWheelchair,
    BikeSharing, CarSharing,
}

// Location — rust/transit-osr/src/types.rs
pub struct Location {
    pub pos: Point,
    pub level: Level,
}

impl Location {
    pub fn from_latlng_no_level(lat: f64, lng: f64) -> Self;
}

// Spatial lookup — rust/transit-osr/src/lookup.rs
impl Lookup {
    pub fn match_location(&self, ways: &Ways, loc: Location, profile: SearchProfile)
        -> Vec<WayCandidate>;
    pub fn find_ways_in_bbox(&self, min: Point, max: Point) -> Vec<WayIdx>;
}

// Platform matching — rust/transit-osr/src/platforms.rs
impl Platforms {
    pub fn find(&self, location: Point, max_distance: f64) -> Option<PlatformIdx>;
}

// GBFS sharing — rust/transit-osr/src/sharing_data.rs
pub struct SharingData<'a> {
    pub start_allowed: Option<&'a BitVec>,
    pub end_allowed: Option<&'a BitVec>,
    pub through_allowed: Option<&'a BitVec>,
    pub additional_node_offset: u32,
    pub additional_node_coordinates: &'a [Point],
    pub additional_edges: &'a AHashMap<NodeIdx, Vec<AdditionalEdge>>,
}
```

## What Remains: Integration Work

### 1. OSM Extraction During Import

Wire the `transit_osr::extract::extract()` function into the import pipeline:

```rust
// transit-import/src/osr_import.rs

use transit_osr::extract::extract;
use transit_osr::OsrData;

pub fn extract_street_network(config: &Config) -> Result<OsrData, TransitError> {
    let osm_path = config.osm.as_ref()
        .ok_or(TransitError::Config("no OSM file configured".into()))?;
    let output_path = config.data_path().join("osr");

    extract(
        true,  // with_platforms
        osm_path,
        &output_path,
        config.street_routing.as_ref()
            .and_then(|sr| sr.elevation_data_dir.as_deref()),
    ).map_err(TransitError::Osr)?;

    OsrData::import(&output_path).map_err(TransitError::Osr)
}
```

### 2. First/Last-Mile Transit Offsets

Compute walking/biking/driving times from user coordinates to nearby transit stops
for feeding into RAPTOR as start/destination offsets:

```rust
// transit-routing/src/street.rs

use transit_osr::{routing, OsrData, Location, SearchProfile, RoutingAlgorithm};

pub fn compute_transit_offsets(
    osr: &OsrData,
    rtree: &RTree<LocationEntry>,
    pos: Position,
    modes: &[LegMode],
    max_seconds: u16,
) -> Vec<StopOffset> {
    let mut offsets = Vec::new();
    let from = Location::from_latlng_no_level(pos.lat, pos.lon);

    for &mode in modes {
        let profile = leg_mode_to_profile(mode);
        let radius = max_beeline_radius(profile); // rough filter

        // Pre-filter candidates via R-tree (much faster than routing all)
        let candidates = rtree.locate_within_distance(
            [pos.lon, pos.lat], radius
        );

        for candidate in candidates {
            let to = Location::from_latlng_no_level(candidate.lat, candidate.lon);

            if let Some(path) = routing::route::route(
                &osr.ways,
                &osr.lookup,
                Some(&osr.elevations),
                profile,
                from.clone(),
                to,
                max_seconds as u32,
                RoutingAlgorithm::Dijkstra,
            ) {
                offsets.push(StopOffset {
                    location_idx: candidate.idx,
                    duration_minutes: (path.cost / 60).min(i16::MAX as u32) as i16,
                    mode,
                });
            }
        }
    }

    offsets
}

fn leg_mode_to_profile(mode: LegMode) -> SearchProfile {
    match mode {
        LegMode::Walk => SearchProfile::Foot,
        LegMode::Wheelchair => SearchProfile::Wheelchair,
        LegMode::PersonalBike => SearchProfile::Bike,
        LegMode::BikeFast => SearchProfile::BikeFast,
        LegMode::SharedBike => SearchProfile::BikeSharing,
        LegMode::SharedCar => SearchProfile::CarSharing,
        LegMode::Car => SearchProfile::Car,
        LegMode::CarParking => SearchProfile::CarParking,
        LegMode::CarDropOff => SearchProfile::CarDropOff,
        LegMode::CarParkingWheelchair => SearchProfile::CarParkingWheelchair,
        LegMode::CarDropOffWheelchair => SearchProfile::CarDropOffWheelchair,
    }
}

fn max_beeline_radius(profile: SearchProfile) -> f64 {
    // Conservative beeline radius (meters) per profile.
    // Actual routing may yield longer paths; this is just an R-tree pre-filter.
    match profile {
        SearchProfile::Foot | SearchProfile::Wheelchair => 2_000.0,
        SearchProfile::Bike | SearchProfile::BikeFast
        | SearchProfile::BikeElevationLow | SearchProfile::BikeElevationHigh
        | SearchProfile::BikeSharing => 5_000.0,
        SearchProfile::Car | SearchProfile::CarDropOff
        | SearchProfile::CarDropOffWheelchair | SearchProfile::CarParking
        | SearchProfile::CarParkingWheelchair | SearchProfile::CarSharing => 30_000.0,
    }
}
```

### 3. Direct (Non-Transit) Routing

For walk-only or bike-only journeys without transit:

```rust
// transit-routing/src/street.rs

pub fn route_direct(
    osr: &OsrData,
    from: Position,
    to: Position,
    profile: SearchProfile,
    max_seconds: u32,
) -> Option<DirectRoute> {
    let from_loc = Location::from_latlng_no_level(from.lat, from.lon);
    let to_loc = Location::from_latlng_no_level(to.lat, to.lon);

    routing::route::route(
        &osr.ways,
        &osr.lookup,
        Some(&osr.elevations),
        profile,
        from_loc,
        to_loc,
        max_seconds,
        RoutingAlgorithm::AStarBi, // faster for longer distances
    ).map(|path| DirectRoute {
        cost_seconds: path.cost,
        distance_meters: path.dist,
        elevation: path.elevation,
        segments: path.segments,
        uses_elevator: path.uses_elevator,
    })
}
```

### 4. Polyline Encoding

Convert OSR `Path` segments into an encoded polyline string for GraphQL:

```rust
pub fn path_to_polyline(path: &transit_osr::routing::path::Path) -> String {
    let coords: Vec<(f64, f64)> = path.segments.iter()
        .flat_map(|seg| seg.polyline.iter().map(|p| (p.lat(), p.lng())))
        .collect();
    polyline::encode_coordinates(coords, 5)
}
```

## Data Flow

```
planJourney(from: {lat: -27.47, lon: 153.02}, to: {stopId: "seq:600022"})
  │
  ├─ Origin is coordinates → need street routing
  │
  ├─ compute_transit_offsets(osr, (-27.47, 153.02), [Walk], 900s)
  │   └─ transit_osr::routing::route::route() per candidate
  │   └─ [{stop_42: 5min walk}, {stop_43: 8min walk}, ...]
  │
  ├─ Destination is stop → direct lookup
  │   └─ tags.resolve_location("seq:600022") → LocationIdx(87)
  │
  ├─ nigiri_route_offsets(tt, start_offsets, [dest:87], options)
  │   └─ RAPTOR with pre-transit walking offsets
  │
  └─ Format legs:
      WalkLeg(origin→stop_42, polyline from osr)
      + TransitLeg(stop_42→stop_87, shapes from nigiri)
```

## Acceptance Criteria

1. `transit_osr::extract::extract()` processes OSM PBF into loadable data
2. `OsrData::import()` loads the extracted street network
3. Walking offset computation finds nearby transit stops and returns durations
4. Direct walk/bike/car routing returns path with polyline
5. All 13 profiles work (foot, wheelchair, bike variants, car variants, sharing)
6. Elevation-aware routing produces correct uphill/downhill costs
7. `planJourney` from coordinates correctly finds walking connections to transit

## motis Source Reference (integration patterns)

| File | Lines | Pattern to Mirror |
|---|---|---|
| `src/osr/street_routing.cc` | ~600 | How offsets are computed and passed to RAPTOR |
| `src/osr/mode_to_profile.cc` | ~100 | Mode → profile mapping table |
| `src/osr/max_distance.cc` | ~80 | Max distance thresholds per profile |
| `src/endpoints/routing.cc` | ~800 | How start/dest offsets feed into routing query |
