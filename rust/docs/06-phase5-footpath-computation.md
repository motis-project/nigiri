# Phase 5 — Footpath Computation

**Priority**: Sixth  
**Type**: Rust orchestration using `transit-osr` crate + nigiri FFI  
**Depends on**: Phase 1 (timetable), Phase 4 (OSR integration)  
**Crate**: `transit-import` (module)  
**Estimated effort**: Medium

## Objective

Compute accurate transfer footpaths between transit stops using the `transit-osr` crate's
street routing, replacing nigiri's default beeline-distance estimates with real walking
times that account for street layout, elevation, and accessibility barriers.

## What motis Does (`src/compute_footpaths.cc`, ~800 LOC)

### Overview

By default, nigiri computes footpaths using beeline (straight-line) distances.
motis replaces these with `transit-osr`-routed walking times that account for:

- Actual street network (no walking through buildings)
- Elevation changes (stairways, hills)
- Accessibility barriers (elevators, wheelchair-inaccessible paths)
- Platform matching (snap stops to the correct street network position)

### Algorithm

```
For each routing profile (foot, wheelchair):
    1. Build R-tree of candidate stops
    2. For each origin stop:
        a. Match stop to street node via osr platform matching
        b. Query R-tree for neighbors within max_radius
        c. transit_osr::routing::route::route(origin, neighbor, profile)
        d. If path.cost ≤ max_footpath_duration: update footpath
        e. If path.uses_elevator: record in ElevatorFootpaths
    3. Update timetable footpaths via nigiri ABI
```

### Key Parameters (from motis config)

```rust
pub struct FootpathConfig {
    pub max_duration_seconds: u32,        // default 900 (15 min)
    pub max_matching_distance: f64,       // default 25.0 m
    pub adjust_footpaths: bool,           // enable/disable OSR routing
    pub profiles: Vec<SearchProfile>,     // [Foot, Wheelchair]
}
```

## Rust Implementation

### Platform Matching

Before computing footpaths, snap each transit stop to its closest OSR platform node.
Uses `transit_osr::platforms::Platforms::find()`:

```rust
// transit-import/src/platform_matching.rs

use transit_osr::OsrData;
use transit_osr::types::Point;

pub struct PlatformMatch {
    pub platform_idx: transit_osr::types::PlatformIdx,
    pub snapped_pos: Position,
    pub distance: f64,
}

pub fn match_platforms(
    tt: &nigiri::Timetable,
    osr: &OsrData,
    max_distance: f64,
) -> HashMap<LocationIdx, PlatformMatch> {
    let mut matches = HashMap::new();

    for loc_idx in tt.enumerate_stops() {
        let loc = tt.get_location(loc_idx).unwrap();
        let point = Point::from_latlng(loc.lat, loc.lon);

        if let Some(platform_idx) = osr.platforms.find(point, max_distance) {
            let platform_pos = osr.platforms.get_position(platform_idx);
            matches.insert(loc_idx, PlatformMatch {
                platform_idx,
                snapped_pos: Position {
                    lat: platform_pos.lat(),
                    lon: platform_pos.lng(),
                },
                distance: point.distance_to(&platform_pos),
            });
        }
    }

    matches
}
```

### Footpath Computer

```rust
// transit-import/src/footpaths.rs

use transit_osr::{routing, OsrData, Location, SearchProfile, RoutingAlgorithm};

pub struct FootpathComputer<'a> {
    tt: &'a mut nigiri::Timetable,
    osr: &'a OsrData,
    rtree: &'a RTree<LocationEntry>,
    config: &'a FootpathConfig,
}

impl<'a> FootpathComputer<'a> {
    /// Compute routed footpaths for all stops
    pub fn compute(&mut self) -> Result<ElevatorFootpaths, TransitError> {
        // 1. Match all stops to OSR platforms
        let matches = match_platforms(
            self.tt, self.osr, self.config.max_matching_distance
        );

        let mut elevator_fps = ElevatorFootpaths::new();

        // 2. Compute footpaths for each profile
        for &profile in &self.config.profiles {
            self.compute_for_profile(profile, &matches, &mut elevator_fps)?;
        }

        Ok(elevator_fps)
    }

    fn compute_for_profile(
        &mut self,
        profile: SearchProfile,
        matches: &HashMap<LocationIdx, PlatformMatch>,
        elevator_fps: &mut ElevatorFootpaths,
    ) -> Result<(), TransitError> {
        let stops = self.tt.enumerate_stops();

        for &origin_idx in &stops {
            let origin_loc = self.stop_to_location(origin_idx, matches);

            // Pre-filter via R-tree with generous beeline radius
            let radius = beeline_radius_for_duration(
                profile, self.config.max_duration_seconds
            );
            let candidates = self.rtree.locate_within_distance(
                [origin_loc.pos.lng(), origin_loc.pos.lat()],
                radius,
            );

            for candidate in candidates {
                if candidate.idx == origin_idx {
                    continue;
                }

                let dest_loc = self.stop_to_location(candidate.idx, matches);

                // Route via osr
                if let Some(path) = routing::route::route(
                    &self.osr.ways,
                    &self.osr.lookup,
                    Some(&self.osr.elevations),
                    profile,
                    origin_loc.clone(),
                    dest_loc,
                    self.config.max_duration_seconds,
                    RoutingAlgorithm::Dijkstra,
                ) {
                    let duration_mins = (path.cost / 60).min(i16::MAX as u32) as i16;

                    // Update nigiri footpath
                    self.tt.set_footpath(
                        origin_idx, candidate.idx, duration_mins
                    )?;

                    // Track elevator-dependent footpaths
                    if path.uses_elevator {
                        elevator_fps.add(origin_idx, candidate.idx);
                    }
                }
            }
        }

        Ok(())
    }

    fn stop_to_location(
        &self,
        idx: LocationIdx,
        matches: &HashMap<LocationIdx, PlatformMatch>,
    ) -> Location {
        if let Some(m) = matches.get(&idx) {
            // Use platform-matched position
            Location::from_latlng_no_level(m.snapped_pos.lat, m.snapped_pos.lon)
        } else {
            // Use original stop coordinates
            let loc = self.tt.get_location(idx).unwrap();
            Location::from_latlng_no_level(loc.lat, loc.lon)
        }
    }
}

fn beeline_radius_for_duration(profile: SearchProfile, max_seconds: u32) -> f64 {
    // Conservative speed assumptions for beeline pre-filter (m/s)
    let speed = match profile {
        SearchProfile::Foot | SearchProfile::Wheelchair => 1.5,
        SearchProfile::Bike | SearchProfile::BikeFast
        | SearchProfile::BikeElevationLow | SearchProfile::BikeElevationHigh
        | SearchProfile::BikeSharing => 5.0,
        _ => 15.0, // car
    };
    speed * max_seconds as f64 * 1.2 // 20% buffer
}
```

### Elevator Footpath Tracking

```rust
// transit-core/src/elevator_footpaths.rs

/// Tracks which footpaths require a working elevator.
/// Used in Phase 10 to adjust footpath availability based on elevator status.
pub struct ElevatorFootpaths {
    pairs: HashSet<(LocationIdx, LocationIdx)>,
}

impl ElevatorFootpaths {
    pub fn new() -> Self {
        Self { pairs: HashSet::new() }
    }

    pub fn add(&mut self, from: LocationIdx, to: LocationIdx) {
        self.pairs.insert((from, to));
    }

    pub fn requires_elevator(&self, from: LocationIdx, to: LocationIdx) -> bool {
        self.pairs.contains(&(from, to))
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }
}
```

## nigiri C ABI Extensions Needed

```c
// Update footpath duration between two stops
bool nigiri_set_footpath(
    nigiri_timetable_t* t,
    uint32_t from_location_idx,
    uint32_t to_location_idx,
    int16_t duration_minutes
);

// Add a new footpath (if no existing one)
bool nigiri_add_footpath(
    nigiri_timetable_t* t,
    uint32_t from_location_idx,
    uint32_t to_location_idx,
    int16_t duration_minutes
);

// Enumerate all locations that are stops
typedef void (*nigiri_stop_cb)(uint32_t location_idx, void* context);
uint32_t nigiri_enumerate_stops(
    nigiri_timetable_t* t,
    nigiri_stop_cb cb,
    void* context
);
```

## Integration with Import Pipeline

```rust
// In transit-import::ImportPipeline::run():

// After loading timetable and OSR data:
if config.footpaths.adjust_footpaths {
    if let Some(ref osr) = data.osr {
        let mut computer = FootpathComputer {
            tt: &mut data.tt,
            osr,
            rtree: &data.location_rtree,
            config: &config.footpaths,
        };
        data.elevator_footpaths = Some(computer.compute()?);
        data.platform_matches = Some(match_platforms(
            &data.tt, osr, config.footpaths.max_matching_distance,
        ));
    }
}
```

## Performance Considerations

- **N² problem**: For S stops, naive approach routes S×S pairs. R-tree pre-filter
  reduces this to S × ~20 candidates (within beeline radius).
- **Parallelism**: Origin stops can be processed in parallel using rayon. Each
  `route()` call is independent. Footpath updates must be synchronized.
- **Caching**: OSR's Dijkstra already uses a node-local priority queue. No additional
  caching needed.
- **Batch approach**: Could also use `transit-osr`'s one-to-many from a single origin to
  reduce overhead, if the crate supports it.

## Acceptance Criteria

1. Platform matching snaps stops to correct street network positions
2. Footpath durations reflect actual walking routes (not beeline)
3. Footpaths exceeding max duration are excluded
4. Elevator-dependent footpaths are tracked for Phase 10
5. Routing results use routed footpaths for transfers
6. Performance: footpath computation completes in reasonable time for thousands of stops

## motis Source Reference

| File | Lines | Pattern to Mirror |
|---|---|---|
| `src/compute_footpaths.cc` | ~800 | `compute_footpaths()`, profile iteration, R-tree filter |
| `src/match_platforms.cc` | ~250 | `match_platforms()`, scoring algorithm |
