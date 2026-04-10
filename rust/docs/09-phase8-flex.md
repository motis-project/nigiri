# Phase 8 — Flex (Demand-Responsive Transit)

**Priority**: Ninth  
**Type**: Pure Rust (using `transit-osr` + nigiri FFI)  
**Depends on**: Phase 1 (timetable), Phase 4 (OSR), Phase 7 (GBFS compression utils)  
**Crate**: `transit-flex`  
**Estimated effort**: Medium–Large

## Objective

Support demand-responsive (flex) transit services where vehicles serve areas or location groups rather than fixed stop sequences. Flex transports allow boarding/alighting within geographic zones or at any stop in a location group, with time windows per stop.

## What motis Does (`src/flex/`, ~600+ LOC)

### Overview

Flex transports in GTFS are encoded in nigiri's timetable with:
- **Flex areas** — GeoJSON polygons defining service zones (stored as outer/inner rings)
- **Location groups** — Sets of timetable locations that flex vehicles serve
- **Time windows** — Per-stop boarding/alighting time ranges
- **Traffic day bitfields** — Which days each flex transport operates

motis computes time-dependent offsets by:
1. Finding flex transports reachable from the query location
2. Building OSR `SharingData` to restrict routing within flex areas
3. Computing OSR path costs from query position to each flex stop
4. Intersecting path costs with stop time windows to produce td_offsets

### Key Encoding: `mode_id` Bitfield

Flex transports are identified by a packed 32-bit ID:

```rust
pub struct FlexModeId(u32);

impl FlexModeId {
    // Bit layout: [MSB=1] [flex_transport_idx: 23 bits] [direction: 1 bit] [stop_idx: 7 bits]
    pub fn new(transport_idx: u32, direction: Direction, stop_idx: u8) -> Self;
    pub fn transport_idx(&self) -> u32;
    pub fn direction(&self) -> Direction;
    pub fn stop_idx(&self) -> u8;
    pub fn is_flex(id: u32) -> bool; // MSB set = flex
}
```

## Rust Implementation

### Flex Area Geometry

```rust
// transit-flex/src/areas.rs

use geo::{MultiPolygon, Contains};
use bitvec::prelude::*;

pub struct FlexAreas {
    /// Per-area bitvector of OSR nodes within the area
    area_nodes: Vec<CompressedBitVec>,
    /// Area geometries (loaded from timetable flex area rings)
    geometries: Vec<MultiPolygon<f64>>,
}

impl FlexAreas {
    /// Build area node bitvectors from timetable flex areas + OSR street network.
    /// Uses rayon for parallel polygon-containment tests.
    pub fn build(
        tt: &nigiri::Timetable,
        osr: &transit_osr::OsrData,
    ) -> Self {
        let areas = tt.flex_areas(); // outer/inner ring coordinates
        let geometries: Vec<MultiPolygon<f64>> = areas.iter()
            .map(|a| build_multipolygon(&a.outer_rings, &a.inner_rings))
            .collect();

        let node_count = osr.ways.node_count();

        // Parallel: for each area, find all OSR nodes inside
        let area_nodes: Vec<CompressedBitVec> = geometries.par_iter()
            .map(|geom| {
                let mut bv = BitVec::repeat(false, node_count);
                let bbox = geom.bounding_rect().unwrap();
                for node_idx in osr.lookup.find_nodes_in_bbox(bbox) {
                    let pos = osr.ways.node_position(node_idx);
                    if geom.contains(&geo::Point::new(pos.lng(), pos.lat())) {
                        bv.set(node_idx.0 as usize, true);
                    }
                }
                compress_bitvec(&bv)
            })
            .collect();

        FlexAreas { area_nodes, geometries }
    }

    pub fn is_in_area(&self, area_idx: u32, node_idx: u32) -> bool {
        let bv = decompress_bitvec(&self.area_nodes[area_idx as usize]);
        bv[node_idx as usize]
    }
}
```

### Flex Routing Discovery

```rust
// transit-flex/src/discovery.rs

use nigiri::Timetable;

pub struct FlexMatch {
    pub transport_idx: u32,
    pub stop_seq_idx: u8,
    pub direction: Direction,
    pub mode_id: FlexModeId,
}

/// Find flex transports reachable from a position within a time interval.
pub fn discover_flex_transports(
    tt: &Timetable,
    flex_areas: &FlexAreas,
    rtree: &RTree<LocationEntry>,
    pos: Position,
    interval: DateTimeInterval,
    direction: Direction,
) -> Vec<FlexMatch> {
    let mut matches = Vec::new();

    // 1. Find nearby flex areas via R-tree on area bounding boxes
    let nearby_areas = rtree.locate_within_distance(
        [pos.lon, pos.lat], MAX_FLEX_RADIUS
    );

    // 2. For each flex transport operating on queried days
    for transport in tt.flex_transports() {
        if !transport.active_on(interval.date()) {
            continue; // traffic day check
        }

        for (stop_idx, stop) in transport.stops().enumerate() {
            // Skip last stop (no onward journey from terminal)
            if direction == Direction::Forward && stop_idx == transport.stop_count() - 1 {
                continue;
            }

            // Check if query position is within this stop's service area
            let in_area = match stop.location_type() {
                FlexStopType::Area(area_idx) => {
                    flex_areas.geometries[area_idx as usize]
                        .contains(&geo::Point::new(pos.lon, pos.lat))
                }
                FlexStopType::LocationGroup(group) => {
                    // Check if any location in group is near enough
                    group.locations().any(|loc_idx| {
                        let loc = tt.get_location(loc_idx);
                        haversine_distance(pos, loc.pos()) < MAX_FLEX_RADIUS
                    })
                }
            };

            if in_area {
                matches.push(FlexMatch {
                    transport_idx: transport.idx(),
                    stop_seq_idx: stop_idx as u8,
                    direction,
                    mode_id: FlexModeId::new(transport.idx(), direction, stop_idx as u8),
                });
            }
        }
    }

    matches
}
```

### Time-Dependent Offset Computation

```rust
// transit-flex/src/offsets.rs

use transit_osr::{OsrData, routing, SearchProfile, RoutingAlgorithm, Location};
use transit_osr::sharing_data::SharingData;

pub struct TdOffset {
    pub location_idx: LocationIdx,
    pub valid_from: DateTime,
    pub valid_to: DateTime,
    pub duration_minutes: i16,
    pub mode_id: u32,
}

/// Compute time-dependent offsets for flex transports.
pub fn compute_flex_offsets(
    tt: &nigiri::Timetable,
    osr: &OsrData,
    flex_areas: &FlexAreas,
    matches: &[FlexMatch],
    pos: Position,
    interval: DateTimeInterval,
) -> Vec<TdOffset> {
    let mut offsets = Vec::new();

    for flex_match in matches {
        let transport = tt.flex_transport(flex_match.transport_idx);

        // Build SharingData restricting routing to flex service area
        let sharing_data = build_flex_sharing_data(
            osr, tt, flex_areas, transport, flex_match.stop_seq_idx
        );

        // Route from query position to each reachable stop
        let from = Location::from_latlng_no_level(pos.lat, pos.lon);
        let profile = SearchProfile::CarSharing; // flex uses car-sharing profile

        for (stop_idx, stop) in transport.stops().enumerate() {
            // Get target locations for this stop
            let target_positions = stop_target_positions(tt, osr, stop);

            for target in &target_positions {
                let to = Location::from_latlng_no_level(target.lat, target.lon);

                if let Some(path) = routing::route::route(
                    &osr.ways, &osr.lookup,
                    Some(&osr.elevations),
                    profile, from.clone(), to,
                    MAX_FLEX_DURATION,
                    RoutingAlgorithm::Dijkstra,
                ) {
                    let travel_seconds = path.cost;

                    // Intersect with stop time windows for each active day
                    for day in interval.days() {
                        if !transport.active_on(day) {
                            continue;
                        }

                        let window = transport.time_window(stop_idx, day);
                        // Effective arrival = departure + travel_seconds
                        // Must fall within stop's time window
                        let effective_start = window.start.saturating_sub(travel_seconds);
                        let effective_end = window.end.saturating_sub(travel_seconds);

                        if effective_start < effective_end {
                            offsets.push(TdOffset {
                                location_idx: target.location_idx,
                                valid_from: day.at(effective_start),
                                valid_to: day.at(effective_end),
                                duration_minutes: (travel_seconds / 60).min(i16::MAX as u32) as i16,
                                mode_id: flex_match.mode_id.to_id(),
                            });
                        }
                    }
                }
            }
        }
    }

    offsets.sort_by_key(|o| o.valid_from);
    offsets
}

fn build_flex_sharing_data(
    osr: &OsrData,
    tt: &nigiri::Timetable,
    flex_areas: &FlexAreas,
    transport: FlexTransport,
    from_stop_idx: u8,
) -> SharingData<'static> {
    // Build bitvecs marking allowed start/end/through nodes
    // based on flex area geometry or location group membership
    // ... (mirrors motis prepare_sharing_data)
    todo!()
}
```

### Flex Output Formatting

```rust
// transit-flex/src/output.rs

pub struct FlexLegInfo {
    pub pickup_type: PickupDropoffType,
    pub dropoff_type: PickupDropoffType,
    pub pickup_window: Option<TimeWindow>,
    pub dropoff_window: Option<TimeWindow>,
    pub flex_area_name: Option<String>,
}

pub fn annotate_flex_leg(
    tt: &nigiri::Timetable,
    mode_id: u32,
    leg: &mut JourneyLeg,
) -> Option<FlexLegInfo> {
    if !FlexModeId::is_flex(mode_id) {
        return None;
    }

    let fid = FlexModeId(mode_id);
    let transport = tt.flex_transport(fid.transport_idx());
    let stop = transport.stop(fid.stop_idx());

    Some(FlexLegInfo {
        pickup_type: stop.pickup_type(),
        dropoff_type: stop.dropoff_type(),
        pickup_window: stop.pickup_window(),
        dropoff_window: stop.dropoff_window(),
        flex_area_name: stop.area_name(tt),
    })
}
```

## Integration with Routing

```rust
// In transit-routing, during query preparation:

// 1. Discover flex transports near origin/destination
let flex_matches = discover_flex_transports(
    &tt, &flex_areas, &rtree, query.from, query.interval, Direction::Forward
);

// 2. Compute time-dependent offsets
let flex_offsets = compute_flex_offsets(
    &tt, &osr, &flex_areas, &flex_matches, query.from, query.interval
);

// 3. Merge flex offsets with regular transit offsets
query.td_start_offsets.extend(flex_offsets);

// 4. Run RAPTOR with combined offsets
let journeys = raptor.route(&query);

// 5. Post-process: annotate flex legs
for journey in &mut journeys {
    for leg in &mut journey.legs {
        if FlexModeId::is_flex(leg.transport_mode_id) {
            leg.flex_info = annotate_flex_leg(&tt, leg.transport_mode_id, leg);
        }
    }
}
```

## Acceptance Criteria

1. Flex areas loaded from timetable and converted to `geo::MultiPolygon`
2. OSR node containment bitvectors built in parallel
3. Flex transports discoverable by position + date
4. Time-dependent offsets computed with correct time window intersection
5. `FlexModeId` encoding/decoding preserves transport/direction/stop
6. Flex legs annotated with pickup/dropoff windows in journey output
7. `SharingData` correctly restricts OSR routing to flex service areas

## motis Source Reference

| File | Lines | Key Pattern |
|---|---|---|
| `src/flex/flex.cc` | ~365 | `add_flex_td_offsets()`, `get_flex_routings()`, discovery + offset computation |
| `src/flex/flex_areas.cc` | ~107 | Parallel geometry loading, point-in-area via TG library |
| `src/flex/flex_output.cc` | ~140 | Leg annotation, place resolution |
| `include/motis/flex/mode_id.h` | ~55 | `mode_id` bitfield layout |
| `include/motis/flex/flex_routing_data.h` | ~35 | `SharingData` construction |
