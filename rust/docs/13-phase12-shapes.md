# Phase 12 — Map Data & Route Shapes

**Priority**: Thirteenth  
**Type**: Rust + `transit-osr` for map-matching  
**Depends on**: Phase 1 (timetable), Phase 4 (OSR)  
**Crate**: `transit-shapes` (module in `transit-import` + `transit-server`)  
**Estimated effort**: Medium

## Objective

Generate and serve route shape polylines (the geographic path a transit vehicle follows between stops). Shapes can come from GTFS `shapes.txt` or be synthesized by map-matching stop sequences through the OSR street network.

## What motis Does (`src/`, ~1,500+ LOC across multiple files)

### Overview

motis handles route shapes through:

1. **GTFS shapes** — If `shapes.txt` is present, use pre-defined polylines
2. **Map-matching** — If no GTFS shape, route the stop sequence through OSR to synthesize a polyline
3. **Shapes storage** — Store polylines + per-segment bounding boxes in `shapes_storage` (LMDB-backed)
4. **Spatial indexing** — Build R-tree on route bounding boxes for zoom-level-based filtering (railviz)
5. **Polyline encoding** — Google encoded polyline format (precision 5 or 7)

### Shapes Storage

```rust
// Per route/trip, store:
pub struct ShapeData {
    pub polyline: Vec<Position>,              // decoded coordinate list
    pub segment_bboxes: Vec<BoundingBox>,     // one bbox per stop-to-stop segment
}

// Encoded for storage:
pub struct EncodedShape {
    pub encoded_polyline: String,             // Google polyline format
    pub segment_offsets: Vec<u32>,            // index into polyline for each segment
}
```

### Railviz Static Index

For map rendering, motis builds a spatial index of all routes with zoom-level filtering:

- Routes indexed by bounding box
- Filtered by transport class + zoom level (high-speed rail shown at low zoom, bus only at high zoom)
- Used by `trips_by_area` endpoint to find routes visible in a map viewport

## Rust Implementation

### Shape Generation

```rust
// transit-shapes/src/generate.rs

use transit_osr::{OsrData, routing, SearchProfile, RoutingAlgorithm, Location};

pub struct ShapeGenerator<'a> {
    osr: &'a OsrData,
    tt: &'a nigiri::Timetable,
}

impl<'a> ShapeGenerator<'a> {
    /// Generate a shape for a route by map-matching its stop sequence.
    pub fn generate_shape(&self, route_idx: RouteIdx) -> Option<ShapeData> {
        let stops = self.tt.route_stops(route_idx);
        if stops.len() < 2 {
            return None;
        }

        let mut polyline = Vec::new();
        let mut segment_bboxes = Vec::new();

        for window in stops.windows(2) {
            let from_loc = self.tt.get_location(window[0]);
            let to_loc = self.tt.get_location(window[1]);

            let from = Location::from_latlng_no_level(from_loc.lat, from_loc.lon);
            let to = Location::from_latlng_no_level(to_loc.lat, to_loc.lon);

            // Use appropriate profile based on transport class
            let profile = route_class_to_profile(
                self.tt.route_class(route_idx)
            );

            match routing::route::route(
                &self.osr.ways, &self.osr.lookup,
                Some(&self.osr.elevations),
                profile, from, to,
                MAX_SHAPE_ROUTING_COST,
                RoutingAlgorithm::Dijkstra,
            ) {
                Some(path) => {
                    let segment_points: Vec<Position> = path.segments.iter()
                        .flat_map(|s| s.polyline.iter().map(|p| Position {
                            lat: p.lat(), lon: p.lng()
                        }))
                        .collect();

                    let bbox = bounding_box(&segment_points);
                    segment_bboxes.push(bbox);
                    polyline.extend(segment_points);
                }
                None => {
                    // Fallback: straight line between stops
                    let segment = vec![
                        Position { lat: from_loc.lat, lon: from_loc.lon },
                        Position { lat: to_loc.lat, lon: to_loc.lon },
                    ];
                    let bbox = bounding_box(&segment);
                    segment_bboxes.push(bbox);
                    polyline.extend(segment);
                }
            }
        }

        Some(ShapeData { polyline, segment_bboxes })
    }

    /// Load shape from GTFS shapes.txt data if available.
    pub fn load_gtfs_shape(&self, trip_idx: TripIdx) -> Option<ShapeData> {
        let shape_points = self.tt.trip_shape_points(trip_idx)?;
        let stops = self.tt.trip_stops(trip_idx);

        // Split shape into segments at nearest points to each stop
        let mut polyline = Vec::new();
        let mut segment_bboxes = Vec::new();
        let mut current_segment = Vec::new();
        let mut stop_iter = stops.iter().peekable();

        for point in shape_points {
            current_segment.push(Position { lat: point.lat, lon: point.lon });

            // Check if this point is closest to the next stop
            if let Some(&&next_stop) = stop_iter.peek() {
                let stop_loc = self.tt.get_location(next_stop);
                let dist = haversine_distance(
                    Position { lat: point.lat, lon: point.lon },
                    Position { lat: stop_loc.lat, lon: stop_loc.lon },
                );
                if dist < SHAPE_STOP_SNAP_DISTANCE {
                    let bbox = bounding_box(&current_segment);
                    segment_bboxes.push(bbox);
                    polyline.extend(current_segment.drain(..));
                    stop_iter.next();
                }
            }
        }

        // Flush remaining
        if !current_segment.is_empty() {
            let bbox = bounding_box(&current_segment);
            segment_bboxes.push(bbox);
            polyline.extend(current_segment);
        }

        Some(ShapeData { polyline, segment_bboxes })
    }
}

fn route_class_to_profile(clasz: TransportClass) -> SearchProfile {
    match clasz {
        TransportClass::Bus | TransportClass::Coach => SearchProfile::Car,
        _ => SearchProfile::Foot, // rail, tram, subway don't need street routing
    }
}
```

### Polyline Encoding

```rust
// transit-shapes/src/polyline.rs

/// Encode coordinates as Google encoded polyline (precision 5).
pub fn encode_polyline(coords: &[Position]) -> String {
    let mut encoded = String::new();
    let mut prev_lat = 0i64;
    let mut prev_lon = 0i64;

    for coord in coords {
        let lat = (coord.lat * 1e5).round() as i64;
        let lon = (coord.lon * 1e5).round() as i64;

        encode_value(lat - prev_lat, &mut encoded);
        encode_value(lon - prev_lon, &mut encoded);

        prev_lat = lat;
        prev_lon = lon;
    }

    encoded
}

fn encode_value(mut value: i64, buf: &mut String) {
    let mut v = if value < 0 { (!value) << 1 | 1 } else { value << 1 };
    while v >= 0x20 {
        buf.push(char::from((((v & 0x1f) | 0x20) + 63) as u8));
        v >>= 5;
    }
    buf.push(char::from((v + 63) as u8));
}

/// Decode Google encoded polyline.
pub fn decode_polyline(encoded: &str) -> Vec<Position> {
    let mut coords = Vec::new();
    let mut lat = 0i64;
    let mut lon = 0i64;
    let bytes = encoded.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        lat += decode_next(bytes, &mut i);
        lon += decode_next(bytes, &mut i);
        coords.push(Position {
            lat: lat as f64 / 1e5,
            lon: lon as f64 / 1e5,
        });
    }

    coords
}

fn decode_next(bytes: &[u8], i: &mut usize) -> i64 {
    let mut result = 0i64;
    let mut shift = 0;
    loop {
        let b = (bytes[*i] - 63) as i64;
        *i += 1;
        result |= (b & 0x1f) << shift;
        shift += 5;
        if b < 0x20 { break; }
    }
    if result & 1 != 0 { !(result >> 1) } else { result >> 1 }
}
```

### Shapes Cache

```rust
// transit-shapes/src/cache.rs

use std::collections::HashMap;
use std::sync::RwLock;

pub struct ShapesCache {
    /// Route shapes (generated or from GTFS)
    route_shapes: RwLock<HashMap<RouteIdx, EncodedShape>>,
    /// Per-trip override shapes (when trips in same route have different shapes)
    trip_shapes: RwLock<HashMap<TripIdx, EncodedShape>>,
}

impl ShapesCache {
    pub fn new() -> Self {
        Self {
            route_shapes: RwLock::new(HashMap::new()),
            trip_shapes: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_shape(&self, route_idx: RouteIdx, trip_idx: Option<TripIdx>) -> Option<String> {
        // Check trip-specific shape first
        if let Some(tid) = trip_idx {
            if let Some(shape) = self.trip_shapes.read().unwrap().get(&tid) {
                return Some(shape.encoded_polyline.clone());
            }
        }
        // Fall back to route shape
        self.route_shapes.read().unwrap().get(&route_idx)
            .map(|s| s.encoded_polyline.clone())
    }

    pub fn insert_route_shape(&self, route_idx: RouteIdx, shape: ShapeData) {
        let encoded = EncodedShape {
            encoded_polyline: encode_polyline(&shape.polyline),
            segment_offsets: compute_segment_offsets(&shape),
        };
        self.route_shapes.write().unwrap().insert(route_idx, encoded);
    }

    /// Get shape points for a specific segment (stop_a → stop_b)
    pub fn get_segment_polyline(
        &self,
        route_idx: RouteIdx,
        segment_idx: usize,
    ) -> Option<String> {
        let shapes = self.route_shapes.read().unwrap();
        let shape = shapes.get(&route_idx)?;
        let start = shape.segment_offsets.get(segment_idx).copied()? as usize;
        let end = shape.segment_offsets.get(segment_idx + 1)
            .copied()
            .unwrap_or(shape.encoded_polyline.len() as u32) as usize;

        // Re-encode segment from full polyline
        let full = decode_polyline(&shape.encoded_polyline);
        let segment = &full[start..end.min(full.len())];
        Some(encode_polyline(segment))
    }
}
```

### Spatial Index for Map Rendering

```rust
// transit-shapes/src/spatial_index.rs

pub struct RouteSpatialIndex {
    rtree: RTree<RouteEntry>,
}

pub struct RouteEntry {
    pub route_idx: RouteIdx,
    pub bbox: BoundingBox,
    pub transport_class: TransportClass,
}

impl RouteSpatialIndex {
    pub fn build(
        tt: &nigiri::Timetable,
        shapes: &ShapesCache,
    ) -> Self {
        let entries: Vec<RouteEntry> = tt.routes()
            .filter_map(|route_idx| {
                let shape = shapes.route_shapes.read().unwrap();
                let encoded = shape.get(&route_idx)?;
                let polyline = decode_polyline(&encoded.encoded_polyline);
                let bbox = bounding_box(&polyline);
                Some(RouteEntry {
                    route_idx,
                    bbox,
                    transport_class: tt.route_class(route_idx),
                })
            })
            .collect();

        RouteSpatialIndex {
            rtree: RTree::bulk_load(entries),
        }
    }

    /// Find routes visible in a map viewport at a given zoom level.
    pub fn routes_in_area(
        &self,
        bbox: &BoundingBox,
        min_class: TransportClass,
    ) -> Vec<RouteIdx> {
        self.rtree.locate_in_envelope(&bbox.to_envelope())
            .filter(|e| e.transport_class <= min_class)
            .map(|e| e.route_idx)
            .collect()
    }
}
```

## GraphQL Integration

```graphql
type TransitLeg {
  # ... other fields
  shape: String          # Google encoded polyline
  intermediateStops: [IntermediateStop!]!
}

type Route {
  shape: String          # Full route shape polyline
  # ...
}
```

```rust
// transit-server/src/resolvers/journey.rs

impl TransitLegResolver {
    async fn shape(&self, ctx: &Context<'_>) -> Option<String> {
        let data = ctx.data::<AppData>().unwrap();
        data.shapes.get_shape(self.route_idx, Some(self.trip_idx))
    }
}
```

## Acceptance Criteria

1. GTFS shapes loaded and split into per-segment polylines
2. Map-matched shapes generated via OSR for routes without GTFS shapes
3. Polyline encoding/decoding matches Google format (precision 5)
4. Shapes cached by route (with per-trip overrides)
5. Per-segment polylines extractable for journey leg rendering
6. Spatial index enables map viewport queries
7. GraphQL `shape` field returns encoded polyline on transit legs

## motis Source Reference

| File | Lines | Key Pattern |
|---|---|---|
| `src/route_shapes.cc` | ~500 | Shape generation via OSR + GTFS shapes |
| `src/railviz.cc` | ~1,000 | Spatial indexing, zoom filtering, trip rendering |
| `include/motis/railviz.h` | ~80 | `railviz_static_index` + spatial types |
