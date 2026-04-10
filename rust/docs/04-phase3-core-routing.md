# Phase 3 — Core Transit Routing (RAPTOR via nigiri)

**Priority**: Fourth  
**Type**: Rust orchestration + nigiri FFI  
**Depends on**: Phase 1 (timetable), Phase 2 (tag lookup)  
**Crate**: `transit-routing`  
**Estimated effort**: Large

## Objective

Expose nigiri's RAPTOR routing engine via the GraphQL `planJourney` query, including departure/arrival board queries (`stopDepartures`), trip details, and reachability analysis.

## What motis Does

### Routing Orchestration (`src/endpoints/routing.cc`, ~800 LOC)

The routing endpoint is the most complex in motis. It:

1. **Parses the query** — origin/destination (coordinates, stop ID, or address), time, modes
2. **Computes pre/post-transit offsets** — walking/biking/driving distance from origin/destination to nearby stops using OSR
3. **Adds GBFS/flex/ODM offsets** — additional first/last-mile connections
4. **Runs RAPTOR** — `nigiri::routing::raptor_search()` with all offsets
5. **Filters results** — remove dominated journeys, apply direct-route filter
6. **Formats response** — `journey_to_response()` converts nigiri journeys to API format

For Phase 3, we implement **steps 1, 4, 5** (transit-only routing without street routing). Street routing offsets come in Phase 4–5.

### nigiri Routing Query

```cpp
struct query {
    location_idx_t start_;           // or offsets
    location_idx_t destination_;     // or offsets
    unixtime_t start_time_;
    bool is_arrival_time_;           // depart-at vs arrive-by
    direction search_dir_;           // FORWARD or BACKWARD
    uint8_t max_transfers_;
    clasz_mask_t allowed_claszes_;   // mode filter
    
    // Pre/post-transit offsets (from street routing)
    vector<offset> start_offsets_;
    vector<offset> dest_offsets_;
    
    // Time-dependent offsets (flex, wheelchair)
    vector<td_offset> td_start_offsets_;
    vector<td_offset> td_dest_offsets_;
    
    unixtime_t prf_dest_;            // for profile queries
    unsigned timeout_seconds_;
};
```

### Existing nigiri C ABI for Routing

Already in `abi.h`:

```c
int64_t nigiri_get_journeys(
    nigiri_timetable_t* t,
    uint32_t from_stop_idx,
    uint32_t to_stop_idx,
    int64_t time,           // unix timestamp
    bool backward,          // arrive-by mode
    nigiri_journey_cb cb,   // callback per journey
    void* context
);
```

The callback receives `nigiri_journey_t` with legs containing transport/walk segments, times, and stop indices.

### What We Need to Add

The existing routing ABI is minimal. We need:

```c
// Extended routing with full query options
typedef struct {
    int64_t time;               // departure or arrival unix timestamp
    bool is_arrival_time;       // false = depart-at, true = arrive-by
    uint8_t max_transfers;      // 0–7, default 7
    uint16_t allowed_claszes;   // bitmask of route types
    uint32_t max_results;       // limit result count
    uint32_t timeout_seconds;   // search timeout
    bool wheelchair;            // accessibility filter
    bool bikes_allowed;         // bike carriage filter
} nigiri_routing_options_t;

// Route between two stops with options
int64_t nigiri_route_stops(
    nigiri_timetable_t* t,
    uint32_t from_stop_idx,
    uint32_t to_stop_idx,
    nigiri_routing_options_t const* options,
    nigiri_journey_cb cb,
    void* context
);

// Route with pre-computed offsets (for street routing integration)
typedef struct {
    uint32_t location_idx;
    int16_t duration_minutes;
    uint8_t transport_mode;     // walk, bike, car, etc.
} nigiri_offset_t;

int64_t nigiri_route_offsets(
    nigiri_timetable_t* t,
    nigiri_offset_t const* start_offsets,
    uint32_t n_start_offsets,
    nigiri_offset_t const* dest_offsets,
    uint32_t n_dest_offsets,
    nigiri_routing_options_t const* options,
    nigiri_journey_cb cb,
    void* context
);

// Extended journey callback with more detail
typedef struct {
    uint32_t from_stop_idx;
    uint32_t to_stop_idx;
    int64_t departure_time;      // unix timestamp
    int64_t arrival_time;
    uint16_t n_transfers;
    uint16_t n_legs;
} nigiri_journey_summary_t;

typedef struct {
    uint32_t from_stop_idx;
    uint32_t to_stop_idx;
    int64_t departure_time;
    int64_t arrival_time;
    bool is_transport;           // true = transit, false = walk/transfer
    uint32_t transport_idx;      // only if is_transport
    uint32_t route_idx;          // only if is_transport
    const char* trip_id;         // GTFS trip_id (only if is_transport)
    uint32_t trip_id_len;
    // RT: scheduled vs estimated times
    int64_t scheduled_departure;
    int64_t scheduled_arrival;
    int32_t departure_delay;     // seconds, 0 if no RT
    int32_t arrival_delay;
    bool is_cancelled;
    bool is_real_time;
} nigiri_leg_detail_t;

typedef void (*nigiri_journey_detail_cb)(
    nigiri_journey_summary_t const* journey,
    nigiri_leg_detail_t const* legs,
    uint32_t n_legs,
    void* context
);

// Stop departures/arrivals
typedef struct {
    int64_t scheduled_time;
    int64_t estimated_time;     // same as scheduled if no RT
    int32_t delay;
    uint32_t transport_idx;
    uint32_t route_idx;
    uint32_t trip_day_idx;
    const char* headsign;
    uint32_t headsign_len;
    uint8_t route_type;
    bool is_cancelled;
    bool is_real_time;
    uint8_t direction;
} nigiri_departure_t;

typedef void (*nigiri_departure_cb)(
    nigiri_departure_t const* dep,
    void* context
);

int64_t nigiri_get_stop_departures(
    nigiri_timetable_t* t,
    uint32_t stop_idx,
    int64_t start_time,
    uint32_t max_results,
    uint16_t allowed_claszes,    // mode filter bitmask
    nigiri_departure_cb cb,
    void* context
);

// One-to-all reachability (isochrone)
typedef struct {
    uint32_t location_idx;
    int16_t duration_minutes;
    uint8_t n_transfers;
} nigiri_reachable_t;

typedef void (*nigiri_reachable_cb)(
    nigiri_reachable_t const* stop,
    void* context
);

int64_t nigiri_one_to_all(
    nigiri_timetable_t* t,
    uint32_t from_stop_idx,
    int64_t departure_time,
    uint16_t max_duration_minutes,
    uint16_t allowed_claszes,
    nigiri_reachable_cb cb,
    void* context
);
```

## Rust Implementation

### Query Builder

```rust
// transit-routing/src/query.rs

pub struct RoutingQuery {
    pub from: LocationSpec,
    pub to: LocationSpec,
    pub time: DateTime<Utc>,
    pub is_arrival_time: bool,
    pub max_transfers: u8,
    pub allowed_modes: Vec<RouteType>,
    pub max_results: u32,
    pub timeout_seconds: u32,
    pub wheelchair: bool,
    pub bikes_allowed: bool,
}

pub enum LocationSpec {
    StopId(String),                        // GraphQL ID
    Coordinates { lat: f64, lon: f64 },    // will need street routing (Phase 4)
    Offsets(Vec<StopOffset>),              // pre-computed
}

pub struct StopOffset {
    pub location_idx: LocationIdx,
    pub duration_minutes: i16,
    pub mode: LegMode,
}
```

### Routing Orchestrator

```rust
// transit-routing/src/router.rs

pub struct TransitRouter<'a> {
    tt: &'a Timetable,
    tags: &'a TagLookup,
    rtree: &'a RTree<LocationEntry>,
}

impl<'a> TransitRouter<'a> {
    /// Plan journeys between two locations
    pub fn plan(&self, query: &RoutingQuery) -> Result<Vec<Journey>, TransitError> {
        // 1. Resolve locations to stop indices
        let from_idx = self.resolve_location(&query.from)?;
        let to_idx = self.resolve_location(&query.to)?;
        
        // 2. Build nigiri routing options
        let options = self.build_options(query);
        
        // 3. Call nigiri RAPTOR via FFI
        let raw_journeys = self.tt.get_journeys_with_options(
            from_idx, to_idx, &options
        )?;
        
        // 4. Convert to domain model
        raw_journeys.into_iter()
            .map(|j| self.to_journey(j))
            .collect()
    }
    
    /// Get departures from a stop
    pub fn stop_departures(&self, stop_id: &str, 
                           start_time: DateTime<Utc>,
                           limit: u32,
                           modes: Option<&[RouteType]>
    ) -> Result<Vec<Departure>, TransitError> {
        let loc_idx = self.tags.resolve_location(self.tt, stop_id)?;
        // Call nigiri FFI for departures
        // Convert to domain model
    }
    
    /// Compute reachability (isochrone)
    pub fn reachable(&self, from: &LocationSpec, 
                     departure_time: DateTime<Utc>,
                     max_duration_mins: u32,
                     modes: Option<&[RouteType]>
    ) -> Result<Vec<ReachableStop>, TransitError> {
        // Call nigiri one-to-all via FFI
    }
}
```

### GraphQL Resolvers

```rust
#[async_graphql::Object]
impl QueryRoot {
    /// Plan a journey between two locations
    async fn plan_journey(&self, ctx: &Context<'_>, 
                          input: JourneyPlanInput) -> Result<Vec<GqlJourney>> {
        let data = ctx.data::<AppData>()?;
        let router = TransitRouter::new(&data.tt, &data.tags, &data.location_rtree);
        
        let query = RoutingQuery::from_graphql(input, &data.tags, &data.tt)?;
        let journeys = router.plan(&query)?;
        
        Ok(journeys.into_iter().map(GqlJourney::from).collect())
    }
    
    /// Get upcoming departures from a stop
    async fn stop_departures(&self, ctx: &Context<'_>,
                             stop_id: ID, limit: i32,
                             modes: Option<Vec<RouteType>>,
                             include_realtime: bool,
                             start_time: Option<DateTime<Utc>>,
                             expand_stops: bool,
    ) -> Result<Vec<GqlDeparture>> {
        let data = ctx.data::<AppData>()?;
        let router = TransitRouter::new(&data.tt, &data.tags, &data.location_rtree);
        
        let start = start_time.unwrap_or_else(Utc::now);
        let deps = router.stop_departures(
            &stop_id, start, limit as u32, modes.as_deref()
        )?;
        
        Ok(deps.into_iter().map(GqlDeparture::from).collect())
    }
    
    /// Compute reachability
    async fn reachable(&self, ctx: &Context<'_>,
                       input: ReachableInput) -> Result<ReachableResult> {
        // ...
    }
    
    /// Get trip by ID
    async fn trip(&self, ctx: &Context<'_>, id: ID) -> Result<Option<GqlTrip>> {
        let data = ctx.data::<AppData>()?;
        let (transport_idx, day_idx) = data.tags.resolve_trip(&data.tt, &id)?;
        // Build GqlTrip from transport data + stop times
    }
}
```

### Domain Types

```rust
// transit-routing/src/types.rs

pub struct Journey {
    pub departure_time: DateTime<Utc>,
    pub arrival_time: DateTime<Utc>,
    pub duration_seconds: i32,
    pub transfers: u16,
    pub legs: Vec<Leg>,
}

pub enum Leg {
    Transit(TransitLeg),
    Walk(WalkLeg),
}

pub struct TransitLeg {
    pub from: StopRef,
    pub to: StopRef,
    pub departure_time: DateTime<Utc>,
    pub arrival_time: DateTime<Utc>,
    pub route: RouteRef,
    pub trip_id: String,
    pub headsign: String,
    pub direction: u8,
    pub intermediate_stops: Vec<IntermediateStop>,
    // RT fields
    pub scheduled_departure: DateTime<Utc>,
    pub scheduled_arrival: DateTime<Utc>,
    pub departure_delay: Option<i32>,
    pub arrival_delay: Option<i32>,
    pub is_real_time: bool,
    pub is_cancelled: bool,
}

pub struct WalkLeg {
    pub from: Place,
    pub to: Place,
    pub duration_seconds: i32,
    pub distance_meters: f64,
}

pub struct Departure {
    pub aimed_departure_time: DateTime<Utc>,
    pub aimed_arrival_time: DateTime<Utc>,
    pub estimated_departure_time: Option<DateTime<Utc>>,
    pub estimated_arrival_time: Option<DateTime<Utc>>,
    pub delay: Option<i32>,
    pub route: RouteRef,
    pub headsign: String,
    pub direction: u8,
    pub trip_id: String,
    pub is_real_time: bool,
    pub is_cancelled: Option<bool>,
}
```

## Data Flow

```
GraphQL: planJourney(from: {stopId: "seq:600011"}, to: {stopId: "seq:600022"}, departureTime: "...")
  │
  ▼
transit-routing::TransitRouter::plan()
  ├─ tags.resolve_location("seq:600011") → LocationIdx(42)
  ├─ tags.resolve_location("seq:600022") → LocationIdx(87)
  │
  ├─ nigiri_route_stops(tt, 42, 87, options, callback) via FFI
  │       └─ RAPTOR search → journey candidates
  │
  ├─ filter dominated journeys
  │
  └─ convert to Journey { legs: [TransitLeg, WalkLeg, ...] }
       │
       ▼
    GqlJourney → JSON response
```

## Acceptance Criteria

1. `planJourney(from: {stopId: "..."}, to: {stopId: "..."}, departureTime: "...")` returns journeys
2. Journeys have correct leg structure (transit + walk transfers)
3. `stopDepartures(stopId: "...", limit: 10)` returns upcoming departures
4. `trip(id: "...")` returns full stop times for a trip
5. `reachable(input: {from: {stopId: "..."}, maxDurationMins: 30})` returns reachable stops
6. Mode filtering works (e.g., `modes: [BUS, RAIL]`)
7. Results respect max_transfers and timeout limits

## motis Source Reference

| File | Lines | Key Functions |
|---|---|---|
| `src/endpoints/routing.cc` | ~800 | Full multimodal routing orchestration |
| `src/endpoints/stop_times.cc` | ~300 | Departure board |
| `src/endpoints/trip.cc` | ~200 | Trip detail |
| `src/endpoints/one_to_all.cc` | ~300 | Reachability/isochrone |
| nigiri `src/routing/raptor_search.cc` | ~500 | RAPTOR engine |
| nigiri `src/routing/journey.cc` | ~200 | Journey data structure |
