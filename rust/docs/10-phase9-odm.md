# Phase 9 — ODM (On-Demand Mobility)

**Priority**: Tenth  
**Type**: Pure Rust (HTTP client + routing orchestration)  
**Depends on**: Phase 3 (routing), Phase 4 (OSR), Phase 7 (GBFS), Phase 10 (elevators, optional)  
**Crate**: `transit-odm`  
**Estimated effort**: Large

## Objective

Integrate on-demand mobility services (taxis, ride-sharing) via the PRIMA API as first/last-mile legs in multimodal journey planning. This involves multi-stage query orchestration: blacklisting infeasible rides, generating multi-variant RAPTOR queries with taxi/rideshare offsets, whitelist refinement, and journey shortening.

## What motis Does (`src/odm/`, ~1,800+ LOC)

### Overview

ODM support adds taxi and ride-sharing as pre-/post-transit legs. The flow is:

1. **Blacklist** — Query PRIMA to eliminate infeasible taxi connections
2. **Rideshare whitelist** — Query available ride-sharing events
3. **Multi-query** — Generate up to 9 RAPTOR query variants (walk / taxi-short / taxi-long / rideshare combinations)
4. **RAPTOR routing** — Execute all variants
5. **Taxi whitelist** — Refine taxi legs with exact pickup/dropoff times
6. **Shorten** — Replace long taxi legs with shorter alternatives

### Key Constants

```rust
const ODM_TRANSFER_BUFFER: Duration = Duration::from_secs(300);  // 5 min padding
const ODM_DIRECT_PERIOD: u32 = 300;        // 300s slot interval for direct taxi
const ODM_DIRECT_FACTOR: f64 = 1.0;        // cost multiplier
const ODM_OFFSET_MIN_IMPROVEMENT: u32 = 60; // taxi must beat PT by ≥ 60s
const ODM_MAX_DURATION: u32 = 3600;         // max 1h taxi ride
const SEARCH_INTERVAL_SIZE: Duration = Duration::from_secs(36000); // 10h search window
```

## Rust Implementation

### PRIMA API Client

```rust
// transit-odm/src/prima.rs

pub struct PrimaClient {
    http: reqwest::Client,
    base_url: String,
}

pub struct PrimaConfig {
    pub blacklist_url: String,   // /api/blacklist
    pub whitelist_url: String,   // /api/whitelist
    pub rideshare_url: String,   // /api/whitelistRideShare
}

#[derive(Serialize)]
pub struct BlacklistRequest {
    pub origin: PrimaLocation,
    pub destination: PrimaLocation,
    pub earliest_departure: DateTime<Utc>,
    pub latest_departure: DateTime<Utc>,
    pub is_arrival: bool,
    pub capacities: Capacities,
}

#[derive(Deserialize)]
pub struct BlacklistResponse {
    pub first_mile: Vec<TaxiOffer>,
    pub last_mile: Vec<TaxiOffer>,
    pub direct: Vec<DirectRide>,
}

#[derive(Deserialize)]
pub struct TaxiOffer {
    pub from: PrimaLocation,
    pub to: PrimaLocation,
    pub duration: u32,                   // seconds
    pub service_times: Vec<TimeInterval>, // when available
}

#[derive(Deserialize)]
pub struct DirectRide {
    pub duration: u32,
    pub departure: DateTime<Utc>,
    pub arrival: DateTime<Utc>,
}

pub struct Capacities {
    pub wheelchairs: u8,
    pub bikes: u8,
    pub passengers: u8,
    pub luggage: u8,
}

impl PrimaClient {
    pub async fn blacklist(
        &self,
        req: &BlacklistRequest,
    ) -> Result<BlacklistResponse, OdmError> {
        let resp = self.http
            .post(&self.config.blacklist_url)
            .json(req)
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp)
    }

    pub async fn whitelist_taxi(
        &self,
        req: &WhitelistRequest,
    ) -> Result<WhitelistResponse, OdmError> {
        // POST to /api/whitelist with specific ride times
        todo!()
    }

    pub async fn whitelist_rideshare(
        &self,
        req: &RideshareRequest,
    ) -> Result<RideshareResponse, OdmError> {
        // POST to /api/whitelistRideShare
        todo!()
    }
}
```

### Service Area Bounds

```rust
// transit-odm/src/bounds.rs

use geo::{MultiPolygon, Contains, Point};

pub struct ServiceBounds {
    pub taxi: Option<MultiPolygon<f64>>,
    pub rideshare: Option<MultiPolygon<f64>>,
}

impl ServiceBounds {
    pub fn load(config: &OdmConfig) -> Result<Self, OdmError> {
        Ok(Self {
            taxi: config.taxi_bounds_file.as_ref()
                .map(|f| load_geojson_polygon(f))
                .transpose()?,
            rideshare: config.rideshare_bounds_file.as_ref()
                .map(|f| load_geojson_polygon(f))
                .transpose()?,
        })
    }

    pub fn taxi_available_at(&self, pos: Position) -> bool {
        self.taxi.as_ref()
            .map_or(false, |p| p.contains(&Point::new(pos.lon, pos.lat)))
    }

    pub fn rideshare_available_at(&self, pos: Position) -> bool {
        self.rideshare.as_ref()
            .map_or(false, |p| p.contains(&Point::new(pos.lon, pos.lat)))
    }
}
```

### Meta-Router (Multi-Query Orchestrator)

```rust
// transit-odm/src/meta_router.rs

pub struct MetaRouter<'a> {
    prima: &'a PrimaClient,
    bounds: &'a ServiceBounds,
    router: &'a TransitRouter<'a>,
    osr: &'a transit_osr::OsrData,
}

pub struct OdmQuery {
    pub from: Position,
    pub to: Position,
    pub time: DateTime<Utc>,
    pub is_arrival: bool,
    pub capacities: Capacities,
    pub modes: Vec<OdmMode>,
}

pub enum OdmMode {
    Walk,
    TaxiShort,
    TaxiLong,
    RideSharing,
}

impl<'a> MetaRouter<'a> {
    pub async fn route(&self, query: &OdmQuery) -> Result<Vec<Journey>, OdmError> {
        let interval = self.extend_interval(query);

        // 1. Blacklist: get service time windows
        let blacklist = self.prima.blacklist(&BlacklistRequest {
            origin: query.from.into(),
            destination: query.to.into(),
            earliest_departure: interval.start,
            latest_departure: interval.end,
            is_arrival: query.is_arrival,
            capacities: query.capacities.clone(),
        }).await?;

        // 2. Rideshare whitelist (if applicable)
        let rideshare_events = if query.modes.contains(&OdmMode::RideSharing)
            && self.bounds.rideshare_available_at(query.from) {
            self.prima.whitelist_rideshare(&RideshareRequest {
                // ...
            }).await.ok()
        } else {
            None
        };

        // 3. Build query variants
        let variants = self.build_query_variants(
            query, &blacklist, &rideshare_events
        );

        // 4. Run RAPTOR for each variant
        let mut all_journeys = Vec::new();
        for variant in &variants {
            let raptor_query = self.to_raptor_query(variant);
            let journeys = self.router.route(&raptor_query)?;
            all_journeys.extend(
                journeys.into_iter().filter(|j| self.uses_odm(j))
            );
        }

        // 5. Add direct taxi/rideshare journeys
        if let Some(direct) = self.compute_direct_rides(query, &blacklist).await? {
            all_journeys.extend(direct);
        }

        // 6. Whitelist: refine taxi availability
        let refined = self.whitelist_refine(&mut all_journeys).await?;

        // 7. Shorten: replace long ODM legs with shorter alternatives
        self.shorten_journeys(&mut refined, &blacklist);

        // 8. Add pure PT journeys
        let pt_journeys = self.router.route(&self.pt_only_query(query))?;
        refined.extend(pt_journeys);

        Ok(pareto_optimal(refined))
    }

    fn build_query_variants(
        &self,
        query: &OdmQuery,
        blacklist: &BlacklistResponse,
        rideshare: &Option<RideshareResponse>,
    ) -> Vec<QueryVariant> {
        let mut variants = vec![QueryVariant::WalkOnly];

        // Split taxi offsets into short/long by median duration
        let (short, long) = split_taxi_offsets(&blacklist.first_mile);

        if !short.is_empty() {
            variants.push(QueryVariant::TaxiShortFirst(short.clone()));
            variants.push(QueryVariant::TaxiShortBoth(short.clone()));
        }
        if !long.is_empty() {
            variants.push(QueryVariant::TaxiLongFirst(long.clone()));
            variants.push(QueryVariant::TaxiLongBoth(long.clone()));
        }

        if let Some(rs) = rideshare {
            variants.push(QueryVariant::RideshareFirst(rs.clone()));
            variants.push(QueryVariant::RideshareLast(rs.clone()));
        }

        variants
    }
}
```

### TD Offset Helpers

```rust
// transit-odm/src/td_offsets.rs

pub fn taxi_to_td_offsets(
    offers: &[TaxiOffer],
    tt: &nigiri::Timetable,
    rtree: &RTree<LocationEntry>,
) -> Vec<TdOffset> {
    let mut offsets = Vec::new();

    for offer in offers {
        // Find closest transit stop to taxi dropoff/pickup
        let stop = rtree.nearest_neighbor(&[offer.to.lon, offer.to.lat]);
        if let Some(stop) = stop {
            for window in &offer.service_times {
                offsets.push(TdOffset {
                    location_idx: stop.idx,
                    valid_from: window.start,
                    valid_to: window.end,
                    duration_minutes: (offer.duration / 60) as i16,
                    mode_id: ODM_TRANSPORT_MODE_ID,
                });
            }
        }
    }

    offsets
}

fn split_taxi_offsets(offers: &[TaxiOffer]) -> (Vec<TaxiOffer>, Vec<TaxiOffer>) {
    if offers.is_empty() {
        return (vec![], vec![]);
    }
    let mut sorted: Vec<_> = offers.to_vec();
    sorted.sort_by_key(|o| o.duration);
    let median = sorted[sorted.len() / 2].duration;
    let short = sorted.iter().filter(|o| o.duration <= median).cloned().collect();
    let long = sorted.iter().filter(|o| o.duration > median).cloned().collect();
    (short, long)
}
```

### Journey Shortening

```rust
// transit-odm/src/shorten.rs

/// Replace long first/last-mile ODM legs with shorter taxi alternatives.
pub fn shorten_journeys(
    journeys: &mut Vec<Journey>,
    whitelist: &WhitelistResponse,
) {
    for journey in journeys.iter_mut() {
        // Check first leg
        if let Some(first) = journey.legs.first() {
            if is_odm_leg(first) {
                if let Some(shorter) = find_shorter_taxi(first, whitelist) {
                    journey.legs[0] = shorter;
                    journey.departure = journey.legs[0].departure;
                }
            }
        }

        // Check last leg
        if let Some(last) = journey.legs.last() {
            if is_odm_leg(last) {
                let idx = journey.legs.len() - 1;
                if let Some(shorter) = find_shorter_taxi(last, whitelist) {
                    journey.legs[idx] = shorter;
                    journey.arrival = journey.legs[idx].arrival;
                }
            }
        }
    }
}
```

## Data Flow

```
ODM Query (from, to, time, modes=[WALK, TAXI, RIDESHARE])
  │
  ├─ 1. Check service area bounds
  │     └─ taxi_available_at(from)? rideshare_available_at(from)?
  │
  ├─ 2. PRIMA /api/blacklist
  │     └─ Get first_mile/last_mile taxi offers + direct rides
  │     └─ Filter by service time windows
  │
  ├─ 3. PRIMA /api/whitelistRideShare
  │     └─ Get available ride-sharing events with tour IDs
  │
  ├─ 4. Build query variants (up to 9)
  │     └─ Walk / TaxiShort / TaxiLong / Rideshare × first/last/both
  │
  ├─ 5. RAPTOR routing per variant
  │     └─ td_offsets from taxi/rideshare injected as start/dest constraints
  │
  ├─ 6. PRIMA /api/whitelist (for taxi legs in results)
  │     └─ Refine pickup/dropoff times
  │
  ├─ 7. Shorten: replace long taxi legs with shorter alternatives
  │
  └─ 8. Merge: ODM journeys + direct rides + pure PT journeys
       └─ Pareto-optimal filter
```

## Acceptance Criteria

1. PRIMA blacklist API queried and parsed correctly
2. PRIMA whitelist (taxi + rideshare) APIs integrated
3. Multi-variant queries generated (walk + taxi combinations)
4. TD offsets correctly computed from taxi service windows
5. Journey shortening replaces long ODM legs when shorter available
6. Direct taxi/rideshare journeys included when faster than PT
7. Service area bounds enforced (GeoJSON polygon containment)
8. Pareto-optimal filtering across all journey types

## motis Source Reference

| File | Lines | Key Pattern |
|---|---|---|
| `src/odm/meta_router.cc` | ~500 | Multi-stage routing orchestration |
| `src/odm/prima.cc` | ~300 | PRIMA JSON API client |
| `src/odm/query_factory.cc` | ~60 | Query variant generation |
| `src/odm/shorten.cc` | ~160 | Journey shortening heuristics |
| `src/odm/blacklist_taxi.cc` | ~130 | Blacklist request/response |
| `src/odm/whitelist_taxi.cc` | ~190 | Whitelist request/response |
| `src/odm/whitelist_ridesharing.cc` | ~160 | Rideshare whitelist |
| `src/odm/bounds.cc` | ~40 | GeoJSON service area loading |
| `src/odm/td_offsets.h` | ~60 | TD offset helpers |
