# Phase 6 — Real-Time Updates

**Priority**: Seventh  
**Type**: Rust + nigiri FFI  
**Depends on**: Phase 1 (timetable), Phase 2 (tag lookup)  
**Crate**: `transit-rt`  
**Estimated effort**: Large

## Objective

Consume GTFS-RT, SIRI, and other real-time feeds, apply delay/cancellation updates to the nigiri `rt_timetable`, and surface real-time data through GraphQL queries (departures, trip status, service alerts).

## What motis Does

### RT Architecture (`src/rt_update.cc`, `src/rt/`, ~2,400 LOC total)

motis uses an async polling loop (Boost ASIO coroutines) to:

1. **Poll multiple feeds** in parallel on a configurable interval (default 60s)
2. **Parse feed data** — GTFS-RT protobuf, SIRI XML, AUSER, SIRI-JSON
3. **Apply updates** to `nigiri::rt_timetable` via `nigiri::rt::gtfsrt_update()`
4. **Atomic swap** — new `rt_timetable` replaces old one atomically
5. **Publish metrics** — per-feed success/failure counts, latencies

### Feed Protocols

| Protocol | Format | Parser | Use Case |
|---|---|---|---|
| `gtfsrt` | Protobuf | `transit_realtime::FeedMessage` | Standard worldwide |
| `siri` | XML | pugixml | European standard |
| `siri_json` | JSON | boost::json | SIRI variant |
| `auser` | Custom | motis-specific | German VDV protocol |

### Update Flow

```cpp
// Simplified from rt_update.cc:

while (running) {
    co_await timer.async_wait();  // sleep(update_interval)
    
    for (auto& feed : config.rt_feeds) {
        auto response = co_await http_get(feed.url, feed.headers);
        
        switch (feed.protocol) {
            case gtfsrt:
                auto msg = transit_realtime::FeedMessage::parse(response);
                nigiri::rt::gtfsrt_update(tt, rtt, source, msg);
                break;
            case siri:
                auto xml = pugixml::parse(response);
                // Convert SIRI to trip updates
                break;
        }
    }
    
    // Atomic swap
    std::atomic_store(&data.rt_, new_rtt);
}
```

### nigiri RT Functions

nigiri already has internal RT support:

```cpp
// In nigiri:
void gtfsrt_update(timetable const&, rt_timetable&, 
                   source_idx_t, transit_realtime::FeedMessage const&);

// rt_timetable holds:
// - delay per stop event (departure/arrival)
// - cancellation flags
// - added trips
// - service alerts
```

### Service Alerts

GTFS-RT `Alert` entities are stored in `rt_timetable` and exposed via:
- Affected routes, stops, trips, agencies
- Active time periods
- Cause (strike, weather, construction, etc.)
- Effect (no_service, reduced_service, detour, etc.)
- Translated header/description text

## nigiri C ABI Extensions Needed

```c
// Create a new rt_timetable for a specific date
bool nigiri_init_rt(nigiri_timetable_t* t, int64_t date_unix);

// Apply GTFS-RT update (protobuf bytes)
bool nigiri_apply_gtfsrt(
    nigiri_timetable_t* t,
    uint32_t source_idx,
    const uint8_t* protobuf_data,
    uint32_t protobuf_len
);

// Get real-time delay for a stop event
typedef struct {
    int32_t departure_delay_seconds;
    int32_t arrival_delay_seconds;
    bool is_cancelled;
    bool is_real_time;
    int64_t estimated_departure;
    int64_t estimated_arrival;
} nigiri_rt_stop_event_t;

bool nigiri_get_rt_stop_event(
    nigiri_timetable_t* t,
    uint32_t transport_idx,
    uint32_t day_idx,
    uint32_t stop_seq,
    nigiri_rt_stop_event_t* out
);

// Enumerate service alerts
typedef struct {
    const char* id;
    uint32_t id_len;
    const char* header_text;
    uint32_t header_text_len;
    const char* description_text;
    uint32_t description_text_len;
    const char* url;
    uint32_t url_len;
    uint8_t cause;          // AlertCause enum
    uint8_t effect;         // AlertEffect enum
    uint32_t n_active_periods;
    uint32_t n_affected_routes;
    uint32_t n_affected_stops;
    uint32_t n_affected_trips;
} nigiri_alert_summary_t;

typedef void (*nigiri_alert_cb)(nigiri_alert_summary_t const* alert, void* context);

uint32_t nigiri_get_alerts(
    nigiri_timetable_t* t,
    nigiri_alert_cb cb,
    void* context
);

// Get alerts for a specific entity
uint32_t nigiri_get_alerts_for_stop(
    nigiri_timetable_t* t,
    uint32_t location_idx,
    nigiri_alert_cb cb,
    void* context
);

uint32_t nigiri_get_alerts_for_route(
    nigiri_timetable_t* t,
    uint32_t route_idx,
    nigiri_alert_cb cb,
    void* context
);

// Get RT transport count (added trips)
uint32_t nigiri_get_rt_transport_count(nigiri_timetable_t* t);  // already exists
```

## Rust Implementation

### GTFS-RT Parsing

Use `prost` for protobuf:

```toml
# transit-rt/Cargo.toml
[dependencies]
prost = "0.13"
prost-types = "0.13"
reqwest = { version = "0.12", features = ["rustls-tls"] }
tokio = { version = "1", features = ["time", "sync"] }
quick-xml = "0.37"  # for SIRI

[build-dependencies]
prost-build = "0.13"
```

```rust
// transit-rt/build.rs
fn main() {
    prost_build::compile_protos(
        &["proto/gtfs-realtime.proto"],
        &["proto/"],
    ).unwrap();
}
```

### RT Update Loop

```rust
// transit-rt/src/updater.rs

pub struct RtUpdater {
    config: TimetableConfig,
    tt: Arc<Timetable>,
    rt_state: Arc<RwLock<Option<RtState>>>,
    tags: Arc<TagLookup>,
    client: reqwest::Client,
}

pub struct RtState {
    pub last_updated: DateTime<Utc>,
    pub alerts: Vec<ServiceAlert>,
    // Delays are stored in nigiri rt_timetable (C++ side)
}

impl RtUpdater {
    /// Start the background update loop
    pub async fn run(&self, mut shutdown: tokio::sync::watch::Receiver<bool>) {
        let interval = Duration::from_secs(self.config.update_interval as u64);
        let mut ticker = tokio::time::interval(interval);
        
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    self.update_all_feeds().await;
                }
                _ = shutdown.changed() => {
                    tracing::info!("RT updater shutting down");
                    break;
                }
            }
        }
    }
    
    async fn update_all_feeds(&self) {
        for (tag, dataset) in &self.config.datasets {
            if let Some(rt_feeds) = &dataset.rt {
                for feed in rt_feeds {
                    match self.update_feed(tag, feed).await {
                        Ok(count) => {
                            tracing::info!(tag, url = %feed.url, count, "RT update applied");
                        }
                        Err(e) => {
                            tracing::warn!(tag, url = %feed.url, error = %e, "RT update failed");
                        }
                    }
                }
            }
        }
    }
    
    async fn update_feed(&self, tag: &str, feed: &RtFeedConfig) -> Result<u32, TransitError> {
        // 1. Fetch feed data
        let mut req = self.client.get(&feed.url);
        if let Some(headers) = &feed.headers {
            for (k, v) in headers {
                req = req.header(k, v);
            }
        }
        let response = req.send().await?.bytes().await?;
        
        // 2. Parse and apply based on protocol
        match feed.protocol {
            RtProtocol::GtfsRt => self.apply_gtfsrt(tag, &response),
            RtProtocol::Siri => self.apply_siri(tag, &response),
            RtProtocol::SiriJson => self.apply_siri_json(tag, &response),
            RtProtocol::Auser => self.apply_auser(tag, &response),
        }
    }
    
    fn apply_gtfsrt(&self, tag: &str, data: &[u8]) -> Result<u32, TransitError> {
        let source_idx = self.tags.get_source(tag)?;
        
        // Option A: Parse in Rust, feed individual updates via FFI
        let feed = transit_realtime::FeedMessage::decode(data)?;
        let mut count = 0u32;
        
        for entity in &feed.entity {
            if let Some(trip_update) = &entity.trip_update {
                // Apply via nigiri FFI
                // ...
                count += 1;
            }
            if let Some(alert) = &entity.alert {
                // Store alerts in Rust-side state
                self.process_alert(tag, alert)?;
                count += 1;
            }
        }
        
        Ok(count)
        
        // Option B: Pass raw protobuf bytes to nigiri for processing
        // self.tt.apply_gtfsrt(source_idx, data)?;
    }
}
```

### SIRI Parser

```rust
// transit-rt/src/siri.rs

use quick_xml::Reader;

pub fn parse_siri_delivery(xml: &[u8]) -> Result<Vec<TripUpdate>, TransitError> {
    let mut reader = Reader::from_reader(xml);
    let mut updates = Vec::new();
    
    // Parse SIRI ServiceDelivery → EstimatedTimetableDelivery
    // Convert to TripUpdate structs
    // ...
    
    Ok(updates)
}
```

### Service Alert Types

```rust
// transit-rt/src/alerts.rs

#[derive(Debug, Clone)]
pub struct ServiceAlert {
    pub id: String,
    pub header_text: String,
    pub description_text: Option<String>,
    pub url: Option<String>,
    pub cause: AlertCause,
    pub effect: AlertEffect,
    pub active_periods: Vec<ActivePeriod>,
    pub affected_routes: Vec<String>,    // route IDs
    pub affected_stops: Vec<String>,     // stop IDs
    pub affected_trips: Vec<String>,     // trip IDs
    pub affected_agencies: Vec<String>,  // agency IDs
}

#[derive(Debug, Clone)]
pub struct ActivePeriod {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, async_graphql::Enum)]
pub enum AlertCause {
    UnknownCause, OtherCause, TechnicalProblem, Strike, Demonstration,
    Accident, Holiday, Weather, Maintenance, Construction,
    PoliceActivity, MedicalEmergency,
}

#[derive(Debug, Clone, Copy, async_graphql::Enum)]
pub enum AlertEffect {
    NoService, ReducedService, SignificantDelays, Detour,
    AdditionalService, ModifiedService, OtherEffect, UnknownEffect,
    StopMoved, NoEffect, AccessibilityIssue,
}
```

### GraphQL Resolvers

```rust
#[async_graphql::Object]
impl QueryRoot {
    /// Get service alerts for a specific entity
    async fn alerts(&self, ctx: &Context<'_>, 
                    filter: AlertFilterInput,
                    start_time: Option<DateTime<Utc>>,
                    end_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<GqlServiceAlert>> {
        let data = ctx.data::<AppData>()?;
        let rt = data.rt.read().await;
        
        let alerts = match filter {
            AlertFilterInput::Stop(id) => rt.alerts_for_stop(&id),
            AlertFilterInput::Route(id) => rt.alerts_for_route(&id),
            AlertFilterInput::Trip(id) => rt.alerts_for_trip(&id),
            AlertFilterInput::Operator(id) => rt.alerts_for_agency(&id),
        };
        
        // Filter by time window
        Ok(alerts.into_iter()
            .filter(|a| a.is_active(start_time, end_time))
            .map(GqlServiceAlert::from)
            .collect())
    }
}
```

## Data Flow

```
                                    ┌─────────────────┐
  GTFS-RT Feed (protobuf)  ─────►  │                 │
  SIRI Feed (XML)           ─────►  │  RtUpdater      │
  SIRI-JSON Feed            ─────►  │  (tokio task)   │
                                    └────────┬────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              │              │              │
                              ▼              ▼              ▼
                    Parse protobuf    Parse XML      Parse JSON
                              │              │              │
                              └──────────────┼──────────────┘
                                             │
                                             ▼
                              nigiri_apply_gtfsrt() via FFI
                              (updates rt_timetable C++ side)
                                             │
                                             ▼
                              Arc<RwLock<RtState>> updated
                              (alerts stored Rust-side)
                                             │
                                             ▼
                              GraphQL queries read updated state
```

## Acceptance Criteria

1. GTFS-RT protobuf feed parsed and applied successfully
2. `stopDepartures` shows delays from RT updates
3. `trip(id)` shows per-stop delays and cancellations
4. `alerts(filter: {route: "seq:GOLD"})` returns active alerts
5. RT update loop runs on configurable interval
6. HTTP headers (Authorization, etc.) supported per feed
7. Feed failures don't crash the server
8. Old RT state remains valid while updates are applied

## motis Source Reference

| File | Lines | Key Functions |
|---|---|---|
| `src/rt_update.cc` | ~400 | Main RT update loop, feed polling |
| `src/rt/auser.cc` | ~300 | AUSER protocol handler |
| `src/rt/rt_metrics.h` | ~50 | Per-feed metrics |
| nigiri `src/rt/gtfsrt_update.cc` | ~500 | GTFS-RT → rt_timetable |
| nigiri `src/rt/gtfsrt_alert.cc` | ~200 | Alert processing |
| nigiri `src/rt/service_alert.cc` | ~100 | Alert storage |
