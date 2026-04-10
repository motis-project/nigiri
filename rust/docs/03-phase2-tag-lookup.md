# Phase 2 — Tag Lookup & ID Resolution

**Priority**: Third  
**Type**: Pure Rust  
**Depends on**: Phase 1 (timetable loaded)  
**Crate**: `transit-core` (module: `tag_lookup`)  
**Estimated effort**: Small–Medium

## Objective

Implement bidirectional mapping between human-readable IDs (used in the GraphQL API) and nigiri's internal numeric indices. This is the foundation for every API query that references a stop, trip, or route by ID.

## What motis Does (`src/tag_lookup.cc`, ~300 LOC)

### ID Format Conventions

| Entity | Format | Example |
|---|---|---|
| **Location ID** | `{tag}_{gtfs_stop_id}` | `seq_600011` |
| **Trip ID** | `{YYYYMMDD}_{HH:MM}_{tag}_{gtfs_trip_id}` | `20240510_14:30_seq_123456` |
| **Route ID** | `{tag}_{gtfs_route_id}` | `seq_GOLD` |

> Our GraphQL schema uses `:` as separator (e.g. `seq:600011`) — adjust format accordingly.

### Core Data Structure

```cpp
struct tag_lookup {
    // Bidirectional tag ↔ source_idx mapping
    vecvec<source_idx_t, char> src_to_tag_;       // source_idx → tag string
    hash_map<string, source_idx_t> tag_to_src_;   // tag string → source_idx
    
    // ID generation
    string id(timetable&, location_idx_t) const;
    string id(timetable&, rt::run_stop, event_type) const;
    string route_id(rt::run_stop, event_type) const;
    
    // ID parsing & resolution
    location_idx_t get_location(timetable&, string_view) const;
    optional<location_idx_t> find_location(timetable&, string_view) const;
    pair<rt::run, trip_idx_t> get_trip(timetable&, rt_timetable*, string_view) const;
    
    // Serialization
    void write(path) const;
    static wrapped<tag_lookup> read(path);
};
```

### Trip ID Components

```cpp
template <typename T>
struct trip_id {
    T start_date_;   // "YYYYMMDD"
    T start_time_;   // "HH:MM"
    T tag_;          // dataset tag
    T trip_id_;      // GTFS trip_id
};
```

The `split_trip_id()` function parses `YYYYMMDD_HH:MM_tag_trip_id` with exactly 3 underscores separating the first 3 components, and the remainder being the trip ID (which may itself contain underscores).

### How IDs Are Built

**Location ID**: `tag_lookup::id(tt, location_idx)`
1. Get `source_idx` from the location
2. Look up tag string for that source
3. Get the GTFS ID from `tt.locations.ids[src][location_idx]`
4. Return `"{tag}_{id}"`

**Trip ID**: `tag_lookup::id(tt, run_stop, event_type)`
1. Get the `trip_idx` from the transport's trip list
2. Get the GTFS trip ID string
3. Get the trip's start date (day_idx → calendar date → YYYYMMDD)
4. Get the departure time of first stop (→ HH:MM, can exceed 24:00)
5. Return `"{date}_{time}_{tag}_{trip_id}"`

### How IDs Are Resolved

**Location lookup**: `get_location(tt, "tag_id")`
1. Split at first `_` → tag, remainder
2. `tag_to_src_[tag]` → `source_idx`
3. `tt.find(location_id{remainder, source_idx})` → `location_idx_t`

**Trip lookup**: `get_trip(tt, rtt, "date_time_tag_trip_id")`
1. `split_trip_id()` → `{date, time, tag, trip_id}` components
2. `tag_to_src_[tag]` → `source_idx`
3. `gtfsrt_resolve_run(tt, rtt, source_idx, trip_id, date, time)`
4. Returns `(rt::run, trip_idx_t)`

## Rust Implementation

### Data Structure

```rust
// transit-core/src/tag_lookup.rs

use std::collections::HashMap;

pub struct TagLookup {
    /// source_idx → tag string (e.g., 0 → "seq")
    src_to_tag: Vec<String>,
    /// tag string → source_idx (e.g., "seq" → 0)
    tag_to_src: HashMap<String, SourceIdx>,
}

#[derive(Debug, Clone)]
pub struct TripId {
    pub start_date: String,   // "YYYYMMDD"
    pub start_time: String,   // "HH:MM"
    pub tag: String,
    pub trip_id: String,
}
```

### ID Separator

The GraphQL schema uses `:` as separator (e.g. `SEQ:600011`). Define this as a constant:

```rust
const ID_SEPARATOR: char = ':';

impl TagLookup {
    /// Build a location ID string: "{tag}:{gtfs_id}"
    pub fn location_id(&self, tt: &Timetable, loc_idx: LocationIdx) -> Option<String> {
        let src = tt.location_source(loc_idx)?;
        let tag = self.src_to_tag.get(src.0 as usize)?;
        let gtfs_id = tt.location_gtfs_id(loc_idx, src)?;
        Some(format!("{}{}{}", tag, ID_SEPARATOR, gtfs_id))
    }
    
    /// Build a trip ID string: "{tag}:{YYYYMMDD}_{HH:MM}_{trip_id}"
    pub fn trip_id(&self, tt: &Timetable, transport_idx: TransportIdx, 
                   day_idx: DayIdx) -> Option<String> {
        let src = tt.transport_source(transport_idx)?;
        let tag = self.src_to_tag.get(src.0 as usize)?;
        let gtfs_trip_id = tt.transport_gtfs_trip_id(transport_idx)?;
        let date = tt.day_to_date(day_idx)?;  // → "YYYYMMDD"
        let time = tt.transport_first_departure(transport_idx)?;  // → "HH:MM"
        Some(format!("{}{}{}_{}_{}",
            tag, ID_SEPARATOR, date, time, gtfs_trip_id))
    }
    
    /// Build a route ID string: "{tag}:{gtfs_route_id}"
    pub fn route_id(&self, tt: &Timetable, route_idx: RouteIdx) -> Option<String> {
        let src = tt.route_source(route_idx)?;
        let tag = self.src_to_tag.get(src.0 as usize)?;
        let gtfs_id = tt.route_gtfs_id(route_idx)?;
        Some(format!("{}{}{}", tag, ID_SEPARATOR, gtfs_id))
    }
}
```

### ID Parsing

```rust
impl TagLookup {
    /// Parse a location ID "tag:gtfs_id" → (source_idx, gtfs_id)
    pub fn parse_location_id(&self, id: &str) -> Result<(SourceIdx, &str), TransitError> {
        let (tag, gtfs_id) = id.split_once(ID_SEPARATOR)
            .ok_or_else(|| TransitError::InvalidId(format!(
                "expected format 'tag:id', got '{}'", id
            )))?;
        let src = self.tag_to_src.get(tag)
            .copied()
            .ok_or_else(|| TransitError::NotFound(format!("unknown tag '{}'", tag)))?;
        Ok((src, gtfs_id))
    }
    
    /// Parse a trip ID "tag:YYYYMMDD_HH:MM_trip_id"
    pub fn parse_trip_id(&self, id: &str) -> Result<TripId, TransitError> {
        let (tag, rest) = id.split_once(ID_SEPARATOR)
            .ok_or_else(|| TransitError::InvalidId(id.to_string()))?;
        
        // rest = "YYYYMMDD_HH:MM_trip_id"
        // First underscore separates date
        let (date, rest) = rest.split_once('_')
            .ok_or_else(|| TransitError::InvalidId(id.to_string()))?;
        
        // Second underscore separates time (HH:MM) from trip_id
        let (time, trip_id) = rest.split_once('_')
            .ok_or_else(|| TransitError::InvalidId(id.to_string()))?;
        
        // Validate date format (8 digits)
        if date.len() != 8 || !date.chars().all(|c| c.is_ascii_digit()) {
            return Err(TransitError::InvalidId(format!("invalid date: {}", date)));
        }
        
        // Validate time format (HH:MM, may exceed 24:00 for overnight)
        if !time.contains(':') {
            return Err(TransitError::InvalidId(format!("invalid time: {}", time)));
        }
        
        Ok(TripId {
            start_date: date.to_string(),
            start_time: time.to_string(),
            tag: tag.to_string(),
            trip_id: trip_id.to_string(),
        })
    }
    
    /// Resolve location ID to nigiri location_idx
    pub fn resolve_location(&self, tt: &Timetable, id: &str) -> Result<LocationIdx, TransitError> {
        let (src, gtfs_id) = self.parse_location_id(id)?;
        tt.find_location_by_source_id(src, gtfs_id)
            .ok_or_else(|| TransitError::NotFound(format!("location not found: {}", id)))
    }
    
    /// Resolve trip ID to nigiri transport + day
    pub fn resolve_trip(&self, tt: &Timetable, id: &str) -> Result<(TransportIdx, DayIdx), TransitError> {
        let tid = self.parse_trip_id(id)?;
        let src = self.tag_to_src.get(&tid.tag)
            .copied()
            .ok_or_else(|| TransitError::NotFound(format!("unknown tag: {}", tid.tag)))?;
        tt.resolve_trip(src, &tid.trip_id, &tid.start_date, &tid.start_time)
            .ok_or_else(|| TransitError::NotFound(format!("trip not found: {}", id)))
    }
}
```

### nigiri C ABI Extensions Needed

To support tag lookup, we need these additional functions in `abi.h`:

```c
// Get the source index for a location
uint32_t nigiri_get_location_source(nigiri_timetable_t* t, uint32_t location_idx);

// Get the GTFS ID for a location at a given source
bool nigiri_get_location_gtfs_id(
    nigiri_timetable_t* t, uint32_t location_idx, uint32_t source_idx,
    const char** id_out, uint32_t* id_len_out);

// Get the GTFS trip ID for a transport
bool nigiri_get_transport_gtfs_trip_id(
    nigiri_timetable_t* t, uint32_t transport_idx,
    const char** id_out, uint32_t* id_len_out);

// Get the GTFS route ID for a route
bool nigiri_get_route_gtfs_id(
    nigiri_timetable_t* t, uint32_t route_idx,
    const char** id_out, uint32_t* id_len_out);

// Get the source for a transport
uint32_t nigiri_get_transport_source(nigiri_timetable_t* t, uint32_t transport_idx);

// Day index → date string (YYYYMMDD, caller provides 9-byte buffer)
bool nigiri_day_to_date_str(
    nigiri_timetable_t* t, uint32_t day_idx,
    char* buf_out);

// Resolve trip from source + trip_id + date + time
bool nigiri_resolve_trip(
    nigiri_timetable_t* t,
    uint32_t source_idx,
    const char* trip_id, uint32_t trip_id_len,
    const char* date_str, uint32_t date_len,    // "YYYYMMDD"
    const char* time_str, uint32_t time_len,    // "HH:MM"
    uint32_t* transport_idx_out,
    uint32_t* day_idx_out);
```

### Serialization

TagLookup should be serializable to disk alongside the timetable:

```rust
impl TagLookup {
    pub fn write(&self, path: &Path) -> Result<(), TransitError> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
    
    pub fn read(path: &Path) -> Result<Self, TransitError> {
        let bytes = std::fs::read(path)?;
        Ok(bincode::deserialize(&bytes)?)
    }
}
```

### Construction During Import

```rust
impl ImportPipeline {
    fn build_tag_lookup(&self, tt: &Timetable) -> Result<TagLookup, TransitError> {
        let mut lookup = TagLookup::new();
        
        for (tag, _dataset) in &self.config.timetable.as_ref().unwrap().datasets {
            let src_idx = SourceIdx(lookup.src_to_tag.len() as u32);
            lookup.src_to_tag.push(tag.clone());
            lookup.tag_to_src.insert(tag.clone(), src_idx);
        }
        
        Ok(lookup)
    }
}
```

## Acceptance Criteria

1. `stop(id: "seq:600011")` resolves to correct nigiri `location_idx_t`
2. `trip(id: "seq:20240510_14:30_trip_123")` resolves to correct transport + day
3. `route(id: "seq:GOLD")` resolves to correct route
4. Round-trip: build ID from index, parse it back, get the same index
5. Multi-source: two datasets with tags "seq" and "cns" can coexist
6. Invalid IDs return clear error messages (not panics)
7. Tag lookup can be serialized/deserialized to disk

## motis Source Reference

| File | Lines | Key Functions |
|---|---|---|
| `src/tag_lookup.cc` | ~300 | `id()`, `get_location()`, `get_trip()`, `split_trip_id()` |
| `include/motis/tag_lookup.h` | ~80 | Struct definition, `trip_id<T>` template |
