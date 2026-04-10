# Phase 1 — Import Pipeline & Timetable Loading

**Priority**: Second (immediately after foundation)  
**Type**: Rust orchestration + nigiri FFI  
**Depends on**: Phase 0 (types, config)  
**Crate**: `transit-import`  
**Estimated effort**: Large

## Objective

Load GTFS/HAFAS datasets into a nigiri timetable via FFI, with Rust-side orchestration for multi-dataset loading, incremental rebuilds, and pre-processing.

## What motis Does

### Import DAG (`src/import.cc`, ~1,000 LOC)

motis organizes the import as a task DAG with hash-based change detection:

1. **Hash inputs** — SHA256 of GTFS files, OSM data, config → stored in `meta/` directory
2. **Load timetable** — `nigiri::loader::load()` with config for:
   - `first_day`, `num_days`, `with_shapes`, `merge_dupes`, `link_stop_distance`
   - Per-dataset: `path`, `bikes_allowed`, `cars_allowed`, `extend_calendar`, `default_timezone`
3. **Build R-tree** — Spatial index over all timetable locations
4. **Initialize tag lookup** — Map dataset tags to `source_idx_t`
5. *(Later phases)*: OSR footpaths, platform matching, shape routing, flex areas, elevators

### Timetable Config → nigiri Loader

The motis config maps to `nigiri::loader::loader_config`:

```cpp
struct loader_config {
  std::string_view first_day;     // "TODAY" or YYYY-MM-DD
  std::uint16_t num_days;
  bool with_shapes;
  bool adjust_footpaths;
  bool merge_dupes_intra_src;
  bool merge_dupes_inter_src;
  unsigned link_stop_distance;
  std::uint16_t max_footpath_length;
  // ... per-dataset settings
};
```

### R-tree Construction

After loading, motis builds a `point_rtree<location_idx_t>` using each location's coordinates. This enables `nearbyStops`, spatial queries, and map bounding-box lookups.

## nigiri C ABI — Existing Functions

Already in `abi.h` (from our previous work):

| Function | Purpose |
|---|---|
| `nigiri_load(path, from_ts, to_ts)` | Load GTFS from directory |
| `nigiri_load_linking_stops(path, from_ts, to_ts, max_distance)` | Load with footpath linking |
| `nigiri_get_transport_count(t)` | Number of transports |
| `nigiri_get_route_count(t)` | Number of routes |
| `nigiri_get_location_count(t)` | Number of locations |
| `nigiri_get_location(t, idx)` | Location name, position, type |
| `nigiri_get_transport(t, idx)` | Transport metadata |
| `nigiri_get_route(t, idx)` | Route metadata |

## nigiri C ABI — New Functions Needed

These new C functions must be added to `include/nigiri/abi.h` and `src/abi.cc`:

```c
// Multi-dataset loading (motis-style with config)
nigiri_timetable_t* nigiri_load_multi(
    const char* data_path,          // output directory for serialized tt
    const char* config_json,        // JSON config string
    uint32_t config_len
);

// Location iteration with full detail
typedef struct {
    double lat;
    double lon;
    const char* name;
    uint32_t name_len;
    const char* id;                // source:gtfs_id
    uint32_t id_len;
    uint8_t location_type;         // 0=stop, 1=station, 2=entrance, ...
    uint32_t parent_idx;           // parent station index (UINT32_MAX = none)
    uint8_t wheelchair_boarding;   // 0=unknown, 1=yes, 2=no
} nigiri_location_detail_t;

bool nigiri_get_location_detail(
    nigiri_timetable_t* t,
    uint32_t location_idx,
    nigiri_location_detail_t* out
);

// Route with full detail
typedef struct {
    const char* short_name;
    uint32_t short_name_len;
    const char* long_name;
    uint32_t long_name_len;
    const char* agency_name;
    uint32_t agency_name_len;
    const char* agency_id;
    uint32_t agency_id_len;
    uint8_t route_type;    // GTFS route_type → clasz enum
    uint32_t color;        // 0xRRGGBB
    uint32_t text_color;
} nigiri_route_detail_t;

bool nigiri_get_route_detail(
    nigiri_timetable_t* t,
    uint32_t route_idx,
    nigiri_route_detail_t* out
);

// Enumerate all locations for a source
uint32_t nigiri_get_source_count(nigiri_timetable_t* t);

// Get the GTFS ID for a location (src:id pair)
bool nigiri_get_location_source_id(
    nigiri_timetable_t* t,
    uint32_t location_idx,
    uint32_t source_idx,
    const char** id_out,
    uint32_t* id_len_out
);

// Serialize timetable to file (mmap-able)
bool nigiri_save(nigiri_timetable_t* t, const char* path, uint32_t path_len);

// Load from serialized file
nigiri_timetable_t* nigiri_load_serialized(const char* path, uint32_t path_len);
```

## Rust Implementation

### Import Orchestrator

```rust
// transit-import/src/lib.rs

pub struct ImportPipeline {
    config: Config,
    data_path: PathBuf,
}

impl ImportPipeline {
    pub fn new(config: Config, data_path: PathBuf) -> Self { ... }
    
    /// Run the full import pipeline
    pub async fn run(&self) -> Result<AppData, TransitError> {
        // 1. Check hashes for incremental rebuild
        let hashes = self.load_hashes()?;
        
        // 2. Load timetable (nigiri FFI)
        let tt = self.load_timetable()?;
        
        // 3. Build location R-tree
        let rtree = self.build_location_rtree(&tt)?;
        
        // 4. Initialize tag lookup
        let tags = self.build_tag_lookup(&tt)?;
        
        // 5. Save hashes
        self.save_hashes(&hashes)?;
        
        Ok(AppData { tt, tags, location_rtree: rtree, ... })
    }
}
```

### Hash-Based Change Detection

```rust
// transit-import/src/hashes.rs
use std::collections::HashMap;
use sha2::{Sha256, Digest};

pub struct ImportHashes {
    hashes: HashMap<String, String>,
    path: PathBuf,
}

impl ImportHashes {
    pub fn load(path: &Path) -> Self { ... }
    pub fn save(&self) -> Result<(), TransitError> { ... }
    
    pub fn needs_rebuild(&self, component: &str, inputs: &[&Path]) -> bool {
        let current = self.hash_inputs(inputs);
        self.hashes.get(component) != Some(&current)
    }
    
    fn hash_inputs(&self, paths: &[&Path]) -> String {
        let mut hasher = Sha256::new();
        for path in paths {
            // Hash file contents or directory listing
        }
        hex::encode(hasher.finalize())
    }
}
```

### R-tree Construction

```rust
use rstar::{RTree, RTreeObject, AABB, PointDistance};

#[derive(Debug, Clone)]
pub struct LocationEntry {
    pub idx: LocationIdx,
    pub lat: f64,
    pub lon: f64,
}

impl RTreeObject for LocationEntry {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.lon, self.lat])
    }
}

pub fn build_location_rtree(tt: &nigiri::Timetable) -> RTree<LocationEntry> {
    let count = tt.location_count();
    let mut entries = Vec::with_capacity(count as usize);
    for i in 0..count {
        if let Ok(loc) = tt.get_location(i) {
            entries.push(LocationEntry {
                idx: LocationIdx(i),
                lat: loc.lat,
                lon: loc.lon,
            });
        }
    }
    RTree::bulk_load(entries)
}
```

### GraphQL Resolvers (Phase 1 additions)

```rust
#[async_graphql::Object]
impl QueryRoot {
    /// Get stop by ID
    async fn stop(&self, ctx: &Context<'_>, id: ID) -> Result<Option<GqlStop>> {
        let data = ctx.data::<AppData>()?;
        let loc_idx = data.tags.get_location(&id)?;
        // Build GqlStop from nigiri location data
    }
    
    /// Find stops near a location
    async fn nearby_stops(&self, ctx: &Context<'_>, 
                          lat: f64, lon: f64, radius: f64) -> Result<Vec<GqlStop>> {
        let data = ctx.data::<AppData>()?;
        let nearby = data.location_rtree.locate_within_distance([lon, lat], radius);
        // Convert to GqlStop
    }
    
    /// Get route by ID
    async fn route(&self, ctx: &Context<'_>, id: ID) -> Result<Option<GqlRoute>> { ... }
    
    /// Search stops
    async fn stops(&self, ctx: &Context<'_>, source: Option<String>,
                   phrase: Option<String>, pagination: Option<PaginationInput>
    ) -> Result<StopsConnection> { ... }
}
```

## Data Flow

```
config.yaml
  └─ datasets: { "gtfs": { path: "/data/gtfs.zip" } }
       │
       ▼
transit-import::ImportPipeline::run()
  │
  ├─ hash_inputs(gtfs_dir) → compare with meta/hashes
  │
  ├─ if changed: nigiri_load(path, from_ts, to_ts) via FFI
  │       └─ Returns nigiri_timetable_t* (opaque C pointer)
  │
  ├─ nigiri::Timetable::from_raw(ptr) → safe Rust wrapper
  │
  ├─ iterate all locations → build RTree<LocationEntry>
  │
  ├─ iterate sources → build TagLookup { tag ↔ source_idx }
  │
  └─ return AppData { tt, tags, location_rtree }
```

## Acceptance Criteria

1. GTFS ZIP/directory loads successfully via nigiri FFI
2. `{ stop(id: "tag:stop_id") { name, position { lat, lon } } }` resolves
3. `{ nearbyStops(lat: -27.47, lon: 153.02, radius: 500) { name } }` returns results
4. `{ route(id: "tag:route_1") { shortName, type } }` resolves
5. Location R-tree supports bounding-box and radius queries
6. Incremental rebuild skips unchanged datasets

## motis Source Reference

| File | Lines | Key Functions |
|---|---|---|
| `src/import.cc` | ~1,000 | `import()` — full DAG orchestration |
| `src/data.cc` | ~300 | `load_tt()`, `init_rtt()` |
| `include/motis/config.h` | ~600 | `timetable::dataset` config |
| `include/nigiri/loader/load.h` | ~100 | `load()` entry point |
| `src/transit-adr_extend_tt.cc` | ~300 | Links stops to places (Phase 11) |
