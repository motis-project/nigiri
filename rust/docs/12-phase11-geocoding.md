# Phase 11 — Geocoding & Search (ADR Integration)

**Priority**: Twelfth  
**Type**: Pure Rust (using existing `transit-adr` crate)  
**Depends on**: Phase 0 (types), Phase 1 (timetable)  
**Crate**: `transit-adr` (exists at `rust/transit-adr/`), integration in `transit-import` + `transit-server`  
**Estimated effort**: Small–Medium (crate already exists, only integration needed)

## Status: `transit-adr` Crate Already Implemented

The `transit-adr` crate at `rust/transit-adr/` is a **complete pure-Rust** geocoding library. No C++ FFI is needed. This phase focuses on:

1. Extending the typeahead database with transit stop data during import
2. Wiring geocoding into the GraphQL API

### Existing Capabilities

| Feature | Status | Details |
|---|---|---|
| Typeahead search | **Done** | Bigram-indexed fuzzy search with `guess()` |
| Reverse geocoding | **Done** | R-tree-based `Reverse::lookup()` |
| Address formatting | **Done** | Country-specific Mustache-like templates |
| Fuzzy matching | **Done** | SIFT4, Levenshtein, bigram cosine similarity |
| Area database | **Done** | Point-in-polygon for admin area lookup |
| Import system | **Done** | `ImportContext` for building the database |
| ~280 amenity categories | **Done** | With scoring and classification |

### Key Types (Already Exist)

```rust
// Main search database — transit-adr/src/typeahead.rs
pub struct Typeahead {
    // Bigram index, place/street/area data, language support
    pub fn guess(&self, input: &str, context: &mut GuessContext) -> Vec<Suggestion>;
}

// Reverse geocoding — transit-adr/src/reverse.rs
pub struct Reverse {
    pub fn lookup(&self, t: &Typeahead, lat: f64, lng: f64,
                  n_guesses: usize, filter: Option<&str>) -> Vec<Suggestion>;
}

// Search result — transit-adr/src/suggestion.rs
pub struct Suggestion {
    pub lat: f64,
    pub lng: f64,
    pub name: String,
    pub area_set: u32,
    // ... matched info, score, etc.
}

// Building the database — transit-adr/src/import_context.rs
pub struct ImportContext {
    pub fn get_or_create_place(&mut self, ...) -> PlaceIdx;
    pub fn get_or_create_street(&mut self, ...) -> StreetIdx;
    pub fn add_place(&mut self, name: &str, lat: f64, lng: f64, ...);
    pub fn add_admin_area(&mut self, ...);
}

// Address formatting — transit-adr/src/formatter.rs
pub struct Formatter {
    pub fn format(&self, suggestion: &Suggestion, lang: &str) -> String;
}
```

## What motis Does (`src/adr_extend_tt.cc`, ~400 LOC)

### Extending Typeahead with Transit Stops

motis enriches the adr database with transit stop data so that stop names appear in search results. The process:

1. **Group equivalent stops** — Stops with similar names within distance threshold are grouped
2. **Compute importance scores** — Weighted by transport class (Air/HighSpeed=300, Bus=2, Tram=3)
3. **Build class masks** — Which transport modes serve each stop (for mode filtering)
4. **Create places** — Add grouped stops as adr places with importance + class mask

### Name Equivalence

```rust
/// Two stops are equivalent if their name similarity and distance meet the threshold.
fn are_equivalent(name_a: &str, name_b: &str, distance_meters: f64) -> bool {
    let str_diff = sift4_distance(name_a, name_b); // normalized 0.0..1.0
    distance_meters < (500.0 - 1750.0 * str_diff)
    // Near identical names: up to 500m apart
    // Slightly different names: closer together required
}
```

### Importance Scoring

```rust
/// Importance based on transport class priority.
fn transport_class_weight(clasz: TransportClass) -> u32 {
    match clasz {
        TransportClass::Air | TransportClass::HighSpeed => 300,
        TransportClass::LongDistance => 200,
        TransportClass::Coach => 100,
        TransportClass::Night => 100,
        TransportClass::RegionalFast => 50,
        TransportClass::Regional => 50,
        TransportClass::Metro => 10,
        TransportClass::Subway => 10,
        TransportClass::Tram => 3,
        TransportClass::Bus => 2,
        TransportClass::Ship => 5,
        TransportClass::Other => 1,
    }
}
```

## Rust Implementation

### ADR Extension During Import

```rust
// transit-import/src/adr_extend.rs

use transit_adr::{ImportContext, Typeahead};

pub struct AdrExtension {
    pub location_to_place: HashMap<LocationIdx, PlaceIdx>,
    pub class_masks: HashMap<PlaceIdx, u32>,
}

pub fn extend_typeahead_with_stops(
    tt: &nigiri::Timetable,
    import_ctx: &mut ImportContext,
) -> AdrExtension {
    let mut ext = AdrExtension {
        location_to_place: HashMap::new(),
        class_masks: HashMap::new(),
    };

    // 1. Build equivalence groups
    let groups = group_equivalent_stops(tt);

    // 2. For each group, compute importance and create place
    for group in &groups {
        let (importance, class_mask) = compute_group_importance(tt, group);
        let representative = &group[0];

        let loc = tt.get_location(representative.idx);
        let place_idx = import_ctx.add_place(
            &loc.name,
            loc.lat,
            loc.lon,
            importance,
        );

        // Map all group members to this place
        for member in group {
            ext.location_to_place.insert(member.idx, place_idx);
        }
        ext.class_masks.insert(place_idx, class_mask);
    }

    ext
}

fn group_equivalent_stops(tt: &nigiri::Timetable) -> Vec<Vec<StopEntry>> {
    let stops: Vec<StopEntry> = tt.enumerate_stops()
        .map(|idx| {
            let loc = tt.get_location(idx);
            StopEntry { idx, name: loc.name.clone(), lat: loc.lat, lon: loc.lon }
        })
        .collect();

    // Union-find grouping by name similarity + distance
    let mut uf = UnionFind::new(stops.len());

    for i in 0..stops.len() {
        for j in (i + 1)..stops.len() {
            let dist = haversine_distance(
                Position { lat: stops[i].lat, lon: stops[i].lon },
                Position { lat: stops[j].lat, lon: stops[j].lon },
            );
            if are_equivalent(&stops[i].name, &stops[j].name, dist) {
                uf.union(i, j);
            }
        }
    }

    // Collect groups
    let mut groups: HashMap<usize, Vec<StopEntry>> = HashMap::new();
    for (i, stop) in stops.into_iter().enumerate() {
        groups.entry(uf.find(i)).or_default().push(stop);
    }
    groups.into_values().collect()
}

fn compute_group_importance(
    tt: &nigiri::Timetable,
    group: &[StopEntry],
) -> (f64, u32) {
    let mut total_weight = 0u64;
    let mut class_mask = 0u32;

    for stop in group {
        for route in tt.routes_at(stop.idx) {
            let clasz = route.transport_class();
            total_weight += transport_class_weight(clasz) as u64
                * route.trip_count() as u64;
            class_mask |= 1 << (clasz as u32);
        }
    }

    // Normalize to [0, 1]
    let max_expected = 300 * 1000; // rough upper bound
    let importance = (total_weight as f64 / max_expected as f64).min(1.0);

    (importance, class_mask)
}
```

### GraphQL Resolvers

```rust
// transit-server/src/resolvers/search.rs

use transit_adr::{Typeahead, Reverse, GuessContext, Suggestion};

#[Object]
impl QueryRoot {
    /// Search for locations by name (typeahead/autocomplete)
    async fn search_locations(
        &self,
        ctx: &Context<'_>,
        query: String,
        #[graphql(default = 10)] limit: usize,
        language: Option<String>,
    ) -> Vec<LocationResult> {
        let data = ctx.data::<AppData>().unwrap();
        let mut guess_ctx = GuessContext::new();

        let suggestions = data.typeahead.guess(&query, &mut guess_ctx);

        suggestions.into_iter()
            .take(limit)
            .map(|s| suggestion_to_location_result(s, &data.formatter, language.as_deref()))
            .collect()
    }

    /// Reverse geocode coordinates to nearest address/place
    async fn reverse_geocode(
        &self,
        ctx: &Context<'_>,
        lat: f64,
        lon: f64,
        #[graphql(default = 5)] limit: usize,
    ) -> Vec<LocationResult> {
        let data = ctx.data::<AppData>().unwrap();

        data.reverse.lookup(&data.typeahead, lat, lon, limit, None)
            .into_iter()
            .map(|s| suggestion_to_location_result(s, &data.formatter, None))
            .collect()
    }
}

fn suggestion_to_location_result(
    suggestion: Suggestion,
    formatter: &Formatter,
    lang: Option<&str>,
) -> LocationResult {
    LocationResult {
        name: suggestion.name.clone(),
        formatted_address: formatter.format(&suggestion, lang.unwrap_or("en")),
        lat: suggestion.lat,
        lon: suggestion.lng,
        type_: location_type_from_suggestion(&suggestion),
    }
}
```

## Data Flow

```
Import Pipeline:
  GTFS stops ─→ group by name+distance ─→ compute importance ─→ add to transit-adr Typeahead

GraphQL Query:
  searchLocations("Central Station")
    └─ transit_adr::Typeahead::guess("Central Station")
    └─ Format results with transit_adr::Formatter
    └─ Return [{name, lat, lon, type, formatted_address}]

  reverseGeocode(lat: -27.47, lon: 153.02)
    └─ transit_adr::Reverse::lookup(lat, lon, 5, None)
    └─ Format results
    └─ Return [{name, lat, lon, type, formatted_address}]
```

## Acceptance Criteria

1. Transit stops added to typeahead database during import
2. Equivalent stops grouped by name similarity + distance
3. Importance scores reflect transport class weights
4. Class masks enable mode-based filtering
5. `searchLocations` GraphQL query returns fuzzy-matched results
6. `reverseGeocode` returns nearest locations to coordinates
7. Address formatting uses country-specific templates
8. Transit stops appear alongside POIs and addresses in search results

## motis Source Reference

| File | Lines | Key Pattern |
|---|---|---|
| `src/adr_extend_tt.cc` | ~400 | Stop grouping, importance scoring, class masks |
| `include/motis/adr_extend_tt.h` | ~40 | `adr_ext` struct, classification masks |
