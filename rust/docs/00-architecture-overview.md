# Rust Multimodal Transit Platform — Architecture Overview

## Goal

Reimplement the core functionality of [motis](https://github.com/motis-project/motis) in Rust, served via a **GraphQL API** (see `schema.generated.graphql`). This replaces motis's REST/OpenAPI endpoint layer entirely — no motis HTTP endpoints are carried forward.

The C++ **nigiri** library remains the transit routing engine, accessed via FFI through the `nigiri-sys` and `nigiri` Rust crates already built in this workspace.

## What We Are Building

| Capability | motis C++ source | Our approach | Status |
|---|---|---|---|
| GTFS/HAFAS import & serialization | `src/import.cc`, `src/timetable/` | Orchestrate via nigiri FFI + Rust preprocessing | Planned |
| Configuration | `include/motis/config.h` | Pure Rust config (YAML/TOML) | Planned |
| Central data layer | `include/motis/data.h` | Rust `AppData` struct owning all subsystems | Planned |
| Tag lookup / ID resolution | `src/tag_lookup.cc` | Pure Rust reimplementation | Planned |
| Core transit routing | nigiri RAPTOR via `abi.h` | Extend nigiri C ABI, call from Rust | Planned |
| Street routing (OSR) | `src/transit-osr/` (C++) | **`transit-osr` Rust crate (exists)** — pure Rust port with 13 profiles, Dijkstra + bidirectional A*, elevation, sharing data | **Done** |
| Footpath computation | `src/compute_footpaths.cc` | Rust orchestration using `transit-osr` crate + nigiri FFI | Planned |
| Real-time updates | `src/rt/`, `src/rt_update.cc` | GTFS-RT protobuf parsing in Rust, feed to nigiri | Planned |
| GBFS (bike/scooter sharing) | `src/gbfs/` | Pure Rust — `transit-osr` crate already has `SharingData` for GBFS integration | Planned |
| Flex (demand-responsive) | `src/flex/` | Rust using `transit-osr` crate for area expansion | Planned |
| ODM (on-demand mobility) | `src/odm/` | Pure Rust (HTTP client + journey filtering) | Planned |
| Elevators / accessibility | `src/elevators/` | Rust parsers (SIRI-FM, FASTA) + `transit-osr` node blocking | Planned |
| Geocoding / search | `src/endpoints/adr/` via `adr` lib | **`transit-adr` Rust crate (exists)** — pure Rust port with typeahead, reverse geocoding, fuzzy matching, address formatting | **Done** |
| Map data / shapes / railviz | `src/route_shapes.cc`, `src/railviz.cc` | Rust shape processing + polyline encoding | Planned |
| Journey formatting | `src/journey_to_response.cc` | Pure Rust: nigiri journey → GraphQL types | Planned |
| **GraphQL server** | *(not in motis)* | `async-graphql` + `axum` | Planned |

## What We Are NOT Building

- motis REST/OpenAPI endpoints (`src/endpoints/`)
- motis HTTP server (`src/server.cc`, `net` library)
- Vector tile generation/serving (`tiles` library) — use external tile server
- OJP protocol adapter (`src/endpoints/ojp.cc`)
- Prometheus metrics endpoint (replace with Rust metrics crate)
- OpenAPI code generation (`openapi-cpp`)
- motis UI (`ui/`)

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      GraphQL Server (axum + async-graphql)    │
│                                                              │
│  Query { planJourney, stop, route, trip, stopDepartures,     │
│          searchLocations, reverseGeocode, nearbyStops,       │
│          alerts, reachable, isochrone, map, ... }            │
└──────────┬───────────────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────────┐
│                      Rust Application Layer                   │
│                                                              │
│  ┌─────────┐ ┌──────────┐ ┌──────┐ ┌─────┐ ┌─────────────┐ │
│  │ Routing  │ │ Journey  │ │  RT  │ │GBFS │ │  Geocoding  │ │
│  │Orchestr. │ │Formatter │ │Update│ │     │ │  / Search   │ │
│  └────┬─────┘ └────┬─────┘ └──┬───┘ └──┬──┘ └──────┬──────┘ │
│       │            │          │        │            │        │
│  ┌────▼────────────▼──────────▼────────▼────────────▼──────┐ │
│  │                    AppData (Rust struct)                  │ │
│  │  tt: Timetable (nigiri FFI)    config: Config            │ │
│  │  rtt: Arc<RwLock<RtTimetable>> tags: TagLookup           │ │
│  │  osr: transit_osr::OsrData (Rust)     gbfs: GbfsData             │ │
│  │  adr: transit_adr::Typeahead (Rust)   flex: FlexAreas             │ │
│  │  shapes: ShapesStorage        odm_bounds: OdmBounds       │ │
│  │  elevators: Elevators         reverse: transit_adr::Reverse       │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
           │
    ┌──────▼──────┐
    │  nigiri FFI │  (only C++ dependency)
    │  (abi.h)    │
    └─────────────┘
```

## Phased Rollout

Each phase builds on the previous. The order is driven by dependency chains — you can't route without a timetable, can't format journeys without routing, etc.

| Phase | Component | Type | Depends On |
|-------|-----------|------|------------|
| **0** | Foundation: types, config, GraphQL scaffold | Pure Rust | — |
| **1** | Import pipeline & timetable loading | Rust + nigiri FFI | Phase 0 |
| **2** | Tag lookup & ID resolution | Pure Rust | Phase 1 |
| **3** | Core routing (RAPTOR via nigiri) | Rust + nigiri FFI | Phases 1, 2 |
| **4** | Street routing (OSR integration) | Rust + OSR FFI | Phase 0 |
| **5** | Footpath computation | Rust + nigiri/OSR FFI | Phases 1, 4 |
| **6** | Real-time updates (GTFS-RT, SIRI) | Rust + nigiri FFI | Phases 1, 2 |
| **7** | GBFS (bike/scooter sharing) | Pure Rust | Phases 3, 4 |
| **8** | Flex routing (demand-responsive) | Rust + OSR FFI | Phases 3, 4 |
| **9** | ODM (on-demand mobility) | Pure Rust + HTTP | Phase 3 |
| **10** | Elevators & accessibility | Rust + OSR FFI | Phases 4, 6 |
| **11** | Geocoding & location search | Rust (or FFI to adr) | Phase 0 |
| **12** | Map data, route shapes, polylines | Rust + nigiri FFI | Phases 1, 4 |
| **13** | Journey formatting (→ GraphQL types) | Pure Rust | Phases 3, 4, 7, 8, 9 |

> **Note**: Phase 13 (journey formatting) is listed last because it needs awareness of all leg types (transit, walk, bike, GBFS, flex, ODM). In practice, a basic version ships with Phase 3 and is extended as each mode is added.

## External Dependencies — C++ via FFI

| C++ Library | Purpose | FFI Strategy |
|---|---|---|
| **nigiri** | Transit timetable + RAPTOR routing | Extend `abi.h` C ABI (already started) |

> **Note**: OSR and ADR are **not** C++ FFI — they are already ported to pure Rust as the `transit-osr` and `transit-adr` crates in this workspace.

## External Dependencies — Rust Crates

### Workspace Crates (Already Exist)

| Crate | Purpose | Status |
|---|---|---|
| `nigiri-sys` | FFI bindings to nigiri C++ | **Done** |
| `nigiri` | Safe Rust wrapper for nigiri | **Done** |
| `transit-osr` | Street routing (13 profiles, Dijkstra + A*, elevation, sharing) | **Done** |
| `transit-adr` | Geocoding (typeahead, reverse, fuzzy matching, formatting) | **Done** |

### New Dependencies

| Crate | Purpose | Replaces |
|---|---|---|
| `async-graphql` | GraphQL server | motis REST endpoints |
| `axum` | HTTP framework | motis `net` library |
| `tokio` | Async runtime | boost::asio |
| `prost` / `prost-build` | Protobuf (GTFS-RT) | C++ protobuf |
| `serde` + `serde_yaml` | Config parsing | reflect-cpp |
| `reqwest` | HTTP client (RT feeds, GBFS) | boost::beast |
| `chrono` / `chrono-tz` | DateTime + IANA timezones | C++ `date` library |
| `tracing` | Structured logging | motis logging |
| `metrics` / `prometheus` | Observability | prometheus-cpp |
| `quick-xml` | SIRI XML parsing | pugixml |
| `lz4_flex` | Compression | C++ lz4 |

> **Note**: `rstar`, `geo`, `rkyv`, `bincode`, `bitvec`, `memmap2` etc. are already transitive dependencies of the `transit-osr` crate.

## Key Design Decisions

1. **nigiri stays C++** — The RAPTOR algorithm, timetable data structure, and cista serialization are too tightly coupled to port. We extend the C ABI and call from Rust.

2. **OSR and ADR are pure Rust** — Both the street routing (`transit-osr` crate) and geocoding (`transit-adr` crate) already exist as complete Rust ports with full feature parity. No C++ FFI needed for these.

4. **Everything else is Rust** — Config, import orchestration, real-time feed processing, GBFS, flex area handling, ODM, journey formatting, and the GraphQL server are all pure Rust or Rust with FFI calls.

5. **GraphQL over REST** — The `schema.generated.graphql` defines the full API surface. Clients get exactly the fields they need, nested resolvers handle complex types efficiently.

6. **Immutable data + atomic RT swap** — Same pattern as motis: the timetable is immutable after load, real-time state is behind `Arc<RwLock<>>` and swapped atomically.

## File Layout (Target)

```
rust/
├── Cargo.toml                    # Workspace root
├── schema.generated.graphql      # GraphQL schema (reference)
├── docs/                         # This documentation
├── nigiri-sys/                   # FFI bindings to nigiri (exists)
├── nigiri/                       # Safe nigiri wrapper (exists)
├── transit-osr/                  # Street routing — pure Rust (EXISTS)
├── transit-adr/                  # Geocoding & search — pure Rust (EXISTS)
├── transit-core/                 # Shared types, config, IDs
├── transit-import/               # Import pipeline orchestration
├── transit-rt/                   # Real-time update processing
├── transit-routing/              # Routing orchestration (multi-modal)
├── transit-gbfs/                 # GBFS integration
├── transit-flex/                 # Flex/DRT routing
├── transit-odm/                  # On-demand mobility
├── transit-elevators/            # Elevator/accessibility
├── transit-geocoding/            # Geocoding & search
├── transit-shapes/               # Route shapes & polylines
├── transit-server/               # GraphQL server (axum + async-graphql)
└── transit-app/                  # Binary entry point
```

## motis Source Reference

The motis codebase at `reference/motis/` contains ~24,000 LOC across these subsystems:

| Component | Files | LOC | Complexity |
|---|---|---|---|
| Endpoints (NOT ported) | 20+ .cc | 5,000+ | — |
| Import pipeline | 3 .cc | 1,000+ | Very Complex |
| Data layer | 1 .cc | 300+ | Complex (façade) |
| Config | 1 .cc | 600+ | Moderate |
| Journey formatting | 1 .cc | 2,000+ | Very Complex |
| Routing orchestration | 2 .cc | 1,500+ | Complex |
| OSR integration | 4 .cc | 1,500+ | Moderate |
| GBFS | 9 .cc | 2,000+ | Complex |
| Flex routing | 6 .cc | 1,000+ | Complex |
| ODM | 11 .cc | 800+ | Moderate |
| RT updates | 10+ .cc | 2,000+ | Complex |
| Elevators | 6 .cc | 1,000+ | Moderate |
| Footpath computation | 1 .cc | 800+ | Complex |
| Tag lookup | 1 .cc | 300+ | Moderate |
| Utilities | 12+ .cc | 3,000+ | Various |
| **Total (ported)** | | **~18,000** | |

Net of the ~5,000 LOC of endpoints we skip, roughly **18,000 LOC of C++** maps to Rust code. Pure-Rust components (config, GBFS, ODM, journey formatting) will likely be comparable in size. FFI orchestration (nigiri, OSR) adds overhead but reuses existing C++ implementations.
