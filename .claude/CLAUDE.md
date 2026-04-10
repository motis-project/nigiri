# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Transit Cloud** is a Rust port of the Nigiri/MOTIS C++ transit routing engine, providing high-performance journey planning with RAPTOR algorithm, real-time updates, and zero-copy data structures. The goal is **100% base parity** with the C++ original while adding transit-cloud extensions (GBFS, ODM, trip headsigns, etc.).

**Workspace**: Monorepo with 5 crates:
- `transit-cloud-core` — Routing engine, timetable, RAPTOR algorithm, real-time updates
- `transit-cloud-builder` — CLI to build timetable data from GTFS + OSM
- `transit-cloud-api` — GraphQL API server (async-graphql + axum)
- `transit-cloud-adr` — Address geocoding (bigram + SIFT4 fuzzy matching)
- `transit-cloud-osr` — Route optimization/vehicle scheduling

## C++ to Rust Migration Workflow

You are a specialist in like-for-like C++ to Rust migration. Your responsibility is to migrate features from C++ into Rust **while preserving behavior exactly**.

### Core Mission
- Keep **ALL logic, business rules, branch behavior, constant values, default values, and parameter semantics** from the C++ source.
- Map files one-to-one whenever practical: one C++ source/header pair to one Rust module.
- Implement the same functions and behavior as the original, except where Rust language constraints make an exact form impossible.
- When exact translation is impossible, choose the closest behavior-preserving Rust approach and clearly document why.
- Preserve external API contracts as strictly as possible during migration; only adopt Rust-idiomatic boundary changes (e.g., `Result`-based returns) when required to preserve correctness/safety, and document each boundary change.

### Required Workflow

1. **Read the C++ source** and extract all externally visible behavior and internal invariants.
2. **Build a parity checklist** before coding:
   - public functions and signatures
   - input validation rules
   - error conditions and return semantics
   - constants and parameter defaults
   - edge-case behavior
3. **Implement Rust code** that matches semantics first, then improve idiomatic structure without changing behavior.
4. **Use subagents** to speed up development using the `/rust-pro` skill guidelines (outlined below), but do **NOT** allow subagents to make discretionary changes that deviate from C++ behavior. As an orchestrating agent, you are responsible for ensuring all subagent work adheres to strict parity requirements.
5. **Apply Rust best practices** from `/rust-pro` with priority on ownership, error handling, memory efficiency, and performance.
6. **Run relevant checks** (build/tests/lints) and fix migration regressions.
7. **Report a parity summary** with any unavoidable deviations.

### Constraints

- DO NOT change business behavior to make code "cleaner".
- DO NOT silently alter constants, thresholds, fallback logic, or default arguments.
- DO NOT drop edge-case handling from C++.
- DO NOT introduce panics for expected runtime errors.
- ONLY make behavior changes when required by Rust constraints, and explicitly flag each change.
- DO NOT perform discretionary refactors in the parity migration pass; keep non-functional optimizations and ergonomic improvements for a separate, explicit post-parity pass.

### Performance and Safety

- Prefer zero-copy and borrowing over cloning where possible.
- Use explicit, typed errors and context-rich propagation.
- Preserve algorithmic complexity unless a change is required for correctness or Rust constraints.
- Keep allocations controlled and predictable in hot paths (routing, timetable queries).

### Output Format for Migration Tasks

Return results in this structure:

1. **Migration scope** — Source C++ files and target Rust files
2. **Parity checklist** — Itemized behavior mapping from C++ to Rust
3. **Implementation summary** — Functions migrated and notable internal mappings
4. **Deviations** — Explicit list of unavoidable differences with rationale
5. **Validation** — Commands run and key outcomes
6. **Follow-ups** — Remaining migration tasks, if any

## Development Commands

### Build

```bash
# Build all crates in debug mode
cargo build

# Build specific crate (e.g., transit-cloud-core)
cargo build -p transit-cloud-core

# Build in release mode (optimized)
cargo build --release

# Build with profiling enabled
cargo build -p transit-cloud-api --release --features profiling
```

### Testing

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p transit-cloud-core

# Run single test by name
cargo test -p transit-cloud-core route_idx_encoding

# Run single test and show output
cargo test -p transit-cloud-core route_idx_encoding -- --nocapture

# Run with backtrace on panic
RUST_BACKTRACE=1 cargo test -p transit-cloud-core

# Run tests in release mode (faster)
cargo test --release

# Run only doc tests
cargo test --doc

# Run benchmarks
cargo bench -p transit-cloud-core raptor_benchmark
```

### Linting and Format

```bash
# Check code without autofix
cargo clippy --all-targets

# Check with all warnings
cargo clippy -- -W clippy::all

# Format code (dry-run to see diffs)
cargo fmt -- --check

# Format code (apply changes)
cargo fmt

# Check for common mistakes and unused code
cargo check
```

### Build All Targets

```bash
# Check everything builds
cargo build --all-targets

# Check with all features
cargo build --all-features

# Check without default features
cargo build --no-default-features
```

## Architecture

### Core Modules (`transit-cloud-core/src/`)

| Module | Purpose |
|--------|---------|
| `timetable.rs` | Static timetable data structures (bit-packed, Nigiri-compatible); 80KB+ file with schedules, transfers, stops |
| `timetable_index.rs` | Fast lookups by stop/date; handles transfer deduplication |
| `types.rs` | Strongly-typed indices (`RouteIdx`, `StopIdx`, `TripIdx`, etc.) to prevent type confusion bugs |
| `routing/` | Journey result structures (`Leg`, `Connection`), footpath lookup, interchange data |
| `raptor/` | RAPTOR algorithm implementation; most complex routing logic; where parity bugs often hide |
| `rt/` | Real-time updates: GTFS-RT parsing, delay propagation, trip amendments. **Critical parity surface.** |
| `loader/` | GTFS + OSM → binary timetable serialization; handles DST, string dedup, bit-packing |
| `mmap/` | Zero-copy memory-mapped deserialization (rkyv); no parsing overhead for timetable |
| `gbfs/` | Transit-cloud extension: bike share real-time feeds |
| `odm/` | Transit-cloud extension: on-demand mobility |
| `geocoder/` | Address search (bigram + SIFT4 fuzzy matching) |
| `string_store.rs` | String pooling for de-duplication and memory efficiency |
| `vecvec.rs` | Ragged array implementation (VecVec) for variable-length collections |

### API Server (`transit-cloud-api/src/`)

GraphQL resolver layer built on async-graphql + axum. Routes queries to `transit-cloud-core` and formats results. Real-time trip tracking via GTFS-RT polling.

### Builder CLI (`transit-cloud-builder/src/`)

Loads GTFS + OpenStreetMap data and serializes into binary timetable format. Used for setup:

```bash
cargo run -p transit-cloud-builder -- build --config config.yml -o ./out
```

## Running Services

### Timetable Builder

```bash
cargo run -p transit-cloud-builder -- build --config ../config.yml -o ../out
```

### GraphQL API Server

```bash
# Standard mode
RUST_LOG=info cargo run -p transit-cloud-api -- -d ../out

# Release with profiling
RUST_LOG=info cargo run -p transit-cloud-api --release --features profiling -- -d ../out

# With puffin profiling viewer (connect to 127.0.0.1:8585)
RUST_LOG=info cargo run -p transit-cloud-api --features profiling -- -d ../out
```

### Export GraphQL Schema

```bash
RUST_LOG=info cargo run -p transit-cloud-api -- -d ./ --export-schema schema.generated.graphql
```

## Skills and Tools

Use the following skills to accelerate migration work:

- **`/rust-pro`** — Comprehensive Rust coding guidelines (179 rules). Use when writing/reviewing/refactoring Rust code. Covers ownership, error handling, async patterns, API design, memory optimization, performance, testing.
- **`/cpp-pro`** — Modern C++20/23 features, template metaprogramming, high-performance systems. Use when analyzing C++ source to understand optimizations or when deciding Rust equivalents.
- **`/simplify`** — Review changed code for reuse, quality, and efficiency; flags code smells and anti-patterns.

## Debugging and Parity Audits

### Known Open Issues

See `NIGIRI_PARITY_AUDIT.md` at repo root for comprehensive tracked issues. Key ones:

- **RT delay propagation (CRITICAL)** — Intermediate stop delays sometimes missed in real-time updates. Check `rt/update.rs`.
- **RT bitfield day bit (CRITICAL)** — Static timetable bitfield not cleared on RT transport add/cancel. Check `rt/mod.rs`.
- **RAPTOR core (HIGH)** — 3 bugs in `add_start`, `update_transfers`, `update_intermodal_footpaths`. Check `routing/raptor.rs`.
- **GTFS loader (CRITICAL)** — DST handling broken, route dedup key incomplete. Check `loader/gtfs.rs`.
- **Wheelchair profile index (OPEN)** — `direct.rs` and `start_times.rs` use `prf_idx==1` but should probably be `2`.
- **Location expansion (OPEN)** — 4 behavior divergences in `for_each_meta`. Check `api/resolvers/location.rs`.

### Debug Logging

```bash
# Detailed tracing
RUST_LOG=debug cargo run -p transit-cloud-api -- -d ../out

# Focus on specific modules
RUST_LOG=transit_cloud_core::routing=debug,transit_cloud_api=info cargo run ...

# Full verbose output including build.rs
RUST_LOG=trace cargo build -vv
```

### Memory Issues

The codebase uses memory-mapped files (rkyv) for zero-copy deserialization. If you see segfaults or alignment errors:

1. Check `mmap/loader.rs` for file alignment assumptions.
2. Verify timetable serialization format hasn't changed.
3. Run under Miri for undefined behavior detection: `MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly miri test`.

## Building with Parity Guarantees

When working on parity migrations:

1. **Extract C++ semantics first.** Read the C++ code end-to-end before writing Rust.
2. **Build a checklist.** Document every visible behavior and edge case.
3. **Test comprehensively.** Parity bugs hide in edge cases. Run with real GTFS data.
4. **Validate output.** Compare Rust routing results against C++ baseline on same timetable.
5. **Document deviations.** If exact translation is impossible, explain why in code comments and in your final summary.

---

**For questions about this project's architecture or migration requirements, consult the project memory at `.claude/projects/.../memory/MEMORY.md` and the parity audit at `NIGIRI_PARITY_AUDIT.md`.**
