# OSR Test Suite

Rust port of the C++ OSR test suite from `osr/test/`.

## Test Files

### Ported Tests

1. **level_tests.rs** - Level representation tests
   - ✅ Fully implemented
   - Tests level rounding and conversion
   
2. **routing_tests.rs** - Basic routing tests
   - ✅ Fully implemented with GeoJSON output comparison
   - Tests: island, ferry, corridor, stop_area
   - Compares segment counts, costs, distances, levels, OSM way IDs vs C++ expected values
   - Requires test data files (`#[ignore]`)
   
3. **algorithm_comparison_tests.rs** - Dijkstra vs A* comparison
   - ✅ Implemented with proper `#[ignore]` annotations
   - Compares routing algorithms for identical results
   - Includes foot and bike profile tests
   
4. **restriction_tests.rs** - Turn restrictions and elevation
   - ✅ Fully implemented with C++ expected values
   - Tests turn restriction enforcement, bike routing distance (~163m), elevation profiles
   - Unit tests for restriction data structures pass without test data
   - Integration test requires test data (`#[ignore]`)

### Required Test Data

Copy from C++ project `osr/test/`:

```
transit-cloud-rust/crates/osr/test/
├── ajaccio-ferry.osm.pbf
├── luisenplatz-darmstadt.osm.pbf
├── london-corridor.osm.pbf
├── station-border.osm.pbf
├── map.osm
└── restriction_test_elevation/
    ├── elevations_1.bil
    └── elevations_1.hdr
```

Optional (for algorithm comparison tests):
```
├── monaco.osm.pbf (larger dataset)
├── hamburg.osm.pbf
└── switzerland.osm.pbf
```

## Running Tests

Run all tests (excluding ignored ones):
```bash
cargo test
```

Run all tests including ignored (requires test data):
```bash
cargo test -- --ignored --test-threads=1
```

Run specific test:
```bash
cargo test test_island -- --ignored
```

Run with debug output:
```bash
RUST_LOG=debug cargo test test_island -- --ignored --nocapture
```

## Test Status

### ✅ Passing (without test data)
- `level_tests.rs` - All level conversion tests pass
- `restriction_tests.rs` - Unit tests for restriction data structures
- `serialization_test.rs` - Serialization round-trip tests
- `multi_way_test.rs` - Multi-way routing tests
- `simple_integration_tests.rs` - Basic integration tests
- `integration_routing.rs` - Integration routing tests

### ✅ Passing (with test data, `#[ignore]`)
- `routing_tests.rs` - island, ferry, corridor, stop_area with full GeoJSON comparison
- `restriction_tests.rs` - Turn restriction + elevation integration test
- `algorithm_comparison_tests.rs` - Dijkstra vs A* on various datasets

### Missing Features (routing engine, not tests)
- Turn restriction enforcement during routing
- Elevation-aware routing costs
- A* bidirectional search algorithm

## Differences from C++

### Architecture
- C++ uses `cista::mmap` for memory mapping
- Rust uses `memmap2` + `rkyv` for zero-copy deserialization
- C++ uses templates for profile dispatch
- Rust uses enum dispatch (`SearchProfile`)

### Missing Features
1. **Turn Restrictions**: Extracted during pass 3 but not enforced in routing
2. **Elevation Costs**: Elevation data loaded but not used in cost calculation
3. **A* Bidirectional**: Only Dijkstra implemented so far

### API Differences
```rust
// C++ API
auto p = osr::route(
    osr::foot<false, osr::elevator_tracking>::parameters{},
    w, l, osr::search_profile::kFoot,
    from, {to}, 900, osr::direction::kForward,
    250.0, nullptr, nullptr, nullptr,
    osr::routing_algorithm::kDijkstra
);

// Rust API (current)
let result = route(
    &ways,
    &lookup,
    SearchProfile::Foot,
    &from,
    &to
)?;
```

Rust API is simpler but missing:
- Multiple destinations support
- Direction (forward/backward) parameter
- Cost limit parameter
- Algorithm selection (Dijkstra vs A*)

## Next Steps

### Phase 1: Complete Basic Routing
1. ✅ Implement profile-based costs
2. ⏳ Debug spatial search (bbox/R-tree)
3. ❌ Add GeoJSON output matching C++ format
4. ❌ Run routing tests and compare outputs

### Phase 2: Advanced Features
1. ❌ Implement turn restriction checking
2. ❌ Add elevation costs to profile calculations
3. ❌ Implement A* bidirectional search
4. ❌ Add OSM ID lookup methods

### Phase 3: Full Parity
1. ❌ Support multiple destinations
2. ❌ Add backward search direction
3. ❌ Add cost/time limits
4. ❌ Performance benchmarking vs C++

## Debugging Tips

### Enable Debug Output
The current code has debug output in `get_way_candidates()`:
```rust
println!("DEBUG get_way_candidates: bbox = ...");
println!("DEBUG get_way_candidates: Found {} ways in bbox", nearby_ways.len());
```

### Compare with C++
Run same test in C++:
```bash
cd osr/build
./test --gtest_filter=routing.island
```

Then run in Rust:
```bash
cargo test test_island -- --ignored --nocapture
```

Compare outputs to find differences.

### Visualize Routes
The C++ tests output GeoJSON that can be visualized at geojson.io.
Copy the GeoJSON string and paste into http://geojson.io to see the route.

## Test Data Setup

### Copy from C++ Project
```bash
# From transit-cloud-rust/crates/osr/
mkdir -p test
cp ../../../osr/test/*.osm.pbf test/
cp ../../../osr/test/*.osm test/
cp -r ../../../osr/test/restriction_test_elevation test/
```

### Verify Test Data
```bash
ls -lh test/
# Should show:
# ajaccio-ferry.osm.pbf (27K)
# luisenplatz-darmstadt.osm.pbf (48K)
# london-corridor.osm.pbf (31K)
# station-border.osm.pbf (142K)
# map.osm (113K)
# restriction_test_elevation/ (directory)
```

## Expected Test Results

Once fully implemented, all tests should produce identical routing results as C++:

- Same waypoints traversed
- Same total cost
- Same total distance
- Same elevation changes (when elevation enabled)
- Turn restrictions respected
