# Phase 7 — GBFS (Bike/Scooter/Car Sharing)

**Priority**: Eighth  
**Type**: Pure Rust (using `transit-osr` crate for `SharingData`)  
**Depends on**: Phase 0 (types), Phase 4 (OSR), Phase 6 (RT infrastructure)  
**Crate**: `transit-gbfs`  
**Estimated effort**: Large

## Objective

Integrate GBFS (General Bikeshare Feed Specification) feeds to support rental bike, scooter, and car-sharing legs in multimodal journey planning. This includes feed fetching, vehicle/station parsing, geofencing zone enforcement, OSR integration via `SharingData`, and periodic real-time updates.

## What motis Does (`src/gbfs/`, ~3,000+ LOC)

### Overview

motis fetches GBFS feeds (v1/v2/v3) from sharing operators, parses station and vehicle availability, maps everything to the OSR street network, and provides routing constraints to RAPTOR via bitvector-based `SharingData`.

### Feed Types Parsed

| GBFS File | Purpose |
|---|---|
| `system_information.json` | Operator name, ID, color |
| `vehicle_types.json` | Form factor, propulsion, return constraints |
| `station_information.json` | Station lat/lng, name, area polygon |
| `station_status.json` | Vehicle/dock availability per station |
| `free_bike_status.json` / `vehicle_status.json` | Free-floating vehicle positions |
| `geofencing_zones.json` | GeoJSON zones with ride start/end/through rules |

### Vehicle Form Factors

```rust
pub enum VehicleFormFactor {
    Bicycle,
    CargoBicycle,
    Car,
    Moped,
    ScooterStanding,
    ScooterSeated,
    Other,
}

pub enum PropulsionType {
    Human,
    ElectricAssist,
    Electric,
    Combustion,
    CombustionDiesel,
    Hybrid,
    PlugInHybrid,
    HydrogenFuelCell,
}

pub enum ReturnConstraint {
    FreeFloating,
    AnyStation,
    RoundtripStation,
}
```

### Key Architecture

```
GBFS Feeds (HTTP/JSON)
  ↓ periodic fetch (TTL-based)
  ↓ hash-based change detection
GbfsProvider (stations + vehicles + geofencing + vehicle_types)
  ↓ partition by routing constraints
Products (groups of vehicle types with identical routing rules)
  ↓ map to OSR street network
SharingData (start/end/through bitvecs + additional edges)
  ↓ inject into routing query
RAPTOR + OSR (sharing-aware street routing)
```

## Rust Implementation

### Core Data Structures

```rust
// transit-gbfs/src/types.rs

pub struct GbfsProvider {
    pub system_info: SystemInfo,
    pub vehicle_types: Vec<VehicleType>,
    pub stations: HashMap<String, Station>,
    pub vehicles: Vec<Vehicle>,
    pub geofencing: Vec<GeofencingZone>,
    pub products: Vec<Product>,
    pub bbox: Option<BoundingBox>,
}

pub struct SystemInfo {
    pub system_id: String,
    pub name: String,
    pub operator: Option<String>,
    pub url: Option<String>,
    pub color: Option<String>,
}

pub struct VehicleType {
    pub id: String,
    pub idx: u32,
    pub name: Option<String>,
    pub form_factor: VehicleFormFactor,
    pub propulsion_type: PropulsionType,
    pub return_constraint: ReturnConstraint,
}

pub struct Station {
    pub id: String,
    pub pos: Position,
    pub name: String,
    pub address: Option<String>,
    pub capacity: Option<u32>,
    pub vehicles_available: HashMap<String, u32>,  // by vehicle_type_id
    pub docks_available: HashMap<String, u32>,
    pub rental_uris: Option<RentalUris>,
    pub area: Option<geo::MultiPolygon<f64>>,
}

pub struct Vehicle {
    pub id: String,
    pub pos: Position,
    pub is_reserved: bool,
    pub is_disabled: bool,
    pub vehicle_type_id: String,
    pub station_id: Option<String>,
}

pub struct GeofencingZone {
    pub geometry: geo::MultiPolygon<f64>,
    pub rules: Vec<GeofencingRule>,
}

pub struct GeofencingRule {
    pub vehicle_type_ids: Option<Vec<String>>,
    pub ride_start_allowed: bool,
    pub ride_end_allowed: bool,
    pub ride_through_allowed: bool,
    pub station_parking: bool,
}
```

### Product Partitioning

Group vehicle types with identical routing constraints into "products" using partition refinement:

```rust
// transit-gbfs/src/partition.rs

pub struct Product {
    pub vehicle_type_indices: Vec<u32>,
    pub form_factor: VehicleFormFactor,
    pub propulsion_type: PropulsionType,
    pub return_constraint: ReturnConstraint,
    pub transport_mode_id: u32,
}

/// Partition vehicle types into groups with identical routing behavior.
/// Refine by: form_factor → propulsion → return_constraint → dock availability → geofencing rules
pub fn partition_provider(provider: &GbfsProvider) -> Vec<Product> {
    let mut partition = Partition::new(provider.vehicle_types.len());

    // Refine by (form_factor, propulsion_type)
    partition.refine_by(|vt| (vt.form_factor, vt.propulsion_type));

    // Refine by return_constraint
    partition.refine_by(|vt| vt.return_constraint);

    // Refine by dock availability patterns per station
    for station in provider.stations.values() {
        partition.refine_by(|vt| station.docks_available.get(&vt.id).copied());
    }

    // Refine by geofencing zone rules
    for zone in &provider.geofencing {
        for rule in &zone.rules {
            partition.refine_by(|vt| {
                rule.vehicle_type_ids.as_ref()
                    .map_or(true, |ids| ids.contains(&vt.id))
            });
        }
    }

    partition.into_products(&provider.vehicle_types)
}
```

### OSR Mapping

Map stations/vehicles to OSR street network nodes and build `SharingData`:

```rust
// transit-gbfs/src/osr_mapping.rs

use transit_osr::{OsrData, Location, SearchProfile};
use transit_osr::sharing_data::{SharingData, AdditionalEdge};

pub struct GbfsRoutingData {
    pub additional_nodes: Vec<AdditionalNode>,
    pub additional_coordinates: Vec<Point>,
    pub additional_edges: AHashMap<NodeIdx, Vec<AdditionalEdge>>,
    pub start_allowed: BitVec,
    pub end_allowed: BitVec,
    pub through_allowed: BitVec,
}

pub enum AdditionalNode {
    Station { id: String },
    Vehicle { idx: u32 },
}

pub fn build_routing_data(
    osr: &OsrData,
    provider: &GbfsProvider,
    product: &Product,
) -> GbfsRoutingData {
    let node_count = osr.ways.node_count();
    let mut data = GbfsRoutingData {
        additional_nodes: Vec::new(),
        additional_coordinates: Vec::new(),
        additional_edges: AHashMap::new(),
        start_allowed: BitVec::repeat(false, node_count),
        end_allowed: BitVec::repeat(false, node_count),
        through_allowed: BitVec::repeat(true, node_count),
    };

    let profile = form_factor_to_profile(product.form_factor);

    // Map stations
    for station in provider.stations.values() {
        let loc = Location::from_latlng_no_level(station.pos.lat, station.pos.lon);
        let candidates = osr.lookup.match_location(&osr.ways, loc, profile);

        for candidate in &candidates {
            let node = candidate.left.node; // closest node

            // Check availability
            let has_vehicles = product.vehicle_type_indices.iter()
                .any(|&vti| station.vehicles_available.get(&provider.vehicle_types[vti as usize].id)
                    .copied().unwrap_or(0) > 0);

            if has_vehicles {
                data.start_allowed.set(node.0 as usize, true);
            }

            // Check return rules
            match product.return_constraint {
                ReturnConstraint::AnyStation | ReturnConstraint::RoundtripStation => {
                    data.end_allowed.set(node.0 as usize, true);
                }
                ReturnConstraint::FreeFloating => {
                    // end anywhere allowed by geofencing
                }
            }

            // Add additional node + edges
            let add_idx = data.additional_nodes.len() as u32 + node_count as u32;
            data.additional_nodes.push(AdditionalNode::Station { id: station.id.clone() });
            data.additional_coordinates.push(station.pos.into());
            data.additional_edges.entry(node)
                .or_default()
                .push(AdditionalEdge { target: NodeIdx(add_idx), cost: 0 });
        }
    }

    // Map free-floating vehicles
    for (idx, vehicle) in provider.vehicles.iter().enumerate() {
        if vehicle.is_reserved || vehicle.is_disabled || vehicle.station_id.is_some() {
            continue;
        }
        if !product.vehicle_type_indices.contains(&vehicle_type_idx(provider, &vehicle.vehicle_type_id)) {
            continue;
        }

        let loc = Location::from_latlng_no_level(vehicle.pos.lat, vehicle.pos.lon);
        let candidates = osr.lookup.match_location(&osr.ways, loc, profile);
        if let Some(c) = candidates.first() {
            data.start_allowed.set(c.left.node.0 as usize, true);
        }
    }

    // Apply geofencing zones
    apply_geofencing(&osr, &provider.geofencing, product, &mut data);

    data
}

fn form_factor_to_profile(ff: VehicleFormFactor) -> SearchProfile {
    match ff {
        VehicleFormFactor::Car => SearchProfile::CarSharing,
        _ => SearchProfile::BikeSharing, // bikes, scooters, mopeds
    }
}

impl GbfsRoutingData {
    pub fn to_sharing_data(&self) -> SharingData<'_> {
        SharingData {
            start_allowed: Some(&self.start_allowed),
            end_allowed: Some(&self.end_allowed),
            through_allowed: Some(&self.through_allowed),
            additional_node_offset: self.additional_nodes.len() as u32,
            additional_node_coordinates: &self.additional_coordinates,
            additional_edges: &self.additional_edges,
        }
    }
}
```

### Feed Fetcher

```rust
// transit-gbfs/src/update.rs

pub struct GbfsFeedConfig {
    pub id: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub oauth: Option<OAuthConfig>,
    pub default_ttl: u32,
}

pub struct OAuthConfig {
    pub token_url: String,
    pub client_id: String,
    pub client_secret: String,
}

pub struct GbfsUpdater {
    client: reqwest::Client,
    feeds: Vec<GbfsFeedConfig>,
    providers: Vec<GbfsProvider>,
    hashes: HashMap<String, u64>,       // URL → content hash
    expiry: HashMap<String, Instant>,   // URL → next fetch time
    oauth_tokens: HashMap<String, OAuthToken>,
}

impl GbfsUpdater {
    pub async fn update(&mut self) -> Result<Vec<GbfsProvider>, GbfsError> {
        let mut updated = Vec::new();

        for feed in &self.feeds {
            // Check TTL
            if let Some(expiry) = self.expiry.get(&feed.id) {
                if Instant::now() < *expiry {
                    continue;
                }
            }

            // Refresh OAuth token if needed
            if let Some(oauth) = &feed.oauth {
                self.refresh_oauth_token(&feed.id, oauth).await?;
            }

            // Fetch and parse provider
            let provider = self.fetch_provider(feed).await?;
            updated.push(provider);
        }

        Ok(updated)
    }

    async fn fetch_provider(&mut self, feed: &GbfsFeedConfig) -> Result<GbfsProvider, GbfsError> {
        // Discover feed URLs (GBFS auto-discovery)
        let discovery = self.fetch_json::<GbfsDiscovery>(&feed.url, feed).await?;

        let mut provider = GbfsProvider::default();

        // Fetch each sub-feed, skip if hash unchanged
        if let Some(url) = discovery.system_information_url() {
            if self.has_changed(&url, feed).await? {
                provider.system_info = self.fetch_json(&url, feed).await?;
            }
        }

        if let Some(url) = discovery.vehicle_types_url() {
            if self.has_changed(&url, feed).await? {
                provider.vehicle_types = self.fetch_json(&url, feed).await?;
            }
        }

        // ... station_information, station_status, vehicle_status, geofencing_zones

        // Partition into products
        provider.products = partition_provider(&provider);

        Ok(provider)
    }
}
```

### LZ4 Compression for Bitvectors

```rust
// transit-gbfs/src/compression.rs

pub fn compress_bitvec(bv: &BitVec) -> Vec<u8> {
    lz4_flex::compress_prepend_size(bv.as_raw_slice().as_bytes())
}

pub fn decompress_bitvec(data: &[u8], bit_count: usize) -> BitVec {
    let decompressed = lz4_flex::decompress_size_prepended(data)
        .expect("invalid lz4 data");
    BitVec::from_vec(decompressed).truncate(bit_count)
}
```

### GBFS Background Update Loop

```rust
// transit-gbfs/src/service.rs

pub async fn run_gbfs_update_loop(
    config: Vec<GbfsFeedConfig>,
    osr: Arc<OsrData>,
    gbfs_data: Arc<RwLock<GbfsData>>,
    interval: Duration,
) {
    let mut updater = GbfsUpdater::new(config);

    loop {
        match updater.update().await {
            Ok(providers) => {
                if !providers.is_empty() {
                    // Rebuild routing data for updated providers
                    let routing_data: Vec<_> = providers.iter()
                        .flat_map(|p| p.products.iter()
                            .map(|prod| build_routing_data(&osr, p, prod)))
                        .collect();

                    let mut data = gbfs_data.write().await;
                    data.providers = providers;
                    data.routing_data = routing_data;
                }
            }
            Err(e) => {
                tracing::warn!("GBFS update error: {e}");
            }
        }

        tokio::time::sleep(interval).await;
    }
}
```

## Data Flow in Routing Query

```
planJourney(modes: [WALK, BIKE_RENTAL])
  │
  ├─ Lookup GBFS products matching BIKE_RENTAL
  │   └─ Find products with form_factor ∈ {Bicycle, ScooterStanding, ...}
  │
  ├─ Decompress routing data for selected products
  │   └─ LZ4 decompress start_allowed, end_allowed, through_allowed bitvecs
  │
  ├─ Build SharingData from product routing data
  │   └─ transit_osr::SharingData { start, end, through, additional_edges }
  │
  ├─ Run OSR routing with SharingData
  │   └─ transit_osr::route(profile=BikeSharing, sharing_data=...)
  │
  ├─ Feed offsets to RAPTOR
  │   └─ Start offsets include walk-to-station + rental-ride segments
  │
  └─ Format rental legs with provider/vehicle info
      └─ Station name, rental URI, vehicle type, return instructions
```

## GraphQL Types

```graphql
type RentalLeg {
  provider: RentalProvider!
  vehicleType: RentalVehicleType!
  from: RentalPlace!
  to: RentalPlace!
  rentalUri: String
}

type RentalProvider {
  id: String!
  name: String!
  url: String
  color: String
}

type RentalVehicleType {
  formFactor: VehicleFormFactor!
  propulsionType: PropulsionType!
  name: String
}

type RentalPlace {
  station: RentalStation
  coordinates: Position!
}

type RentalStation {
  id: String!
  name: String!
  vehiclesAvailable: Int!
  docksAvailable: Int
}
```

## Acceptance Criteria

1. GBFS v1/v2/v3 feeds parsed correctly (stations, vehicles, geofencing)
2. Feed updates respect TTL and hash-based change detection
3. OAuth token refresh works for authenticated feeds
4. Vehicle types partitioned into products with correct routing constraints
5. OSR `SharingData` built with correct start/end/through bitvecs
6. Geofencing zones enforce ride restrictions
7. Rental legs appear in routing results with provider info
8. LZ4 compression reduces memory for cached routing data

## motis Source Reference

| File | Lines | Key Pattern |
|---|---|---|
| `src/gbfs/update.cc` | ~1,100 | Async feed fetch + OAuth + hash detection |
| `src/gbfs/parser.cc` | ~600 | GBFS v1/v2/v3 JSON parsing |
| `src/gbfs/osr_mapping.cc` | ~380 | Station/vehicle → OSR node mapping + bitvecs |
| `src/gbfs/data.cc` | ~60 | Routing data decompression + caching |
| `src/gbfs/gbfs_output.cc` | ~130 | Rental leg annotation |
| `include/motis/gbfs/data.h` | ~250 | All GBFS data structures |
