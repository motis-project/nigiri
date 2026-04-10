# Phase 10 — Elevators / Accessibility

**Priority**: Eleventh  
**Type**: Pure Rust (feed parsing + OSR node blocking)  
**Depends on**: Phase 4 (OSR), Phase 5 (footpaths + `ElevatorFootpaths`)  
**Crate**: `transit-elevators`  
**Estimated effort**: Medium

## Objective

Track real-time elevator status from external feeds (DB FaSta JSON API, SIRI-FM XML) and dynamically update time-dependent footpaths so that wheelchair routing avoids broken elevators.

## What motis Does (`src/elevators/`, ~400+ LOC)

### Overview

1. Parse elevator data from FaSta (JSON) and/or SIRI-FM (XML)
2. Match each elevator to its OSR street node (by DIID or geo-proximity)
3. Build a `blocked` bitvector marking inactive elevator nodes
4. Periodically poll for updates; when elevator status changes, recompute affected td-footpaths
5. Wheelchair profile routing respects blocked elevators via td-footpath constraints

### Matching Strategy

- **SIRI-FM**: Has a `DIID` (facility ID) → match to OSM node via CSV mapping file (OSM ID ↔ DIID)
- **FaSta**: Has coordinates → match by geo-proximity (within 20m of OSR elevator node)
- SIRI takes priority when both have data for the same elevator

## Rust Implementation

### Elevator Data Model

```rust
// transit-elevators/src/types.rs

pub type ElevatorIdx = u32;

pub struct Elevator {
    pub id: u64,
    pub id_str: Option<String>,    // DIID for SIRI-FM matching
    pub pos: Position,
    pub status: ElevatorStatus,
    pub description: Option<String>,
    pub out_of_service: Vec<TimeInterval>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ElevatorStatus {
    Active,
    Inactive,
    Unknown,
}

pub struct TimeInterval {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

pub struct Elevators {
    pub entries: Vec<Elevator>,
    pub rtree: RTree<ElevatorEntry>,
    pub blocked: BitVec,              // indexed by OSR NodeIdx
    pub node_to_elevator: HashMap<NodeIdx, ElevatorIdx>,
}
```

### FaSta JSON Parser

```rust
// transit-elevators/src/parse_fasta.rs

use serde::Deserialize;

#[derive(Deserialize)]
struct FastaFacility {
    #[serde(rename = "equipmentnumber")]
    id: u64,
    #[serde(rename = "geocoordX")]
    lon: f64,
    #[serde(rename = "geocoordY")]
    lat: f64,
    description: Option<String>,
    state: String,                   // "ACTIVE" | "INACTIVE"
    #[serde(rename = "stateExplanation")]
    state_explanation: Option<String>,
    #[serde(rename = "outOfService")]
    out_of_service: Vec<FastaInterval>,
}

#[derive(Deserialize)]
struct FastaInterval {
    start: String,  // ISO 8601
    end: String,
}

pub fn parse_fasta(json: &str) -> Result<Vec<Elevator>, ElevatorError> {
    let facilities: Vec<FastaFacility> = serde_json::from_str(json)?;

    facilities.into_iter().map(|f| {
        Ok(Elevator {
            id: f.id,
            id_str: None,
            pos: Position { lat: f.lat, lon: f.lon },
            status: match f.state.as_str() {
                "ACTIVE" => ElevatorStatus::Active,
                _ => ElevatorStatus::Inactive,
            },
            description: f.description,
            out_of_service: f.out_of_service.iter()
                .map(|i| Ok(TimeInterval {
                    start: DateTime::parse_from_rfc3339(&i.start)?.with_timezone(&Utc),
                    end: DateTime::parse_from_rfc3339(&i.end)?.with_timezone(&Utc),
                }))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }).collect()
}
```

### SIRI-FM XML Parser

```rust
// transit-elevators/src/parse_siri.rs

pub fn parse_siri_fm(xml: &str) -> Result<Vec<Elevator>, ElevatorError> {
    let doc = quick_xml::Reader::from_str(xml);
    let mut elevators = Vec::new();

    // Parse <FacilityCondition> elements
    // Extract: <FacilityRef> (DIID), <FacilityStatus> (available/notAvailable)
    // Map to Elevator { id_str: Some(diid), status: ... }

    // ... XML traversal using quick-xml events ...

    Ok(elevators)
}
```

### OSM ID Mapping

```rust
// transit-elevators/src/osm_mapping.rs

/// Parse CSV mapping: OSM node ID → DIID string
pub fn parse_osm_mapping(csv: &str) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    for line in csv.lines().skip(1) {  // skip header
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(osm_id) = parts[0].trim().parse::<u64>() {
                map.insert(parts[1].trim().to_string(), osm_id);
            }
        }
    }
    map
}
```

### Elevator Matching

```rust
// transit-elevators/src/matching.rs

use transit_osr::OsrData;

const ELEVATOR_MATCH_RADIUS: f64 = 20.0; // meters

/// Match elevator data to OSR street graph nodes.
pub fn match_elevators(
    elevators: &[Elevator],
    osr: &OsrData,
    osm_mapping: &HashMap<String, u64>,
) -> Elevators {
    // 1. Get all OSR nodes tagged as elevator
    let elevator_nodes = osr.ways.elevator_nodes();

    // 2. Build R-tree on elevator feed entries
    let rtree = RTree::bulk_load(
        elevators.iter().enumerate()
            .map(|(i, e)| ElevatorEntry { idx: i as u32, pos: e.pos })
            .collect()
    );

    let mut blocked = BitVec::repeat(false, osr.ways.node_count());
    let mut node_to_elevator = HashMap::new();

    // 3. For each OSR elevator node, find matching feed elevator
    for &node_idx in &elevator_nodes {
        let node_pos = osr.ways.node_position(node_idx);

        // Try DIID matching first (SIRI-FM)
        let matched = elevators.iter().enumerate().find(|(_, e)| {
            if let Some(diid) = &e.id_str {
                osm_mapping.get(diid).map_or(false, |&osm_id| {
                    osr.ways.node_osm_id(node_idx) == osm_id
                })
            } else {
                false
            }
        });

        // Fall back to geo-proximity (FaSta)
        let matched = matched.or_else(|| {
            rtree.nearest_neighbor(&[node_pos.lng(), node_pos.lat()])
                .filter(|entry| {
                    haversine_distance(
                        Position { lat: node_pos.lat(), lon: node_pos.lng() },
                        elevators[entry.idx as usize].pos,
                    ) <= ELEVATOR_MATCH_RADIUS
                })
                .map(|entry| (entry.idx as usize, &elevators[entry.idx as usize]))
        });

        if let Some((elev_idx, elevator)) = matched {
            node_to_elevator.insert(node_idx, elev_idx as u32);
            if elevator.status == ElevatorStatus::Inactive {
                blocked.set(node_idx.0 as usize, true);
            }
        }
    }

    Elevators {
        entries: elevators.to_vec(),
        rtree,
        blocked,
        node_to_elevator,
    }
}
```

### Real-Time Update

```rust
// transit-elevators/src/update.rs

pub struct ElevatorUpdater {
    client: reqwest::Client,
    config: ElevatorConfig,
    osm_mapping: HashMap<String, u64>,
}

pub struct ElevatorConfig {
    pub fasta_url: Option<String>,
    pub siri_url: Option<String>,
    pub osm_mapping_csv: Option<PathBuf>,
    pub update_interval: Duration,
}

impl ElevatorUpdater {
    pub async fn update(
        &self,
        old: &Elevators,
        osr: &transit_osr::OsrData,
        tt: &mut nigiri::Timetable,
        elevator_footpaths: &ElevatorFootpaths,
    ) -> Result<Elevators, ElevatorError> {
        // 1. Fetch new data
        let mut new_elevators = Vec::new();

        if let Some(url) = &self.config.fasta_url {
            let json = self.client.get(url).send().await?.text().await?;
            new_elevators.extend(parse_fasta(&json)?);
        }

        if let Some(url) = &self.config.siri_url {
            let xml = self.client.get(url).send().await?.text().await?;
            new_elevators.extend(parse_siri_fm(&xml)?);
        }

        // 2. Re-match elevators to nodes
        let new = match_elevators(&new_elevators, osr, &self.osm_mapping);

        // 3. Detect changes and update td-footpaths
        let changed_nodes = detect_changes(old, &new);
        if !changed_nodes.is_empty() {
            update_td_footpaths(tt, osr, elevator_footpaths, &changed_nodes, &new)?;
        }

        Ok(new)
    }
}

fn detect_changes(old: &Elevators, new: &Elevators) -> Vec<NodeIdx> {
    let mut changed = Vec::new();

    for (&node, &new_elev_idx) in &new.node_to_elevator {
        let new_status = new.entries[new_elev_idx as usize].status;

        let old_status = old.node_to_elevator.get(&node)
            .map(|&idx| old.entries[idx as usize].status)
            .unwrap_or(ElevatorStatus::Active); // absent = assumed active

        if new_status != old_status {
            changed.push(node);
        }
    }

    // Elevators that disappeared (assume now active)
    for (&node, &old_elev_idx) in &old.node_to_elevator {
        if !new.node_to_elevator.contains_key(&node) {
            if old.entries[old_elev_idx as usize].status == ElevatorStatus::Inactive {
                changed.push(node);
            }
        }
    }

    changed
}

fn update_td_footpaths(
    tt: &mut nigiri::Timetable,
    osr: &transit_osr::OsrData,
    elevator_fps: &ElevatorFootpaths,
    changed_nodes: &[NodeIdx],
    elevators: &Elevators,
) -> Result<(), ElevatorError> {
    // For each footpath that uses an elevator at a changed node,
    // update the td-footpath (time-dependent footpath) to reflect
    // the new elevator state.

    // 1. Find affected location pairs from ElevatorFootpaths
    // 2. For each pair, recompute wheelchair footpath with new blocked set
    // 3. Update nigiri rt_timetable td_footpaths

    // This integrates with Phase 5's ElevatorFootpaths tracking
    todo!()
}
```

### Background Update Loop

```rust
// transit-elevators/src/service.rs

pub async fn run_elevator_update_loop(
    config: ElevatorConfig,
    osr: Arc<transit_osr::OsrData>,
    tt: Arc<RwLock<nigiri::Timetable>>,
    elevator_fps: Arc<ElevatorFootpaths>,
    elevators: Arc<RwLock<Elevators>>,
) {
    let updater = ElevatorUpdater::new(config.clone());

    loop {
        {
            let old = elevators.read().await;
            let mut tt = tt.write().await;
            match updater.update(&old, &osr, &mut tt, &elevator_fps).await {
                Ok(new) => {
                    drop(old);
                    *elevators.write().await = new;
                }
                Err(e) => tracing::warn!("Elevator update error: {e}"),
            }
        }

        tokio::time::sleep(config.update_interval).await;
    }
}
```

## State Change Model

For out-of-service intervals, elevators can have scheduled downtime:

```
Time ──────────────────────────────────────────→
         ┌─── out_of_service[0] ───┐  ┌── out_of_service[1] ──┐
         │       INACTIVE          │  │       INACTIVE         │
─ ACTIVE ┘                         └──┘                        └── ACTIVE ─
```

The state-change generator merges intervals from multiple elevators and emits `(time, state_vector)` tuples for td-footpath computation.

## Integration Points

- **Phase 5** (`ElevatorFootpaths`) — Tracks which footpaths need elevators; Phase 10 uses this to know which footpaths to update
- **Phase 4** (`transit-osr`) — `blocked` bitvec passed to wheelchair routing via OSR; Routes avoid blocked elevator nodes
- **Phase 6** (RT) — Elevator updates trigger td-footpath updates in the rt_timetable

## Acceptance Criteria

1. FaSta JSON API parsed (elevator ID, coordinates, status, out-of-service intervals)
2. SIRI-FM XML parsed (facility ID, status)
3. OSM ID ↔ DIID CSV mapping loaded
4. Elevator-to-node matching works (DIID first, then geo-proximity within 20m)
5. `blocked` bitvec correctly marks inactive elevator nodes
6. Real-time status changes detected via delta comparison
7. TD-footpaths updated when elevator status changes
8. Wheelchair routing avoids broken elevators

## motis Source Reference

| File | Lines | Key Pattern |
|---|---|---|
| `src/elevators/elevators.cc` | ~43 | Constructor, R-tree + blocked set |
| `src/elevators/match_elevators.cc` | ~98 | DIID + geo-proximity matching |
| `src/elevators/parse_fasta.cc` | ~70 | FaSta JSON parsing |
| `src/elevators/parse_siri_fm.cc` | ~53 | SIRI-FM XML parsing |
| `src/elevators/update_elevators.cc` | ~110 | RT update + delta detection |
| `include/motis/elevators/get_state_changes.h` | ~90 | State-change generator template |
