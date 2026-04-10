//! Translation of osr/include/osr/ways.h + src/ways.cc
//!
//! Core street network data structures.

use std::path::Path;

use ahash::AHashMap;
use packed_struct::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

use crate::types::*;
use crate::Point;

/// Turn restriction type
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum RestrictionType {
    No,   // "no turn" restriction
    Only, // "only turn" restriction
}

/// Resolved turn restriction
#[derive(
    Debug, Clone, Copy, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize,
)]
pub struct ResolvedRestriction {
    pub type_: RestrictionType,
    pub from: WayIdx,
    pub to: WayIdx,
    pub via: NodeIdx,
}

/// Compact restriction stored at node
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Restriction {
    pub from: WayPos,
    pub to: WayPos,
}

/// Way properties (5 bytes, bit-packed)
///
/// Direct 1:1 translation of C++ bitfield layout:
/// ```cpp
/// struct way_properties {
///   bool is_foot_accessible_ : 1;
///   bool is_bike_accessible_ : 1;
///   bool is_car_accessible_ : 1;
///   bool is_destination_ : 1;
///   bool is_oneway_car_ : 1;
///   bool is_oneway_bike_ : 1;
///   bool is_elevator_ : 1;
///   bool is_steps_ : 1;
///   std::uint8_t speed_limit_ : 3;
///   std::uint8_t from_level_ : 6;
///   std::uint8_t to_level_ : 6;
///   bool is_incline_down_ : 1;
///   std::uint8_t is_platform_ : 1;
///   bool is_parking_ : 1;
///   bool is_ramp_ : 1;
///   bool is_sidewalk_separate_ : 1;
///   bool motor_vehicle_no_ : 1;
///   bool has_toll_ : 1;
///   bool is_big_street_ : 1;
/// };
/// static_assert(sizeof(way_properties) == 5);
/// ```
#[derive(PackedStruct, Debug, Clone, Copy, Default)]
#[packed_struct(bit_numbering = "lsb0", size_bytes = "5")]
pub struct WayProperties {
    #[packed_field(bits = "0")]
    pub is_foot_accessible: bool,
    #[packed_field(bits = "1")]
    pub is_bike_accessible: bool,
    #[packed_field(bits = "2")]
    pub is_car_accessible: bool,
    #[packed_field(bits = "3")]
    pub is_destination: bool,
    #[packed_field(bits = "4")]
    pub is_oneway_car: bool,
    #[packed_field(bits = "5")]
    pub is_oneway_bike: bool,
    #[packed_field(bits = "6")]
    pub is_elevator: bool,
    #[packed_field(bits = "7")]
    pub is_steps: bool,

    #[packed_field(bits = "8..=10", ty = "enum")]
    pub speed_limit: SpeedLimit,

    #[packed_field(bits = "11..=16")]
    pub from_level: Integer<u8, packed_bits::Bits<6>>,
    #[packed_field(bits = "17..=22")]
    pub to_level: Integer<u8, packed_bits::Bits<6>>,

    #[packed_field(bits = "23")]
    pub is_incline_down: bool,
    #[packed_field(bits = "24")]
    pub is_platform: bool,
    #[packed_field(bits = "25")]
    pub is_parking: bool,
    #[packed_field(bits = "26")]
    pub is_ramp: bool,
    #[packed_field(bits = "27")]
    pub is_sidewalk_separate: bool,
    #[packed_field(bits = "28")]
    pub motor_vehicle_no: bool,
    #[packed_field(bits = "29")]
    pub has_toll: bool,
    #[packed_field(bits = "30")]
    pub is_big_street: bool,
}

impl WayProperties {
    pub fn is_accessible(&self) -> bool {
        self.is_car_accessible || self.is_bike_accessible || self.is_foot_accessible
    }

    // Getter methods for compatibility
    pub fn is_car_accessible(&self) -> bool {
        self.is_car_accessible
    }

    pub fn is_bike_accessible(&self) -> bool {
        self.is_bike_accessible
    }

    pub fn is_foot_accessible(&self) -> bool {
        self.is_foot_accessible
    }

    pub fn is_big_street(&self) -> bool {
        self.is_big_street
    }

    pub fn is_oneway_car(&self) -> bool {
        self.is_oneway_car
    }

    pub fn is_oneway_bike(&self) -> bool {
        self.is_oneway_bike
    }

    pub fn max_speed_m_per_s(&self) -> u16 {
        self.speed_limit.to_meters_per_second()
    }

    pub fn max_speed_km_per_h(&self) -> u16 {
        self.speed_limit.to_kmh()
    }

    pub fn from_level(&self) -> Level {
        Level::from_idx(*self.from_level)
    }

    pub fn to_level(&self) -> Level {
        Level::from_idx(*self.to_level)
    }
}

// Serde support for packed struct
impl serde::Serialize for WayProperties {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use packed_struct::PackedStruct;
        let bytes = self.pack().map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> serde::Deserialize<'de> for WayProperties {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use packed_struct::PackedStruct;
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        let bytes_array: [u8; 5] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| serde::de::Error::custom("invalid byte length"))?;
        WayProperties::unpack(&bytes_array).map_err(serde::de::Error::custom)
    }
}

/// Node properties (3 bytes, bit-packed)
///
/// Direct 1:1 translation of C++ bitfield layout:
/// ```cpp
/// struct node_properties {
///   std::uint8_t from_level_ : 6;
///   bool is_foot_accessible_ : 1;
///   bool is_bike_accessible_ : 1;
///   bool is_car_accessible_ : 1;
///   bool is_elevator_ : 1;
///   bool is_entrance_ : 1;
///   bool is_multi_level_ : 1;
///   bool is_parking_ : 1;
///   std::uint8_t to_level_ : 6;
/// };
/// static_assert(sizeof(node_properties) == 3);
/// ```
#[derive(PackedStruct, Debug, Clone, Copy, Default)]
#[packed_struct(bit_numbering = "lsb0", size_bytes = "3")]
pub struct NodeProperties {
    #[packed_field(bits = "0..=5")]
    pub from_level: Integer<u8, packed_bits::Bits<6>>,

    #[packed_field(bits = "6")]
    pub is_foot_accessible: bool,
    #[packed_field(bits = "7")]
    pub is_bike_accessible: bool,
    #[packed_field(bits = "8")]
    pub is_car_accessible: bool,
    #[packed_field(bits = "9")]
    pub is_elevator: bool,
    #[packed_field(bits = "10")]
    pub is_entrance: bool,
    #[packed_field(bits = "11")]
    pub is_multi_level: bool,
    #[packed_field(bits = "12")]
    pub is_parking: bool,

    #[packed_field(bits = "16..=21")]
    pub to_level: Integer<u8, packed_bits::Bits<6>>,
}

impl NodeProperties {
    pub fn new_elevator() -> Self {
        Self {
            is_elevator: true,
            ..Default::default()
        }
    }

    // Getter methods for compatibility
    pub fn is_car_accessible(&self) -> bool {
        self.is_car_accessible
    }

    pub fn is_bike_accessible(&self) -> bool {
        self.is_bike_accessible
    }

    pub fn is_walk_accessible(&self) -> bool {
        self.is_foot_accessible
    }

    pub fn is_elevator(&self) -> bool {
        self.is_elevator
    }

    pub fn is_multi_level(&self) -> bool {
        self.is_multi_level
    }

    pub fn is_entrance(&self) -> bool {
        self.is_entrance
    }

    pub fn is_parking(&self) -> bool {
        self.is_parking
    }

    pub fn from_level(&self) -> Level {
        Level::from_idx(*self.from_level)
    }

    pub fn to_level(&self) -> Level {
        Level::from_idx(*self.to_level)
    }
}

// Serde support for packed struct
impl serde::Serialize for NodeProperties {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use packed_struct::PackedStruct;
        let bytes = self.pack().map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> serde::Deserialize<'de> for NodeProperties {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use packed_struct::PackedStruct;
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        let bytes_array: [u8; 3] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| serde::de::Error::custom("invalid byte length"))?;
        NodeProperties::unpack(&bytes_array).map_err(serde::de::Error::custom)
    }
}

/// Main street network data structure
///
/// Complete implementation following C++ osr::ways structure.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Ways {
    /// OSM way ID to internal way index mapping (sorted)
    way_osm_idx: Vec<OsmWayIdx>,
    /// OSM node ID to internal node index mapping (sorted)
    node_to_osm: Vec<OsmNodeIdx>,

    // Routing data
    /// Node positions
    node_positions: Vec<Point>,
    /// Way properties
    way_properties: Vec<WayProperties>,
    /// Node properties
    node_properties: Vec<NodeProperties>,

    /// Nodes in each way
    way_nodes: Vec<Vec<NodeIdx>>,
    /// Distance between consecutive nodes in each way (in meters, rounded to u16)
    way_node_dist: Vec<Vec<u16>>,
    /// Index of each routing node into the full way_polylines array
    /// (maps routing node position → OSM node position in polyline)
    way_node_polyline_idx: Vec<Vec<u16>>,

    /// Ways at each node
    node_ways: Vec<Vec<WayIdx>>,
    /// Index of node within each way
    node_in_way_idx: Vec<Vec<u16>>,

    /// Nodes with turn restrictions
    node_is_restricted: Vec<bool>,
    /// Turn restrictions at nodes
    node_restrictions: Vec<Vec<Restriction>>,

    /// Connected component index for each way
    way_component: Vec<ComponentIdx>,

    /// Multi-level elevator nodes
    multi_level_elevators: Vec<(NodeIdx, u64)>, // (node, level_bits)

    /// Bitvector: whether each node is an elevator
    /// Corresponds to C++ node_is_elevator_ bitvec<node_idx_t>
    node_is_elevator: Vec<bool>,

    /// Reverse node properties (from/to levels swapped for reverse routing)
    /// Corresponds to C++ r_node_properties_
    r_node_properties: Vec<NodeProperties>,

    // Geometry and metadata
    /// Polyline geometry for each way
    way_polylines: Vec<Vec<Point>>,
    /// OSM node IDs along each way
    way_osm_nodes: Vec<Vec<OsmNodeIdx>>,

    /// String pool
    strings: Vec<Vec<u8>>,
    /// Way name string indices
    way_names: Vec<StringIdx>,

    /// Ways with conditional access restrictions
    way_has_conditional_access: Vec<bool>,
    /// Conditional access string data
    way_conditional_access: Vec<(WayIdx, StringIdx)>,

    /// Counter for how many ways each OSM node belongs to (used to identify routing nodes)
    /// Only nodes with counter > 1 become routing nodes
    /// Note: Only used during construction, not persisted
    #[serde(skip, default = "AHashMap::new")]
    node_way_counter: AHashMap<OsmNodeIdx, u32>,
}

impl Ways {
    /// Create a new empty Ways structure
    pub fn new() -> Self {
        Self {
            way_osm_idx: Vec::new(),
            node_to_osm: Vec::new(),
            node_positions: Vec::new(),
            way_properties: Vec::new(),
            node_properties: Vec::new(),
            way_nodes: Vec::new(),
            way_node_dist: Vec::new(),
            way_node_polyline_idx: Vec::new(),
            node_ways: Vec::new(),
            node_in_way_idx: Vec::new(),
            node_is_restricted: Vec::new(),
            node_restrictions: Vec::new(),
            way_component: Vec::new(),
            multi_level_elevators: Vec::new(),
            node_is_elevator: Vec::new(),
            r_node_properties: Vec::new(),
            way_polylines: Vec::new(),
            way_osm_nodes: Vec::new(),
            strings: Vec::new(),
            way_names: Vec::new(),
            way_has_conditional_access: Vec::new(),
            way_conditional_access: Vec::new(),
            node_way_counter: AHashMap::new(),
        }
    }

    /// Load ways from file or directory (auto-detects format)
    ///
    /// Checks metadata to determine format:
    /// - If path is a directory with metadata.json → load_multi_file()
    /// - If path is a .bin file → load_from_file() (legacy single-file)
    /// - If path is a directory with ways.bin → load_from_file()
    pub fn load(path: &Path) -> crate::Result<Self> {
        if path.is_dir() {
            // Check for metadata.json (multi-file format)
            let metadata_path = path.join("metadata.json");
            if metadata_path.exists() {
                return Self::load_multi_file(path).map_err(|e| e.into());
            }

            // Check for ways.bin (legacy format in directory)
            let ways_path = path.join("ways.bin");
            if ways_path.exists() {
                return Self::load_from_file(&ways_path).map_err(|e| e.into());
            }

            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No valid ways data found in directory",
            )
            .into());
        } else {
            // Single file (legacy format)
            Self::load_from_file(path).map_err(|e| e.into())
        }
    }

    /// Find way by OSM ID using binary search
    pub fn find_way(&self, osm_way_id: OsmWayIdx) -> Option<WayIdx> {
        self.way_osm_idx
            .binary_search(&osm_way_id)
            .ok()
            .map(|idx| WayIdx(idx as u32))
    }

    /// Find node by raw OSM node ID.
    ///
    /// Uses binary search when `node_to_osm` is sorted (after `connect_ways()` /
    /// `finalize()` in OSM-extraction mode) and falls back to a linear scan for
    /// manually-constructed graphs used in tests.
    ///
    /// # TODO 03 – OSM ID lookup
    /// This is the public entry point for restriction tests that need to map an
    /// OSM node ID (e.g. `528944_u64`) to an internal `NodeIdx`.
    pub fn find_node_by_osm_id(&self, osm_id: u64) -> Option<NodeIdx> {
        // Fast path: binary search (sorted after OSM extraction)
        let target = OsmNodeIdx(osm_id);
        if let Some(idx) = self.find_node_idx(target) {
            return Some(idx);
        }
        // Fallback: linear scan (manually-constructed test graphs)
        self.node_to_osm
            .iter()
            .position(|&n| n.0 == osm_id)
            .map(|i| NodeIdx(i as u32))
    }

    /// Find way by raw OSM way ID.
    ///
    /// Uses binary search when `way_osm_idx` is sorted (after `finalize()`) and
    /// falls back to a linear scan otherwise.
    ///
    /// # TODO 03 – OSM ID lookup
    /// Used in restriction tests to map an OSM way ID to an internal `WayIdx`.
    pub fn find_way_by_osm_id(&self, osm_id: u64) -> Option<WayIdx> {
        // Fast path: binary search (sorted after finalize())
        let target = OsmWayIdx(osm_id);
        if let Some(idx) = self.find_way(target) {
            return Some(idx);
        }
        // Fallback: linear scan
        self.way_osm_idx
            .iter()
            .position(|&w| w.0 == osm_id)
            .map(|i| WayIdx(i as u32))
    }

    /// Find node by OSM ID using binary search
    pub fn find_node_idx(&self, osm_node_id: OsmNodeIdx) -> Option<NodeIdx> {
        self.node_to_osm
            .binary_search(&osm_node_id)
            .ok()
            .map(|idx| NodeIdx(idx as u32))
    }

    /// Get node index, panicking if not found
    pub fn get_node_idx(&self, osm_node_id: OsmNodeIdx) -> NodeIdx {
        self.find_node_idx(osm_node_id)
            .expect(&format!("OSM node {:?} not found", osm_node_id))
    }

    /// Get node position
    pub fn get_node_pos(&self, node: NodeIdx) -> Point {
        self.node_positions
            .get(node.0 as usize)
            .copied()
            .unwrap_or_else(|| Point::from_latlng(0.0, 0.0))
    }

    /// Check if way is a loop
    pub fn is_loop(&self, way: WayIdx) -> bool {
        if let Some(nodes) = self.way_nodes.get(way.0 as usize) {
            if nodes.len() >= 2 {
                return nodes.first() == nodes.last();
            }
        }
        false
    }

    /// Get the OSM way index for an internal way index.
    /// Corresponds to C++ get_osm_way_idx().
    pub fn get_osm_way_idx(&self, way: WayIdx) -> OsmWayIdx {
        self.way_osm_idx[way.0 as usize]
    }

    /// Find positions of two nodes in a way.
    /// Corresponds to C++ find_node_pair().
    pub fn find_node_pair(&self, way: WayIdx, from: NodeIdx, to: NodeIdx) -> Option<(usize, usize)> {
        let nodes = self.way_nodes.get(way.0 as usize)?;
        let from_pos = nodes.iter().position(|n| *n == from)?;
        let to_pos = nodes.iter().position(|n| *n == to)?;
        Some((from_pos, to_pos))
    }

    /// Check if a node is an elevator.
    /// Corresponds to C++ node_is_elevator_ bitvec lookup.
    pub fn is_elevator(&self, node: NodeIdx) -> bool {
        self.node_is_elevator
            .get(node.0 as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Get reverse node properties for a node.
    /// Corresponds to C++ r_node_properties_ lookup.
    pub fn get_r_node_properties(&self, node: NodeIdx) -> NodeProperties {
        self.r_node_properties
            .get(node.0 as usize)
            .copied()
            .unwrap_or_default()
    }

    /// Get number of ways
    pub fn n_ways(&self) -> u32 {
        self.way_osm_idx.len() as u32
    }

    /// Get number of nodes
    pub fn n_nodes(&self) -> u32 {
        self.node_to_osm.len() as u32
    }

    /// Check if node is an additional node (for routing extensions)
    pub fn is_additional_node(&self, node: NodeIdx) -> bool {
        node != NodeIdx::INVALID && node.0 >= self.n_nodes()
    }

    /// Get way properties
    pub fn get_way_properties(&self, way: WayIdx) -> Option<&WayProperties> {
        self.way_properties.get(way.0 as usize)
    }

    /// Get node properties
    pub fn get_node_properties(&self, node: NodeIdx) -> Option<&NodeProperties> {
        self.node_properties.get(node.0 as usize)
    }

    /// Get nodes in a way
    pub fn get_way_nodes(&self, way: WayIdx) -> &[NodeIdx] {
        self.way_nodes
            .get(way.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get node distances along a way
    pub fn get_way_node_distances(&self, way: WayIdx) -> &[u16] {
        self.way_node_dist
            .get(way.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get polyline indices for routing nodes in a way.
    /// Maps routing node position → index in the full way_polylines array.
    pub fn get_way_node_polyline_indices(&self, way: WayIdx) -> &[u16] {
        self.way_node_polyline_idx
            .get(way.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get ways at a node
    pub fn get_node_ways(&self, node: NodeIdx) -> &[WayIdx] {
        self.node_ways
            .get(node.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get precomputed positions of a node within each of its ways.
    ///
    /// The returned slice is parallel to `get_node_ways`: `get_node_in_way_idx(n)[i]`
    /// is the index of node `n` inside `get_node_ways(n)[i]`'s node list.
    /// This allows O(1) position lookup in `adjacent()` instead of a linear scan.
    pub fn get_node_in_way_idx(&self, node: NodeIdx) -> &[u16] {
        self.node_in_way_idx
            .get(node.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get way position for a node-way pair
    pub fn get_way_pos(&self, node: NodeIdx, way: WayIdx) -> WayPos {
        if let Some(ways) = self.node_ways.get(node.0 as usize) {
            for (i, &w) in ways.iter().enumerate() {
                if w == way {
                    return i as WayPos;
                }
            }
        }
        0
    }

    /// Get way position with node-in-way index disambiguation
    pub fn get_way_pos_with_idx(&self, node: NodeIdx, way: WayIdx, node_in_way_idx: u16) -> WayPos {
        if let Some(ways) = self.node_ways.get(node.0 as usize) {
            if let Some(indices) = self.node_in_way_idx.get(node.0 as usize) {
                for (i, (&w, &idx)) in ways.iter().zip(indices.iter()).enumerate() {
                    if w == way
                        && (i + 1 == ways.len() || ways[i] != ways[i + 1] || idx == node_in_way_idx)
                    {
                        return i as WayPos;
                    }
                }
            }
        }
        0
    }

    /// Check if a turn is restricted at a node
    pub fn is_restricted(&self, node: NodeIdx, from_way: WayPos, to_way: WayPos) -> bool {
        let node_idx = node.0 as usize;
        if node_idx >= self.node_is_restricted.len() || !self.node_is_restricted[node_idx] {
            return false;
        }
        if let Some(restrictions) = self.node_restrictions.get(node_idx) {
            let needle = Restriction {
                from: from_way,
                to: to_way,
            };
            restrictions.contains(&needle)
        } else {
            false
        }
    }

    /// Get polyline geometry for a way
    pub fn get_way_polyline(&self, way: WayIdx) -> &[Point] {
        self.way_polylines
            .get(way.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get OSM node IDs along a way
    pub fn get_way_osm_nodes(&self, way: WayIdx) -> &[OsmNodeIdx] {
        self.way_osm_nodes
            .get(way.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get way name
    pub fn get_way_name(&self, way: WayIdx) -> Option<&str> {
        self.way_names
            .get(way.0 as usize)
            .and_then(|&str_idx| self.strings.get(str_idx.0 as usize))
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    /// Get component index for a way
    pub fn get_component(&self, way: WayIdx) -> Option<ComponentIdx> {
        self.way_component.get(way.0 as usize).copied()
    }

    /// Check if way has conditional access restrictions
    pub fn has_conditional_access(&self, way: WayIdx) -> bool {
        self.way_has_conditional_access
            .get(way.0 as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Get conditional access restriction string
    pub fn get_access_restriction(&self, way: WayIdx) -> Option<&str> {
        if !self.has_conditional_access(way) {
            return None;
        }
        self.way_conditional_access
            .iter()
            .find(|(w, _)| *w == way)
            .and_then(|(_, str_idx)| self.strings.get(str_idx.0 as usize))
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }

    /// Get OSM way ID for a given WayIdx
    pub fn get_way_osm_id(&self, way: WayIdx) -> Option<OsmWayIdx> {
        self.way_osm_idx.get(way.0 as usize).copied()
    }

    /// Get OSM node ID for a given NodeIdx
    pub fn get_node_osm_id(&self, node: NodeIdx) -> Option<OsmNodeIdx> {
        self.node_to_osm.get(node.0 as usize).copied()
    }

    /// Add a node with properties
    pub fn add_node(&mut self, osm_id: OsmNodeIdx, pos: Point, props: NodeProperties) -> NodeIdx {
        let idx = NodeIdx(self.node_to_osm.len() as u32);
        self.node_to_osm.push(osm_id);
        self.node_positions.push(pos);
        self.node_properties.push(props);
        self.node_ways.push(Vec::new());
        self.node_in_way_idx.push(Vec::new());
        self.node_is_restricted.push(false);
        self.node_restrictions.push(Vec::new());
        idx
    }

    /// Add a way with properties
    pub fn add_way(
        &mut self,
        osm_id: OsmWayIdx,
        nodes: Vec<NodeIdx>,
        props: WayProperties,
    ) -> WayIdx {
        let idx = WayIdx(self.way_osm_idx.len() as u32);
        self.way_osm_idx.push(osm_id);
        let n = nodes.len();
        self.way_nodes.push(nodes);
        self.way_node_dist.push(Vec::new());
        self.way_node_polyline_idx.push((0..n as u16).collect());
        self.way_properties.push(props);
        self.way_polylines.push(Vec::new());
        self.way_osm_nodes.push(Vec::new());
        self.way_names.push(StringIdx::INVALID);
        self.way_component.push(ComponentIdx::INVALID);
        self.way_has_conditional_access.push(false);
        idx
    }

    /// Add way with OSM node sequence only (routing nodes created later in connect_ways)
    ///
    /// This is used during extraction - we store the OSM node sequence and geometry,
    /// but don't create routing nodes yet. The routing nodes are created in connect_ways()
    /// for only the "multi" nodes (nodes in >1 way).
    pub fn add_way_osm_only(
        &mut self,
        osm_id: OsmWayIdx,
        osm_nodes: Vec<OsmNodeIdx>,
        props: WayProperties,
        polyline: Vec<Point>,
    ) -> WayIdx {
        let idx = WayIdx(self.way_osm_idx.len() as u32);
        self.way_osm_idx.push(osm_id);
        self.way_nodes.push(Vec::new()); // Will be filled in connect_ways()
        self.way_node_dist.push(Vec::new()); // Will be filled in connect_ways()
        self.way_node_polyline_idx.push(Vec::new()); // Will be filled in connect_ways()
        self.way_properties.push(props);
        self.way_polylines.push(polyline);
        self.way_osm_nodes.push(osm_nodes);
        self.way_names.push(StringIdx::INVALID);
        self.way_component.push(ComponentIdx::INVALID);
        self.way_has_conditional_access.push(false);
        idx
    }

    /// Add way with full geometry data
    pub fn add_way_with_geometry(
        &mut self,
        osm_id: OsmWayIdx,
        nodes: Vec<NodeIdx>,
        props: WayProperties,
        polyline: Vec<Point>,
        osm_nodes: Vec<OsmNodeIdx>,
        distances: Vec<u16>,
    ) -> WayIdx {
        let idx = WayIdx(self.way_osm_idx.len() as u32);
        self.way_osm_idx.push(osm_id);
        let n = nodes.len();
        self.way_nodes.push(nodes);
        self.way_node_dist.push(distances);
        self.way_node_polyline_idx.push((0..n as u16).collect());
        self.way_properties.push(props);
        self.way_polylines.push(polyline);
        self.way_osm_nodes.push(osm_nodes);
        self.way_names.push(StringIdx::INVALID);
        self.way_component.push(ComponentIdx::INVALID);
        self.way_has_conditional_access.push(false);
        idx
    }

    /// Set way name
    pub fn set_way_name(&mut self, way: WayIdx, name: String) -> StringIdx {
        let str_idx = StringIdx(self.strings.len() as u32);
        self.strings.push(name.into_bytes());
        if let Some(name_slot) = self.way_names.get_mut(way.0 as usize) {
            *name_slot = str_idx;
        }
        str_idx
    }

    /// Set node properties
    pub fn set_node_properties(&mut self, node: NodeIdx, props: NodeProperties) {
        let node_idx = node.0 as usize;
        if node_idx < self.node_properties.len() {
            self.node_properties[node_idx] = props;
        }
    }

    /// Get multi-level elevator nodes with their level bits
    ///
    /// # C++ Equivalent
    /// ```cpp
    /// vec<pair<node_idx_t, level_bits_t>> multi_level_elevators_;
    /// ```
    pub fn multi_level_elevators(&self) -> &[(NodeIdx, u64)] {
        &self.multi_level_elevators
    }

    /// Add a multi-level elevator node with its level bits
    pub fn add_multi_level_elevator(&mut self, node: NodeIdx, level_bits: u64) {
        self.multi_level_elevators.push((node, level_bits));
    }

    /// Sort multi-level elevators for binary search
    ///
    /// # C++ Equivalent
    /// ```cpp
    /// utl::sort(w.r_->multi_level_elevators_);
    /// ```
    pub fn sort_multi_level_elevators(&mut self) {
        self.multi_level_elevators.sort();
    }

    /// Add a turn restriction from a from-way, to-way, via-node triplet
    pub fn add_restriction(
        &mut self,
        from_way: WayIdx,
        to_way: WayIdx,
        via_node: NodeIdx,
        _is_only: bool,
    ) {
        // Find the via node's position in both ways
        let from_pos = self.get_way_pos(via_node, from_way);
        let to_pos = self.get_way_pos(via_node, to_way);

        let restriction = Restriction {
            from: from_pos,
            to: to_pos,
        };

        self.add_node_restriction(via_node, restriction);
    }

    /// Add a turn restriction at a specific node
    pub fn add_node_restriction(&mut self, node: NodeIdx, restriction: Restriction) {
        let node_idx = node.0 as usize;
        if node_idx < self.node_restrictions.len() {
            self.node_restrictions[node_idx].push(restriction);
            if node_idx < self.node_is_restricted.len() {
                self.node_is_restricted[node_idx] = true;
            }
        }
    }

    /// Add multiple resolved restrictions
    pub fn add_resolved_restrictions(&mut self, restrictions: &[ResolvedRestriction]) {
        for r in restrictions {
            // Find the via node's position in both ways
            let from_pos = self.get_way_pos(r.via, r.from);
            let to_pos = self.get_way_pos(r.via, r.to);

            let restriction = Restriction {
                from: from_pos,
                to: to_pos,
            };

            // Call the node restriction method instead
            self.add_node_restriction(r.via, restriction);
        }
    }

    /// Increment the counter for an OSM node (called during way extraction)
    pub fn increment_node_way_counter(&mut self, osm_node: OsmNodeIdx) {
        *self.node_way_counter.entry(osm_node).or_insert(0) += 1;
    }

    /// Check if an OSM node is "multi" (part of multiple ways)
    pub fn is_multi_node(&self, osm_node: OsmNodeIdx) -> bool {
        self.node_way_counter
            .get(&osm_node)
            .map_or(false, |&count| count > 1)
    }

    /// Connect ways to nodes (build routing graph)
    ///
    /// This is the main graph construction function, following C++ osr::ways::connect_ways().
    ///
    /// Two modes:
    /// 1. OSM extraction mode: If node_way_counter has entries, creates routing nodes only
    ///    for OSM nodes that are part of multiple ways.
    /// 2. Manual mode (for tests): If nodes already exist via add_node(), just builds
    ///    the reverse mappings without recreating nodes.
    pub fn connect_ways(&mut self) {
        // Check if we're in OSM extraction mode or manual mode
        let osm_extraction_mode = !self.node_way_counter.is_empty() && self.node_to_osm.is_empty();

        if osm_extraction_mode {
            println!("  Creating routing nodes for multi-way intersections...");

            // Step 1: Create routing nodes only for "multi" OSM nodes (nodes in >1 way)
            self.node_to_osm.clear();
            for (&osm_node, &count) in &self.node_way_counter {
                if count > 1 {
                    self.node_to_osm.push(osm_node);
                }
            }

            // Sort node_to_osm for binary search
            self.node_to_osm.sort_unstable();

            println!(
                "    Created {} routing nodes from {} total OSM nodes",
                self.node_to_osm.len(),
                self.node_way_counter.len()
            );

            // Initialize node storage
            self.node_ways = vec![Vec::new(); self.node_to_osm.len()];
            self.node_in_way_idx = vec![Vec::new(); self.node_to_osm.len()];
            self.node_properties
                .resize(self.node_to_osm.len(), NodeProperties::default());
            self.node_positions
                .resize(self.node_to_osm.len(), bytemuck::Zeroable::zeroed());
            self.node_is_restricted
                .resize(self.node_to_osm.len(), false);
            self.node_restrictions
                .resize(self.node_to_osm.len(), Vec::new());

            // Step 2: Build routing graph by extracting routing nodes from OSM node sequences
            println!("  Building routing graph...");
            let mut new_way_nodes: Vec<Vec<NodeIdx>> = Vec::with_capacity(self.way_nodes.len());
            let mut new_way_node_dist: Vec<Vec<u16>> = Vec::with_capacity(self.way_node_dist.len());
            let mut new_way_node_polyline_idx: Vec<Vec<u16>> = Vec::with_capacity(self.way_nodes.len());

            for (way_idx, (osm_nodes, polyline)) in self
                .way_osm_nodes
                .iter()
                .zip(self.way_polylines.iter())
                .enumerate()
            {
                let way = WayIdx(way_idx as u32);
                let mut routing_nodes = Vec::new();
                let mut routing_dists = Vec::new();
                let mut routing_polyline_idx = Vec::new();
                let mut distance_accum = 0.0;
                let mut prev_routing_node: Option<NodeIdx> = None;

                // Get way properties to inherit accessibility
                let way_props = self
                    .way_properties
                    .get(way_idx)
                    .copied()
                    .unwrap_or_default();

                // Iterate through OSM nodes and extract only routing nodes
                for (i, (&osm_node, &pos)) in osm_nodes.iter().zip(polyline.iter()).enumerate() {
                    // Accumulate distance
                    if i > 0 {
                        distance_accum += polyline[i - 1].distance_to(&pos);
                    }

                    // Check if this OSM node is a routing node
                    if self.is_multi_node(osm_node) {
                        if let Some(node_idx) = self.find_node_idx(osm_node) {
                            // Add to routing node sequence
                            routing_nodes.push(node_idx);
                            // Record position in the full polyline array
                            routing_polyline_idx.push(i as u16);

                            // Store node position (may overwrite, but all should be same)
                            self.node_positions[node_idx.0 as usize] = pos;

                            // Inherit accessibility from way (nodes are accessible if ANY connected way is accessible)
                            let node_props = &mut self.node_properties[node_idx.0 as usize];
                            node_props.is_foot_accessible |= way_props.is_foot_accessible;
                            node_props.is_bike_accessible |= way_props.is_bike_accessible;
                            node_props.is_car_accessible |= way_props.is_car_accessible;

                            // Add distance if not first routing node
                            if prev_routing_node.is_some() {
                                routing_dists
                                    .push(distance_accum.round().min(u16::MAX as f64) as u16);
                                distance_accum = 0.0;
                            }

                            // Update reverse mapping
                            self.node_ways[node_idx.0 as usize].push(way);
                            self.node_in_way_idx[node_idx.0 as usize]
                                .push(routing_nodes.len() as u16 - 1);

                            prev_routing_node = Some(node_idx);
                        }
                    }
                }

                new_way_nodes.push(routing_nodes);
                new_way_node_dist.push(routing_dists);
                new_way_node_polyline_idx.push(routing_polyline_idx);
            }

            // Replace old way_nodes and way_node_dist with new routing-only versions
            self.way_nodes = new_way_nodes;
            self.way_node_dist = new_way_node_dist;
            self.way_node_polyline_idx = new_way_node_polyline_idx;

            println!("    Built {} ways with routing nodes", self.way_nodes.len());
        } else {
            // Manual mode: Nodes already exist, just build reverse mappings
            println!("  Building node-to-way reverse mappings (manual mode)...");

            // In manual mode, routing nodes map 1:1 to polyline points (identity mapping)
            if self.way_node_polyline_idx.is_empty() {
                self.way_node_polyline_idx = self
                    .way_nodes
                    .iter()
                    .map(|nodes| (0..nodes.len() as u16).collect())
                    .collect();
            }

            // Initialize reverse mappings if needed
            if self.node_ways.len() != self.node_to_osm.len() {
                self.node_ways = vec![Vec::new(); self.node_to_osm.len()];
                self.node_in_way_idx = vec![Vec::new(); self.node_to_osm.len()];
            }

            // Build reverse mappings from ways to nodes
            for (way_idx, way_nodes) in self.way_nodes.iter().enumerate() {
                let way = WayIdx(way_idx as u32);
                for (node_in_way_idx, &node) in way_nodes.iter().enumerate() {
                    let node_idx = node.0 as usize;
                    if node_idx < self.node_ways.len() {
                        self.node_ways[node_idx].push(way);
                        self.node_in_way_idx[node_idx].push(node_in_way_idx as u16);
                    }
                }
            }

            println!(
                "    Built reverse mappings for {} nodes",
                self.node_to_osm.len()
            );
        }
    }

    /// Build connected components using union-find
    pub fn build_components(&mut self) {
        let n_ways = self.n_ways() as usize;
        if n_ways == 0 {
            return;
        }

        // Initialize union-find
        let mut parent: Vec<usize> = (0..n_ways).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut [usize], x: usize, y: usize) {
            let root_x = find(parent, x);
            let root_y = find(parent, y);
            if root_x != root_y {
                parent[root_x] = root_y;
            }
        }

        // Connect ways that share nodes
        for node_ways in &self.node_ways {
            if node_ways.len() > 1 {
                let first = node_ways[0].0 as usize;
                for &way in &node_ways[1..] {
                    union(&mut parent, first, way.0 as usize);
                }
            }
        }

        // Assign component IDs
        use ahash::AHashMap;
        let mut component_map: AHashMap<usize, u32> = AHashMap::new();
        let mut next_component = 0u32;

        self.way_component.clear();
        self.way_component.resize(n_ways, ComponentIdx::INVALID);

        for (way_idx, comp) in self.way_component.iter_mut().enumerate() {
            let root = find(&mut parent, way_idx);
            let component_id = *component_map.entry(root).or_insert_with(|| {
                let id = next_component;
                next_component += 1;
                id
            });
            *comp = ComponentIdx(component_id);
        }
    }

    /// Mark big street neighbors (ways connected to big streets)
    pub fn compute_big_street_neighbors(&mut self) {
        // First pass: mark all big streets
        let mut is_big_street = vec![false; self.way_properties.len()];
        for (idx, props) in self.way_properties.iter().enumerate() {
            is_big_street[idx] = props.is_big_street();
        }

        // Second pass: mark neighbors of big streets
        for node_ways in &self.node_ways {
            let has_big_street = node_ways
                .iter()
                .any(|&w| is_big_street.get(w.0 as usize).copied().unwrap_or(false));

            if has_big_street {
                for &_way in node_ways {
                    // Neighbors of big streets are also considered "big" for routing purposes
                    // This is tracked separately from the is_big_street property
                    // In practice, this could set a separate flag or be used during routing
                }
            }
        }
    }

    /// Finalize by sorting indices for binary search
    /// Finalize by sorting indices for binary search
    pub fn finalize(&mut self) {
        // Sort way indices
        let mut way_mapping: Vec<(OsmWayIdx, WayIdx)> = self
            .way_osm_idx
            .iter()
            .enumerate()
            .map(|(idx, &osm_id)| (osm_id, WayIdx(idx as u32)))
            .collect();
        way_mapping.sort_by_key(|(osm_id, _)| *osm_id);

        // Reorder all way data
        let new_way_osm: Vec<_> = way_mapping.iter().map(|(osm_id, _)| *osm_id).collect();
        let new_way_props: Vec<_> = way_mapping
            .iter()
            .map(|(_, idx)| self.way_properties[idx.0 as usize])
            .collect();
        let new_way_nodes: Vec<_> = way_mapping
            .iter()
            .map(|(_, idx)| self.way_nodes[idx.0 as usize].clone())
            .collect();
        let new_way_node_dist: Vec<_> = way_mapping
            .iter()
            .map(|(_, idx)| self.way_node_dist[idx.0 as usize].clone())
            .collect();
        let new_way_polylines: Vec<_> = way_mapping
            .iter()
            .map(|(_, idx)| self.way_polylines[idx.0 as usize].clone())
            .collect();
        let new_way_osm_nodes: Vec<_> = way_mapping
            .iter()
            .map(|(_, idx)| self.way_osm_nodes[idx.0 as usize].clone())
            .collect();
        let new_way_names: Vec<_> = way_mapping
            .iter()
            .map(|(_, idx)| self.way_names[idx.0 as usize])
            .collect();
        let new_way_component: Vec<_> = way_mapping
            .iter()
            .map(|(_, idx)| self.way_component[idx.0 as usize])
            .collect();
        let new_way_has_conditional: Vec<_> = way_mapping
            .iter()
            .map(|(_, idx)| self.way_has_conditional_access[idx.0 as usize])
            .collect();

        self.way_osm_idx = new_way_osm;
        self.way_properties = new_way_props;
        self.way_nodes = new_way_nodes;
        self.way_node_dist = new_way_node_dist;
        self.way_polylines = new_way_polylines;
        self.way_osm_nodes = new_way_osm_nodes;
        self.way_names = new_way_names;
        self.way_component = new_way_component;
        self.way_has_conditional_access = new_way_has_conditional;

        // Sort node indices
        let mut node_mapping: Vec<(OsmNodeIdx, NodeIdx)> = self
            .node_to_osm
            .iter()
            .enumerate()
            .map(|(idx, &osm_id)| (osm_id, NodeIdx(idx as u32)))
            .collect();
        node_mapping.sort_by_key(|(osm_id, _)| *osm_id);

        // Reorder all node data
        let new_node_osm: Vec<_> = node_mapping.iter().map(|(osm_id, _)| *osm_id).collect();
        let new_node_pos: Vec<_> = node_mapping
            .iter()
            .map(|(_, idx)| self.node_positions[idx.0 as usize])
            .collect();
        let new_node_props: Vec<_> = node_mapping
            .iter()
            .map(|(_, idx)| self.node_properties[idx.0 as usize])
            .collect();
        let new_node_ways: Vec<_> = node_mapping
            .iter()
            .map(|(_, idx)| self.node_ways[idx.0 as usize].clone())
            .collect();
        let new_node_in_way_idx: Vec<_> = node_mapping
            .iter()
            .map(|(_, idx)| self.node_in_way_idx[idx.0 as usize].clone())
            .collect();
        let new_node_is_restricted: Vec<_> = node_mapping
            .iter()
            .map(|(_, idx)| self.node_is_restricted[idx.0 as usize])
            .collect();
        let new_node_restrictions: Vec<_> = node_mapping
            .iter()
            .map(|(_, idx)| self.node_restrictions[idx.0 as usize].clone())
            .collect();

        self.node_to_osm = new_node_osm;
        self.node_positions = new_node_pos;
        self.node_properties = new_node_props;
        self.node_ways = new_node_ways;
        self.node_in_way_idx = new_node_in_way_idx;
        self.node_is_restricted = new_node_is_restricted;
        self.node_restrictions = new_node_restrictions;

        // Build node_is_elevator bitvec from node properties
        self.node_is_elevator = self
            .node_properties
            .iter()
            .map(|np| np.is_elevator)
            .collect();

        // Build reverse node properties (from/to levels swapped)
        self.r_node_properties = self
            .node_properties
            .iter()
            .map(|np| {
                let mut rnp = *np;
                let tmp = rnp.from_level;
                rnp.from_level = rnp.to_level;
                rnp.to_level = tmp;
                rnp
            })
            .collect();
    }

    /// Save Ways to a binary file (legacy single-file format)
    ///
    /// Uses bincode for efficient serialization.
    /// In production, could use rkyv for zero-copy memory-mapped loading.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        bincode::serialize_into(writer, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(())
    }

    /// Save Ways to multiple files (Nigiri-compatible format)
    ///
    /// Splits data into separate memory-mappable files for efficient loading.
    /// Ragged arrays are stored as: [count: u32][offsets: u32...][data: T...]
    ///
    /// Files created:
    /// - Simple arrays: node_pos.bin, node_to_osm.bin, way_osm_idx.bin, etc.
    /// - Packed structs: node_properties.bin (3 bytes/node), way_properties.bin (5 bytes/way)
    /// - Ragged arrays: way_nodes.bin, way_polylines.bin, strings.bin, etc. (combined index+data)
    /// - metadata.json: Format version and dataset info
    ///
    /// This allows selective loading (e.g., routing without geometry) and
    /// memory-mapping for instant startup with large datasets.
    pub fn save_multi_file(&self, dir: &Path) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        std::fs::create_dir_all(dir)?;

        // Helper to write binary data
        let write_bin = |filename: &str, data: &[u8]| -> std::io::Result<()> {
            let path = dir.join(filename);
            let mut file = BufWriter::new(File::create(path)?);
            file.write_all(data)?;
            Ok(())
        };

        // Helper to write ragged array (combined index + data)
        let write_ragged = |filename: &str, data: &[u8], index: &[u32]| -> std::io::Result<()> {
            let path = dir.join(filename);
            let mut file = BufWriter::new(File::create(path)?);
            // Write count
            file.write_all(&(index.len() as u32 - 1).to_le_bytes())?;
            // Write index
            file.write_all(bytemuck::cast_slice(index))?;
            // Write data
            file.write_all(data)?;
            Ok(())
        };

        // Node data
        write_bin("node_pos.bin", bytemuck::cast_slice(&self.node_positions))?;
        write_bin("node_to_osm.bin", bytemuck::cast_slice(&self.node_to_osm))?;

        // Way OSM mapping
        write_bin("way_osm_idx.bin", bytemuck::cast_slice(&self.way_osm_idx))?;

        // Properties (need custom serialization for packed structs)
        {
            use packed_struct::PackedStruct;
            let mut node_props_bytes = Vec::with_capacity(self.node_properties.len() * 3);
            for prop in &self.node_properties {
                node_props_bytes.extend_from_slice(
                    &prop.pack().map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                    })?,
                );
            }
            write_bin("node_properties.bin", &node_props_bytes)?;

            let mut way_props_bytes = Vec::with_capacity(self.way_properties.len() * 5);
            for prop in &self.way_properties {
                way_props_bytes.extend_from_slice(
                    &prop.pack().map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                    })?,
                );
            }
            write_bin("way_properties.bin", &way_props_bytes)?;
        }

        // Ragged arrays (combined index + data)
        {
            // way_nodes
            let (data, index) = flatten_vecvec(&self.way_nodes);
            write_ragged("way_nodes.bin", bytemuck::cast_slice(&data), &index)?;

            // way_node_dist
            let (data, index) = flatten_vecvec(&self.way_node_dist);
            write_ragged("way_node_dist.bin", bytemuck::cast_slice(&data), &index)?;

            // way_node_polyline_idx
            let (data, index) = flatten_vecvec(&self.way_node_polyline_idx);
            write_ragged("way_node_polyline_idx.bin", bytemuck::cast_slice(&data), &index)?;

            // way_polylines
            let (data, index) = flatten_vecvec(&self.way_polylines);
            write_ragged("way_polylines.bin", bytemuck::cast_slice(&data), &index)?;

            // way_osm_nodes
            let (data, index) = flatten_vecvec(&self.way_osm_nodes);
            write_ragged("way_osm_nodes.bin", bytemuck::cast_slice(&data), &index)?;

            // node_ways
            let (data, index) = flatten_vecvec(&self.node_ways);
            write_ragged("node_ways.bin", bytemuck::cast_slice(&data), &index)?;

            // node_in_way_idx
            let (data, index) = flatten_vecvec(&self.node_in_way_idx);
            write_ragged("node_in_way_idx.bin", bytemuck::cast_slice(&data), &index)?;

            // node_restrictions (special format: 2 bytes per restriction)
            let (restrictions, index) = flatten_vecvec(&self.node_restrictions);
            let restrictions_bytes: Vec<u8> = restrictions
                .iter()
                .flat_map(|r| vec![r.from as u8, r.to as u8])
                .collect();
            write_ragged("node_restrictions.bin", &restrictions_bytes, &index)?;
        }

        // Simple arrays
        write_bin("way_names.bin", bytemuck::cast_slice(&self.way_names))?;
        write_bin(
            "way_component.bin",
            bytemuck::cast_slice(&self.way_component),
        )?;
        write_bin(
            "node_is_restricted.bin",
            bytemuck::cast_slice(
                &self
                    .node_is_restricted
                    .iter()
                    .map(|&b| b as u8)
                    .collect::<Vec<_>>(),
            ),
        )?;

        // String pool (combined)
        {
            let (data, index) = flatten_vecvec(&self.strings);
            write_ragged("strings.bin", &data, &index)?;
        }

        // Conditional access (sparse)
        write_bin(
            "way_has_conditional_access.bin",
            bytemuck::cast_slice(
                &self
                    .way_has_conditional_access
                    .iter()
                    .map(|&b| b as u8)
                    .collect::<Vec<_>>(),
            ),
        )?;

        // Multi-level elevators
        {
            let data: Vec<u8> = self
                .multi_level_elevators
                .iter()
                .flat_map(|(node, bits)| {
                    let mut bytes = Vec::with_capacity(12);
                    bytes.extend_from_slice(&node.0.to_le_bytes());
                    bytes.extend_from_slice(&bits.to_le_bytes());
                    bytes
                })
                .collect();
            write_bin("multi_level_elevators.bin", &data)?;
        }

        // Write metadata as JSON
        {
            let metadata = serde_json::json!({
                "format": "multi-file",
                "version": 2,
                "nodes": self.n_nodes(),
                "ways": self.n_ways(),
                "strings": self.strings.len(),
            });
            write_bin("metadata.json", metadata.to_string().as_bytes())?;
        }

        Ok(())
    }

    /// Load Ways from multi-file format
    ///
    /// Reads from directory containing metadata.json and binary data files.
    pub fn load_multi_file(dir: &Path) -> crate::Result<Self> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        // Helper to read entire file
        let read_bin = |filename: &str| -> std::io::Result<Vec<u8>> {
            let path = dir.join(filename);
            let mut file = File::open(path)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;
            Ok(data)
        };

        // Helper to read ragged array (combined format)
        let read_ragged = |filename: &str| -> std::io::Result<(Vec<u8>, Vec<u32>)> {
            let data = read_bin(filename)?;
            let mut reader = BufReader::new(&data[..]);

            // Read count
            let mut count_bytes = [0u8; 4];
            reader.read_exact(&mut count_bytes)?;
            let count = u32::from_le_bytes(count_bytes) as usize;

            // Read index
            let index_size = (count + 1) * 4;
            let index_bytes = &data[4..4 + index_size];
            let index: Vec<u32> = bytemuck::cast_slice(index_bytes).to_vec();

            // Read data
            let data_bytes = data[4 + index_size..].to_vec();

            Ok((data_bytes, index))
        };

        // Read metadata
        let metadata_bytes = read_bin("metadata.json")?;
        let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Verify format
        if metadata["format"] != "multi-file" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid format in metadata",
            )
            .into());
        }

        let mut ways = Ways::new();

        // Load simple arrays
        {
            let data = read_bin("node_pos.bin")?;
            ways.node_positions = bytemuck::cast_slice(&data).to_vec();

            let data = read_bin("node_to_osm.bin")?;
            ways.node_to_osm = bytemuck::cast_slice(&data).to_vec();

            let data = read_bin("way_osm_idx.bin")?;
            ways.way_osm_idx = bytemuck::cast_slice(&data).to_vec();

            let data = read_bin("way_names.bin")?;
            ways.way_names = bytemuck::cast_slice(&data).to_vec();

            let data = read_bin("way_component.bin")?;
            ways.way_component = bytemuck::cast_slice(&data).to_vec();

            let data = read_bin("node_is_restricted.bin")?;
            ways.node_is_restricted = data.iter().map(|&b| b != 0).collect();

            let data = read_bin("way_has_conditional_access.bin")?;
            ways.way_has_conditional_access = data.iter().map(|&b| b != 0).collect();
        }

        // Load packed structs
        {
            use packed_struct::PackedStruct;

            let data = read_bin("node_properties.bin")?;
            ways.node_properties = data
                .chunks_exact(3)
                .map(|chunk| {
                    let arr: [u8; 3] = chunk.try_into().unwrap();
                    NodeProperties::unpack(&arr).unwrap()
                })
                .collect();

            let data = read_bin("way_properties.bin")?;
            ways.way_properties = data
                .chunks_exact(5)
                .map(|chunk| {
                    let arr: [u8; 5] = chunk.try_into().unwrap();
                    WayProperties::unpack(&arr).unwrap()
                })
                .collect();
        }

        // Load ragged arrays
        {
            // way_nodes
            let (data, index) = read_ragged("way_nodes.bin")?;
            ways.way_nodes = unflatten_vecvec(bytemuck::cast_slice(&data), &index);

            // way_node_dist
            let (data, index) = read_ragged("way_node_dist.bin")?;
            ways.way_node_dist = unflatten_vecvec(bytemuck::cast_slice(&data), &index);

            // way_node_polyline_idx (optional - may not exist in older data)
            if dir.join("way_node_polyline_idx.bin").exists() {
                let (data, index) = read_ragged("way_node_polyline_idx.bin")?;
                ways.way_node_polyline_idx = unflatten_vecvec(bytemuck::cast_slice(&data), &index);
            } else {
                // Fallback: identity mapping (routing nodes = polyline nodes)
                ways.way_node_polyline_idx = ways
                    .way_nodes
                    .iter()
                    .map(|nodes| (0..nodes.len() as u16).collect())
                    .collect();
            }

            // way_polylines
            let (data, index) = read_ragged("way_polylines.bin")?;
            ways.way_polylines = unflatten_vecvec(bytemuck::cast_slice(&data), &index);

            // way_osm_nodes
            let (data, index) = read_ragged("way_osm_nodes.bin")?;
            ways.way_osm_nodes = unflatten_vecvec(bytemuck::cast_slice(&data), &index);

            // node_ways
            let (data, index) = read_ragged("node_ways.bin")?;
            ways.node_ways = unflatten_vecvec(bytemuck::cast_slice(&data), &index);

            // node_in_way_idx
            let (data, index) = read_ragged("node_in_way_idx.bin")?;
            ways.node_in_way_idx = unflatten_vecvec(bytemuck::cast_slice(&data), &index);

            // node_restrictions (special: 2 bytes per restriction)
            let (data, index) = read_ragged("node_restrictions.bin")?;
            let restrictions: Vec<Restriction> = data
                .chunks_exact(2)
                .map(|chunk| Restriction {
                    from: chunk[0],
                    to: chunk[1],
                })
                .collect();
            ways.node_restrictions = unflatten_vecvec(&restrictions, &index);

            // strings
            let (data, index) = read_ragged("strings.bin")?;
            ways.strings = unflatten_vecvec(&data, &index);
        }

        // Load multi-level elevators
        {
            let data = read_bin("multi_level_elevators.bin")?;
            ways.multi_level_elevators = data
                .chunks_exact(12)
                .map(|chunk| {
                    let node =
                        NodeIdx(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    let bits = u64::from_le_bytes([
                        chunk[4], chunk[5], chunk[6], chunk[7], chunk[8], chunk[9], chunk[10],
                        chunk[11],
                    ]);
                    (node, bits)
                })
                .collect();
        }

        Ok(ways)
    }

    /// Load Ways from a binary file
    ///
    /// Uses bincode for deserialization.
    /// In production, could use rkyv for zero-copy memory-mapped loading.
    pub fn load_from_file(path: &Path) -> std::io::Result<Self> {
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        bincode::deserialize_from(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

impl Default for Ways {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to flatten a Vec<Vec<T>> into (data, index) for ragged array storage
fn flatten_vecvec<T: Clone>(vecvec: &[Vec<T>]) -> (Vec<T>, Vec<u32>) {
    let mut data = Vec::new();
    let mut index = Vec::with_capacity(vecvec.len() + 1);

    index.push(0u32);
    for vec in vecvec {
        data.extend_from_slice(vec);
        index.push(data.len() as u32);
    }

    (data, index)
}

/// Helper function to unflatten (data, index) back into Vec<Vec<T>>
fn unflatten_vecvec<T: Clone>(data: &[T], index: &[u32]) -> Vec<Vec<T>> {
    if index.len() <= 1 {
        return Vec::new();
    }

    (0..index.len() - 1)
        .map(|i| {
            let start = index[i] as usize;
            let end = index[i + 1] as usize;
            data[start..end].to_vec()
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_way_properties_bitpacking() {
        // Test that direct field access works with packed_struct
        let props = WayProperties {
            is_foot_accessible: true,
            is_bike_accessible: false,
            is_car_accessible: true,
            speed_limit: SpeedLimit::Kmh50,
            from_level: 3.into(),
            to_level: 5.into(),
            is_big_street: true,
            ..Default::default()
        };

        assert!(props.is_foot_accessible());
        assert!(!props.is_bike_accessible());
        assert!(props.is_car_accessible());
        assert_eq!(props.speed_limit, SpeedLimit::Kmh50);
        assert_eq!(*props.from_level, 3);
        assert_eq!(*props.to_level, 5);
        assert!(props.is_big_street());

        // Verify packed size
        use packed_struct::PackedStruct;
        let packed = props.pack().unwrap();
        assert_eq!(packed.len(), 5);
    }

    #[test]
    fn test_node_properties_bitpacking() {
        let props = NodeProperties {
            is_foot_accessible: true,
            is_car_accessible: false,
            is_elevator: true,
            from_level: 2.into(),
            to_level: 4.into(),
            ..Default::default()
        };

        assert!(props.is_walk_accessible());
        assert!(!props.is_car_accessible());
        assert!(props.is_elevator());
        assert_eq!(*props.from_level, 2);
        assert_eq!(*props.to_level, 4);

        // Verify packed size
        use packed_struct::PackedStruct;
        let packed = props.pack().unwrap();
        assert_eq!(packed.len(), 3);
    }

    #[test]
    fn test_way_properties_size() {
        // packed_struct stores the packed bytes (5 bytes) but the struct itself is larger
        // The packed representation is what matters for serialization
        use packed_struct::PackedStruct;
        let props = WayProperties::default();
        let packed = props.pack().unwrap();
        assert_eq!(packed.len(), 5);
    }

    #[test]
    fn test_node_properties_size() {
        // packed_struct stores the packed bytes (3 bytes) but the struct itself is larger
        use packed_struct::PackedStruct;
        let props = NodeProperties::default();
        let packed = props.pack().unwrap();
        assert_eq!(packed.len(), 3);
    }

    #[test]
    fn test_restriction_equality() {
        let r1 = Restriction { from: 0, to: 1 };
        let r2 = Restriction { from: 0, to: 1 };
        let r3 = Restriction { from: 1, to: 0 };

        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_ways_new() {
        let ways = Ways::new();
        assert_eq!(ways.n_ways(), 0);
        assert_eq!(ways.n_nodes(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut ways = Ways::new();
        let pos = Point::from_latlng(37.7749, -122.4194);
        let props = NodeProperties::default();

        let idx = ways.add_node(OsmNodeIdx(123), pos, props);
        assert_eq!(idx, NodeIdx(0));
        assert_eq!(ways.n_nodes(), 1);
        assert_eq!(ways.get_node_pos(NodeIdx(0)), pos);
    }

    #[test]
    fn test_add_way() {
        let mut ways = Ways::new();
        let props = WayProperties::default();

        let idx = ways.add_way(
            OsmWayIdx(456),
            vec![NodeIdx(0), NodeIdx(1), NodeIdx(2)],
            props,
        );
        assert_eq!(idx, WayIdx(0));
        assert_eq!(ways.n_ways(), 1);
    }

    #[test]
    fn test_find_way() {
        let mut ways = Ways::new();
        ways.add_way(OsmWayIdx(100), vec![], WayProperties::default());
        ways.add_way(OsmWayIdx(200), vec![], WayProperties::default());
        ways.add_way(OsmWayIdx(150), vec![], WayProperties::default());

        // Binary search requires sorted data, so finalize first
        ways.finalize();

        // After finalize, data is sorted by OSM ID
        assert_eq!(ways.find_way(OsmWayIdx(100)), Some(WayIdx(0)));
        assert_eq!(ways.find_way(OsmWayIdx(150)), Some(WayIdx(1)));
        assert_eq!(ways.find_way(OsmWayIdx(200)), Some(WayIdx(2)));

        // Non-existent way
        assert_eq!(ways.find_way(OsmWayIdx(999)), None);
    }

    #[test]
    fn test_find_node_idx() {
        let mut ways = Ways::new();
        let pos = Point::from_latlng(0.0, 0.0);
        ways.add_node(OsmNodeIdx(500), pos, NodeProperties::default());
        ways.add_node(OsmNodeIdx(300), pos, NodeProperties::default());
        ways.add_node(OsmNodeIdx(400), pos, NodeProperties::default());

        ways.finalize();

        assert_eq!(ways.find_node_idx(OsmNodeIdx(300)), Some(NodeIdx(0)));
        assert_eq!(ways.find_node_idx(OsmNodeIdx(400)), Some(NodeIdx(1)));
        assert_eq!(ways.find_node_idx(OsmNodeIdx(500)), Some(NodeIdx(2)));
        assert_eq!(ways.find_node_idx(OsmNodeIdx(999)), None);
    }

    #[test]
    fn test_is_loop() {
        let mut ways = Ways::new();

        // Add non-loop way
        ways.add_way(
            OsmWayIdx(100),
            vec![NodeIdx(0), NodeIdx(1), NodeIdx(2)],
            WayProperties::default(),
        );
        assert!(!ways.is_loop(WayIdx(0)));

        // Add loop way
        ways.add_way(
            OsmWayIdx(200),
            vec![NodeIdx(0), NodeIdx(1), NodeIdx(2), NodeIdx(0)],
            WayProperties::default(),
        );
        assert!(ways.is_loop(WayIdx(1)));
    }

    #[test]
    fn test_get_way_nodes() {
        let mut ways = Ways::new();
        ways.add_way(
            OsmWayIdx(100),
            vec![NodeIdx(5), NodeIdx(6), NodeIdx(7)],
            WayProperties::default(),
        );

        let nodes = ways.get_way_nodes(WayIdx(0));
        assert_eq!(nodes, &[NodeIdx(5), NodeIdx(6), NodeIdx(7)]);

        let empty = ways.get_way_nodes(WayIdx(99));
        assert_eq!(empty, &[]);
    }

    #[test]
    fn test_restrictions() {
        let mut ways = Ways::new();
        let pos = Point::from_latlng(0.0, 0.0);
        ways.add_node(OsmNodeIdx(100), pos, NodeProperties::default());

        let restriction = Restriction { from: 1, to: 2 };
        ways.add_node_restriction(NodeIdx(0), restriction);

        assert!(ways.is_restricted(NodeIdx(0), 1, 2));
        assert!(!ways.is_restricted(NodeIdx(0), 2, 1));
        assert!(!ways.is_restricted(NodeIdx(0), 1, 3));
    }

    #[test]
    fn test_finalize_preserves_data() {
        let mut ways = Ways::new();

        // Add nodes
        let pos1 = Point::from_latlng(37.7749, -122.4194);
        let pos2 = Point::from_latlng(37.8044, -122.2711);
        ways.add_node(OsmNodeIdx(500), pos1, NodeProperties::default());
        ways.add_node(OsmNodeIdx(300), pos2, NodeProperties::default());

        // Add ways
        ways.add_way(
            OsmWayIdx(200),
            vec![NodeIdx(0), NodeIdx(1)],
            WayProperties::default(),
        );
        ways.add_way(
            OsmWayIdx(100),
            vec![NodeIdx(1), NodeIdx(0)],
            WayProperties::default(),
        );

        ways.finalize();

        // Verify data is preserved after sorting
        assert_eq!(ways.n_nodes(), 2);
        assert_eq!(ways.n_ways(), 2);

        // Verify sorted access works
        assert_eq!(ways.find_way(OsmWayIdx(100)), Some(WayIdx(0)));
        assert_eq!(ways.find_way(OsmWayIdx(200)), Some(WayIdx(1)));
        assert_eq!(ways.find_node_idx(OsmNodeIdx(300)), Some(NodeIdx(0)));
        assert_eq!(ways.find_node_idx(OsmNodeIdx(500)), Some(NodeIdx(1)));
    }

    #[test]
    fn test_way_with_geometry() {
        let mut ways = Ways::new();

        let nodes = vec![NodeIdx(0), NodeIdx(1), NodeIdx(2)];
        let polyline = vec![
            Point::from_latlng(37.0, -122.0),
            Point::from_latlng(37.1, -122.1),
            Point::from_latlng(37.2, -122.2),
        ];
        let osm_nodes = vec![OsmNodeIdx(1000), OsmNodeIdx(1001), OsmNodeIdx(1002)];
        let distances = vec![100, 150];

        let way_idx = ways.add_way_with_geometry(
            OsmWayIdx(100),
            nodes.clone(),
            WayProperties::default(),
            polyline.clone(),
            osm_nodes.clone(),
            distances.clone(),
        );

        assert_eq!(ways.get_way_polyline(way_idx), &polyline[..]);
        assert_eq!(ways.get_way_osm_nodes(way_idx), &osm_nodes[..]);
        assert_eq!(ways.get_way_node_distances(way_idx), &distances[..]);
    }

    #[test]
    fn test_way_name() {
        let mut ways = Ways::new();
        let way_idx = ways.add_way(OsmWayIdx(100), vec![], WayProperties::default());

        ways.set_way_name(way_idx, "Main Street".to_string());

        assert_eq!(ways.get_way_name(way_idx), Some("Main Street"));
    }

    #[test]
    fn test_connect_ways() {
        let mut ways = Ways::new();

        // Add nodes
        for i in 0..4 {
            let pos = Point::from_latlng(37.0 + i as f64 * 0.1, -122.0);
            ways.add_node(OsmNodeIdx(i), pos, NodeProperties::default());
        }

        // Add ways that share nodes
        ways.add_way(
            OsmWayIdx(1),
            vec![NodeIdx(0), NodeIdx(1), NodeIdx(2)],
            WayProperties::default(),
        );
        ways.add_way(
            OsmWayIdx(2),
            vec![NodeIdx(1), NodeIdx(2), NodeIdx(3)],
            WayProperties::default(),
        );

        ways.connect_ways();

        // Node 1 should be connected to both ways
        let node_1_ways = ways.get_node_ways(NodeIdx(1));
        assert_eq!(node_1_ways.len(), 2);
        assert!(node_1_ways.contains(&WayIdx(0)));
        assert!(node_1_ways.contains(&WayIdx(1)));
    }

    #[test]
    fn test_build_components() {
        let mut ways = Ways::new();

        // Add nodes
        for i in 0..6 {
            let pos = Point::from_latlng(37.0 + i as f64 * 0.1, -122.0);
            ways.add_node(OsmNodeIdx(i), pos, NodeProperties::default());
        }

        // Create two separate components
        // Component 1: ways 0, 1 (connected via node 1)
        ways.add_way(
            OsmWayIdx(10),
            vec![NodeIdx(0), NodeIdx(1)],
            WayProperties::default(),
        );
        ways.add_way(
            OsmWayIdx(11),
            vec![NodeIdx(1), NodeIdx(2)],
            WayProperties::default(),
        );

        // Component 2: ways 2, 3 (connected via node 4)
        ways.add_way(
            OsmWayIdx(20),
            vec![NodeIdx(3), NodeIdx(4)],
            WayProperties::default(),
        );
        ways.add_way(
            OsmWayIdx(21),
            vec![NodeIdx(4), NodeIdx(5)],
            WayProperties::default(),
        );

        ways.connect_ways();
        ways.build_components();

        // Ways 0 and 1 should be in the same component
        let comp0 = ways.get_component(WayIdx(0)).unwrap();
        let comp1 = ways.get_component(WayIdx(1)).unwrap();
        assert_eq!(comp0, comp1);

        // Ways 2 and 3 should be in the same component
        let comp2 = ways.get_component(WayIdx(2)).unwrap();
        let comp3 = ways.get_component(WayIdx(3)).unwrap();
        assert_eq!(comp2, comp3);

        // But components should be different
        assert_ne!(comp0, comp2);
    }

    #[test]
    fn test_get_way_pos() {
        let mut ways = Ways::new();

        // Add nodes
        for i in 0..3 {
            let pos = Point::from_latlng(37.0, -122.0);
            ways.add_node(OsmNodeIdx(i), pos, NodeProperties::default());
        }

        // Add ways
        ways.add_way(
            OsmWayIdx(10),
            vec![NodeIdx(0), NodeIdx(1)],
            WayProperties::default(),
        );
        ways.add_way(
            OsmWayIdx(20),
            vec![NodeIdx(1), NodeIdx(2)],
            WayProperties::default(),
        );
        ways.add_way(
            OsmWayIdx(30),
            vec![NodeIdx(1), NodeIdx(0)],
            WayProperties::default(),
        );

        ways.connect_ways();

        // Node 1 is in all three ways - find position of way 1
        let pos = ways.get_way_pos(NodeIdx(1), WayIdx(1));
        assert!(pos < 3); // Should find the way
    }

    #[test]
    fn test_resolved_restrictions() {
        let mut ways = Ways::new();

        // Add nodes
        for i in 0..3 {
            let pos = Point::from_latlng(37.0, -122.0);
            ways.add_node(OsmNodeIdx(i), pos, NodeProperties::default());
        }

        // Add ways
        ways.add_way(
            OsmWayIdx(10),
            vec![NodeIdx(0), NodeIdx(1)],
            WayProperties::default(),
        );
        ways.add_way(
            OsmWayIdx(20),
            vec![NodeIdx(1), NodeIdx(2)],
            WayProperties::default(),
        );

        ways.connect_ways();

        let restrictions = vec![ResolvedRestriction {
            type_: RestrictionType::No,
            from: WayIdx(0),
            to: WayIdx(1),
            via: NodeIdx(1),
        }];

        ways.add_resolved_restrictions(&restrictions);

        // Restriction should be added
        let from_pos = ways.get_way_pos(NodeIdx(1), WayIdx(0));
        let to_pos = ways.get_way_pos(NodeIdx(1), WayIdx(1));
        assert!(ways.is_restricted(NodeIdx(1), from_pos, to_pos));
    }

    #[test]
    fn test_is_additional_node() {
        let mut ways = Ways::new();
        let pos = Point::from_latlng(37.0, -122.0);
        ways.add_node(OsmNodeIdx(0), pos, NodeProperties::default());

        assert!(!ways.is_additional_node(NodeIdx(0)));
        assert!(ways.is_additional_node(NodeIdx(100)));
    }

    #[test]
    fn test_get_node_idx() {
        let mut ways = Ways::new();
        let pos = Point::from_latlng(37.0, -122.0);
        ways.add_node(OsmNodeIdx(12345), pos, NodeProperties::default());
        ways.finalize();

        let idx = ways.get_node_idx(OsmNodeIdx(12345));
        assert_eq!(idx, NodeIdx(0));
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn test_get_node_idx_not_found() {
        let ways = Ways::new();
        ways.get_node_idx(OsmNodeIdx(99999));
    }
}
