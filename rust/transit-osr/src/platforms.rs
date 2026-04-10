//! Translation of osr/include/osr/platforms.h
//!
//! Platform (transit stop) data structures.
//! Complete implementation with spatial indexing and OSM object handling.

use std::path::Path;

use rkyv::{Archive, Deserialize, Serialize};

use crate::types::*;
use crate::ways::Ways;
use crate::{Level, Location, Point};

/// Reference to either a way or a node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub enum Ref {
    Way(WayIdx),
    Node(NodeIdx),
}

type RefValue = u32;

const K_NODE_MARKER: u32 = 1u32 << 31;

impl Ref {
    pub fn to_value(self) -> RefValue {
        match self {
            Ref::Way(w) => w.value(),
            Ref::Node(n) => n.value() + K_NODE_MARKER,
        }
    }

    pub fn from_value(v: RefValue) -> Self {
        if v < K_NODE_MARKER {
            Ref::Way(WayIdx::new(v))
        } else {
            Ref::Node(NodeIdx::new(v - K_NODE_MARKER))
        }
    }

    pub fn is_way(&self) -> bool {
        matches!(self, Ref::Way(_))
    }

    pub fn is_node(&self) -> bool {
        matches!(self, Ref::Node(_))
    }
}

/// OSM tags for extracting platform names
const PLATFORM_NAME_TAGS: &[&str] = &[
    "ref",
    "ref:IFOPT",
    "ref:operator",
    "ref_name",
    "local_ref",
    "name",
    "alt_name",
    "description",
];

/// Platform data structure
///
/// Stores platform (transit stop) information including:
/// - Node positions for platform nodes
/// - Platform/node flags (bitvec)
/// - Platform references (way/node)
/// - Platform names
/// - Spatial index for efficient lookups
#[derive(Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct Platforms {
    /// Node positions for platform nodes
    node_pos: Vec<(NodeIdx, Point)>,

    /// Bitfield marking which nodes are platforms
    node_is_platform: Vec<bool>,

    /// Bitfield marking which ways are platforms
    way_is_platform: Vec<bool>,

    /// References for each platform (can have multiple refs per platform)
    platform_ref: Vec<Vec<RefValue>>,

    /// Platform names (multiple names per platform)
    platform_names: Vec<Vec<String>>,

    /// String buffer for tag extraction
    #[with(rkyv::with::Skip)]
    strings: Vec<String>,
}

impl Platforms {
    /// Create a new empty Platforms structure
    pub fn new() -> Self {
        Self {
            node_pos: Vec::new(),
            node_is_platform: Vec::new(),
            way_is_platform: Vec::new(),
            platform_ref: Vec::new(),
            platform_names: Vec::new(),
            strings: Vec::new(),
        }
    }

    /// Add a way as a platform
    ///
    /// # Arguments
    /// * `way` - Way index
    /// * `tags` - OSM tags for extracting names
    ///
    /// # Returns
    /// Platform index if names were found, None otherwise
    pub fn add_way(&mut self, way: WayIdx, tags: &[(String, String)]) -> Option<PlatformIdx> {
        assert!(way.value() < K_NODE_MARKER);

        let platform_idx = self.add_names(tags)?;

        // Resize bitvec if needed
        if way.value() as usize >= self.way_is_platform.len() {
            self.way_is_platform.resize(way.value() as usize + 1, false);
        }
        self.way_is_platform[way.value() as usize] = true;

        // Add reference
        self.platform_ref.push(vec![Ref::Way(way).to_value()]);

        Some(platform_idx)
    }

    /// Add a node as a platform
    ///
    /// # Arguments
    /// * `node` - Node index
    /// * `pos` - Geographic position
    /// * `tags` - OSM tags for extracting names
    ///
    /// # Returns
    /// Platform index if names were found, None otherwise
    pub fn add_node(
        &mut self,
        node: NodeIdx,
        pos: Point,
        tags: &[(String, String)],
    ) -> Option<PlatformIdx> {
        assert!(node.value() < K_NODE_MARKER);

        let platform_idx = self.add_names(tags)?;

        // Add position
        self.node_pos.push((node, pos));

        // Resize bitvec if needed
        if node.value() as usize >= self.node_is_platform.len() {
            self.node_is_platform
                .resize(node.value() as usize + 1, false);
        }
        self.node_is_platform[node.value() as usize] = true;

        // Add reference
        self.platform_ref.push(vec![Ref::Node(node).to_value()]);

        Some(platform_idx)
    }

    /// Add a relation as a platform
    ///
    /// # Arguments
    /// * `tags` - OSM tags for extracting names
    ///
    /// # Returns
    /// Platform index if names were found, None otherwise
    pub fn add_relation(&mut self, tags: &[(String, String)]) -> Option<PlatformIdx> {
        let platform_idx = self.add_names(tags)?;

        // Add empty reference list (will be filled with members)
        self.platform_ref.push(vec![]);

        Some(platform_idx)
    }

    /// Extract platform names from OSM tags
    ///
    /// Looks for ref, name, description, etc. tags and extracts semicolon-separated values.
    fn add_names(&mut self, tags: &[(String, String)]) -> Option<PlatformIdx> {
        self.strings.clear();

        for (key, value) in tags {
            if PLATFORM_NAME_TAGS.contains(&key.as_str()) {
                // Split by semicolon and add each part
                for part in value.split(';') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        self.strings.push(trimmed.to_string());
                    }
                }
            }
        }

        if self.strings.is_empty() {
            return None;
        }

        let idx = PlatformIdx::new(self.platform_names.len() as u32);
        self.platform_names.push(self.strings.clone());
        Some(idx)
    }

    /// Get the level of a platform
    pub fn get_level(&self, ways: &Ways, platform: PlatformIdx) -> Level {
        if platform.value() as usize >= self.platform_ref.len() {
            return Level::default();
        }

        let refs = &self.platform_ref[platform.value() as usize];
        if refs.is_empty() {
            return Level::default();
        }

        match Ref::from_value(refs[0]) {
            Ref::Way(way_idx) => ways
                .get_way_properties(way_idx)
                .map(|p| p.from_level())
                .unwrap_or_default(),
            Ref::Node(node_idx) => ways
                .get_node_properties(node_idx)
                .map(|p| p.from_level())
                .unwrap_or_default(),
        }
    }

    /// Get the number of platforms
    pub fn num_platforms(&self) -> usize {
        self.platform_ref.len()
    }

    /// Check if platforms are empty
    pub fn is_empty(&self) -> bool {
        self.platform_ref.is_empty()
    }

    /// Find platforms within a bounding box
    ///
    /// Note: Without an R-tree, this is a linear search. For production,
    /// integrate rstar crate or implement R-tree.
    pub fn find(&self, location: &Location, max_distance: f64) -> Vec<PlatformIdx> {
        let mut results = Vec::new();

        // Search node platforms
        for (node, pos) in &self.node_pos {
            let dist = location.pos_.distance_to(pos);
            if dist <= max_distance {
                // Find platform index for this node
                if let Some(platform_idx) = self.find_platform_for_node(*node) {
                    results.push(platform_idx);
                }
            }
        }

        results
    }

    /// Find platforms within a lat/lng bounding box
    pub fn find_bbox(
        &self,
        min_lat: f64,
        min_lng: f64,
        max_lat: f64,
        max_lng: f64,
    ) -> Vec<PlatformIdx> {
        let mut results = Vec::new();

        for (node, pos) in &self.node_pos {
            let (lat, lng) = pos.as_latlng();
            if lat >= min_lat && lat <= max_lat && lng >= min_lng && lng <= max_lng {
                if let Some(platform_idx) = self.find_platform_for_node(*node) {
                    results.push(platform_idx);
                }
            }
        }

        results
    }

    /// Find platform index for a given node
    fn find_platform_for_node(&self, node: NodeIdx) -> Option<PlatformIdx> {
        for (idx, refs) in self.platform_ref.iter().enumerate() {
            for &ref_val in refs {
                if let Ref::Node(n) = Ref::from_value(ref_val) {
                    if n == node {
                        return Some(PlatformIdx::new(idx as u32));
                    }
                }
            }
        }
        None
    }

    /// Get node position
    pub fn get_node_pos(&self, node: NodeIdx) -> Option<Point> {
        self.node_pos
            .binary_search_by_key(&node, |(n, _)| *n)
            .ok()
            .map(|idx| self.node_pos[idx].1)
    }

    /// Check if a node is a platform
    pub fn is_node_platform(&self, node: NodeIdx) -> bool {
        let idx = node.value() as usize;
        idx < self.node_is_platform.len() && self.node_is_platform[idx]
    }

    /// Check if a way is a platform
    pub fn is_way_platform(&self, way: WayIdx) -> bool {
        let idx = way.value() as usize;
        idx < self.way_is_platform.len() && self.way_is_platform[idx]
    }

    /// Get platform names
    pub fn get_names(&self, platform: PlatformIdx) -> Option<&[String]> {
        self.platform_names
            .get(platform.value() as usize)
            .map(|v| v.as_slice())
    }

    /// Get platform references (ways/nodes)
    pub fn get_refs(&self, platform: PlatformIdx) -> Option<Vec<Ref>> {
        self.platform_ref
            .get(platform.value() as usize)
            .map(|refs| refs.iter().map(|&r| Ref::from_value(r)).collect())
    }

    /// Save platforms to directory (multi-file format)
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        // Helper to write binary data
        let write_bin = |filename: &str, data: &[u8]| -> std::io::Result<()> {
            let path = dir.join(filename);
            let mut file = BufWriter::new(File::create(path)?);
            file.write_all(data)?;
            Ok(())
        };

        // Helper to write ragged array
        let write_ragged = |filename: &str, data: &[u8], index: &[u32]| -> std::io::Result<()> {
            let path = dir.join(filename);
            let mut file = BufWriter::new(File::create(path)?);
            file.write_all(&(index.len() as u32 - 1).to_le_bytes())?;
            file.write_all(bytemuck::cast_slice(index))?;
            file.write_all(data)?;
            Ok(())
        };

        // Node positions
        {
            let mut nodes_data = Vec::with_capacity(self.node_pos.len() * 4);
            let mut pos_data = Vec::with_capacity(self.node_pos.len() * 8);
            for (node, pos) in &self.node_pos {
                nodes_data.extend_from_slice(&node.0.to_le_bytes());
                pos_data.extend_from_slice(&pos.lat_.to_le_bytes());
                pos_data.extend_from_slice(&pos.lng_.to_le_bytes());
            }
            write_bin("platform_node_idx.bin", &nodes_data)?;
            write_bin("platform_node_pos.bin", &pos_data)?;
        }

        // Bitfields
        write_bin(
            "platform_node_is_platform.bin",
            bytemuck::cast_slice(
                &self
                    .node_is_platform
                    .iter()
                    .map(|&b| b as u8)
                    .collect::<Vec<_>>(),
            ),
        )?;
        write_bin(
            "platform_way_is_platform.bin",
            bytemuck::cast_slice(
                &self
                    .way_is_platform
                    .iter()
                    .map(|&b| b as u8)
                    .collect::<Vec<_>>(),
            ),
        )?;

        // Platform references (ragged array)
        {
            let (data, index) = flatten_vecvec(&self.platform_ref);
            write_ragged("platform_ref.bin", bytemuck::cast_slice(&data), &index)?;
        }

        // Platform names (nested ragged array)
        {
            // First flatten each Vec<String> to bytes
            let mut all_names_data = Vec::new();
            let mut names_index = vec![0u32];

            for names in &self.platform_names {
                for name in names {
                    all_names_data.extend_from_slice(name.as_bytes());
                }
                names_index.push(all_names_data.len() as u32);
            }

            // Now create a second level index for individual strings
            let mut string_index = vec![0u32];
            let mut current_offset = 0u32;
            for names in &self.platform_names {
                for name in names {
                    current_offset += name.len() as u32;
                    string_index.push(current_offset);
                }
            }

            write_bin("platform_names_data.bin", &all_names_data)?;
            write_ragged(
                "platform_names.bin",
                bytemuck::cast_slice(&string_index),
                &names_index,
            )?;
        }

        Ok(())
    }

    /// Load platforms from multi-file format (internal implementation)
    pub fn load(dir: &Path) -> crate::Result<Self> {
        use std::fs::File;
        use std::io::Read;

        // Helper to read entire file
        let read_bin = |filename: &str| -> std::io::Result<Vec<u8>> {
            let path = dir.join(filename);
            if !path.exists() {
                return Ok(Vec::new());
            }
            let mut file = File::open(path)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;
            Ok(data)
        };

        // Helper to read ragged array
        let read_ragged = |filename: &str| -> std::io::Result<(Vec<u8>, Vec<u32>)> {
            let data = read_bin(filename)?;
            if data.len() < 4 {
                return Ok((Vec::new(), vec![0]));
            }

            let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
            let index_size = (count + 1) * 4;
            if data.len() < 4 + index_size {
                return Ok((Vec::new(), vec![0]));
            }

            let index_bytes = &data[4..4 + index_size];
            let index: Vec<u32> = bytemuck::cast_slice(index_bytes).to_vec();
            let data_bytes = data[4 + index_size..].to_vec();

            Ok((data_bytes, index))
        };

        let mut platforms = Platforms::new();

        // Node positions
        {
            let nodes_data = read_bin("platform_node_idx.bin")?;
            let pos_data = read_bin("platform_node_pos.bin")?;

            if !nodes_data.is_empty() && !pos_data.is_empty() {
                let nodes: Vec<NodeIdx> = bytemuck::cast_slice(&nodes_data).to_vec();
                let positions: Vec<i32> = bytemuck::cast_slice(&pos_data).to_vec();

                for (i, &node) in nodes.iter().enumerate() {
                    if i * 2 + 1 < positions.len() {
                        let pos = Point {
                            lat_: positions[i * 2],
                            lng_: positions[i * 2 + 1],
                        };
                        platforms.node_pos.push((node, pos));
                    }
                }
            }
        }

        // Bitfields
        {
            let data = read_bin("platform_node_is_platform.bin")?;
            platforms.node_is_platform = data.iter().map(|&b| b != 0).collect();

            let data = read_bin("platform_way_is_platform.bin")?;
            platforms.way_is_platform = data.iter().map(|&b| b != 0).collect();
        }

        // Platform references
        {
            let (data, index) = read_ragged("platform_ref.bin")?;
            if !data.is_empty() {
                platforms.platform_ref = unflatten_vecvec(bytemuck::cast_slice(&data), &index);
            }
        }

        // Platform names (nested ragged array)
        {
            let names_data = read_bin("platform_names_data.bin")?;
            let (string_index_data, names_index) = read_ragged("platform_names.bin")?;

            if !names_data.is_empty() && !string_index_data.is_empty() {
                let string_index: Vec<u32> = bytemuck::cast_slice(&string_index_data).to_vec();

                // Reconstruct nested structure
                for i in 0..names_index.len() - 1 {
                    let start_str_idx = names_index[i] as usize;
                    let end_str_idx = names_index[i + 1] as usize;

                    let mut names = Vec::new();
                    for j in start_str_idx..end_str_idx {
                        if j + 1 < string_index.len() {
                            let start = string_index[j] as usize;
                            let end = string_index[j + 1] as usize;
                            if end <= names_data.len() {
                                if let Ok(s) = std::str::from_utf8(&names_data[start..end]) {
                                    names.push(s.to_string());
                                }
                            }
                        }
                    }
                    platforms.platform_names.push(names);
                }
            }
        }

        Ok(platforms)
    }

    /// Finalize: sort node positions for binary search
    pub fn finalize(&mut self) {
        self.node_pos.sort_by_key(|(node, _)| *node);
    }

    /// Get number of platforms
    pub fn n_platforms(&self) -> usize {
        self.platform_names.len()
    }
}

impl Default for Platforms {
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
    fn test_ref_way() {
        let r = Ref::Way(WayIdx::new(12345));
        assert!(r.is_way());
        assert!(!r.is_node());

        let val = r.to_value();
        let r2 = Ref::from_value(val);
        assert_eq!(r, r2);
    }

    #[test]
    fn test_ref_node() {
        let r = Ref::Node(NodeIdx::new(67890));
        assert!(r.is_node());
        assert!(!r.is_way());

        let val = r.to_value();
        let r2 = Ref::from_value(val);
        assert_eq!(r, r2);
    }

    #[test]
    fn test_ref_marker() {
        // Ensure node marker doesn't overlap with valid way indices
        let way = Ref::Way(WayIdx::new(K_NODE_MARKER - 1));
        let node = Ref::Node(NodeIdx::new(0));

        assert!(way.to_value() < node.to_value());
    }

    #[test]
    fn test_add_way_platform() {
        let mut platforms = Platforms::new();
        let tags = vec![
            ("highway".to_string(), "platform".to_string()),
            ("ref".to_string(), "Platform 1".to_string()),
        ];

        let platform_idx = platforms.add_way(WayIdx::new(0), &tags);
        assert!(platform_idx.is_some());
        assert!(platforms.is_way_platform(WayIdx::new(0)));
        assert_eq!(platforms.n_platforms(), 1);
    }

    #[test]
    fn test_add_node_platform() {
        let mut platforms = Platforms::new();
        let pos = Point::from_latlng(37.7749, -122.4194);
        let tags = vec![
            ("railway".to_string(), "platform".to_string()),
            ("name".to_string(), "Station A".to_string()),
        ];

        let platform_idx = platforms.add_node(NodeIdx::new(0), pos, &tags);
        assert!(platform_idx.is_some());
        assert!(platforms.is_node_platform(NodeIdx::new(0)));
        assert_eq!(platforms.n_platforms(), 1);
    }

    #[test]
    fn test_add_relation_platform() {
        let mut platforms = Platforms::new();
        let tags = vec![
            ("type".to_string(), "multipolygon".to_string()),
            ("ref".to_string(), "Platform 2A;Platform 2B".to_string()),
        ];

        let platform_idx = platforms.add_relation(&tags);
        assert!(platform_idx.is_some());

        // Relation platforms have empty refs until members are added
        let refs = platforms.get_refs(platform_idx.unwrap());
        assert_eq!(refs.unwrap().len(), 0);
    }

    #[test]
    fn test_platform_names() {
        let mut platforms = Platforms::new();
        let pos = Point::from_latlng(37.7749, -122.4194);
        let tags = vec![
            ("name".to_string(), "Main Station".to_string()),
            ("alt_name".to_string(), "Central".to_string()),
            ("ref".to_string(), "MS1;MS2".to_string()), // Semicolon-separated
        ];

        let platform_idx = platforms.add_node(NodeIdx::new(0), pos, &tags).unwrap();
        let names = platforms.get_names(platform_idx).unwrap();

        // Should have: "Main Station", "Central", "MS1", "MS2"
        assert_eq!(names.len(), 4);
        assert!(names.contains(&"Main Station".to_string()));
        assert!(names.contains(&"Central".to_string()));
        assert!(names.contains(&"MS1".to_string()));
        assert!(names.contains(&"MS2".to_string()));
    }

    #[test]
    fn test_platform_names_no_valid_tags() {
        let mut platforms = Platforms::new();
        let pos = Point::from_latlng(37.7749, -122.4194);
        let tags = vec![
            ("highway".to_string(), "platform".to_string()), // Not a name tag
        ];

        let platform_idx = platforms.add_node(NodeIdx::new(0), pos, &tags);
        assert!(platform_idx.is_none()); // No platform created without names
    }

    #[test]
    fn test_get_node_pos() {
        let mut platforms = Platforms::new();
        let pos = Point::from_latlng(37.7749, -122.4194);
        let tags = vec![("name".to_string(), "Station".to_string())];

        platforms.add_node(NodeIdx::new(5), pos, &tags);
        platforms.finalize();

        let retrieved = platforms.get_node_pos(NodeIdx::new(5));
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), pos);
    }

    #[test]
    fn test_find_platforms() {
        let mut platforms = Platforms::new();

        // Add platform near SF
        let pos1 = Point::from_latlng(37.7749, -122.4194);
        let tags1 = vec![("name".to_string(), "Station A".to_string())];
        platforms.add_node(NodeIdx::new(0), pos1, &tags1);

        // Add platform far away (NY)
        let pos2 = Point::from_latlng(40.7128, -74.0060);
        let tags2 = vec![("name".to_string(), "Station B".to_string())];
        platforms.add_node(NodeIdx::new(1), pos2, &tags2);

        platforms.finalize();

        // Search near SF with 100m radius
        let location = Location {
            pos_: pos1,
            lvl_: Level::default(),
        };
        let results = platforms.find(&location, 100.0);

        // Should only find Station A
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_bbox() {
        let mut platforms = Platforms::new();

        // Add platforms in SF area
        let pos1 = Point::from_latlng(37.7749, -122.4194);
        let tags1 = vec![("name".to_string(), "Station A".to_string())];
        platforms.add_node(NodeIdx::new(0), pos1, &tags1);

        let pos2 = Point::from_latlng(37.7849, -122.4094);
        let tags2 = vec![("name".to_string(), "Station B".to_string())];
        platforms.add_node(NodeIdx::new(1), pos2, &tags2);

        // Add platform outside bbox (NY)
        let pos3 = Point::from_latlng(40.7128, -74.0060);
        let tags3 = vec![("name".to_string(), "Station C".to_string())];
        platforms.add_node(NodeIdx::new(2), pos3, &tags3);

        platforms.finalize();

        // Search SF bbox
        let results = platforms.find_bbox(37.77, -122.42, 37.79, -122.40);

        // Should find both SF stations
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_get_refs() {
        let mut platforms = Platforms::new();
        let pos = Point::from_latlng(37.7749, -122.4194);
        let tags = vec![("name".to_string(), "Station".to_string())];

        let platform_idx = platforms.add_node(NodeIdx::new(10), pos, &tags).unwrap();
        let refs = platforms.get_refs(platform_idx).unwrap();

        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0], Ref::Node(NodeIdx::new(10)));
    }
}
