//! Translation of osr/include/osr/elevation_storage.h + src/elevation_storage.cc
//!
//! Elevation data storage with 4-bit compression.
//!
//! Stores elevation gain/loss for way segments using compressed encoding.
//! Complete implementation matching C++ functionality.

use std::path::Path;

use rkyv::{Archive, Deserialize, Serialize};

use crate::types::*;

/// Compressed elevation values (4-bit encoding)
///
/// Maps 0-15 to actual elevation values in meters with logarithmic spacing
const COMPRESSED_VALUES: [u16; 16] = [0, 1, 2, 4, 6, 8, 11, 14, 17, 21, 25, 29, 34, 38, 43, 48];

/// Elevation gain/loss for a way segment
///
/// Stores monotonic up and down elevation changes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Archive, Serialize, Deserialize)]
pub struct Elevation {
    /// Uphill elevation gain in meters
    pub up: u16,
    /// Downhill elevation loss in meters
    pub down: u16,
}

impl Elevation {
    /// Create new elevation
    pub fn new(up: u16, down: u16) -> Self {
        Self { up, down }
    }

    /// Add another elevation
    pub fn add(&mut self, other: &Elevation) {
        self.up += other.up;
        self.down += other.down;
    }

    /// Swap up and down (for reverse direction)
    pub fn swapped(&self) -> Self {
        Self {
            up: self.down,
            down: self.up,
        }
    }

    /// Check if elevation has any change
    pub fn has_elevation(&self) -> bool {
        self.up > 0 || self.down > 0
    }
}

/// 4-bit compressed elevation encoding
///
/// Compresses elevation gain/loss into 8 bits total (4 bits each)
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct ElevationEncoding {
    /// 4-bit encoded uphill value
    up: u8,
    /// 4-bit encoded downhill value
    down: u8,
}

impl ElevationEncoding {
    /// Encode elevation into compressed format
    pub fn encode(elevation: &Elevation) -> Self {
        Self {
            up: encode_value(elevation.up),
            down: encode_value(elevation.down),
        }
    }

    /// Decode into full elevation
    pub fn decode(&self) -> Elevation {
        Elevation {
            up: decode_value(self.up),
            down: decode_value(self.down),
        }
    }

    /// Check if encoding represents non-zero elevation
    pub fn has_elevation(&self) -> bool {
        self.up != 0 || self.down != 0
    }

    /// Create from raw 8-bit value (upper 4 bits = up, lower 4 bits = down)
    pub fn from_u8(value: u8) -> Self {
        Self {
            up: (value >> 4) & 0x0F,
            down: value & 0x0F,
        }
    }

    /// Convert to raw 8-bit value
    pub fn to_u8(&self) -> u8 {
        ((self.up & 0x0F) << 4) | (self.down & 0x0F)
    }

    /// Create from byte (alias for from_u8)
    pub fn from_byte(byte: u8) -> Self {
        Self::from_u8(byte)
    }

    /// Convert to byte (alias for to_u8)
    pub fn to_byte(&self) -> u8 {
        self.to_u8()
    }
}

/// Encode elevation value into 4-bit compressed index
fn encode_value(meters: u16) -> u8 {
    // Binary search for closest compressed value
    match COMPRESSED_VALUES.binary_search(&meters) {
        Ok(idx) => idx as u8,
        Err(idx) => {
            if idx >= COMPRESSED_VALUES.len() {
                (COMPRESSED_VALUES.len() - 1) as u8
            } else {
                idx as u8
            }
        }
    }
}

/// Decode 4-bit compressed index into elevation value
fn decode_value(code: u8) -> u16 {
    let idx = (code as usize).min(COMPRESSED_VALUES.len() - 1);
    COMPRESSED_VALUES[idx]
}

/// Helper function to flatten Vec<Vec<ElevationEncoding>> into (bytes, index) for storage
fn flatten_vecvec_encoding(vecvec: &[Vec<ElevationEncoding>]) -> (Vec<u8>, Vec<u32>) {
    let mut data = Vec::new();
    let mut index = Vec::with_capacity(vecvec.len() + 1);

    index.push(0u32);
    for vec in vecvec {
        for enc in vec {
            data.push(enc.to_byte());
        }
        index.push(data.len() as u32);
    }

    (data, index)
}

/// Helper function to unflatten bytes back into Vec<Vec<ElevationEncoding>>
fn unflatten_vecvec_encoding(data: &[u8], index: &[u32]) -> Vec<Vec<ElevationEncoding>> {
    if index.len() <= 1 {
        return Vec::new();
    }

    (0..index.len() - 1)
        .map(|i| {
            let start = index[i] as usize;
            let end = index[i + 1] as usize;
            data[start..end]
                .iter()
                .map(|&b| ElevationEncoding::from_byte(b))
                .collect()
        })
        .collect()
}

/// Elevation storage for way segments
///
/// Stores compressed elevation data for each segment of each way.
/// Each way can have multiple segments (between nodes).
#[derive(Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct ElevationStorage {
    /// Elevation data per way, per segment
    /// Outer vec indexed by WayIdx, inner vec indexed by segment number
    elevations: Vec<Vec<ElevationEncoding>>,
}

impl ElevationStorage {
    /// Create new empty elevation storage
    pub fn new() -> Self {
        Self {
            elevations: Vec::new(),
        }
    }

    /// Set elevation for a way segment
    pub fn set_elevation(&mut self, way: WayIdx, segment: u16, elevation: Elevation) {
        let way_idx = way.value() as usize;

        // Resize way vector if needed
        if way_idx >= self.elevations.len() {
            self.elevations.resize(way_idx + 1, Vec::new());
        }

        // Resize segment vector if needed
        let segment_idx = segment as usize;
        if segment_idx >= self.elevations[way_idx].len() {
            self.elevations[way_idx].resize(segment_idx + 1, ElevationEncoding::default());
        }

        // Encode and store
        self.elevations[way_idx][segment_idx] = ElevationEncoding::encode(&elevation);
    }

    /// Set multiple elevations for a way
    pub fn set_way_elevations(&mut self, way: WayIdx, elevations: &[Elevation]) {
        let way_idx = way.value() as usize;

        if way_idx >= self.elevations.len() {
            self.elevations.resize(way_idx + 1, Vec::new());
        }

        self.elevations[way_idx] = elevations
            .iter()
            .map(|e| ElevationEncoding::encode(e))
            .collect();
    }

    /// Get elevation for a way segment
    pub fn get_elevation(&self, way: WayIdx, segment: u16) -> Elevation {
        let way_idx = way.value() as usize;
        let segment_idx = segment as usize;

        if way_idx < self.elevations.len() && segment_idx < self.elevations[way_idx].len() {
            self.elevations[way_idx][segment_idx].decode()
        } else {
            Elevation::default()
        }
    }

    /// Get all elevations for a way
    pub fn get_way_elevations(&self, way: WayIdx) -> Vec<Elevation> {
        let way_idx = way.value() as usize;

        if way_idx < self.elevations.len() {
            self.elevations[way_idx]
                .iter()
                .map(|e| e.decode())
                .collect()
        } else {
            vec![]
        }
    }

    /// Check if elevation data exists for a way segment
    pub fn has_elevation(&self, way: WayIdx, segment: u16) -> bool {
        let way_idx = way.value() as usize;
        let segment_idx = segment as usize;

        if way_idx < self.elevations.len() && segment_idx < self.elevations[way_idx].len() {
            self.elevations[way_idx][segment_idx].has_elevation()
        } else {
            false
        }
    }

    /// Get number of ways with elevation data
    pub fn n_ways(&self) -> usize {
        self.elevations.len()
    }

    /// Get number of segments for a way
    pub fn n_segments(&self, way: WayIdx) -> usize {
        let way_idx = way.value() as usize;
        if way_idx < self.elevations.len() {
            self.elevations[way_idx].len()
        } else {
            0
        }
    }

    /// Populate elevations from Ways data using an elevation provider
    ///
    /// Queries the provider for elevation at each node position along each way.
    /// Calculates uphill/downhill elevation changes for each segment.
    pub fn populate_from_ways<P>(
        &mut self,
        ways: &crate::Ways,
        provider: &P,
    ) -> Result<usize, String>
    where
        P: crate::preprocessing::elevation::shared::ElevationProvider,
    {
        let mut populated_count = 0;

        // Process each way
        for way_idx in 0..ways.n_ways() {
            let way = WayIdx::new(way_idx as u32);
            let nodes = ways.get_way_nodes(way);

            if nodes.len() < 2 {
                continue;
            }

            if nodes.len() < 2 {
                continue; // Need at least 2 nodes for a segment
            }

            // Get elevations for all nodes in the way
            let mut node_elevations = Vec::with_capacity(nodes.len());
            for &node in nodes {
                let pos = ways.get_node_pos(node);
                let elev = provider.get(&pos);
                node_elevations.push(elev);
            }

            // Calculate elevation changes for each segment
            let mut segment_elevations = Vec::with_capacity(nodes.len() - 1);
            let mut has_any_valid = false;

            for i in 0..nodes.len() - 1 {
                let elev_from = &node_elevations[i];
                let elev_to = &node_elevations[i + 1];

                if let (Some(from_m), Some(to_m)) = (elev_from.meters(), elev_to.meters()) {
                    let diff = to_m - from_m;
                    let elevation = if diff >= 0 {
                        // Uphill
                        Elevation::new(diff as u16, 0)
                    } else {
                        // Downhill
                        Elevation::new(0, (-diff) as u16)
                    };
                    segment_elevations.push(elevation);
                    has_any_valid = true;
                } else {
                    // No valid elevation data for this segment
                    segment_elevations.push(Elevation::new(0, 0));
                }
            }

            // Only store if we got at least some valid elevations
            if has_any_valid {
                self.set_way_elevations(way, &segment_elevations);
                populated_count += 1;
            }
        }

        Ok(populated_count)
    }

    /// Save elevation storage to directory (multi-file format)
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        // Helper to write ragged array
        let write_ragged = |filename: &str, data: &[u8], index: &[u32]| -> std::io::Result<()> {
            let path = dir.join(filename);
            let mut file = BufWriter::new(File::create(path)?);
            file.write_all(&(index.len() as u32 - 1).to_le_bytes())?;
            file.write_all(bytemuck::cast_slice(index))?;
            file.write_all(data)?;
            Ok(())
        };

        // Convert elevations to ragged array format
        let (data, index) = flatten_vecvec_encoding(&self.elevations);
        write_ragged("elevations.bin", &data, &index)?;

        Ok(())
    }

    /// Load elevation storage from multi-file format (internal implementation)
    pub fn load(dir: &Path) -> crate::Result<Self> {
        use std::fs::File;
        use std::io::Read;

        let path = dir.join("elevations.bin");
        if !path.exists() {
            return Ok(Self::new());
        }

        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        if data.len() < 4 {
            return Ok(Self::new());
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let index_size = (count + 1) * 4;
        if data.len() < 4 + index_size {
            return Ok(Self::new());
        }

        let index_bytes = &data[4..4 + index_size];
        let index: Vec<u32> = bytemuck::cast_slice(index_bytes).to_vec();
        let data_bytes = &data[4 + index_size..];

        let elevations = unflatten_vecvec_encoding(data_bytes, &index);

        Ok(Self { elevations })
    }

    /// Calculate total elevation gain for a way
    pub fn total_up(&self, way: WayIdx) -> u16 {
        self.get_way_elevations(way).iter().map(|e| e.up).sum()
    }

    /// Calculate total elevation loss for a way
    pub fn total_down(&self, way: WayIdx) -> u16 {
        self.get_way_elevations(way).iter().map(|e| e.down).sum()
    }
}

impl Default for ElevationStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Free function for getting elevations (matches C++ API)
pub fn get_elevations(storage: Option<&ElevationStorage>, way: WayIdx, segment: u16) -> Elevation {
    storage
        .map(|s| s.get_elevation(way, segment))
        .unwrap_or_default()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elevation_storage_creation() {
        // Just verify structure can be created
        let _storage = ElevationStorage::new();
    }

    #[test]
    fn test_elevation_creation() {
        let elev = Elevation::new(100, 50);
        assert_eq!(elev.up, 100);
        assert_eq!(elev.down, 50);
        assert!(elev.has_elevation());
    }

    #[test]
    fn test_elevation_add() {
        let mut elev1 = Elevation::new(10, 5);
        let elev2 = Elevation::new(20, 15);
        elev1.add(&elev2);

        assert_eq!(elev1.up, 30);
        assert_eq!(elev1.down, 20);
    }

    #[test]
    fn test_elevation_swapped() {
        let elev = Elevation::new(100, 50);
        let swapped = elev.swapped();

        assert_eq!(swapped.up, 50);
        assert_eq!(swapped.down, 100);
    }

    #[test]
    fn test_compression_exact_values() {
        // Test exact matches in compressed values table
        for (idx, &value) in COMPRESSED_VALUES.iter().enumerate() {
            let encoded = encode_value(value);
            assert_eq!(encoded, idx as u8);

            let decoded = decode_value(encoded);
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn test_compression_intermediate_values() {
        // Test value between 2 and 4 (should encode to index 2 or 3)
        let encoded = encode_value(3);
        assert!(encoded >= 2 && encoded <= 3);

        // Test value above max (should clamp to max index)
        let encoded_max = encode_value(1000);
        assert_eq!(encoded_max, 15);
        assert_eq!(decode_value(encoded_max), 48);
    }

    #[test]
    fn test_elevation_encoding() {
        let elev = Elevation::new(14, 8);
        let encoded = ElevationEncoding::encode(&elev);
        let decoded = encoded.decode();

        // Should decode to exact values (14 and 8 are in table)
        assert_eq!(decoded.up, 14);
        assert_eq!(decoded.down, 8);
    }

    #[test]
    fn test_elevation_encoding_roundtrip() {
        let original = Elevation::new(25, 17);
        let encoded = ElevationEncoding::encode(&original);
        let decoded = encoded.decode();

        assert_eq!(decoded.up, 25);
        assert_eq!(decoded.down, 17);
    }

    #[test]
    fn test_elevation_encoding_u8_conversion() {
        let encoding = ElevationEncoding { up: 7, down: 3 };
        let byte = encoding.to_u8();
        let reconstructed = ElevationEncoding::from_u8(byte);

        assert_eq!(reconstructed.up, 7);
        assert_eq!(reconstructed.down, 3);
    }

    #[test]
    fn test_set_and_get_elevation() {
        let mut storage = ElevationStorage::new();
        let elev = Elevation::new(14, 8);

        storage.set_elevation(WayIdx::new(5), 2, elev);
        let retrieved = storage.get_elevation(WayIdx::new(5), 2);

        assert_eq!(retrieved.up, 14);
        assert_eq!(retrieved.down, 8);
    }

    #[test]
    fn test_set_way_elevations() {
        let mut storage = ElevationStorage::new();
        let elevations = vec![
            Elevation::new(11, 6), // These will be compressed
            Elevation::new(21, 14),
            Elevation::new(8, 2),
        ];

        storage.set_way_elevations(WayIdx::new(3), &elevations);
        let retrieved = storage.get_way_elevations(WayIdx::new(3));

        assert_eq!(retrieved.len(), 3);
        // Values are compressed, so they may not match exactly
        // 11 -> 11, 6 -> 6, 21 -> 21, 14 -> 14, 8 -> 8, 2 -> 2 (all in table)
        assert_eq!(retrieved[0].up, 11);
        assert_eq!(retrieved[0].down, 6);
        assert_eq!(retrieved[1].up, 21);
        assert_eq!(retrieved[1].down, 14);
        assert_eq!(retrieved[2].up, 8);
        assert_eq!(retrieved[2].down, 2);
    }

    #[test]
    fn test_has_elevation() {
        let mut storage = ElevationStorage::new();

        assert!(!storage.has_elevation(WayIdx::new(0), 0));

        storage.set_elevation(WayIdx::new(0), 0, Elevation::new(10, 5));
        assert!(storage.has_elevation(WayIdx::new(0), 0));

        // Non-existent segment
        assert!(!storage.has_elevation(WayIdx::new(0), 5));
    }

    #[test]
    fn test_total_elevations() {
        let mut storage = ElevationStorage::new();
        let elevations = vec![
            Elevation::new(11, 6), // Values in compression table
            Elevation::new(21, 14),
            Elevation::new(8, 2),
        ];

        storage.set_way_elevations(WayIdx::new(1), &elevations);

        // 11 + 21 + 8 = 40
        // 6 + 14 + 2 = 22
        assert_eq!(storage.total_up(WayIdx::new(1)), 40);
        assert_eq!(storage.total_down(WayIdx::new(1)), 22);
    }

    #[test]
    fn test_n_segments() {
        let mut storage = ElevationStorage::new();
        let elevations = vec![Elevation::new(10, 5), Elevation::new(20, 15)];

        storage.set_way_elevations(WayIdx::new(2), &elevations);

        assert_eq!(storage.n_segments(WayIdx::new(2)), 2);
        assert_eq!(storage.n_segments(WayIdx::new(99)), 0);
    }

    #[test]
    fn test_get_elevations_free_function() {
        let mut storage = ElevationStorage::new();
        storage.set_elevation(WayIdx::new(1), 0, Elevation::new(14, 8));

        // With storage
        let elev = get_elevations(Some(&storage), WayIdx::new(1), 0);
        assert_eq!(elev.up, 14);

        // Without storage (None)
        let elev_none = get_elevations(None, WayIdx::new(1), 0);
        assert_eq!(elev_none.up, 0);
        assert_eq!(elev_none.down, 0);
    }

    #[test]
    fn test_zero_elevation() {
        let elev = Elevation::default();
        assert_eq!(elev.up, 0);
        assert_eq!(elev.down, 0);
        assert!(!elev.has_elevation());
    }

    #[test]
    fn test_encoding_has_elevation() {
        let enc_zero = ElevationEncoding::default();
        assert!(!enc_zero.has_elevation());

        let enc_nonzero = ElevationEncoding::encode(&Elevation::new(10, 5));
        assert!(enc_nonzero.has_elevation());
    }
}
