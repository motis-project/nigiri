//! Translation of osr/include/osr/routing/sharing_data.h
//!
//! Shared mobility data structures (bike/car sharing).
//! Matches C++ implementation exactly.

use ahash::AHashMap;

use bitvec::prelude::*;

use crate::routing::additional_edge::AdditionalEdge;
use crate::types::NodeIdx;
use crate::Point;

/// Sharing data for routing with shared mobility
///
/// Contains information about bike/car sharing stations and allowed nodes.
/// Uses lifetime-bound references to avoid unnecessary copying.
///
/// # C++ Equivalent
/// ```cpp
/// struct sharing_data {
///   bitvec<node_idx_t> const* start_allowed_;
///   bitvec<node_idx_t> const* end_allowed_;
///   bitvec<node_idx_t> const* through_allowed_;
///   node_idx_t::value_t additional_node_offset_{};
///   std::vector<geo::latlng> const& additional_node_coordinates_;
///   hash_map<node_idx_t, std::vector<additional_edge>> const& additional_edges_;
/// };
/// ```
#[derive(Debug)]
pub struct SharingData<'a> {
    /// Nodes where starting a journey is allowed (nullptr = all allowed)
    pub start_allowed: Option<&'a BitVec>,

    /// Nodes where ending a journey is allowed (nullptr = all allowed)
    pub end_allowed: Option<&'a BitVec>,

    /// Nodes that can be passed through (nullptr = all allowed)
    pub through_allowed: Option<&'a BitVec>,

    /// Offset for additional node indices
    pub additional_node_offset: u32,

    /// Coordinates of additional nodes (sharing stations)
    pub additional_node_coordinates: &'a [Point],

    /// Additional edges connecting stations to the graph
    pub additional_edges: &'a AHashMap<NodeIdx, Vec<AdditionalEdge>>,
}

impl<'a> SharingData<'a> {
    /// Create new sharing data with references to externally owned data
    pub fn new(
        start_allowed: Option<&'a BitVec>,
        end_allowed: Option<&'a BitVec>,
        through_allowed: Option<&'a BitVec>,
        additional_node_offset: u32,
        additional_node_coordinates: &'a [Point],
        additional_edges: &'a AHashMap<NodeIdx, Vec<AdditionalEdge>>,
    ) -> Self {
        Self {
            start_allowed,
            end_allowed,
            through_allowed,
            additional_node_offset,
            additional_node_coordinates,
            additional_edges,
        }
    }

    /// Get coordinates of an additional node (sharing station)
    ///
    /// # C++ Equivalent
    /// ```cpp
    /// geo::latlng get_additional_node_coordinates(node_idx_t const n) const {
    ///   return additional_node_coordinates_.at(to_idx(n) - additional_node_offset_);
    /// }
    /// ```
    pub fn get_additional_node_coordinates(&self, n: NodeIdx) -> Point {
        let idx = (n.value() - self.additional_node_offset) as usize;
        self.additional_node_coordinates[idx]
    }
}

/// Check if a node is allowed for the given purpose
///
/// Returns true if bitvec is None (nullptr in C++) or if the bit is set.
///
/// # C++ Equivalent
/// ```cpp
/// inline bool is_allowed(bitvec<node_idx_t> const* b, node_idx_t const n) {
///   return b == nullptr || b->test(n);
/// }
/// ```
#[inline]
pub fn is_allowed(b: Option<&BitVec>, n: NodeIdx) -> bool {
    b.map_or(true, |bits| {
        bits.get(n.value() as usize).map_or(false, |bit| *bit)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharing_data_creation() {
        let coords = vec![Point::from_latlng(0.0, 0.0)];
        let edges = AHashMap::new();

        let data = SharingData::new(None, None, None, 0, &coords, &edges);

        assert!(data.start_allowed.is_none());
        assert!(data.end_allowed.is_none());
        assert!(data.through_allowed.is_none());
        assert_eq!(data.additional_node_offset, 0);
        assert_eq!(data.additional_node_coordinates.len(), 1);
        assert!(data.additional_edges.is_empty());
    }

    #[test]
    fn test_get_additional_node_coordinates() {
        let coords = vec![
            Point::from_latlng(10.0, 20.0),
            Point::from_latlng(30.0, 40.0),
            Point::from_latlng(50.0, 60.0),
        ];
        let edges = AHashMap::new();

        let data = SharingData::new(None, None, None, 100, &coords, &edges);

        // Node 100 -> index 0
        let pt = data.get_additional_node_coordinates(NodeIdx::new(100));
        assert_eq!(pt.lat(), 10.0);
        assert_eq!(pt.lng(), 20.0);

        // Node 101 -> index 1
        let pt = data.get_additional_node_coordinates(NodeIdx::new(101));
        assert_eq!(pt.lat(), 30.0);
        assert_eq!(pt.lng(), 40.0);

        // Node 102 -> index 2
        let pt = data.get_additional_node_coordinates(NodeIdx::new(102));
        assert_eq!(pt.lat(), 50.0);
        assert_eq!(pt.lng(), 60.0);
    }

    #[test]
    fn test_is_allowed_none() {
        // None (nullptr in C++) means all nodes are allowed
        assert!(is_allowed(None, NodeIdx::new(0)));
        assert!(is_allowed(None, NodeIdx::new(123)));
        assert!(is_allowed(None, NodeIdx::new(9999)));
    }

    #[test]
    fn test_is_allowed_some() {
        let mut bits = BitVec::new();
        bits.resize(1000, false);
        bits.set(123, true);
        bits.set(456, true);
        bits.set(789, true);

        assert!(is_allowed(Some(&bits), NodeIdx::new(123)));
        assert!(is_allowed(Some(&bits), NodeIdx::new(456)));
        assert!(is_allowed(Some(&bits), NodeIdx::new(789)));
        assert!(!is_allowed(Some(&bits), NodeIdx::new(0)));
        assert!(!is_allowed(Some(&bits), NodeIdx::new(100)));
        assert!(!is_allowed(Some(&bits), NodeIdx::new(999)));
    }

    #[test]
    fn test_is_allowed_out_of_bounds() {
        let mut bits = BitVec::new();
        bits.resize(100, false);
        bits.set(50, true);

        // Out of bounds should return false
        assert!(!is_allowed(Some(&bits), NodeIdx::new(100)));
        assert!(!is_allowed(Some(&bits), NodeIdx::new(999)));
    }

    #[test]
    fn test_with_bitvecs() {
        let mut start_bits = BitVec::new();
        start_bits.resize(1000, false);
        start_bits.set(10, true);
        start_bits.set(20, true);

        let mut end_bits = BitVec::new();
        end_bits.resize(1000, false);
        end_bits.set(30, true);
        end_bits.set(40, true);

        let mut through_bits = BitVec::new();
        through_bits.resize(1000, false);
        through_bits.set(50, true);

        let coords = vec![Point::from_latlng(0.0, 0.0)];
        let edges = AHashMap::new();

        let data = SharingData::new(
            Some(&start_bits),
            Some(&end_bits),
            Some(&through_bits),
            0,
            &coords,
            &edges,
        );

        // Verify start_allowed
        assert!(is_allowed(data.start_allowed, NodeIdx::new(10)));
        assert!(is_allowed(data.start_allowed, NodeIdx::new(20)));
        assert!(!is_allowed(data.start_allowed, NodeIdx::new(30)));

        // Verify end_allowed
        assert!(is_allowed(data.end_allowed, NodeIdx::new(30)));
        assert!(is_allowed(data.end_allowed, NodeIdx::new(40)));
        assert!(!is_allowed(data.end_allowed, NodeIdx::new(10)));

        // Verify through_allowed
        assert!(is_allowed(data.through_allowed, NodeIdx::new(50)));
        assert!(!is_allowed(data.through_allowed, NodeIdx::new(10)));
    }

    #[test]
    fn test_additional_edges() {
        let coords = vec![Point::from_latlng(0.0, 0.0)];
        let mut edges = AHashMap::new();
        edges.insert(
            NodeIdx::new(100),
            vec![
                AdditionalEdge::new(NodeIdx::new(200), 150),
                AdditionalEdge::new(NodeIdx::new(300), 250),
            ],
        );

        let data = SharingData::new(None, None, None, 0, &coords, &edges);

        assert_eq!(data.additional_edges.len(), 1);
        assert_eq!(
            data.additional_edges.get(&NodeIdx::new(100)).unwrap().len(),
            2
        );
    }
}
