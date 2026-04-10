//! Translation of osr/include/osr/routing/additional_edge.h
//!
//! Additional edges for routing (e.g., start/end connections).

use crate::types::{Distance, NodeIdx};

/// An additional edge connecting to the routing graph
///
/// Used to connect arbitrary locations to nearby nodes in the routing graph.
/// For example, connecting a start/end location to the nearest street node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdditionalEdge {
    /// Target node in the routing graph
    pub node: NodeIdx,
    /// Distance to the node (in meters)
    pub distance: Distance,
}

impl AdditionalEdge {
    pub fn new(node: NodeIdx, distance: Distance) -> Self {
        Self { node, distance }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additional_edge_creation() {
        let edge = AdditionalEdge::new(NodeIdx::new(123), 45);
        assert_eq!(edge.node, NodeIdx::new(123));
        assert_eq!(edge.distance, 45);
    }

    #[test]
    fn test_additional_edge_equality() {
        let edge1 = AdditionalEdge::new(NodeIdx::new(100), 50);
        let edge2 = AdditionalEdge::new(NodeIdx::new(100), 50);
        let edge3 = AdditionalEdge::new(NodeIdx::new(101), 50);

        assert_eq!(edge1, edge2);
        assert_ne!(edge1, edge3);
    }
}
