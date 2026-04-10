//! Translation of osr/include/osr/routing/algorithms.h
//!
//! Available routing algorithms.

use std::fmt;

/// Routing algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RoutingAlgorithm {
    /// Dijkstra's algorithm (single-source shortest path)
    Dijkstra = 0,
    /// Bidirectional A* search
    AStarBi = 1,
}

impl RoutingAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            RoutingAlgorithm::Dijkstra => "dijkstra",
            RoutingAlgorithm::AStarBi => "astar_bi",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "dijkstra" => Some(RoutingAlgorithm::Dijkstra),
            "astar_bi" | "astar" => Some(RoutingAlgorithm::AStarBi),
            _ => None,
        }
    }
}

impl fmt::Display for RoutingAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for RoutingAlgorithm {
    fn default() -> Self {
        RoutingAlgorithm::Dijkstra
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_to_str() {
        assert_eq!(RoutingAlgorithm::Dijkstra.as_str(), "dijkstra");
        assert_eq!(RoutingAlgorithm::AStarBi.as_str(), "astar_bi");
    }

    #[test]
    fn test_algorithm_from_str() {
        assert_eq!(
            RoutingAlgorithm::from_str("dijkstra"),
            Some(RoutingAlgorithm::Dijkstra)
        );
        assert_eq!(
            RoutingAlgorithm::from_str("astar_bi"),
            Some(RoutingAlgorithm::AStarBi)
        );
        assert_eq!(
            RoutingAlgorithm::from_str("astar"),
            Some(RoutingAlgorithm::AStarBi)
        );
        assert_eq!(RoutingAlgorithm::from_str("invalid"), None);
    }

    #[test]
    fn test_default() {
        assert_eq!(RoutingAlgorithm::default(), RoutingAlgorithm::Dijkstra);
    }
}
