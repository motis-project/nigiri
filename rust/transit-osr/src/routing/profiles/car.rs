//! Translation of osr/include/osr/routing/profiles/car.h
//!
//! Car routing profile with strict oneway enforcement, speed limits,
//! destination restrictions, and turn restriction support.

use crate::routing::mode::Mode;
use crate::routing::parameters::CarParameters;
use crate::types::{Cost, Direction, NodeIdx, WayPos};
use crate::ways::{NodeProperties, WayProperties};
use crate::{ElevationStorage, Ways};
use std::hash::{Hash, Hasher};

/// Maximum cost value for infeasible routes
pub const INFEASIBLE: Cost = Cost::MAX;

/// U-turn penalty in seconds
pub const UTURN_PENALTY: Cost = 120;

/// Car routing profile
#[derive(Debug, Clone)]
pub struct CarProfile {
    pub params: CarParameters,
}

impl CarProfile {
    pub fn new() -> Self {
        Self {
            params: CarParameters::default(),
        }
    }

    pub fn mode(&self) -> Mode {
        Mode::Car
    }

    /// Maximum distance to match a location to the network (meters)
    pub fn max_match_distance(&self) -> u32 {
        Self::MAX_MATCH_DISTANCE
    }

    /// Maximum distance to match a location to the network (meters)
    pub const MAX_MATCH_DISTANCE: u32 = 200;
}

impl Default for CarProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Car routing node with way position and direction
///
/// Unlike bike (just direction) or foot (level), car routing tracks:
/// - Node index
/// - Which way at that node (way_pos)
/// - Direction on that way
///
/// This is needed for turn restrictions and tracking the specific way
/// being traversed at intersections with multiple ways.
#[derive(Debug, Clone, Copy)]
pub struct CarNode {
    pub n: NodeIdx,
    pub way: WayPos,
    pub dir: Direction,
}

impl CarNode {
    pub fn invalid() -> Self {
        Self {
            n: NodeIdx::INVALID,
            way: 0,
            dir: Direction::Forward,
        }
    }

    pub fn get_node(&self) -> NodeIdx {
        self.n
    }

    pub fn get_key(&self) -> NodeIdx {
        self.n
    }

    pub fn reverse(&self) -> Self {
        Self {
            n: self.n,
            way: self.way,
            dir: self.dir.opposite(),
        }
    }
}

impl PartialEq for CarNode {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.way == other.way && self.dir == other.dir
    }
}

impl Eq for CarNode {}

impl Hash for CarNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n.hash(state);
        self.way.hash(state);
        self.dir.hash(state);
    }
}

/// Car routing label for search state
#[derive(Debug, Clone)]
pub struct CarLabel {
    pub n: NodeIdx,
    pub way: WayPos,
    pub dir: Direction,
    pub cost: Cost,
}

impl CarLabel {
    pub fn new(node: CarNode, cost: Cost) -> Self {
        Self {
            n: node.n,
            way: node.way,
            dir: node.dir,
            cost,
        }
    }

    pub fn get_node(&self) -> CarNode {
        CarNode {
            n: self.n,
            way: self.way,
            dir: self.dir,
        }
    }

    pub fn cost(&self) -> Cost {
        self.cost
    }
}

/// Car routing entry storing best paths for up to 16 ways × 2 directions
///
/// Car routing needs to track which specific way is being used at a node
/// (for turn restrictions). Each node can have multiple ways, and we need
/// to store costs/predecessors for each way-direction combination.
///
/// Storage: 32 slots = 16 ways (MAX_WAYS) × 2 directions (forward/backward)
#[derive(Debug, Clone)]
pub struct CarEntry {
    pub pred: Vec<NodeIdx>,
    pub pred_way: Vec<WayPos>,
    pub pred_dir: Vec<bool>, // false=Forward, true=Backward
    pub cost: Vec<Cost>,
}

impl CarEntry {
    /// Maximum ways per node (matching C++ kMaxWays)
    pub const MAX_WAYS: usize = 16;
    /// Total slots: MAX_WAYS × 2 directions
    pub const N: usize = Self::MAX_WAYS * 2;

    pub fn new() -> Self {
        Self {
            pred: vec![NodeIdx::INVALID; Self::N],
            pred_way: vec![0; Self::N],
            pred_dir: vec![false; Self::N],
            cost: vec![INFEASIBLE; Self::N],
        }
    }

    /// Get linear index from node (way_pos + direction offset)
    fn get_index(node: CarNode) -> usize {
        let dir_offset = match node.dir {
            Direction::Forward => 0,
            Direction::Backward => Self::MAX_WAYS,
        };
        dir_offset + node.way as usize
    }

    /// Convert direction to bool (Forward=false, Backward=true)
    fn dir_to_bool(dir: Direction) -> bool {
        matches!(dir, Direction::Backward)
    }

    /// Convert bool to direction (false=Forward, true=Backward)
    fn bool_to_dir(b: bool) -> Direction {
        if b {
            Direction::Backward
        } else {
            Direction::Forward
        }
    }

    pub fn pred(&self, node: CarNode) -> Option<CarNode> {
        let idx = Self::get_index(node);
        if idx >= Self::N || self.pred[idx] == NodeIdx::INVALID {
            None
        } else {
            Some(CarNode {
                n: self.pred[idx],
                way: self.pred_way[idx],
                dir: Self::bool_to_dir(self.pred_dir[idx]),
            })
        }
    }

    pub fn cost(&self, node: CarNode) -> Cost {
        let idx = Self::get_index(node);
        if idx >= Self::N {
            INFEASIBLE
        } else {
            self.cost[idx]
        }
    }

    /// Update entry if new cost is better
    pub fn update(&mut self, node: CarNode, new_cost: Cost, pred: CarNode) -> bool {
        let idx = Self::get_index(node);
        if idx >= Self::N {
            return false;
        }

        if new_cost < self.cost[idx] {
            self.cost[idx] = new_cost;
            self.pred[idx] = pred.n;
            self.pred_way[idx] = pred.way;
            self.pred_dir[idx] = Self::dir_to_bool(pred.dir);
            true
        } else {
            false
        }
    }

    /// Get node from index
    pub fn get_node(n: NodeIdx, index: usize) -> CarNode {
        CarNode {
            n,
            way: (index % Self::MAX_WAYS) as u8,
            dir: Self::bool_to_dir((index / Self::MAX_WAYS) != 0),
        }
    }
}

impl Default for CarEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate way cost for car routing
///
/// Considers:
/// - Car accessibility
/// - Strict oneway enforcement (no exceptions like bikes)
/// - Speed limit for time calculation
/// - Destination restrictions (access=destination): 5× time + 120s penalty
pub fn way_cost(props: &WayProperties, dir: Direction, dist: u16) -> Cost {
    // Check car accessibility and strict oneway
    if !props.is_car_accessible || (dir == Direction::Backward && props.is_oneway_car) {
        return INFEASIBLE;
    }

    // Calculate base time cost from speed limit
    let speed_m_per_s = props.max_speed_m_per_s();
    if speed_m_per_s == 0 {
        return INFEASIBLE;
    }

    // BUG-12 fix: round instead of truncate to avoid 0-cost short segments
    let time_cost = (dist as f32 / speed_m_per_s as f32).round() as Cost;

    // Apply destination penalties (5× time multiplier + 120s fixed penalty)
    if props.is_destination {
        time_cost * 5 + 120
    } else {
        time_cost
    }
}

/// Calculate node cost for car routing
pub fn node_cost(props: &NodeProperties) -> Cost {
    if !props.is_car_accessible {
        INFEASIBLE
    } else {
        0
    }
}

/// Heuristic for A* search (optimistic estimate)
/// Uses highway speed: 130 km/h = 36.11 m/s
pub fn heuristic(dist: f64) -> f64 {
    dist / (130.0 / 3.6) // 130 km/h converted to m/s
}

// ============================================================================
// Bidirectional Search Support
// ============================================================================

use crate::routing::bidirectional_profile::{BidirectionalEntry, BidirectionalLabel, BidirectionalProfile};

/// Label for bidirectional car routing
#[derive(Debug, Clone, Copy)]
pub struct BidirectionalCarLabel {
    pub n: NodeIdx,
    pub way: WayPos,
    pub dir: Direction,
    pub cost: Cost,
}

impl BidirectionalLabel for BidirectionalCarLabel {
    type Node = CarNode;

    fn new(node: Self::Node, cost: Cost) -> Self {
        Self {
            n: node.n,
            way: node.way,
            dir: node.dir,
            cost,
        }
    }

    fn get_node(&self) -> Self::Node {
        CarNode {
            n: self.n,
            way: self.way,
            dir: self.dir,
        }
    }

    fn cost(&self) -> Cost {
        self.cost
    }
}

/// Entry for bidirectional car routing (32 states per node: 16 ways × 2 directions)
#[derive(Debug, Clone)]
pub struct BidirectionalCarEntry {
    pred: [NodeIdx; 32],
    pred_way: [WayPos; 32],
    pred_dir: [bool; 32],  // false=Forward, true=Backward
    cost: [Cost; 32],
}

impl Default for BidirectionalCarEntry {
    fn default() -> Self {
        Self {
            pred: [NodeIdx::INVALID; 32],
            pred_way: [0; 32],
            pred_dir: [false; 32],
            cost: [Cost::MAX; 32],
        }
    }
}

impl BidirectionalEntry for BidirectionalCarEntry {
    type Node = CarNode;

    fn pred(&self, node: Self::Node) -> Option<Self::Node> {
        let idx = Self::get_index(node);
        if self.pred[idx] == NodeIdx::INVALID {
            None
        } else {
            Some(CarNode {
                n: self.pred[idx],
                way: self.pred_way[idx],
                dir: if self.pred_dir[idx] {
                    Direction::Backward
                } else {
                    Direction::Forward
                },
            })
        }
    }

    fn cost(&self, node: Self::Node) -> Cost {
        let idx = Self::get_index(node);
        self.cost[idx]
    }

    fn update(&mut self, node: Self::Node, cost: Cost, pred: Self::Node) -> bool {
        let idx = Self::get_index(node);
        if cost < self.cost[idx] {
            self.cost[idx] = cost;
            self.pred[idx] = pred.n;
            self.pred_way[idx] = pred.way;
            self.pred_dir[idx] = pred.dir == Direction::Backward;
            true
        } else {
            false
        }
    }
}

impl BidirectionalCarEntry {
    /// Get linear index from node (way_pos + direction offset)
    fn get_index(node: CarNode) -> usize {
        let dir_offset = if node.dir == Direction::Backward {
            16
        } else {
            0
        };
        dir_offset + (node.way as usize)
    }
}

impl BidirectionalProfile for CarProfile {
    type Node = CarNode;
    type Key = NodeIdx; // Car uses simplified key (just node index)
    type Label = BidirectionalCarLabel;
    type Entry = BidirectionalCarEntry;
    type Parameters = CarProfile;  // Or create separate Parameters struct

    fn get_key(node: &Self::Node) -> Self::Key {
        node.n
    }

    fn get_node_idx(node: &Self::Node) -> NodeIdx {
        node.n
    }

    fn heuristic(_params: &Self::Parameters, distance_meters: f64) -> Cost {
        // BUG-05 fix: use 130 km/h (max motorway speed) for admissible lower bound
        const SPEED_MPS: f64 = 130.0 / 3.6; // 130 km/h = ~36.11 m/s
        (distance_meters / SPEED_MPS).round() as Cost
    }

    fn adjacent<F>(
        _params: &Self::Parameters,
        ways: &Ways,
        _elevations: Option<&ElevationStorage>,
        node: Self::Node,
        search_dir: Direction,
        mut callback: F,
    ) where
        F: FnMut(Self::Node, Cost),
    {
        let node_ways = ways.get_node_ways(node.n);
        let node_way_positions = ways.get_node_in_way_idx(node.n);
        for (outgoing_way_pos, (&way, &pos_in_way)) in node_ways.iter()
            .zip(node_way_positions)
            .enumerate()
        {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let way_props = ways.get_way_properties(way);

            if way_nodes.is_empty() || way_props.is_none() {
                continue;
            }

            let props = way_props.unwrap();

            // BUG-02 fix: check turn restrictions before expanding this outgoing way
            if ways.is_restricted(node.n, node.way, outgoing_way_pos as WayPos) {
                continue;
            }

            let i = pos_in_way as usize;
            if i >= way_nodes.len() {
                continue;
            }

            // Forward neighbor along way (i -> i+1)
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = if search_dir == Direction::Forward {
                        Direction::Forward
                    } else {
                        Direction::Backward
                    };

                    let edge_cost = way_cost(props, travel_dir, dist);
                    if edge_cost != INFEASIBLE {
                        // BUG-01 fix: check TARGET node cost
                        let t_cost = ways.get_node_properties(next_node).map(node_cost).unwrap_or(0);
                        if t_cost != INFEASIBLE {
                            // BUG-03 fix: target's way_pos = position of this way in TARGET node's way list
                            let target_way_pos = ways.get_way_pos(next_node, way);
                            // U-turn check: same way, opposite direction
                            let is_uturn = outgoing_way_pos as WayPos == node.way
                                && travel_dir != node.dir;
                            let total_cost = edge_cost
                                .saturating_add(t_cost)
                                .saturating_add(if is_uturn { UTURN_PENALTY } else { 0 });
                            if total_cost < INFEASIBLE {
                                callback(CarNode { n: next_node, way: target_way_pos, dir: travel_dir }, total_cost);
                            }
                        }
                    }
                }
            }

            // Backward neighbor along way (i -> i-1)
            if i > 0 {
                let prev_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = if search_dir == Direction::Forward {
                        Direction::Backward
                    } else {
                        Direction::Forward
                    };

                    let edge_cost = way_cost(props, travel_dir, dist);
                    if edge_cost != INFEASIBLE {
                        // BUG-01 fix: check TARGET node cost
                        let t_cost = ways.get_node_properties(prev_node).map(node_cost).unwrap_or(0);
                        if t_cost != INFEASIBLE {
                            // BUG-03 fix: target's way_pos = position of this way in TARGET node's way list
                            let target_way_pos = ways.get_way_pos(prev_node, way);
                            let is_uturn = outgoing_way_pos as WayPos == node.way
                                && travel_dir != node.dir;
                            let total_cost = edge_cost
                                .saturating_add(t_cost)
                                .saturating_add(if is_uturn { UTURN_PENALTY } else { 0 });
                            if total_cost < INFEASIBLE {
                                callback(CarNode { n: prev_node, way: target_way_pos, dir: travel_dir }, total_cost);
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_meeting_point(
        fwd_entry: &Self::Entry,
        bwd_entry: &Self::Entry,
        node: Self::Node,
    ) -> Option<(Self::Node, Self::Node, Cost)> {
        let flipped = CarNode {
            n: node.n,
            way: node.way,
            dir: node.dir.opposite(),
        };

        let f = fwd_entry.cost(node);
        let b = bwd_entry.cost(flipped);

        if f != INFEASIBLE && b != INFEASIBLE {
            Some((node, flipped, f.saturating_add(b)))
        } else {
            // Also check same directed node
            let b2 = bwd_entry.cost(node);
            if f != INFEASIBLE && b2 != INFEASIBLE {
                Some((node, node, f.saturating_add(b2)))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_car_profile_creation() {
        let profile = CarProfile::default();
        assert_eq!(profile.mode(), Mode::Car);
    }

    #[test]
    fn test_max_match_distance() {
        assert_eq!(CarProfile::MAX_MATCH_DISTANCE, 200);
        let profile = CarProfile::default();
        assert_eq!(profile.max_match_distance(), 200);
    }

    #[test]
    fn test_car_node_creation() {
        let node = CarNode {
            n: NodeIdx::from(42),
            way: 3,
            dir: Direction::Forward,
        };
        assert_eq!(node.n, NodeIdx::from(42));
        assert_eq!(node.way, 3);
        assert_eq!(node.dir, Direction::Forward);
    }

    #[test]
    fn test_car_node_invalid() {
        let node = CarNode::invalid();
        assert_eq!(node.n, NodeIdx::INVALID);
        assert_eq!(node.way, 0);
    }

    #[test]
    fn test_car_node_equality() {
        let n1 = CarNode {
            n: NodeIdx::from(10),
            way: 2,
            dir: Direction::Forward,
        };
        let n2 = CarNode {
            n: NodeIdx::from(10),
            way: 2,
            dir: Direction::Forward,
        };
        let n3 = CarNode {
            n: NodeIdx::from(10),
            way: 2,
            dir: Direction::Backward,
        };
        let n4 = CarNode {
            n: NodeIdx::from(10),
            way: 3,
            dir: Direction::Forward,
        };

        assert_eq!(n1, n2);
        assert_ne!(n1, n3); // Different direction
        assert_ne!(n1, n4); // Different way
    }

    #[test]
    fn test_car_node_reverse() {
        let node = CarNode {
            n: NodeIdx::from(5),
            way: 1,
            dir: Direction::Forward,
        };
        let reversed = node.reverse();
        assert_eq!(reversed.n, NodeIdx::from(5));
        assert_eq!(reversed.way, 1);
        assert_eq!(reversed.dir, Direction::Backward);
    }

    #[test]
    fn test_car_label_creation() {
        let node = CarNode {
            n: NodeIdx::from(20),
            way: 5,
            dir: Direction::Backward,
        };
        let label = CarLabel::new(node, 150);
        assert_eq!(label.n, NodeIdx::from(20));
        assert_eq!(label.way, 5);
        assert_eq!(label.dir, Direction::Backward);
        assert_eq!(label.cost, 150);
    }

    #[test]
    fn test_car_entry_creation() {
        let entry = CarEntry::new();
        assert_eq!(entry.pred.len(), CarEntry::N);
        assert_eq!(entry.cost.len(), CarEntry::N);
        assert!(entry.pred.iter().all(|&p| p == NodeIdx::INVALID));
        assert!(entry.cost.iter().all(|&c| c == INFEASIBLE));
    }

    #[test]
    fn test_car_entry_index_calculation() {
        // Forward direction: index = way_pos
        let fwd_node = CarNode {
            n: NodeIdx::from(10),
            way: 5,
            dir: Direction::Forward,
        };
        assert_eq!(CarEntry::get_index(fwd_node), 5);

        // Backward direction: index = MAX_WAYS + way_pos
        let bwd_node = CarNode {
            n: NodeIdx::from(10),
            way: 5,
            dir: Direction::Backward,
        };
        assert_eq!(CarEntry::get_index(bwd_node), CarEntry::MAX_WAYS + 5);
    }

    #[test]
    fn test_car_entry_update() {
        let mut entry = CarEntry::new();
        let node = CarNode {
            n: NodeIdx::from(10),
            way: 3,
            dir: Direction::Forward,
        };
        let pred = CarNode {
            n: NodeIdx::from(5),
            way: 2,
            dir: Direction::Forward,
        };

        // First update should succeed
        assert!(entry.update(node, 100, pred));
        assert_eq!(entry.cost(node), 100);
        assert_eq!(entry.pred(node).unwrap().n, NodeIdx::from(5));
        assert_eq!(entry.pred(node).unwrap().way, 2);

        // Better cost should update
        assert!(entry.update(node, 50, pred));
        assert_eq!(entry.cost(node), 50);

        // Worse cost should not update
        assert!(!entry.update(node, 80, pred));
        assert_eq!(entry.cost(node), 50);
    }

    #[test]
    fn test_car_entry_multiple_ways() {
        let mut entry = CarEntry::new();
        let pred = CarNode {
            n: NodeIdx::from(1),
            way: 0,
            dir: Direction::Forward,
        };

        // Update different ways at same node
        for way_idx in 0..5 {
            let node = CarNode {
                n: NodeIdx::from(10),
                way: way_idx,
                dir: Direction::Forward,
            };
            entry.update(node, 100 + way_idx as Cost, pred);
        }

        // Verify each way has independent cost
        for way_idx in 0..5 {
            let node = CarNode {
                n: NodeIdx::from(10),
                way: way_idx,
                dir: Direction::Forward,
            };
            assert_eq!(entry.cost(node), 100 + way_idx as Cost);
        }
    }

    #[test]
    fn test_car_entry_directions() {
        let mut entry = CarEntry::new();
        let node = CarNode {
            n: NodeIdx::from(10),
            way: 2,
            dir: Direction::Forward,
        };
        let backward_node = CarNode {
            n: NodeIdx::from(10),
            way: 2,
            dir: Direction::Backward,
        };
        let pred = CarNode {
            n: NodeIdx::from(5),
            way: 1,
            dir: Direction::Forward,
        };

        // Update forward direction
        entry.update(node, 100, pred);
        assert_eq!(entry.cost(node), 100);
        assert_eq!(entry.cost(backward_node), INFEASIBLE);

        // Update backward direction
        entry.update(backward_node, 200, pred);
        assert_eq!(entry.cost(node), 100);
        assert_eq!(entry.cost(backward_node), 200);
    }

    #[test]
    fn test_way_cost_accessible() {
        let props = WayProperties {
            is_car_accessible: true,
            is_oneway_car: false,
            is_destination: false,
            ..Default::default()
        };
        let cost = way_cost(&props, Direction::Forward, 100);
        assert!(cost < INFEASIBLE);
        assert!(cost > 0);
    }

    #[test]
    fn test_way_cost_not_accessible() {
        let props = WayProperties {
            is_car_accessible: false,
            ..Default::default()
        };
        let cost = way_cost(&props, Direction::Forward, 100);
        assert_eq!(cost, INFEASIBLE);
    }

    #[test]
    fn test_way_cost_oneway_forward() {
        let props = WayProperties {
            is_car_accessible: true,
            is_oneway_car: true,
            ..Default::default()
        };

        // Forward should work
        let cost_fwd = way_cost(&props, Direction::Forward, 100);
        assert!(cost_fwd < INFEASIBLE);

        // Backward should be infeasible
        let cost_bwd = way_cost(&props, Direction::Backward, 100);
        assert_eq!(cost_bwd, INFEASIBLE);
    }

    #[test]
    fn test_way_cost_destination_penalty() {
        let normal = WayProperties {
            is_car_accessible: true,
            is_destination: false,
            ..Default::default()
        };
        let destination = WayProperties {
            is_car_accessible: true,
            is_destination: true,
            ..Default::default()
        };

        let cost_normal = way_cost(&normal, Direction::Forward, 100);
        let cost_dest = way_cost(&destination, Direction::Forward, 100);

        // Destination should have 5× time + 120s penalty
        assert!(cost_dest > cost_normal);
        assert!(cost_dest >= cost_normal * 5 + 120);
    }

    #[test]
    fn test_node_cost_accessible() {
        let props = NodeProperties {
            is_car_accessible: true,
            ..Default::default()
        };
        assert_eq!(node_cost(&props), 0);
    }

    #[test]
    fn test_node_cost_inaccessible() {
        let props = NodeProperties {
            is_car_accessible: false,
            ..Default::default()
        };
        assert_eq!(node_cost(&props), INFEASIBLE);
    }

    #[test]
    fn test_heuristic() {
        let dist = 1000.0; // 1km
        let h = heuristic(dist);
        // Should be optimistic (faster than actual at any legal speed)
        let actual_time = dist / (50.0 / 3.6); // 50 km/h
        assert!(h < actual_time);
    }

    #[test]
    fn test_uturn_penalty() {
        assert_eq!(UTURN_PENALTY, 120);
    }
}
