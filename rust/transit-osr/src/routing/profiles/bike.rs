//! Translation of osr/include/osr/routing/profiles/bike.h
//!
//! Bike routing profiles with elevation cost and direction-aware routing.
//! Supports oneway streets (bikes can sometimes go against oneway) and
//! different cost strategies (Safe vs Fast).

use crate::routing::mode::Mode;
use crate::routing::parameters::BikeParameters;
use crate::types::{Cost, Direction, NodeIdx};
use crate::ways::{NodeProperties, WayProperties};
use crate::{ElevationStorage, Ways};
use std::hash::{Hash, Hasher};

/// Maximum cost value for infeasible routes
pub const INFEASIBLE: Cost = Cost::MAX;

/// Elevation cost constants (cost per meter of elevation)
pub const ELEVATION_NO_COST: u32 = 0;
pub const ELEVATION_LOW_COST: u32 = 570;
pub const ELEVATION_HIGH_COST: u32 = 3700;

/// Bike cost strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BikeCostStrategy {
    /// Prefer safe routes (bike lanes, car-free paths)
    Safe,
    /// Prefer fast routes (direct, higher speed)
    Fast,
}

/// Elevation cost level (kept for backward compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElevationCost {
    /// No elevation cost
    None,
    /// Low elevation penalty
    Low,
    /// High elevation penalty
    High,
}

impl From<ElevationCost> for u32 {
    fn from(cost: ElevationCost) -> Self {
        match cost {
            ElevationCost::None => ELEVATION_NO_COST,
            ElevationCost::Low => ELEVATION_LOW_COST,
            ElevationCost::High => ELEVATION_HIGH_COST,
        }
    }
}

/// Bike routing profile with elevation cost
///
/// Template parameters in C++: bike_costing, ElevationUpCost, ElevationExponentThousandth
/// In Rust, we use fields for runtime configuration.
#[derive(Debug, Clone)]
pub struct BikeProfile {
    pub params: BikeParameters,
    pub cost_strategy: BikeCostStrategy,
    pub elevation_cost: ElevationCost,
    pub elevation_up_cost: u32,
    pub elevation_exponent_thousandth: u32, // Exponent * 1000 (e.g., 2100 = 2.1)
}

impl BikeProfile {
    pub fn new(
        cost_strategy: BikeCostStrategy,
        elevation_up_cost: u32,
        elevation_exponent_thousandth: u32,
    ) -> Self {
        let elevation_cost = match elevation_up_cost {
            ELEVATION_NO_COST => ElevationCost::None,
            ELEVATION_LOW_COST => ElevationCost::Low,
            ELEVATION_HIGH_COST => ElevationCost::High,
            _ => ElevationCost::Low,
        };
        Self {
            params: BikeParameters::default(),
            cost_strategy,
            elevation_cost,
            elevation_up_cost,
            elevation_exponent_thousandth,
        }
    }

    /// Safe profile with low elevation cost
    pub fn safe() -> Self {
        Self::new(BikeCostStrategy::Safe, ELEVATION_LOW_COST, 2100)
    }

    /// Fast profile with no elevation cost
    pub fn fast() -> Self {
        Self::new(BikeCostStrategy::Fast, ELEVATION_NO_COST, 1000)
    }

    /// Maximum distance to match a location to the network (meters)
    pub fn max_match_distance(&self) -> u32 {
        Self::MAX_MATCH_DISTANCE
    }

    pub fn mode(&self) -> Mode {
        Mode::Bike
    }

    /// Maximum distance to match a location to the network (meters)
    pub const MAX_MATCH_DISTANCE: u32 = 100;
}

impl Default for BikeProfile {
    fn default() -> Self {
        Self::safe()
    }
}

/// Bike routing node with direction
///
/// Unlike foot routing, bike routing is direction-aware to handle oneway streets.
/// Bikes can sometimes go against oneway, so we need to track direction.
#[derive(Debug, Clone, Copy)]
pub struct BikeNode {
    pub n: NodeIdx,
    pub dir: Direction,
}

impl BikeNode {
    pub fn invalid() -> Self {
        Self {
            n: NodeIdx::INVALID,
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
            dir: self.dir.opposite(),
        }
    }
}

impl PartialEq for BikeNode {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.dir == other.dir
    }
}

impl Eq for BikeNode {}

impl Hash for BikeNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n.hash(state);
        self.dir.hash(state);
    }
}

/// Bike routing label for search state
#[derive(Debug, Clone)]
pub struct BikeLabel {
    pub n: NodeIdx,
    pub dir: Direction,
    pub cost: Cost,
}

impl BikeLabel {
    pub fn new(node: BikeNode, cost: Cost) -> Self {
        Self {
            n: node.n,
            dir: node.dir,
            cost,
        }
    }

    pub fn get_node(&self) -> BikeNode {
        BikeNode {
            n: self.n,
            dir: self.dir,
        }
    }

    pub fn cost(&self) -> Cost {
        self.cost
    }
}

/// Bike routing entry storing best paths for both directions
///
/// Since bikes can travel in both directions (even against some oneways),
/// we store costs and predecessors for forward and backward directions.
#[derive(Debug, Clone)]
pub struct BikeEntry {
    pub pred: [NodeIdx; 2],
    pub pred_dir: [Direction; 2],
    pub cost: [Cost; 2],
}

impl BikeEntry {
    pub fn new() -> Self {
        Self {
            pred: [NodeIdx::INVALID, NodeIdx::INVALID],
            pred_dir: [Direction::Forward, Direction::Backward],
            cost: [INFEASIBLE, INFEASIBLE],
        }
    }

    /// Get direction index (0 = Forward, 1 = Backward)
    fn get_index(node: BikeNode) -> usize {
        match node.dir {
            Direction::Forward => 0,
            Direction::Backward => 1,
        }
    }

    pub fn pred(&self, node: BikeNode) -> Option<BikeNode> {
        let idx = Self::get_index(node);
        if self.pred[idx] == NodeIdx::INVALID {
            None
        } else {
            Some(BikeNode {
                n: self.pred[idx],
                dir: self.pred_dir[idx],
            })
        }
    }

    pub fn cost(&self, node: BikeNode) -> Cost {
        self.cost[Self::get_index(node)]
    }

    /// Update entry if new cost is better
    pub fn update(&mut self, node: BikeNode, new_cost: Cost, pred: BikeNode) -> bool {
        let idx = Self::get_index(node);
        if new_cost < self.cost[idx] {
            self.cost[idx] = new_cost;
            self.pred[idx] = pred.n;
            self.pred_dir[idx] = pred.dir;
            true
        } else {
            false
        }
    }

    /// Get node from index
    pub fn get_node(n: NodeIdx, index: usize) -> BikeNode {
        BikeNode {
            n,
            dir: if index == 0 {
                Direction::Forward
            } else {
                Direction::Backward
            },
        }
    }
}

impl Default for BikeEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate way cost for bike routing
///
/// Considers:
/// - Bike accessibility (bike lanes, cycleways, shared roads)
/// - Oneway restrictions (bikes can sometimes go against oneway)
/// - Cost strategy (Safe: penalize big streets, bonus for car-free; Fast: direct)
/// - Distance and speed
pub fn way_cost(
    params: &BikeParameters,
    props: &WayProperties,
    dir: Direction,
    cost_strategy: BikeCostStrategy,
    dist: u16,
) -> Cost {
    // Check bike accessibility and oneway
    if !props.is_bike_accessible || (dir == Direction::Backward && props.is_oneway_bike) {
        return INFEASIBLE;
    }

    // Calculate base speed with strategy modifiers
    let mut speed = params.speed_meters_per_second;

    if cost_strategy == BikeCostStrategy::Safe {
        // Safe strategy: avoid big streets, prefer car-free
        if props.is_big_street {
            speed -= 0.7;
        }
        if props.motor_vehicle_no {
            speed += 0.5;
        }
    }
    // Fast strategy: no speed modifiers, just direct routing

    (dist as f32 / speed).round() as Cost
}

/// Calculate node cost for bike routing
pub fn node_cost(props: &NodeProperties) -> Cost {
    if !props.is_bike_accessible {
        INFEASIBLE
    } else {
        0
    }
}

/// Heuristic for A* search (optimistic estimate)
pub fn heuristic(params: &BikeParameters, dist: f64) -> f64 {
    dist / (params.speed_meters_per_second as f64 + 0.5)
}

/// Calculate elevation cost for a segment
///
/// Matches C++ implementation exactly:
/// - If ElevationUpCost > 0 and dist > 0:
///   - If exponent > 1000: ElevationUpCost * pow(elevation.up / dist, exponent/1000.0)
///   - Else: ElevationUpCost * elevation.up / dist
/// - Else: 0
///
/// # Arguments
/// * `elevation_up_cost` - Cost per meter of elevation gain
/// * `elevation_exponent_thousandth` - Exponent * 1000 (e.g., 2100 = 2.1)
/// * `elevation` - Elevation gain/loss for this segment
/// * `dist` - Distance of the segment in meters
pub fn calculate_elevation_cost(
    elevation_up_cost: u32,
    elevation_exponent_thousandth: u32,
    elevation: &crate::elevation_storage::Elevation,
    dist: u16,
) -> Cost {
    if elevation_up_cost > 0 && dist > 0 {
        let up = elevation.up as f64;
        let dist_f = dist as f64;

        if elevation_exponent_thousandth > 1000 {
            // Apply exponential penalty for steeper gradients
            let exponent = elevation_exponent_thousandth as f64 / 1000.0;
            let ratio = up / dist_f;
            (elevation_up_cost as f64 * ratio.powf(exponent)).round() as Cost
        } else {
            // Linear penalty
            ((elevation_up_cost as f64 * up) / dist_f).round() as Cost
        }
    } else {
        0
    }
}

// ============================================================================
// Bidirectional Search Support
// ============================================================================

use crate::routing::bidirectional_profile::{BidirectionalEntry, BidirectionalLabel, BidirectionalProfile};

/// Label for bidirectional bike routing
#[derive(Debug, Clone, Copy)]
pub struct BidirectionalBikeLabel {
    pub n: NodeIdx,
    pub dir: Direction,
    pub cost: Cost,
}

impl BidirectionalLabel for BidirectionalBikeLabel {
    type Node = BikeNode;

    fn new(node: Self::Node, cost: Cost) -> Self {
        Self {
            n: node.n,
            dir: node.dir,
            cost,
        }
    }

    fn get_node(&self) -> Self::Node {
        BikeNode {
            n: self.n,
            dir: self.dir,
        }
    }

    fn cost(&self) -> Cost {
        self.cost
    }
}

/// Entry for bidirectional bike routing (two directional states per node)
#[derive(Debug, Clone)]
pub struct BidirectionalBikeEntry {
    // Index 0 = Forward, Index 1 = Backward
    pred: [NodeIdx; 2],
    pred_dir: [Direction; 2],
    cost: [Cost; 2],
}

impl Default for BidirectionalBikeEntry {
    fn default() -> Self {
        Self {
            pred: [NodeIdx::INVALID, NodeIdx::INVALID],
            pred_dir: [Direction::Forward, Direction::Forward],
            cost: [Cost::MAX, Cost::MAX],
        }
    }
}

impl BidirectionalEntry for BidirectionalBikeEntry {
    type Node = BikeNode;

    fn pred(&self, node: Self::Node) -> Option<Self::Node> {
        let idx = match node.dir {
            Direction::Forward => 0,
            Direction::Backward => 1,
        };
        if self.pred[idx] == NodeIdx::INVALID {
            None
        } else {
            Some(BikeNode {
                n: self.pred[idx],
                dir: self.pred_dir[idx],
            })
        }
    }

    fn cost(&self, node: Self::Node) -> Cost {
        let idx = match node.dir {
            Direction::Forward => 0,
            Direction::Backward => 1,
        };
        self.cost[idx]
    }

    fn update(&mut self, node: Self::Node, cost: Cost, pred: Self::Node) -> bool {
        let idx = match node.dir {
            Direction::Forward => 0,
            Direction::Backward => 1,
        };
        if cost < self.cost[idx] {
            self.cost[idx] = cost;
            self.pred[idx] = pred.n;
            self.pred_dir[idx] = pred.dir;
            true
        } else {
            false
        }
    }
}

impl BidirectionalProfile for BikeProfile {
    type Node = BikeNode;
    type Key = NodeIdx; // Bike uses simplified key (just node index)
    type Label = BidirectionalBikeLabel;
    type Entry = BidirectionalBikeEntry;
    type Parameters = BikeProfile;

    fn get_key(node: &Self::Node) -> Self::Key {
        node.n
    }

    fn get_node_idx(node: &Self::Node) -> NodeIdx {
        node.n
    }

    fn heuristic(params: &Self::Parameters, distance_meters: f64) -> Cost {
        let speed = (params.params.speed_meters_per_second + 0.5) as f64;
        (distance_meters / speed).round() as Cost
    }

    fn adjacent<F>(
        params: &Self::Parameters,
        ways: &Ways,
        elevations: Option<&ElevationStorage>,
        node: Self::Node,
        search_dir: Direction,
        mut callback: F,
    ) where
        F: FnMut(Self::Node, Cost),
    {
        for (&way, &pos_in_way) in ways.get_node_ways(node.n).iter()
            .zip(ways.get_node_in_way_idx(node.n))
        {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let way_props = ways.get_way_properties(way);

            if way_nodes.is_empty() || way_props.is_none() {
                continue;
            }

            let props = way_props.unwrap();

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

                    let edge_cost = way_cost(&params.params, props, travel_dir, params.cost_strategy, dist);

                    if edge_cost != INFEASIBLE {
                        // BUG-01 fix: check TARGET node cost
                        let t_cost = ways.get_node_properties(next_node).map(node_cost).unwrap_or(0);
                        if t_cost != INFEASIBLE {
                            let elevation = elevations.map(|e| {
                                let mut el = e.get_elevation(way, i as u16);
                                if travel_dir == Direction::Backward { el = el.swapped(); }
                                el
                            }).unwrap_or_default();

                            let elev_cost = calculate_elevation_cost(
                                params.elevation_up_cost,
                                params.elevation_exponent_thousandth,
                                &elevation,
                                dist as u16,
                            );

                            let total_cost = edge_cost.saturating_add(t_cost).saturating_add(elev_cost);
                            if total_cost < INFEASIBLE {
                                callback(BikeNode { n: next_node, dir: travel_dir }, total_cost);
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

                    let edge_cost = way_cost(&params.params, props, travel_dir, params.cost_strategy, dist);

                    if edge_cost != INFEASIBLE {
                        // BUG-01 fix: check TARGET node cost
                        let t_cost = ways.get_node_properties(prev_node).map(node_cost).unwrap_or(0);
                        if t_cost != INFEASIBLE {
                            let elevation = elevations.map(|e| {
                                let mut el = e.get_elevation(way, (i - 1) as u16);
                                if travel_dir == Direction::Backward { el = el.swapped(); }
                                el
                            }).unwrap_or_default();

                            let elev_cost = calculate_elevation_cost(
                                params.elevation_up_cost,
                                params.elevation_exponent_thousandth,
                                &elevation,
                                dist as u16,
                            );

                            let total_cost = edge_cost.saturating_add(t_cost).saturating_add(elev_cost);
                            if total_cost < INFEASIBLE {
                                callback(BikeNode { n: prev_node, dir: travel_dir }, total_cost);
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
        let flipped = BikeNode {
            n: node.n,
            dir: node.dir.opposite(),
        };

        let f = fwd_entry.cost(node);
        let b = bwd_entry.cost(flipped);

        if f != INFEASIBLE && b != INFEASIBLE {
            Some((node, flipped, f.saturating_add(b)))
        } else {
            // Also check same directed node (e.g. they meet on a directed edge)
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
    fn test_bike_profile_creation() {
        let profile = BikeProfile::default();
        assert_eq!(profile.cost_strategy, BikeCostStrategy::Safe);
        assert_eq!(profile.elevation_up_cost, ELEVATION_LOW_COST);
        assert_eq!(profile.mode(), Mode::Bike);
    }

    #[test]
    fn test_safe_profile() {
        let profile = BikeProfile::safe();
        assert_eq!(profile.cost_strategy, BikeCostStrategy::Safe);
        assert_eq!(profile.elevation_up_cost, ELEVATION_LOW_COST);
    }

    #[test]
    fn test_fast_profile() {
        let profile = BikeProfile::fast();
        assert_eq!(profile.cost_strategy, BikeCostStrategy::Fast);
        assert_eq!(profile.elevation_up_cost, ELEVATION_NO_COST);
    }

    #[test]
    fn test_max_match_distance() {
        assert_eq!(BikeProfile::MAX_MATCH_DISTANCE, 100);
    }

    #[test]
    fn test_bike_node_creation() {
        let node = BikeNode {
            n: NodeIdx::from(42),
            dir: Direction::Forward,
        };
        assert_eq!(node.n, NodeIdx::from(42));
        assert_eq!(node.dir, Direction::Forward);
    }

    #[test]
    fn test_bike_node_invalid() {
        let node = BikeNode::invalid();
        assert_eq!(node.n, NodeIdx::INVALID);
    }

    #[test]
    fn test_bike_node_equality() {
        let n1 = BikeNode {
            n: NodeIdx::from(10),
            dir: Direction::Forward,
        };
        let n2 = BikeNode {
            n: NodeIdx::from(10),
            dir: Direction::Forward,
        };
        let n3 = BikeNode {
            n: NodeIdx::from(10),
            dir: Direction::Backward,
        };
        assert_eq!(n1, n2);
        assert_ne!(n1, n3); // Different direction
    }

    #[test]
    fn test_bike_node_reverse() {
        let node = BikeNode {
            n: NodeIdx::from(5),
            dir: Direction::Forward,
        };
        let reversed = node.reverse();
        assert_eq!(reversed.n, NodeIdx::from(5));
        assert_eq!(reversed.dir, Direction::Backward);
    }

    #[test]
    fn test_bike_label_creation() {
        let node = BikeNode {
            n: NodeIdx::from(20),
            dir: Direction::Backward,
        };
        let label = BikeLabel::new(node, 100);
        assert_eq!(label.n, NodeIdx::from(20));
        assert_eq!(label.dir, Direction::Backward);
        assert_eq!(label.cost, 100);
    }

    #[test]
    fn test_bike_entry_creation() {
        let entry = BikeEntry::new();
        assert_eq!(entry.pred[0], NodeIdx::INVALID);
        assert_eq!(entry.pred[1], NodeIdx::INVALID);
        assert_eq!(entry.cost[0], INFEASIBLE);
        assert_eq!(entry.cost[1], INFEASIBLE);
    }

    #[test]
    fn test_bike_entry_update() {
        let mut entry = BikeEntry::new();
        let node = BikeNode {
            n: NodeIdx::from(10),
            dir: Direction::Forward,
        };
        let pred = BikeNode {
            n: NodeIdx::from(5),
            dir: Direction::Forward,
        };

        // First update should succeed
        assert!(entry.update(node, 100, pred));
        assert_eq!(entry.cost(node), 100);
        assert_eq!(entry.pred(node).unwrap().n, NodeIdx::from(5));

        // Better cost should update
        assert!(entry.update(node, 50, pred));
        assert_eq!(entry.cost(node), 50);

        // Worse cost should not update
        assert!(!entry.update(node, 80, pred));
        assert_eq!(entry.cost(node), 50);
    }

    #[test]
    fn test_bike_entry_directions() {
        let mut entry = BikeEntry::new();
        let forward_node = BikeNode {
            n: NodeIdx::from(10),
            dir: Direction::Forward,
        };
        let backward_node = BikeNode {
            n: NodeIdx::from(10),
            dir: Direction::Backward,
        };
        let pred = BikeNode {
            n: NodeIdx::from(5),
            dir: Direction::Forward,
        };

        // Update forward direction
        entry.update(forward_node, 100, pred);
        assert_eq!(entry.cost(forward_node), 100);
        assert_eq!(entry.cost(backward_node), INFEASIBLE);

        // Update backward direction
        entry.update(backward_node, 200, pred);
        assert_eq!(entry.cost(forward_node), 100);
        assert_eq!(entry.cost(backward_node), 200);
    }

    #[test]
    fn test_way_cost_bike_accessible() {
        let params = BikeParameters::default();
        let props = WayProperties {
            is_bike_accessible: true,
            is_oneway_bike: false,
            is_big_street: false,
            motor_vehicle_no: false,
            ..Default::default()
        };
        let cost = way_cost(
            &params,
            &props,
            Direction::Forward,
            BikeCostStrategy::Safe,
            100,
        );
        assert!(cost < INFEASIBLE);
        assert!(cost > 0);
    }

    #[test]
    fn test_way_cost_not_accessible() {
        let params = BikeParameters::default();
        let props = WayProperties {
            is_bike_accessible: false,
            ..Default::default()
        };
        let cost = way_cost(
            &params,
            &props,
            Direction::Forward,
            BikeCostStrategy::Safe,
            100,
        );
        assert_eq!(cost, INFEASIBLE);
    }

    #[test]
    fn test_way_cost_oneway_forward() {
        let params = BikeParameters::default();
        let props = WayProperties {
            is_bike_accessible: true,
            is_oneway_bike: true,
            ..Default::default()
        };
        // Forward direction should work
        let cost_fwd = way_cost(
            &params,
            &props,
            Direction::Forward,
            BikeCostStrategy::Fast,
            100,
        );
        assert!(cost_fwd < INFEASIBLE);

        // Backward direction should be infeasible
        let cost_bwd = way_cost(
            &params,
            &props,
            Direction::Backward,
            BikeCostStrategy::Fast,
            100,
        );
        assert_eq!(cost_bwd, INFEASIBLE);
    }

    #[test]
    fn test_way_cost_safe_big_street_penalty() {
        let params = BikeParameters::default();
        let normal = WayProperties {
            is_bike_accessible: true,
            is_big_street: false,
            ..Default::default()
        };
        let big_street = WayProperties {
            is_bike_accessible: true,
            is_big_street: true,
            ..Default::default()
        };

        let cost_normal = way_cost(
            &params,
            &normal,
            Direction::Forward,
            BikeCostStrategy::Safe,
            100,
        );
        let cost_big = way_cost(
            &params,
            &big_street,
            Direction::Forward,
            BikeCostStrategy::Safe,
            100,
        );

        // Big street should be slower (higher cost) in Safe mode
        assert!(cost_big > cost_normal);
    }

    #[test]
    fn test_way_cost_safe_car_free_bonus() {
        let params = BikeParameters::default();
        let normal = WayProperties {
            is_bike_accessible: true,
            motor_vehicle_no: false,
            ..Default::default()
        };
        let car_free = WayProperties {
            is_bike_accessible: true,
            motor_vehicle_no: true,
            ..Default::default()
        };

        let cost_normal = way_cost(
            &params,
            &normal,
            Direction::Forward,
            BikeCostStrategy::Safe,
            100,
        );
        let cost_car_free = way_cost(
            &params,
            &car_free,
            Direction::Forward,
            BikeCostStrategy::Safe,
            100,
        );

        // Car-free should be faster (lower cost) in Safe mode
        assert!(cost_car_free < cost_normal);
    }

    #[test]
    fn test_way_cost_fast_no_modifiers() {
        let params = BikeParameters::default();
        let big_street = WayProperties {
            is_bike_accessible: true,
            is_big_street: true,
            motor_vehicle_no: false,
            ..Default::default()
        };
        let car_free = WayProperties {
            is_bike_accessible: true,
            is_big_street: false,
            motor_vehicle_no: true,
            ..Default::default()
        };

        let cost_big = way_cost(
            &params,
            &big_street,
            Direction::Forward,
            BikeCostStrategy::Fast,
            100,
        );
        let cost_car_free = way_cost(
            &params,
            &car_free,
            Direction::Forward,
            BikeCostStrategy::Fast,
            100,
        );

        // Fast mode should not differentiate (same distance = same cost)
        assert_eq!(cost_big, cost_car_free);
    }

    #[test]
    fn test_node_cost_accessible() {
        let props = NodeProperties {
            is_bike_accessible: true,
            ..Default::default()
        };
        assert_eq!(node_cost(&props), 0);
    }

    #[test]
    fn test_node_cost_inaccessible() {
        let props = NodeProperties {
            is_bike_accessible: false,
            ..Default::default()
        };
        assert_eq!(node_cost(&props), INFEASIBLE);
    }

    #[test]
    fn test_heuristic() {
        let params = BikeParameters::default();
        let dist = 1000.0; // 1km
        let h = heuristic(&params, dist);
        // Should be optimistic (faster than actual)
        let actual_time = dist / params.speed_meters_per_second as f64;
        assert!(h < actual_time);
    }

    #[test]
    fn test_elevation_cost_calculation_zero() {
        use crate::elevation_storage::Elevation;

        // No elevation cost
        let elev = Elevation::new(0, 0);
        let cost = calculate_elevation_cost(ELEVATION_NO_COST, 1000, &elev, 100);
        assert_eq!(cost, 0);

        // Elevation gain but cost is zero
        let elev = Elevation::new(10, 0);
        let cost = calculate_elevation_cost(ELEVATION_NO_COST, 1000, &elev, 100);
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_elevation_cost_calculation_linear() {
        use crate::elevation_storage::Elevation;

        // Linear cost (exponent = 1000 = 1.0)
        // 10m elevation gain over 100m distance
        // Cost = 570 * 10 / 100 = 57
        let elev = Elevation::new(10, 0);
        let cost = calculate_elevation_cost(ELEVATION_LOW_COST, 1000, &elev, 100);
        assert_eq!(cost, 57);
    }

    #[test]
    fn test_elevation_cost_calculation_exponential() {
        use crate::elevation_storage::Elevation;

        // Exponential cost (exponent = 2100 = 2.1)
        // 10m elevation gain over 100m distance
        // ratio = 10/100 = 0.1
        // Cost = 570 * (0.1)^2.1 ≈ 3.6 (rounds to 4)
        let elev = Elevation::new(10, 0);
        let cost = calculate_elevation_cost(ELEVATION_LOW_COST, 2100, &elev, 100);
        assert!(cost >= 3 && cost <= 5, "Expected ~4, got {}", cost);

        // Higher gradient should cost more with exponential
        // 20m over 100m: ratio = 0.2, cost = 570 * (0.2)^2.1 ≈ 19
        let elev_steep = Elevation::new(20, 0);
        let cost_steep = calculate_elevation_cost(ELEVATION_LOW_COST, 2100, &elev_steep, 100);
        assert!(cost_steep > cost, "Steeper gradient should cost more");
    }

    #[test]
    fn test_elevation_cost_calculation_high() {
        use crate::elevation_storage::Elevation;

        // High elevation cost constant
        // 10m over 100m with linear
        // Cost = 3700 * 10 / 100 = 370
        let elev = Elevation::new(10, 0);
        let cost = calculate_elevation_cost(ELEVATION_HIGH_COST, 1000, &elev, 100);
        assert_eq!(cost, 370);
    }
}
