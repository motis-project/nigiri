//! Translation of osr/include/osr/routing/profiles/foot.h
//!
//! Foot and wheelchair routing profiles with complete multi-level support.
//! Includes elevator handling, level transitions, and tracking integration.
//!
//! # C++ Template Translation
//! ```cpp
//! template <bool IsWheelchair, typename Tracking = noop_tracking>
//! struct foot { ... };
//! ```
//! Rust uses trait generics instead of template parameters for flexibility.

use std::hash::{Hash, Hasher};

use crate::routing::mode::Mode;
use crate::routing::parameters::FootParameters;
use crate::routing::tracking::Tracking;
use crate::types::{Cost, Direction, Level, NodeIdx, WayIdx};
use crate::ways::{NodeProperties, WayProperties, Ways};
use crate::ElevationStorage;

/// Maximum routing cost (infeasible)
pub const INFEASIBLE: Cost = Cost::MAX;

/// Maximum distance to match a location to the network (meters)
pub const MAX_MATCH_DISTANCE: u32 = 100;

/// Foot routing profile with multi-level support
///
/// # Type Parameters
/// - `T`: Tracking implementation (NoopTracking, ElevatorTracking, etc.)
///
/// # C++ Equivalent
/// ```cpp
/// template <bool IsWheelchair, typename Tracking>
/// struct foot { ... };
/// ```
#[derive(Debug, Clone)]
pub struct FootProfile<T: Tracking = crate::routing::tracking::NoopTracking> {
    pub params: FootParameters,
    pub is_wheelchair: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tracking> FootProfile<T> {
    pub fn new(is_wheelchair: bool) -> Self {
        Self {
            params: if is_wheelchair {
                FootParameters::wheelchair()
            } else {
                FootParameters::default()
            },
            is_wheelchair,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn mode(&self) -> Mode {
        if self.is_wheelchair {
            Mode::Wheelchair
        } else {
            Mode::Foot
        }
    }

    /// Maximum distance to match a location to the network (meters)
    pub fn max_match_distance(&self) -> u32 {
        MAX_MATCH_DISTANCE
    }
}

impl<T: Tracking> Default for FootProfile<T> {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Routing node with level information
///
/// # C++ Equivalent
/// ```cpp
/// struct node {
///   node_idx_t n_;
///   level_t lvl_;
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FootNode {
    pub n: NodeIdx,
    pub lvl: Level,
}

impl FootNode {
    pub const fn invalid() -> Self {
        Self {
            n: NodeIdx::INVALID,
            lvl: Level::from_idx(Level::NO_LEVEL),
        }
    }

    pub const fn new(n: NodeIdx, lvl: Level) -> Self {
        Self { n, lvl }
    }

    pub const fn get_node(self) -> NodeIdx {
        self.n
    }

    pub const fn get_key(self) -> Self {
        self
    }

    pub fn get_mode(is_wheelchair: bool) -> Mode {
        if is_wheelchair {
            Mode::Wheelchair
        } else {
            Mode::Foot
        }
    }
}

impl PartialEq for FootNode {
    fn eq(&self, other: &Self) -> bool {
        let is_zero = |l: Level| l.to_idx() == Level::NO_LEVEL || l.to_float() == 0.0;
        self.n == other.n && (self.lvl == other.lvl || (is_zero(self.lvl) && is_zero(other.lvl)))
    }
}

impl Eq for FootNode {}

impl Hash for FootNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n.hash(state);
        let lvl = if self.lvl.to_idx() == Level::NO_LEVEL {
            Level::from_float(0.0)
        } else {
            self.lvl
        };
        lvl.to_idx().hash(state);
    }
}

/// Search label for Dijkstra/A*
///
/// # C++ Equivalent
/// ```cpp
/// struct label {
///   node_idx_t n_;
///   cost_t cost_;
///   level_t lvl_;
///   [[no_unique_address]] Tracking tracking_;
/// };
/// ```
#[derive(Debug, Clone)]
pub struct FootLabel<T: Tracking> {
    pub n: NodeIdx,
    pub cost: Cost,
    pub lvl: Level,
    pub tracking: T,
}

impl<T: Tracking> FootLabel<T> {
    pub fn new(node: FootNode, cost: Cost) -> Self {
        Self {
            n: node.n,
            cost,
            lvl: node.lvl,
            tracking: T::default(),
        }
    }

    pub const fn get_node(&self) -> FootNode {
        FootNode {
            n: self.n,
            lvl: self.lvl,
        }
    }

    pub const fn cost(&self) -> Cost {
        self.cost
    }

    /// Track edge traversal for path reconstruction
    pub fn track(&mut self, prev: &Self, ways: &Ways, way: WayIdx, node: NodeIdx) {
        self.tracking.track(&prev.tracking, ways, way, node, false);
    }
}

/// Dijkstra entry (best path to node)
///
/// # C++ Equivalent
/// ```cpp
/// struct entry {
///   node_idx_t pred_;
///   cost_t cost_;
///   level_t pred_lvl_;
///   [[no_unique_address]] Tracking tracking_;
/// };
/// ```
#[derive(Debug, Clone)]
pub struct FootEntry<T: Tracking> {
    pub pred: NodeIdx,
    pub cost: Cost,
    pub pred_lvl: Level,
    pub tracking: T,
}

impl<T: Tracking> FootEntry<T> {
    pub fn new() -> Self {
        Self {
            pred: NodeIdx::INVALID,
            cost: INFEASIBLE,
            pred_lvl: Level::from_idx(Level::NO_LEVEL),
            tracking: T::default(),
        }
    }

    pub fn pred(&self, _node: FootNode) -> Option<FootNode> {
        if self.pred == NodeIdx::INVALID {
            None
        } else {
            Some(FootNode::new(self.pred, self.pred_lvl))
        }
    }

    pub const fn cost(&self, _node: FootNode) -> Cost {
        self.cost
    }

    /// Update entry if new path is better
    pub fn update(
        &mut self,
        label: &FootLabel<T>,
        _node: FootNode,
        cost: Cost,
        pred: FootNode,
    ) -> bool {
        if cost < self.cost {
            self.tracking = label.tracking.clone();
            self.cost = cost;
            self.pred = pred.n;
            self.pred_lvl = pred.lvl;
            true
        } else {
            false
        }
    }
}

impl<T: Tracking> Default for FootEntry<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate way cost in seconds
///
/// # Arguments
/// * `dist` - Distance in meters (rounded to u16)
///
/// # C++ Equivalent
/// ```cpp
/// static constexpr cost_t way_cost(parameters const& params,
///                                  way_properties const e,
///                                  direction, std::uint16_t const dist);
/// ```
pub fn way_cost(
    params: &FootParameters,
    props: &WayProperties,
    is_wheelchair: bool,
    dist: u16,
) -> Cost {
    if (props.is_foot_accessible || (!props.is_sidewalk_separate && props.is_bike_accessible))
        && (!is_wheelchair || !props.is_steps)
    {
        // Base cost from distance and speed
        let mut speed = params.speed_meters_per_second;

        // Adjust speed for road characteristics
        if props.is_big_street {
            speed -= 0.2; // Slower on big streets (less pleasant)
        }
        if props.motor_vehicle_no {
            speed += 0.1; // Faster on car-free streets
        }

        let time_cost = (dist as f32 / speed).round() as Cost;

        // Add penalty for non-dedicated pedestrian infrastructure
        let penalty = if !props.is_foot_accessible || props.is_sidewalk_separate {
            90 // 90 second penalty for bike paths without foot access
        } else {
            0
        };

        penalty + time_cost
    } else {
        return INFEASIBLE;
    }
}

/// Calculate node cost in seconds
///
/// # C++ Equivalent
/// ```cpp
/// static constexpr cost_t node_cost(node_properties const n);
/// ```
pub fn node_cost(props: &NodeProperties) -> Cost {
    if !props.is_foot_accessible {
        INFEASIBLE
    } else if props.is_elevator {
        90 // 90 second penalty for using elevator
    } else {
        0
    }
}

/// Calculate heuristic for A* (lower bound on remaining distance)
///
/// # C++ Equivalent
/// ```cpp
/// static constexpr double heuristic(parameters const& params, double const dist);
/// ```
pub fn heuristic(params: &FootParameters, dist: f64) -> f64 {
    // Optimistic speed estimate (slightly faster than params.speed)
    dist / (params.speed_meters_per_second as f64 + 0.1)
}

// ============================================================================
// Level transition helpers (BUG-07)
// ============================================================================

/// Determine the target level when traversing from `from_level` along a way with `props`.
///
/// Returns `None` if the traversal is not level-compatible (e.g. wheelchair on steps,
/// or the way connects different floors that don't match the current floor).
///
/// Matches C++ `foot::get_target_level()`.
pub fn foot_get_target_level(
    from_level: Level,
    props: &WayProperties,
    is_wheelchair: bool,
) -> Option<Level> {
    let no = Level::NO_LEVEL;
    let from_idx = from_level.to_idx();
    let way_from: u8 = props.from_level.into();
    let way_to: u8 = props.to_level.into();

    // Wheelchair cannot use steps
    if is_wheelchair && props.is_steps {
        return None;
    }

    if props.is_steps || props.is_ramp {
        // Staircase / ramp: transitions between two known levels
        if from_idx == no {
            // Unknown level: pick whichever end is non-ground/non-zero
            return Some(Level::from_idx(if way_from == 0 || way_from == no { way_to } else { way_from }));
        } else if from_idx == way_from {
            return Some(Level::from_idx(way_to));
        } else if from_idx == way_to {
            return Some(Level::from_idx(way_from));
        } else {
            return None; // Level mismatch for stairs/ramp
        }
    }

    // Elevator way: keep current level (actual level change happens through elevator node)
    if props.is_elevator {
        return Some(from_level);
    }

    // Normal way: accept if from_level or way level is unspecified, or they match
    if from_idx == no || way_from == no || from_idx == way_from {
        return Some(Level::from_idx(way_from));
    }

    None
}

// ============================================================================
// Bidirectional Search Support
// ============================================================================

use crate::routing::bidirectional_profile::{BidirectionalEntry, BidirectionalLabel, BidirectionalProfile};
use crate::routing::tracking::NoopTracking;

/// Label for bidirectional foot routing (simplified, no tracking)
#[derive(Debug, Clone, Copy)]
pub struct BidirectionalFootLabel {
    pub n: NodeIdx,
    pub lvl: Level,
    pub cost: Cost,
}

impl BidirectionalLabel for BidirectionalFootLabel {
    type Node = FootNode;

    fn new(node: Self::Node, cost: Cost) -> Self {
        Self {
            n: node.n,
            lvl: node.lvl,
            cost,
        }
    }

    fn get_node(&self) -> Self::Node {
        FootNode {
            n: self.n,
            lvl: self.lvl,
        }
    }

    fn cost(&self) -> Cost {
        self.cost
    }
}

/// Entry for bidirectional foot routing (single state - no directional variants)
#[derive(Debug, Clone)]
pub struct BidirectionalFootEntry {
    pred: Option<NodeIdx>,
    pred_lvl: Level,
    cost: Cost,
}

impl Default for BidirectionalFootEntry {
    fn default() -> Self {
        Self {
            pred: None,
            pred_lvl: Level::from_idx(Level::NO_LEVEL),
            cost: Cost::MAX,
        }
    }
}

impl BidirectionalEntry for BidirectionalFootEntry {
    type Node = FootNode;

    fn pred(&self, _node: Self::Node) -> Option<Self::Node> {
        self.pred.map(|n| FootNode { n, lvl: self.pred_lvl })
    }

    fn cost(&self, _node: Self::Node) -> Cost {
        self.cost
    }

    fn update(&mut self, _node: Self::Node, cost: Cost, pred: Self::Node) -> bool {
        if cost < self.cost {
            self.cost = cost;
            self.pred = Some(pred.n);
            self.pred_lvl = pred.lvl;
            true
        } else {
            false
        }
    }
}

impl BidirectionalProfile for FootProfile<NoopTracking> {
    type Node = FootNode;
    type Key = FootNode; // Foot uses full node as key (includes level)
    type Label = BidirectionalFootLabel;
    type Entry = BidirectionalFootEntry;
    type Parameters = FootProfile<NoopTracking>;

    fn get_key(node: &Self::Node) -> Self::Key {
        *node
    }

    fn get_node_idx(node: &Self::Node) -> NodeIdx {
        node.n
    }

    fn heuristic(params: &Self::Parameters, distance_meters: f64) -> Cost {
        // BUG-09 fix: use speed + 0.1 (optimistic lower bound, matching C++)
        let speed = params.params.speed_meters_per_second as f64 + 0.1;
        (distance_meters / speed).round() as Cost
    }

    fn adjacent<F>(
        params: &Self::Parameters,
        ways: &Ways,
        _elevations: Option<&ElevationStorage>,
        node: Self::Node,
        _search_dir: Direction,
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

            // BUG-07 fix: determine target level from way properties
            // Returns None if this way is impassable from current level
            let target_lvl = foot_get_target_level(node.lvl, props, params.is_wheelchair);
            let Some(target_lvl) = target_lvl else {
                continue;
            };

            // Expand to preceding node (i-1)
            if i > 0 {
                let prev_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0);
                if dist > 0 {
                    // BUG-01 fix: check TARGET node cost, not current
                    let target_props = ways.get_node_properties(prev_node);
                    let t_cost = target_props.map(node_cost).unwrap_or(0);
                    if t_cost != INFEASIBLE {
                        // BUG-09 fix: use params.is_wheelchair directly
                        let edge_cost = way_cost(&params.params, props, params.is_wheelchair, dist);
                        if edge_cost != INFEASIBLE {
                            let total = edge_cost.saturating_add(t_cost);
                            if total < INFEASIBLE {
                                callback(FootNode { n: prev_node, lvl: target_lvl }, total);
                            }
                        }
                    }
                }
            }

            // Expand to following node (i+1)
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0);
                if dist > 0 {
                    // BUG-01 fix: check TARGET node cost, not current
                    let target_props = ways.get_node_properties(next_node);
                    let t_cost = target_props.map(node_cost).unwrap_or(0);
                    if t_cost != INFEASIBLE {
                        // BUG-09 fix: use params.is_wheelchair directly
                        let edge_cost = way_cost(&params.params, props, params.is_wheelchair, dist);
                        if edge_cost != INFEASIBLE {
                            let total = edge_cost.saturating_add(t_cost);
                            if total < INFEASIBLE {
                                callback(FootNode { n: next_node, lvl: target_lvl }, total);
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
        let f = fwd_entry.cost(node);
        let b = bwd_entry.cost(node);
        if f != INFEASIBLE && b != INFEASIBLE {
            Some((node, node, f.saturating_add(b)))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foot_profile_creation() {
        let profile: FootProfile = FootProfile::default();
        assert!(!profile.is_wheelchair);
        assert_eq!(profile.mode(), Mode::Foot);
    }

    #[test]
    fn test_wheelchair_profile() {
        let profile: FootProfile = FootProfile::new(true);
        assert!(profile.is_wheelchair);
        assert_eq!(profile.mode(), Mode::Wheelchair);
    }

    #[test]
    fn test_max_match_distance() {
        let profile: FootProfile = FootProfile::default();
        assert_eq!(profile.max_match_distance(), 100);
    }

    #[test]
    fn test_foot_node_equality() {
        let n1 = FootNode::new(NodeIdx(5), Level::from_float(0.0));
        let n2 = FootNode::new(NodeIdx(5), Level::from_idx(Level::NO_LEVEL));
        assert_eq!(n1, n2); // Level 0.0 == NO_LEVEL

        let n3 = FootNode::new(NodeIdx(5), Level::from_float(1.0));
        assert_ne!(n1, n3); // Different non-zero levels
    }

    #[test]
    fn test_foot_node_invalid() {
        let n = FootNode::invalid();
        assert_eq!(n.n, NodeIdx::INVALID);
        assert_eq!(n.lvl.to_idx(), Level::NO_LEVEL);
    }

    #[test]
    fn test_foot_label_creation() {
        let node = FootNode::new(NodeIdx(10), Level::from_float(0.0));
        let label: FootLabel<crate::routing::tracking::NoopTracking> = FootLabel::new(node, 100);
        assert_eq!(label.n, NodeIdx(10));
        assert_eq!(label.cost, 100);
        assert_eq!(label.lvl, Level::from_float(0.0));
    }

    #[test]
    fn test_foot_entry_update() {
        let mut entry: FootEntry<crate::routing::tracking::NoopTracking> = FootEntry::new();
        assert_eq!(entry.cost, INFEASIBLE);

        let node = FootNode::new(NodeIdx(5), Level::from_float(0.0));
        let label = FootLabel::new(node, 50);
        let pred = FootNode::new(NodeIdx(4), Level::from_float(0.0));

        assert!(entry.update(&label, node, 100, pred));
        assert_eq!(entry.cost, 100);
        assert_eq!(entry.pred, NodeIdx(4));

        // Better cost should update
        assert!(entry.update(&label, node, 80, pred));
        assert_eq!(entry.cost, 80);

        // Worse cost should not update
        assert!(!entry.update(&label, node, 90, pred));
        assert_eq!(entry.cost, 80);
    }

    #[test]
    fn test_way_cost_foot_accessible() {
        let params = FootParameters::default();
        let mut props = WayProperties::default();
        props.is_foot_accessible = true;

        // 100 meters at 1.2 m/s ≈ 83 seconds
        let cost = way_cost(&params, &props, false, 100);
        assert!(cost >= 80 && cost <= 86);
    }

    #[test]
    fn test_way_cost_wheelchair_steps() {
        let params = FootParameters::wheelchair();
        let mut props = WayProperties::default();
        props.is_foot_accessible = true;
        props.is_steps = true;

        // Wheelchair cannot use steps
        let cost = way_cost(&params, &props, true, 100);
        assert_eq!(cost, INFEASIBLE);
    }

    #[test]
    fn test_way_cost_big_street_penalty() {
        let params = FootParameters::default();
        let mut props = WayProperties::default();
        props.is_foot_accessible = true;
        props.is_big_street = true;

        // Big street should be slower (speed reduced by 0.2 m/s)
        let cost = way_cost(&params, &props, false, 100);
        let normal_cost = way_cost(
            &params,
            &WayProperties {
                is_foot_accessible: true,
                ..Default::default()
            },
            false,
            100,
        );
        assert!(cost > normal_cost);
    }

    #[test]
    fn test_way_cost_car_free_bonus() {
        let params = FootParameters::default();
        let mut props = WayProperties::default();
        props.is_foot_accessible = true;
        props.motor_vehicle_no = true;

        // Car-free should be faster (speed increased by 0.1 m/s)
        let cost = way_cost(&params, &props, false, 100);
        let normal_cost = way_cost(
            &params,
            &WayProperties {
                is_foot_accessible: true,
                ..Default::default()
            },
            false,
            100,
        );
        assert!(cost < normal_cost);
    }

    #[test]
    fn test_node_cost_normal() {
        let props = NodeProperties {
            is_foot_accessible: true,
            is_elevator: false,
            ..Default::default()
        };
        assert_eq!(node_cost(&props), 0);
    }

    #[test]
    fn test_node_cost_elevator() {
        let props = NodeProperties {
            is_foot_accessible: true,
            is_elevator: true,
            ..Default::default()
        };
        assert_eq!(node_cost(&props), 90);
    }

    #[test]
    fn test_node_cost_inaccessible() {
        let props = NodeProperties {
            is_foot_accessible: false,
            ..Default::default()
        };
        assert_eq!(node_cost(&props), INFEASIBLE);
    }

    #[test]
    fn test_heuristic() {
        let params = FootParameters::default();
        // 1000 meters / 1.3 m/s ≈ 769 seconds
        let h = heuristic(&params, 1000.0);
        assert!(h >= 760.0 && h <= 780.0);
    }
}
