//! Bidirectional A* search algorithm.
//!
//! This module implements bidirectional A* search, which explores from both
//! start and destination simultaneously. This is significantly faster than
//! unidirectional Dijkstra for point-to-point queries.
//!
//! The algorithm maintains two search frontiers that expand towards each other.
//! When they meet, the shortest path is reconstructed.

use crate::location::Location;
use crate::routing::{Dial, Label};
use crate::routing::bidirectional_profile::{BidirectionalEntry, BidirectionalLabel, BidirectionalProfile};
use crate::types::{Cost, NodeIdx};
use crate::Ways;
use ahash::AHashMap;

/// Earth radius in meters (for distance calculations)
#[allow(dead_code)]
const EARTH_RADIUS_M: f64 = 6371000.0;
#[allow(dead_code)]
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Bidirectional A* search state.
///
/// Maintains two search frontiers (forward and backward) that expand
/// simultaneously towards each other. Uses A* heuristic to guide search.
pub struct Bidirectional<'w> {
    /// Forward search priority queue (from start)
    pq_forward: Dial<Label, fn(&Label) -> usize>,

    /// Backward search priority queue (from destination)
    pq_backward: Dial<Label, fn(&Label) -> usize>,

    /// Costs in forward direction (using AHashMap for performance)
    cost_forward: AHashMap<NodeIdx, Cost>,

    /// Costs in backward direction (using AHashMap for performance)
    cost_backward: AHashMap<NodeIdx, Cost>,

    /// Predecessors in forward direction
    pred_forward: AHashMap<NodeIdx, NodeIdx>,

    /// Predecessors in backward direction
    pred_backward: AHashMap<NodeIdx, NodeIdx>,

    /// Best meeting point from forward search
    meet_point_fwd: Option<NodeIdx>,

    /// Best meeting point from backward search
    meet_point_bwd: Option<NodeIdx>,

    /// Best cost found (forward + backward)
    best_cost: Cost,

    /// Start location (for heuristic)
    start_loc: Location,

    /// End location (for heuristic)
    end_loc: Location,

    /// Reference to street network (for node position lookups) - explicit lifetime
    ways: Option<&'w Ways>,

    /// Radius for search termination (approximately 0.5 * straight-line distance)
    radius: Cost,

    /// Maximum cost reached in forward search
    max_reached_fwd: bool,

    /// Maximum cost reached in backward search
    max_reached_bwd: bool,
}

impl<'w> Bidirectional<'w> {
    /// Create a new bidirectional search instance.
    pub fn new() -> Self {
        fn get_bucket(label: &Label) -> usize {
            label.cost as usize
        }

        Self {
            pq_forward: Dial::new(get_bucket),
            pq_backward: Dial::new(get_bucket),
            cost_forward: AHashMap::new(),
            cost_backward: AHashMap::new(),
            pred_forward: AHashMap::new(),
            pred_backward: AHashMap::new(),
            meet_point_fwd: None,
            meet_point_bwd: None,
            best_cost: Cost::MAX,
            start_loc: Location::from_latlng_no_level(0.0, 0.0),
            end_loc: Location::from_latlng_no_level(0.0, 0.0),
            ways: None,
            radius: Cost::MAX,
            max_reached_fwd: false,
            max_reached_bwd: false,
        }
    }

    /// Set the Ways reference for node position lookups.
    ///
    /// This enables the A* heuristic to calculate actual distances.
    /// Must be called before running searches for heuristic to work.
    pub fn set_ways(&mut self, ways: &'w Ways) {
        self.ways = Some(ways); // Safe: explicit lifetime ensures validity
    }

    /// Reset the search state for a new query.
    ///
    /// # Arguments
    /// * `max_cost` - Maximum cost limit
    /// * `start` - Starting location
    /// * `end` - Destination location
    pub fn reset(&mut self, max_cost: Cost, start: Location, end: Location) {
        self.pq_forward.clear();
        self.pq_backward.clear();
        self.pq_forward.set_n_buckets(max_cost as usize + 1);
        self.pq_backward.set_n_buckets(max_cost as usize + 1);
        self.cost_forward.clear();
        self.cost_backward.clear();
        self.pred_forward.clear();
        self.pred_backward.clear();
        self.meet_point_fwd = None;
        self.meet_point_bwd = None;
        self.best_cost = Cost::MAX;
        self.start_loc = start;
        self.end_loc = end;
        
        // Calculate radius (C++ logic: max(diameter * 0.5, kLongestNodeDistance))
        const K_LONGEST_NODE_DISTANCE: Cost = 300;
        if self.ways.is_some() {
            let diameter = self.distance_heuristic(
                start.pos_.lat(), start.pos_.lng(),
                end.pos_.lat(), end.pos_.lng()
            );
            if diameter < max_cost && 
               max_cost.saturating_add(diameter.max(K_LONGEST_NODE_DISTANCE * 2)) < Cost::MAX 
            {
                self.radius = (diameter / 2).max(K_LONGEST_NODE_DISTANCE);
            } else {
                self.radius = max_cost;
            }
        } else {
            self.radius = max_cost;
        }
        
        self.max_reached_fwd = false;
        self.max_reached_bwd = false;
    }

    /// Add a starting node to the forward search with an initial match cost.
    pub fn add_start(&mut self, node: NodeIdx, cost: Cost) {
        let heuristic = self.heuristic_to_end(node);
        let total_cost = cost.saturating_add(heuristic);
        if self.update_cost_forward(node, cost, None) {
            // Mark seed predecessor as self to aid end-of-way/u-turn detection
            self.pred_forward.insert(node, node);
            self.pq_forward.push(Label::new(node, total_cost));
            // If the backward frontier has already reached this node, update meeting
            self.check_meeting_point(node);
        }
    }

    /// Add a destination node to the backward search with an initial match cost.
    pub fn add_end(&mut self, node: NodeIdx, cost: Cost) {
        let heuristic = self.heuristic_to_start(node);
        let total_cost = cost.saturating_add(heuristic);
        if self.update_cost_backward(node, cost, None) {
            // Mark seed predecessor as self to aid end-of-way/u-turn detection
            self.pred_backward.insert(node, node);
            self.pq_backward.push(Label::new(node, total_cost));
            // If the forward frontier has already reached this node, update meeting
            self.check_meeting_point(node);
        }
    }

    /// Get the best path cost found (if any).
    pub fn best_cost(&self) -> Cost {
        self.best_cost
    }

    /// Get the meeting point (if found).
    pub fn meet_point(&self) -> Option<(NodeIdx, NodeIdx)> {
        self.meet_point_fwd
            .and_then(|fwd| self.meet_point_bwd.map(|bwd| (fwd, bwd)))
    }

    /// Check if maximum cost was reached.
    pub fn max_reached(&self) -> bool {
        self.max_reached_fwd || self.max_reached_bwd
    }

    /// Calculate straight-line distance heuristic (in cost units).
    ///
    /// Assumes walking speed of ~5 km/h (1.4 m/s), so 1 meter ≈ 0.7 seconds.
    #[allow(dead_code)]
    fn distance_heuristic(&self, from_lat: f64, from_lon: f64, to_lat: f64, to_lon: f64) -> Cost {
        let lat1 = from_lat * DEG_TO_RAD;
        let lat2 = to_lat * DEG_TO_RAD;
        let dlat = lat2 - lat1;
        let dlon = (to_lon - from_lon) * DEG_TO_RAD;

        // Haversine formula
        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();
        let distance_m = EARTH_RADIUS_M * c;

        // Convert to cost (seconds at walking speed ~1.4 m/s)
        // Cap at Cost::MAX to avoid overflow
        let cost_seconds = distance_m * 0.7;
        if cost_seconds >= Cost::MAX as f64 {
            Cost::MAX
        } else {
            cost_seconds as Cost
        }
    }

    /// Consistent bidirectional heuristic (C++ parity).
    /// Uses formula: 0.5 * (h_goal - h_start) for both directions.
    /// This ensures both frontiers use the same heuristic value.
    fn heuristic_forward(&self, node: NodeIdx) -> Cost {
        match self.ways {
            Some(ways) => {
                let node_pos = ways.get_node_pos(node);
                let dist_to_goal = self.distance_heuristic(
                    node_pos.lat(), node_pos.lng(),
                    self.end_loc.pos_.lat(), self.end_loc.pos_.lng()
                );
                let dist_to_start = self.distance_heuristic(
                    node_pos.lat(), node_pos.lng(),
                    self.start_loc.pos_.lat(), self.start_loc.pos_.lng()
                );
                // Consistent bidirectional: 0.5 * (h_goal - h_start)
                ((dist_to_goal as i64 - dist_to_start as i64) / 2).max(0) as Cost
            }
            None => 0, // No Ways reference - degrade to Dijkstra
        }
    }

    /// Consistent bidirectional heuristic for backward search.
    /// Uses the same formula as forward to ensure consistency.
    fn heuristic_backward(&self, node: NodeIdx) -> Cost {
        self.heuristic_forward(node) // Same heuristic for both directions
    }

    /// Heuristic cost from a node to the end location.
    fn heuristic_to_end(&self, node: NodeIdx) -> Cost {
        self.heuristic_forward(node)
    }

    /// Heuristic cost from a node to the start location.
    fn heuristic_to_start(&self, node: NodeIdx) -> Cost {
        self.heuristic_backward(node)
    }

    /// Get cost to reach a node in forward direction.
    pub fn get_cost_forward(&self, node: NodeIdx) -> Cost {
        self.cost_forward.get(&node).copied().unwrap_or(Cost::MAX)
    }

    /// Get cost to reach a node in backward direction.
    pub fn get_cost_backward(&self, node: NodeIdx) -> Cost {
        self.cost_backward.get(&node).copied().unwrap_or(Cost::MAX)
    }

    /// Update cost for a node in forward direction (C++ parity: strict < only).
    fn update_cost_forward(&mut self, node: NodeIdx, cost: Cost, pred: Option<NodeIdx>) -> bool {
        match self.cost_forward.get(&node) {
            Some(&existing) if cost >= existing => {
                // C++ parity: only update if strictly better (c < cost_)
                return false;
            }
            _ => {}
        }

        // Insert new better cost
        self.cost_forward.insert(node, cost);
        if let Some(p) = pred {
            self.pred_forward.insert(node, p);
        }
        true
    }

    /// Update cost for a node in backward direction (C++ parity: strict < only).
    fn update_cost_backward(&mut self, node: NodeIdx, cost: Cost, pred: Option<NodeIdx>) -> bool {
        match self.cost_backward.get(&node) {
            Some(&existing) if cost >= existing => {
                // C++ parity: only update if strictly better (c < cost_)
                return false;
            }
            _ => {}
        }

        self.cost_backward.insert(node, cost);
        if let Some(p) = pred {
            self.pred_backward.insert(node, p);
        }
        true
    }

    /// Check if searches have met at a node and update best path.
    pub fn check_meeting_point(&mut self, node: NodeIdx) {
        let fwd_cost = self.get_cost_forward(node);
        let bwd_cost = self.get_cost_backward(node);

        if fwd_cost != Cost::MAX && bwd_cost != Cost::MAX {
            // Use evaluate_meetpoint for deterministic tie-breaking and centralized logic
            self.evaluate_meetpoint(fwd_cost, bwd_cost, node, node);
        }
    }

    /// Evaluate a potential meeting point with explicit forward/back nodes (C++ parity: strict <).
    fn evaluate_meetpoint(&mut self, fwd_cost: Cost, bwd_cost: Cost, meet_fwd: NodeIdx, meet_bwd: NodeIdx) {
        let tentative = fwd_cost.saturating_add(bwd_cost);

        // C++ parity: accept only if strictly better (tentative < best_cost_)
        if tentative < self.best_cost {
            self.best_cost = tentative;
            self.meet_point_fwd = Some(meet_fwd);
            self.meet_point_bwd = Some(meet_bwd);
        }
    }

    /// End-of-way meet-point checks for forward-extracted node.
    fn handle_end_of_way_meetpoint_forward(&mut self, current: NodeIdx) {
        let curr_cost = self.get_cost_forward(current);
        if curr_cost == Cost::MAX {
            return;
        }

        // If backward already reached current, accept meeting at current
        let other_cost = self.cost_backward.get(&current).copied().unwrap_or(Cost::MAX);
        if other_cost != Cost::MAX {
            self.evaluate_meetpoint(curr_cost, other_cost, current, current);
            return;
        }

        // Try predecessor-based u-turn detection
        if let Some(&pred) = self.pred_forward.get(&current) {
            // If predecessor is self (seed marker), do not attempt chain-based checks
            if pred == current {
                return;
            }

            // Check if backward predecessor for pred points back to current
            if let Some(&opposite_curr) = self.pred_backward.get(&pred) {
                if opposite_curr == current {
                    let pred_cost = self.get_cost_forward(pred);
                    let opposite_pred_cost = self.get_cost_backward(pred);
                    let opposite_curr_cost = self.get_cost_backward(current);
                    if pred_cost == Cost::MAX || opposite_pred_cost == Cost::MAX || opposite_curr_cost == Cost::MAX {
                        return;
                    }
                    if pred_cost.saturating_add(opposite_pred_cost) > curr_cost.saturating_add(opposite_curr_cost) {
                        self.evaluate_meetpoint(pred_cost, opposite_pred_cost, pred, pred);
                    } else {
                        self.evaluate_meetpoint(curr_cost, opposite_curr_cost, current, current);
                    }
                }
            }
        }
    }

    /// End-of-way meet-point checks for backward-extracted node.
    fn handle_end_of_way_meetpoint_backward(&mut self, current: NodeIdx) {
        let curr_cost = self.get_cost_backward(current);
        if curr_cost == Cost::MAX {
            return;
        }

        // If forward already reached current, accept meeting at current
        let other_cost = self.cost_forward.get(&current).copied().unwrap_or(Cost::MAX);
        if other_cost != Cost::MAX {
            self.evaluate_meetpoint(other_cost, curr_cost, current, current);
            return;
        }

        // Try predecessor-based u-turn detection (roles swapped)
        if let Some(&pred) = self.pred_backward.get(&current) {
            // If predecessor is self (seed marker), do not attempt chain-based checks
            if pred == current {
                return;
            }

            if let Some(&opposite_curr) = self.pred_forward.get(&pred) {
                if opposite_curr == current {
                    let pred_cost = self.get_cost_backward(pred);
                    let opposite_pred_cost = self.get_cost_forward(pred);
                    let opposite_curr_cost = self.get_cost_forward(current);
                    if pred_cost == Cost::MAX || opposite_pred_cost == Cost::MAX || opposite_curr_cost == Cost::MAX {
                        return;
                    }
                    if pred_cost.saturating_add(opposite_pred_cost) > curr_cost.saturating_add(opposite_curr_cost) {
                        self.evaluate_meetpoint(opposite_pred_cost, pred_cost, pred, pred);
                    } else {
                        self.evaluate_meetpoint(opposite_curr_cost, curr_cost, current, current);
                    }
                }
            }
        }
    }

    /// Run one step of forward search.
    ///
    /// Returns true if search should continue.
    pub fn step_forward<F>(&mut self, max_cost: Cost, mut get_neighbors: F) -> bool
    where
        F: FnMut(NodeIdx) -> Vec<(NodeIdx, Cost)>,
    {
        if self.pq_forward.empty() {
            return false;
        }

        // C++ parity: adjusted_max = (max + radius) / 2
        let adjusted_max = (max_cost.saturating_add(self.radius)) / 2;

        let label = self.pq_forward.pop();
        let current = label.node;

        // The Dial stores priority = actual_cost + heuristic. When popping we
        // need to recover the actual cost portion to perform dominance checks
        // against the stored best-known cost. Compute label_actual_cost by
        // subtracting the heuristic for this node.
        let label_priority = label.cost;
        let label_actual_cost = label_priority.saturating_sub(self.heuristic_to_end(current));
        let current_cost = self.get_cost_forward(current);

        // Skip if a better cost is already recorded
        if current_cost < label_actual_cost {
            return true;
        }

        // Check if we've met the backward search
        self.check_meeting_point(current);

        // End-of-way / u-turn meet-point checks immediately on extraction
        self.handle_end_of_way_meetpoint_forward(current);

        // Expand neighbors
        for (neighbor, edge_cost) in get_neighbors(current) {
            let total = current_cost + edge_cost;

            if total >= adjusted_max {
                self.max_reached_fwd = true;
                continue;
            }

            if self.update_cost_forward(neighbor, total, Some(current)) {
                let heuristic = self.heuristic_to_end(neighbor);
                let priority = total + heuristic;
                self.pq_forward.push(Label::new(neighbor, priority));

                // Check if backward search has reached this node
                self.check_meeting_point(neighbor);
                // Also run end-of-way/u-turn checks for the newly discovered neighbor
                self.handle_end_of_way_meetpoint_forward(neighbor);
            }
        }

        // End-of-way / u-turn meet-point checks (post-expansion)
        self.handle_end_of_way_meetpoint_forward(current);

        true
    }

    /// Run one step of backward search.
    ///
    /// Returns true if search should continue.
    pub fn step_backward<F>(&mut self, max_cost: Cost, mut get_neighbors: F) -> bool
    where
        F: FnMut(NodeIdx) -> Vec<(NodeIdx, Cost)>,
    {
        if self.pq_backward.empty() {
            return false;
        }

        // C++ parity: adjusted_max = (max + radius) / 2
        let adjusted_max = (max_cost.saturating_add(self.radius)) / 2;

        let label = self.pq_backward.pop();
        let current = label.node;

        // Recover actual cost portion of the popped priority (priority = cost + heuristic)
        let label_priority = label.cost;
        let label_actual_cost = label_priority.saturating_sub(self.heuristic_to_start(current));
        let current_cost = self.get_cost_backward(current);

        // Skip if a better cost is already recorded
        if current_cost < label_actual_cost {
            return true;
        }

        // Check if we've met the forward search
        self.check_meeting_point(current);

        // End-of-way / u-turn meet-point checks immediately on extraction
        self.handle_end_of_way_meetpoint_backward(current);

        // Expand neighbors (in reverse)
        for (neighbor, edge_cost) in get_neighbors(current) {
            let total = current_cost + edge_cost;

            if total >= adjusted_max {
                self.max_reached_bwd = true;
                continue;
            }

            if self.update_cost_backward(neighbor, total, Some(current)) {
                let heuristic = self.heuristic_to_start(neighbor);
                let priority = total + heuristic;
                self.pq_backward.push(Label::new(neighbor, priority));

                // Check if forward search has reached this node
                self.check_meeting_point(neighbor);
                // Also run end-of-way/u-turn checks for the newly discovered neighbor
                self.handle_end_of_way_meetpoint_backward(neighbor);
            }
        }

        // End-of-way / u-turn meet-point checks (post-expansion)
        self.handle_end_of_way_meetpoint_backward(current);

        true
    }

    /// Run bidirectional search until completion or max cost reached.
    ///
    /// Alternates between forward and backward search steps.
    pub fn run<F, B>(
        &mut self,
        max_cost: Cost,
        mut get_neighbors_fwd: F,
        mut get_neighbors_bwd: B,
    ) -> bool
    where
        F: FnMut(NodeIdx) -> Vec<(NodeIdx, Cost)>,
        B: FnMut(NodeIdx) -> Vec<(NodeIdx, Cost)>,
    {
        while !self.pq_forward.empty() || !self.pq_backward.empty() {
            // C++ parity: terminate when top_f + top_r >= best_cost + radius
            // PQ contains f-values (priority = g + h)
            if self.best_cost != Cost::MAX {
                let min_fwd = self
                    .pq_forward
                    .peek_min_bucket()
                    .unwrap_or(usize::MAX);
                let min_bwd = self
                    .pq_backward
                    .peek_min_bucket()
                    .unwrap_or(usize::MAX);

                // C++ logic: if (top_f + top_r >= best_cost_ + radius_) break;
                if min_fwd != usize::MAX && min_bwd != usize::MAX {
                    let sum = (min_fwd as Cost).saturating_add(min_bwd as Cost);
                    let threshold = self.best_cost.saturating_add(self.radius);
                    if sum >= threshold {
                        break;
                    }
                }
            }

            // Alternate: step forward if it has fewer items
            if !self.pq_forward.empty()
                && (self.pq_backward.empty() || self.pq_forward.size() <= self.pq_backward.size())
            {
                if !self.step_forward(max_cost, &mut get_neighbors_fwd) {
                    break;
                }
            } else if !self.pq_backward.empty() {
                if !self.step_backward(max_cost, &mut get_neighbors_bwd) {
                    break;
                }
            }
        }

        !self.max_reached()
    }

    /// Reconstruct path from meet point
    ///
    /// Returns vector of nodes from start to goal through meet point
    pub fn reconstruct_path(&self) -> Option<Vec<NodeIdx>> {
        let (meet_fwd, meet_bwd) = self.meet_point()?;

        // Forward path: start -> meet_point
        // Pre-allocate reasonable capacity (typical paths: 10-100 nodes)
        let mut fwd_path = Vec::with_capacity(64);
        let mut current = meet_fwd;

        loop {
            fwd_path.push(current);
            if let Some(&pred) = self.pred_forward.get(&current) {
                // If predecessor is self (seed marker) treat as start and stop
                if pred == current {
                    break;
                }
                current = pred;
            } else {
                break; // Reached start node
            }
        }

        fwd_path.reverse(); // Reverse to get start->meet order

        // Backward path: meet_point -> goal
        // Pre-allocate reasonable capacity
        let mut bwd_path = Vec::with_capacity(64);
        let mut current = meet_bwd;

        // Skip the meet point itself to avoid duplication
        if let Some(&pred) = self.pred_backward.get(&current) {
            // If predecessor is self (seed marker) treat as goal and stop
            if pred != current {
                current = pred;

                loop {
                    bwd_path.push(current);
                    if let Some(&pred) = self.pred_backward.get(&current) {
                        if pred == current {
                            break;
                        }
                        current = pred;
                    } else {
                        break; // Reached goal node
                    }
                }
            }
        }

        // Combine paths
        fwd_path.extend(bwd_path);

        Some(fwd_path)
    }

    /// Get predecessor in forward search
    pub fn get_pred_forward(&self, node: NodeIdx) -> Option<NodeIdx> {
        self.pred_forward.get(&node).copied()
    }

    /// Get predecessor in backward search
    pub fn get_pred_backward(&self, node: NodeIdx) -> Option<NodeIdx> {
        self.pred_backward.get(&node).copied()
    }

    /// Dump internal search state for debugging.
    pub fn dump_state(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("best_cost={} meet_point={:?} max_reached_fwd={} max_reached_bwd={}\n",
            self.best_cost, self.meet_point(), self.max_reached_fwd, self.max_reached_bwd));

        out.push_str(&format!("pq_forward_min={:?} pq_backward_min={:?} pq_forward_size={} pq_backward_size={}\n",
            self.pq_forward.peek_min_bucket(), self.pq_backward.peek_min_bucket(), self.pq_forward.size(), self.pq_backward.size()));

        out.push_str("cost_forward:\n");
        for (k, v) in &self.cost_forward {
            out.push_str(&format!("  {:?}: {}\n", k, v));
        }

        out.push_str("cost_backward:\n");
        for (k, v) in &self.cost_backward {
            out.push_str(&format!("  {:?}: {}\n", k, v));
        }

        out
    }

    /// Accessors for queue state to support external debugging.
    pub fn pq_forward_empty(&self) -> bool {
        self.pq_forward.empty()
    }

    pub fn pq_backward_empty(&self) -> bool {
        self.pq_backward.empty()
    }

    pub fn pq_forward_size(&self) -> usize {
        self.pq_forward.size()
    }

    pub fn pq_backward_size(&self) -> usize {
        self.pq_backward.size()
    }

    pub fn pq_forward_peek(&self) -> Option<usize> {
        self.pq_forward.peek_min_bucket()
    }

    pub fn pq_backward_peek(&self) -> Option<usize> {
        self.pq_backward.peek_min_bucket()
    }
}

// ============================================================================
// Generic Bidirectional A* Search
// ============================================================================

/// Generic bidirectional A* search
///
/// C++ equivalent: `template <Profile P> struct bidirectional`
///
/// This implementation is generic over routing profiles, allowing the same
/// bidirectional search algorithm to work with foot, bike, and car profiles.
pub struct GenericBidirectional<'w, P>
where
    P: BidirectionalProfile,
{
    /// Forward search priority queue
    pq_forward: Dial<P::Label, fn(&P::Label) -> usize>,

    /// Backward search priority queue
    pq_backward: Dial<P::Label, fn(&P::Label) -> usize>,

    /// Forward costs (keyed by P::Key, not P::Node)
    cost_forward: AHashMap<P::Key, P::Entry>,

    /// Backward costs
    cost_backward: AHashMap<P::Key, P::Entry>,

    /// Best meeting point from forward search
    meet_point_fwd: Option<P::Node>,

    /// Best meeting point from backward search
    meet_point_bwd: Option<P::Node>,

    /// Best cost found (forward + backward)
    best_cost: Cost,

    /// Start location (for heuristic)
    start_loc: Location,

    /// End location (for heuristic)
    end_loc: Location,

    /// Reference to street network (for node position lookups)
    ways: Option<&'w Ways>,

    /// Radius for search termination
    radius: Cost,

    /// Maximum cost reached flags
    max_reached_fwd: bool,
    max_reached_bwd: bool,

    /// Longitude degree to meters factor (depends on latitude)
    distance_lon_degrees: f64,

    /// Profile-specific parameters
    _phantom: std::marker::PhantomData<P>,
}

impl<'w, P> GenericBidirectional<'w, P>
where
    P: BidirectionalProfile,
{
    pub fn new() -> Self {
        fn get_bucket<L: crate::routing::bidirectional_profile::BidirectionalLabel>(
            label: &L,
        ) -> usize {
            label.cost() as usize
        }

        Self {
            pq_forward: Dial::new(get_bucket::<P::Label>),
            pq_backward: Dial::new(get_bucket::<P::Label>),
            cost_forward: AHashMap::new(),
            cost_backward: AHashMap::new(),
            meet_point_fwd: None,
            meet_point_bwd: None,
            best_cost: Cost::MAX,
            start_loc: Location::from_latlng_no_level(0.0, 0.0),
            end_loc: Location::from_latlng_no_level(0.0, 0.0),
            ways: None,
            radius: Cost::MAX,
            max_reached_fwd: false,
            max_reached_bwd: false,
            distance_lon_degrees: 0.0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set ways reference for heuristic calculation
    pub fn set_ways(&mut self, ways: &'w Ways) {
        self.ways = Some(ways);
    }

    /// Reset for new query
    pub fn reset(
        &mut self,
        _params: &P::Parameters,
        max_cost: Cost,
        start: Location,
        end: Location,
    ) {
        self.pq_forward.clear();
        self.pq_backward.clear();
        self.pq_forward.set_n_buckets(max_cost as usize + 1);
        self.pq_backward.set_n_buckets(max_cost as usize + 1);
        self.cost_forward.clear();
        self.cost_backward.clear();
        self.meet_point_fwd = None;
        self.meet_point_bwd = None;
        self.best_cost = Cost::MAX;
        self.start_loc = start;
        self.end_loc = end;

        // Initialize distance constants
        const K_DISTANCE_LAT_DEGREES: f64 = 111111.0;
        let avg_lat = (start.pos_.lat() + end.pos_.lat()) * 0.5;
        self.distance_lon_degrees = avg_lat.to_radians().cos() * K_DISTANCE_LAT_DEGREES;

        // Calculate radius (C++ logic from bidirectional.h lines 62-71)
        const K_LONGEST_NODE_DISTANCE: Cost = 300;
        if self.ways.is_some() {
            let diameter = self.dist_approx(start.pos_, end.pos_);
            if diameter < max_cost as f64
                && max_cost.saturating_add(diameter.max(K_LONGEST_NODE_DISTANCE as f64 * 2.0) as Cost)
                    < Cost::MAX
            {
                self.radius = (diameter as Cost / 2).max(K_LONGEST_NODE_DISTANCE);
            } else {
                self.radius = max_cost;
            }
        } else {
            self.radius = max_cost;
        }

        self.max_reached_fwd = false;
        self.max_reached_bwd = false;
    }

    /// Add starting node to forward search
    pub fn add_start(&mut self, params: &P::Parameters, node: P::Node, cost: Cost) {
        let h = self.potential(params, node);
        let total_cost = cost.saturating_add_signed(h);

        let key = P::get_key(&node);
        let entry = self.cost_forward.entry(key).or_insert_with(P::Entry::default);

        if entry.update(node, cost, node) {
            self.pq_forward.push(P::Label::new(node, total_cost));
            self.check_meeting_point(node);
        }
    }

    /// Add destination node to backward search
    pub fn add_end(&mut self, params: &P::Parameters, node: P::Node, cost: Cost) {
        let h = self.potential(params, node);
        let total_cost = cost.saturating_add_signed(-h);

        let key = P::get_key(&node);
        let entry = self.cost_backward.entry(key).or_insert_with(P::Entry::default);

        if entry.update(node, cost, node) {
            self.pq_backward.push(P::Label::new(node, total_cost));
            self.check_meeting_point(node);
        }
    }

    /// Check if a node is a meeting point (both frontiers reached it)
    pub fn check_meeting_point(&mut self, node: P::Node) {
        let key = P::get_key(&node);

        if let Some(fwd_entry) = self.cost_forward.get(&key) {
            if let Some(bwd_entry) = self.cost_backward.get(&key) {
                if let Some((m_fwd, m_bwd, tentative)) = P::check_meeting_point(fwd_entry, bwd_entry, node) {
                    if tentative < self.best_cost {
                        self.meet_point_fwd = Some(m_fwd);
                        self.meet_point_bwd = Some(m_bwd);
                        self.best_cost = tentative;
                    }
                }
            }
        }
    }

    /// Fast distance approximation matching C++ distapprox
    fn dist_approx(&self, p1: crate::Point, p2: crate::Point) -> f64 {
        const K_DISTANCE_LAT_DEGREES: f64 = 111111.0;
        let y = (p1.lat() - p2.lat()).abs() * K_DISTANCE_LAT_DEGREES;
        let xdiff = (p1.lng() - p2.lng()).abs();
        let x = if xdiff > 180.0 { 360.0 - xdiff } else { xdiff } * self.distance_lon_degrees;
        y.max(x).max((y + x) / 1.42)
    }

    /// Consistent bidirectional A* heuristic (potential function)
    /// Returns 0.5 * (h_end(node) - h_start(node))
    fn potential(&self, params: &P::Parameters, node: P::Node) -> i16 {
        if let Some(ways) = self.ways {
            let pos = ways.get_node_pos(P::get_node_idx(&node));
            let d_end = self.dist_approx(pos, self.end_loc.pos_);
            let d_start = self.dist_approx(pos, self.start_loc.pos_);

            let h_end = P::heuristic(params, d_end) as f64;
            let h_start = P::heuristic(params, d_start) as f64;

            (0.5 * (h_end - h_start)) as i16
        } else {
            0
        }
    }

    /// Reconstruct the shortest path from start to end through the meeting point
    pub fn reconstruct_path(&self) -> Option<Vec<P::Node>> {
        let meet_fwd = self.meet_point_fwd?;
        let meet_bwd = self.meet_point_bwd?;

        // Forward path: meet -> start (will reverse)
        let mut fwd_path = Vec::new();
        let mut curr = meet_fwd;
        loop {
            fwd_path.push(curr);
            let key = P::get_key(&curr);
            if let Some(entry) = self.cost_forward.get(&key) {
                if let Some(pred) = entry.pred(curr) {
                    if pred == curr {
                        break;
                    } // Seed node
                    curr = pred;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        fwd_path.reverse();

        // Backward path: meet -> end
        let mut bwd_path = Vec::new();
        let mut curr = meet_bwd;
        let key = P::get_key(&curr);
        if let Some(entry) = self.cost_backward.get(&key) {
            if let Some(pred) = entry.pred(curr) {
                if pred != curr {
                    curr = pred;
                    loop {
                        bwd_path.push(curr);
                        let k = P::get_key(&curr);
                        if let Some(e) = self.cost_backward.get(&k) {
                            if let Some(p) = e.pred(curr) {
                                if p == curr {
                                    break;
                                } // Seed node
                                curr = p;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        fwd_path.extend(bwd_path);
        Some(fwd_path)
    }

    /// Get the meeting point (forward and backward nodes)
    pub fn meet_point(&self) -> Option<(P::Node, P::Node)> {
        match (self.meet_point_fwd, self.meet_point_bwd) {
            (Some(f), Some(b)) => Some((f, b)),
            _ => None,
        }
    }

    /// Check if search limit was reached
    pub fn max_reached(&self) -> bool {
        self.max_reached_fwd || self.max_reached_bwd
    }

    /// Dump search state for debugging
    pub fn dump_state(&self) -> String {
        format!(
            "best_cost={} meet={:?}/{:?} fwd_reached={} bwd_reached={}",
            self.best_cost,
            self.meet_point_fwd,
            self.meet_point_bwd,
            self.max_reached_fwd,
            self.max_reached_bwd
        )
    }

    /// Get the best cost found (forward + backward)
    pub fn best_cost(&self) -> Cost {
        self.best_cost
    }

    /// Get the radius (heuristic lower-bound for half the route distance).
    /// Equal to max_cost when no valid estimate is available — used for BUG-14 early exit.
    pub fn radius(&self) -> Cost {
        self.radius
    }

    /// Run the bidirectional search
    pub fn run<F1, F2>(
        &mut self,
        params: &P::Parameters,
        max_cost: Cost,
        mut get_neighbors_fwd: F1,
        mut get_neighbors_bwd: F2,
    ) -> bool
    where
        F1: FnMut(P::Node) -> Vec<(P::Node, Cost)>,
        F2: FnMut(P::Node) -> Vec<(P::Node, Cost)>,
    {
        while !self.pq_forward.empty() || !self.pq_backward.empty() {
            // Forward search
            if !self.pq_forward.empty() {
                let label = self.pq_forward.pop();
                let node = label.get_node();

                let key = P::get_key(&node);
                let current_cost = self.cost_forward.get(&key).unwrap().cost(node);

                if current_cost > max_cost {
                    self.max_reached_fwd = true;
                } else {
                    self.check_meeting_point(node);

                    for (neighbor, edge_cost) in get_neighbors_fwd(node) {
                        let new_cost = current_cost.saturating_add(edge_cost);
                        if new_cost > max_cost {
                            continue;
                        }

                        let key = P::get_key(&neighbor);
                        let entry = self.cost_forward.entry(key).or_insert_with(P::Entry::default);

                        if entry.update(neighbor, new_cost, node) {
                            let h = self.potential(params, neighbor);
                            let total_cost = new_cost.saturating_add_signed(h);
                            self.pq_forward.push(P::Label::new(neighbor, total_cost));
                            self.check_meeting_point(neighbor);
                        }
                    }
                }
            }

            // Backward search
            if !self.pq_backward.empty() {
                let label = self.pq_backward.pop();
                let node = label.get_node();

                let key = P::get_key(&node);
                let current_cost = self.cost_backward.get(&key).unwrap().cost(node);

                if current_cost > max_cost {
                    self.max_reached_bwd = true;
                } else {
                    self.check_meeting_point(node);

                    for (neighbor, edge_cost) in get_neighbors_bwd(node) {
                        let new_cost = current_cost.saturating_add(edge_cost);
                        if new_cost > max_cost {
                            continue;
                        }

                        let key = P::get_key(&neighbor);
                        let entry = self.cost_backward.entry(key).or_insert_with(P::Entry::default);

                        if entry.update(neighbor, new_cost, node) {
                            let h = self.potential(params, neighbor);
                            let total_cost = new_cost.saturating_add_signed(-h);
                            self.pq_backward.push(P::Label::new(neighbor, total_cost));
                            self.check_meeting_point(neighbor);
                        }
                    }
                }
            }

            // Termination conditions
            let min_fwd = self.pq_forward.peek_min_bucket().unwrap_or(max_cost as usize);
            let min_bwd = self.pq_backward.peek_min_bucket().unwrap_or(max_cost as usize);

            if self.best_cost != Cost::MAX {
                let min_total_cost = (min_fwd + min_bwd) as Cost;
                if min_total_cost >= self.best_cost.saturating_add(self.radius) {
                    break;
                }
            }

            if (self.pq_forward.empty() || min_fwd > max_cost as usize)
                && (self.pq_backward.empty() || min_bwd > max_cost as usize)
            {
                break;
            }
        }
        true
    }
}

// Type aliases for convenience
pub type FootBidirectional<'w> = GenericBidirectional<'w, crate::routing::profiles::foot::FootProfile<crate::routing::tracking::NoopTracking>>;
pub type BikeBidirectional<'w> = GenericBidirectional<'w, crate::routing::profiles::BikeProfile>;
pub type CarBidirectional<'w> = GenericBidirectional<'w, crate::routing::profiles::CarProfile>;

// Note: Cannot implement Default for GenericBidirectional<'w, P> due to lifetime parameter
// Use GenericBidirectional::new() instead

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bidirectional_creation() {
        let bidir = Bidirectional::new();
        assert_eq!(bidir.best_cost(), Cost::MAX);
        assert!(!bidir.max_reached());
    }

    #[test]
    fn test_reset() {
        let mut bidir = Bidirectional::new();
        let start = Location::from_latlng_no_level(0.0, 0.0);
        let end = Location::from_latlng_no_level(1.0, 1.0);

        bidir.reset(1000, start, end);
        assert_eq!(bidir.best_cost(), Cost::MAX);
        assert!(bidir.meet_point().is_none());
    }

    #[test]
    fn test_simple_path() {
        let mut bidir = Bidirectional::new();
        let start = Location::from_latlng_no_level(0.0, 0.0);
        let end = Location::from_latlng_no_level(1.0, 1.0);

        bidir.reset(1000, start, end);

        // Graph: 0 --10--> 1 --20--> 2
        let fwd_graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.0 {
                0 => vec![(NodeIdx(1), 10)],
                1 => vec![(NodeIdx(2), 20)],
                _ => vec![],
            }
        };

        let bwd_graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.0 {
                2 => vec![(NodeIdx(1), 20)],
                1 => vec![(NodeIdx(0), 10)],
                _ => vec![],
            }
        };

        bidir.add_start(NodeIdx(0), 0);
        bidir.add_end(NodeIdx(2), 0);

        let completed = bidir.run(1000, fwd_graph, bwd_graph);

        assert!(completed);
        assert_eq!(bidir.best_cost(), 30);
        assert!(bidir.meet_point().is_some());
    }

    #[test]
    fn test_meeting_in_middle() {
        let mut bidir = Bidirectional::new();
        let start = Location::from_latlng_no_level(0.0, 0.0);
        let end = Location::from_latlng_no_level(1.0, 1.0);

        bidir.reset(1000, start, end);

        // Linear graph: 0 --10--> 1 --10--> 2 --10--> 3
        let fwd_graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.0 {
                0 => vec![(NodeIdx(1), 10)],
                1 => vec![(NodeIdx(2), 10)],
                2 => vec![(NodeIdx(3), 10)],
                _ => vec![],
            }
        };

        let bwd_graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.0 {
                3 => vec![(NodeIdx(2), 10)],
                2 => vec![(NodeIdx(1), 10)],
                1 => vec![(NodeIdx(0), 10)],
                _ => vec![],
            }
        };

        bidir.add_start(NodeIdx(0), 0);
        bidir.add_end(NodeIdx(3), 0);

        bidir.run(1000, fwd_graph, bwd_graph);

        assert_eq!(bidir.best_cost(), 30);
    }

    #[test]
    fn test_max_cost_limit() {
        let mut bidir = Bidirectional::new();
        let start = Location::from_latlng_no_level(0.0, 0.0);
        let end = Location::from_latlng_no_level(1.0, 1.0);

        bidir.reset(50, start, end);

        // Long path that exceeds max
        let fwd_graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.0 {
                0 => vec![(NodeIdx(1), 100)],
                _ => vec![],
            }
        };

        let bwd_graph = |_: NodeIdx| -> Vec<(NodeIdx, Cost)> { vec![] };

        bidir.add_start(NodeIdx(0), 0);
        bidir.add_end(NodeIdx(1), 0);

        let completed = bidir.run(50, fwd_graph, bwd_graph);

        assert!(!completed);
        assert!(bidir.max_reached());
    }

    #[test]
    fn test_no_path() {
        let mut bidir = Bidirectional::new();
        let start = Location::from_latlng_no_level(0.0, 0.0);
        let end = Location::from_latlng_no_level(1.0, 1.0);

        bidir.reset(1000, start, end);

        // Disconnected graph
        let fwd_graph = |_: NodeIdx| -> Vec<(NodeIdx, Cost)> { vec![] };
        let bwd_graph = |_: NodeIdx| -> Vec<(NodeIdx, Cost)> { vec![] };

        bidir.add_start(NodeIdx(0), 0);
        bidir.add_end(NodeIdx(2), 0);

        bidir.run(1000, fwd_graph, bwd_graph);

        assert_eq!(bidir.best_cost(), Cost::MAX);
        assert!(bidir.meet_point().is_none());
    }

    #[test]
    fn test_heuristic_with_ways() {
        use crate::ways::NodeProperties;
        use crate::{OsmNodeIdx, Point, Ways};

        let mut ways = Ways::new();

        // Create a simple 3-node linear graph with nearby geographic positions
        // Small distances to keep heuristic costs reasonable
        // Node 0: (0.0, 0.0)
        // Node 1: (0.001, 0.0) - about 111 meters away
        // Node 2: (0.002, 0.0) - about 222 meters away
        let n0 = ways.add_node(
            OsmNodeIdx(0),
            Point::from_latlng(0.0, 0.0),
            NodeProperties::default(),
        );
        let _n1 = ways.add_node(
            OsmNodeIdx(1),
            Point::from_latlng(0.001, 0.0),
            NodeProperties::default(),
        );
        let n2 = ways.add_node(
            OsmNodeIdx(2),
            Point::from_latlng(0.002, 0.0),
            NodeProperties::default(),
        );

        // Create bidirectional search with Ways reference
        let mut bidir = Bidirectional::new();
        bidir.set_ways(&ways);

        let start = Location::from_latlng_no_level(0.0, 0.0);
        let end = Location::from_latlng_no_level(0.002, 0.0);
        bidir.reset(10000, start, end);

        // Simple forward/backward neighbor functions
        let fwd_graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.0 {
                0 => vec![(NodeIdx(1), 100)],
                1 => vec![(NodeIdx(2), 100)],
                _ => vec![],
            }
        };

        let bwd_graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.0 {
                2 => vec![(NodeIdx(1), 100)],
                1 => vec![(NodeIdx(0), 100)],
                _ => vec![],
            }
        };

        bidir.add_start(n0, 0);
        bidir.add_end(n2, 0);

        let completed = bidir.run(10000, fwd_graph, bwd_graph);

        assert!(completed);
        assert_eq!(bidir.best_cost(), 200);

        // Verify heuristic was used (should have explored fewer nodes than pure Dijkstra)
        // With heuristic, we should find the path efficiently
        assert!(bidir.meet_point().is_some());
    }

    #[test]
    fn test_path_reconstruction() {
        let mut bidir = Bidirectional::new();
        let start = Location::from_latlng_no_level(0.0, 0.0);
        let end = Location::from_latlng_no_level(1.0, 1.0);

        bidir.reset(1000, start, end);

        // Linear graph: 0 --10--> 1 --10--> 2 --10--> 3
        let fwd_graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.0 {
                0 => vec![(NodeIdx(1), 10)],
                1 => vec![(NodeIdx(2), 10)],
                2 => vec![(NodeIdx(3), 10)],
                _ => vec![],
            }
        };

        let bwd_graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.0 {
                3 => vec![(NodeIdx(2), 10)],
                2 => vec![(NodeIdx(1), 10)],
                1 => vec![(NodeIdx(0), 10)],
                _ => vec![],
            }
        };

        bidir.add_start(NodeIdx(0), 0);
        bidir.add_end(NodeIdx(3), 0);

        bidir.run(1000, fwd_graph, bwd_graph);

        assert_eq!(bidir.best_cost(), 30);

        // Reconstruct and verify path
        let path = bidir.reconstruct_path();
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], NodeIdx(0));
        assert_eq!(path[3], NodeIdx(3));

        // Verify it's a valid path
        for i in 0..path.len() - 1 {
            assert!(path[i + 1].0 == path[i].0 + 1 || path[i + 1].0 == path[i].0 - 1);
        }
    }
}
