//! Traits for bidirectional routing profiles
//!
//! This module defines the trait system that allows bidirectional A* search
//! to be generic over different routing profiles (foot, bike, car, etc.).
//!
//! Each profile defines its own node representation, cost storage strategy,
//! and heuristic calculation, enabling efficient bidirectional search for
//! different transportation modes.

use std::fmt::Debug;
use std::hash::Hash;

use crate::types::{Cost, Direction, NodeIdx};
use crate::{ElevationStorage, Ways};

/// Core trait that all routing profiles must implement for bidirectional search
///
/// This trait mirrors the C++ profile template system, allowing each profile
/// to define its own node representation and storage strategy.
///
/// C++ equivalent: `template <typename P> struct bidirectional` requires `P` to have:
/// - `node`, `key`, `label`, `entry` types
/// - `heuristic()` method
/// - `adjacent()` method
pub trait BidirectionalProfile: Sized {
    /// Node representation (may include state like direction, level, way)
    ///
    /// C++ equivalent: `typename P::node`
    type Node: Copy + Eq + Hash + Debug;

    /// Map key type (simplified node representation for hashmap lookups)
    /// Often just `NodeIdx`, but foot uses full `Node` including level
    ///
    /// C++ equivalent: `typename P::key`
    type Key: Copy + Eq + Hash;

    /// Priority queue label (node + cost)
    ///
    /// C++ equivalent: `typename P::label`
    type Label: BidirectionalLabel<Node = Self::Node>;

    /// Cost/predecessor entry storage strategy
    ///
    /// C++ equivalent: `typename P::entry`
    type Entry: BidirectionalEntry<Node = Self::Node>;

    /// Parameters for routing (cost calculations, speed, etc.)
    type Parameters: Clone;

    /// Extract the key from a node (for hashmap lookups)
    ///
    /// C++: `node.get_key()`
    fn get_key(node: &Self::Node) -> Self::Key;

    /// Get the underlying NodeIdx from a node (for position lookups)
    ///
    /// This is needed for heuristic calculations that require geographic positions.
    fn get_node_idx(node: &Self::Node) -> NodeIdx;

    /// Calculate the heuristic distance (in cost units)
    ///
    /// C++: `P::heuristic(params, dist)`
    /// From bidirectional.h lines 75-85
    fn heuristic(params: &Self::Parameters, distance_meters: f64) -> Cost;

    /// Generate adjacent nodes in the given search direction
    ///
    /// C++ equivalent: `P::template adjacent<SearchDir, WithBlocked>`
    fn adjacent<F>(
        params: &Self::Parameters,
        ways: &Ways,
        elevations: Option<&ElevationStorage>,
        node: Self::Node,
        search_dir: Direction,
        callback: F,
    ) where
        F: FnMut(Self::Node, Cost);

    /// Check for a meeting point between forward and backward search.
    ///
    /// Given entries from both frontiers for the same key, and the node that was
    /// just reached, find the best meeting point (if any).
    fn check_meeting_point(
        fwd_entry: &Self::Entry,
        bwd_entry: &Self::Entry,
        node: Self::Node,
    ) -> Option<(Self::Node, Self::Node, Cost)>;
}

/// Trait for labels used in bidirectional search
///
/// Labels combine a node with its total cost (actual + heuristic) for
/// priority queue ordering.
///
/// C++ equivalent: `typename P::label`
pub trait BidirectionalLabel: Copy + Debug {
    type Node: Copy + Eq + Hash;

    fn new(node: Self::Node, cost: Cost) -> Self;
    fn get_node(&self) -> Self::Node;
    fn cost(&self) -> Cost;
}

/// Trait for entry storage (cost + predecessor tracking)
///
/// Each profile has different storage requirements:
/// - Foot: Single cost (no directional variants)
/// - Bike: 2 costs (forward/backward directions)
/// - Car: 32 costs (16 ways × 2 directions)
///
/// C++ equivalent: `typename P::entry`
pub trait BidirectionalEntry: Default + Clone {
    type Node: Copy + Eq + Hash;

    /// Get the predecessor of a node
    ///
    /// C++: `entry.pred(node)`
    fn pred(&self, node: Self::Node) -> Option<Self::Node>;

    /// Get the cost to reach a node
    ///
    /// C++: `entry.cost(node)`
    fn cost(&self, node: Self::Node) -> Cost;

    /// Update the entry if the new cost is better
    /// Returns true if updated
    ///
    /// C++: `entry.update(label, node, cost, pred)`
    fn update(&mut self, node: Self::Node, cost: Cost, pred: Self::Node) -> bool;
}