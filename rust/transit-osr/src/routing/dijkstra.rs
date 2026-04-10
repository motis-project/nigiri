//! Dijkstra's shortest path algorithm implementation.
//!
//! Complete implementation matching C++ osr/include/osr/routing/dijkstra.h
//!
//! This module provides a generic Dijkstra implementation with:
//! - Label-based state tracking
//! - Entry-based predecessor/cost storage
//! - Dial queue for efficient priority queue operations
//! - Path reconstruction support
//! - Max cost limit handling
//!
//! # C++ Equivalent
//! ```cpp
//! template <Profile P>
//! struct dijkstra {
//!   using label = typename P::label;
//!   using entry = typename P::entry;
//!   dial<label, get_bucket> pq_;
//!   ankerl::unordered_dense::map<key, entry, hash> cost_;
//!   bool max_reached_;
//! };
//! ```

use std::hash::Hash;

use crate::routing::Dial;
use crate::types::{Cost, NodeIdx};
use ahash::AHashMap;

/// Label for Dijkstra search - combines a node with its cost
///
/// In the C++ version, this is profile-specific (typename P::label).
/// This generic version works with any node type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Label<N = NodeIdx>
where
    N: Copy + Eq + Hash,
{
    pub node: N,
    pub cost: Cost,
}

impl<N> Label<N>
where
    N: Copy + Eq + Hash,
{
    pub fn new(node: N, cost: Cost) -> Self {
        Self { node, cost }
    }

    pub fn get_node(&self) -> N {
        self.node
    }

    pub fn get_cost(&self) -> Cost {
        self.cost
    }
}

/// Entry for tracking predecessor and cost in Dijkstra search
///
/// Stores the best known cost to reach a node and its predecessor
/// for path reconstruction.
///
/// # C++ Equivalent
/// ```cpp
/// struct entry {
///   std::optional<node> pred(node) const noexcept;
///   cost_t cost(node) const noexcept;
///   bool update(label const&, node, cost_t, node) noexcept;
///   void write(node, path&) const;
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Entry<N = NodeIdx>
where
    N: Copy + Eq + Hash,
{
    /// Predecessor node (None if this is a start node)
    pub pred: Option<N>,
    /// Best cost to reach this node
    pub cost: Cost,
}

impl<N> Entry<N>
where
    N: Copy + Eq + Hash,
{
    /// Create a new entry with infinite cost
    pub fn new() -> Self {
        Self {
            pred: None,
            cost: Cost::MAX,
        }
    }

    /// Update the entry if the new cost is better
    ///
    /// # Returns
    /// `true` if the entry was updated (new cost is better)
    pub fn update(&mut self, new_cost: Cost, predecessor: Option<N>) -> bool {
        if new_cost < self.cost {
            self.cost = new_cost;
            self.pred = predecessor;
            true
        } else {
            false
        }
    }

    /// Get the cost for this entry
    pub fn get_cost(&self) -> Cost {
        self.cost
    }

    /// Get the predecessor for this entry
    pub fn get_pred(&self) -> Option<N> {
        self.pred
    }
}

impl<N> Default for Entry<N>
where
    N: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Bucket function for Dial queue - extracts cost from label
fn get_bucket<N>(label: &Label<N>) -> usize
where
    N: Copy + Eq + Hash,
{
    label.cost as usize
}

/// A complete Dijkstra shortest path search implementation.
///
/// Generic over node type N (typically NodeIdx or profile-specific node).
/// Maintains priority queue, cost/predecessor map, and search state.
///
/// # Type Parameters
/// * `N` - Node type (must be Copy + Eq + Hash)
///
/// # C++ Equivalent
/// ```cpp
/// template <Profile P>
/// struct dijkstra {
///   void reset(cost_t const max);
///   void add_start(ways const& w, label const l);
///   cost_t get_cost(node const n) const;
///   bool run(/*...*/);
///   
///   dial<label, get_bucket> pq_;
///   map<key, entry, hash> cost_;
///   bool max_reached_;
/// };
/// ```
pub struct Dijkstra<N = NodeIdx>
where
    N: Copy + Eq + Hash,
{
    /// Dial priority queue for efficient cost-based extraction
    pq: Dial<Label<N>, fn(&Label<N>) -> usize>,

    /// Map from node to entry (cost + predecessor) - using AHashMap for performance
    cost: AHashMap<N, Entry<N>>,

    /// Flag indicating if the maximum cost limit was reached
    max_reached: bool,
}

impl<N> Dijkstra<N>
where
    N: Copy + Eq + Hash,
{
    /// Create a new Dijkstra search instance.
    pub fn new() -> Self {
        Self {
            pq: Dial::new(get_bucket),
            cost: AHashMap::new(),
            max_reached: false,
        }
    }

    /// Reset the search state for a new query.
    ///
    /// # Arguments
    /// * `max_cost` - Maximum cost limit for the search
    ///
    /// # C++ Equivalent
    /// ```cpp
    /// void reset(cost_t const max) {
    ///   pq_.clear();
    ///   pq_.n_buckets(max + 1U);
    ///   cost_.clear();
    ///   max_reached_ = false;
    /// }
    /// ```
    pub fn reset(&mut self, max_cost: Cost) {
        self.pq.clear();
        self.pq.set_n_buckets(max_cost as usize + 1);
        self.cost.clear();
        self.max_reached = false;
    }

    /// Add a starting node to the search.
    ///
    /// # Arguments
    /// * `node` - Starting node
    /// * `cost` - Initial cost (usually 0)
    ///
    /// # C++ Equivalent
    /// ```cpp
    /// void add_start(ways const& w, label const l) {
    ///   if (cost_[l.get_node().get_key()].update(l, l.get_node(), l.cost(), node::invalid())) {
    ///     pq_.push(l);
    ///   }
    /// }
    /// ```
    pub fn add_start(&mut self, node: N, cost: Cost) {
        let entry = self.cost.entry(node).or_insert_with(Entry::new);
        if entry.update(cost, None) {
            self.pq.push(Label::new(node, cost));
        }
    }

    /// Get the best cost to reach a node.
    ///
    /// # Arguments
    /// * `node` - Node to query
    ///
    /// # Returns
    /// Best cost to reach the node, or `Cost::MAX` if not yet reached
    ///
    /// # C++ Equivalent
    /// ```cpp
    /// cost_t get_cost(node const n) const {
    ///   auto const it = cost_.find(n.get_key());
    ///   return it != end(cost_) ? it->second.cost(n) : kInfeasible;
    /// }
    /// ```
    pub fn get_cost(&self, node: N) -> Cost {
        self.cost
            .get(&node)
            .map(|e| e.get_cost())
            .unwrap_or(Cost::MAX)
    }

    /// Get the predecessor of a node in the shortest path tree.
    ///
    /// # Arguments
    /// * `node` - Node to query
    ///
    /// # Returns
    /// Predecessor node if found, None otherwise
    pub fn get_predecessor(&self, node: N) -> Option<N> {
        self.cost.get(&node).and_then(|e| e.get_pred())
    }

    /// Check if the maximum cost limit was reached during search.
    pub fn max_reached(&self) -> bool {
        self.max_reached
    }

    /// Check if the priority queue is empty.
    pub fn is_empty(&self) -> bool {
        self.pq.empty()
    }

    /// Get the number of nodes explored.
    pub fn num_explored(&self) -> usize {
        self.cost.len()
    }

    /// Reconstruct the path from start to target.
    ///
    /// # Arguments
    /// * `target` - Target node
    ///
    /// # Returns
    /// Path as a vector of nodes from start to target, or empty if unreachable
    pub fn reconstruct_path(&self, target: N) -> Vec<N> {
        // Pre-allocate reasonable capacity (typical paths: 10-100 nodes)
        let mut path = Vec::with_capacity(64);
        let mut current = Some(target);

        while let Some(node) = current {
            path.push(node);
            current = self.get_predecessor(node);
        }

        path.reverse();
        path
    }

    /// Run Dijkstra's algorithm with a neighbor function.
    ///
    /// This is the main search loop. Extracts labels from the priority queue,
    /// explores neighbors, and updates costs when better paths are found.
    ///
    /// # Arguments
    /// * `max_cost` - Maximum cost limit
    /// * `get_neighbors` - Function that returns (neighbor_node, edge_cost) pairs
    ///
    /// # Returns
    /// `true` if search completed without hitting max cost limit
    ///
    /// # C++ Equivalent
    /// ```cpp
    /// template <direction SearchDir, bool WithBlocked>
    /// bool run(P::parameters const& params, /*...*/) {
    ///   while (!pq_.empty()) {
    ///     auto l = pq_.pop();
    ///     if (get_cost(l.get_node()) < l.cost()) { continue; }
    ///     P::template adjacent<SearchDir, WithBlocked>(/*...*/, [&](/*...*/) {
    ///       auto const total = l.cost() + cost;
    ///       if (total >= max) { max_reached_ = true; return; }
    ///       if (total < max && cost_[neighbor.get_key()].update(/*...*/)) {
    ///         auto next = label{neighbor, static_cast<cost_t>(total)};
    ///         pq_.push(std::move(next));
    ///       }
    ///     });
    ///   }
    ///   return !max_reached_;
    /// }
    /// ```
    pub fn run<F>(&mut self, max_cost: Cost, mut get_neighbors: F) -> bool
    where
        F: FnMut(N) -> Vec<(N, Cost)>,
    {
        while !self.pq.empty() {
            let label = self.pq.pop();

            // Skip if we've found a better path to this node
            // (label may be outdated if we pushed multiple times)
            if self.get_cost(label.node) < label.cost {
                continue;
            }

            // Explore neighbors
            for (neighbor, edge_cost) in get_neighbors(label.node) {
                let total_cost = label.cost.saturating_add(edge_cost);

                // Check max cost limit
                if total_cost >= max_cost {
                    self.max_reached = true;
                    continue;
                }

                // Update if better path found
                let entry = self.cost.entry(neighbor).or_insert_with(Entry::new);
                if entry.update(total_cost, Some(label.node)) {
                    self.pq.push(Label::new(neighbor, total_cost));
                }
            }
        }

        !self.max_reached
    }

    /// Run Dijkstra until a target is reached or search exhausted.
    ///
    /// # Arguments
    /// * `target` - Target node to find
    /// * `max_cost` - Maximum cost limit
    /// * `get_neighbors` - Function that returns (neighbor_node, edge_cost) pairs
    ///
    /// # Returns
    /// `Some(cost)` if target was reached, None otherwise
    pub fn run_to_target<F>(
        &mut self,
        target: N,
        max_cost: Cost,
        mut get_neighbors: F,
    ) -> Option<Cost>
    where
        F: FnMut(N) -> Vec<(N, Cost)>,
    {
        while !self.pq.empty() {
            let label = self.pq.pop();

            // Found target!
            if label.node == target {
                return Some(label.cost);
            }

            // Skip if we've found a better path to this node
            if self.get_cost(label.node) < label.cost {
                continue;
            }

            // Explore neighbors
            for (neighbor, edge_cost) in get_neighbors(label.node) {
                let total_cost = label.cost.saturating_add(edge_cost);

                // Check max cost limit
                if total_cost >= max_cost {
                    self.max_reached = true;
                    continue;
                }

                // Update if better path found
                let entry = self.cost.entry(neighbor).or_insert_with(Entry::new);
                if entry.update(total_cost, Some(label.node)) {
                    self.pq.push(Label::new(neighbor, total_cost));
                }
            }
        }

        // Target not reached
        None
    }
}

impl<N> Default for Dijkstra<N>
where
    N: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_creation() {
        let label = Label::new(NodeIdx::new(42), 100);
        assert_eq!(label.get_node(), NodeIdx::new(42));
        assert_eq!(label.get_cost(), 100);
    }

    #[test]
    fn test_entry_creation() {
        let entry: Entry<NodeIdx> = Entry::new();
        assert_eq!(entry.get_cost(), Cost::MAX);
        assert_eq!(entry.get_pred(), None);
    }

    #[test]
    fn test_entry_update() {
        let mut entry: Entry<NodeIdx> = Entry::new();

        // First update should succeed
        assert!(entry.update(100, Some(NodeIdx::new(1))));
        assert_eq!(entry.get_cost(), 100);
        assert_eq!(entry.get_pred(), Some(NodeIdx::new(1)));

        // Better cost should update
        assert!(entry.update(50, Some(NodeIdx::new(2))));
        assert_eq!(entry.get_cost(), 50);
        assert_eq!(entry.get_pred(), Some(NodeIdx::new(2)));

        // Worse cost should not update
        assert!(!entry.update(75, Some(NodeIdx::new(3))));
        assert_eq!(entry.get_cost(), 50);
        assert_eq!(entry.get_pred(), Some(NodeIdx::new(2)));
    }

    #[test]
    fn test_dijkstra_creation() {
        let dijkstra: Dijkstra<NodeIdx> = Dijkstra::new();
        assert!(!dijkstra.max_reached());
        assert!(dijkstra.is_empty());
        assert_eq!(dijkstra.num_explored(), 0);
    }

    #[test]
    fn test_add_start_and_get_cost() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);

        let start = NodeIdx::new(0);
        dijkstra.add_start(start, 0);

        assert_eq!(dijkstra.get_cost(start), 0);
        assert_eq!(dijkstra.get_cost(NodeIdx::new(999)), Cost::MAX);
    }

    #[test]
    fn test_simple_path() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);

        // Graph: 0 --10--> 1 --20--> 2
        let graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.value() {
                0 => vec![(NodeIdx::new(1), 10)],
                1 => vec![(NodeIdx::new(2), 20)],
                _ => vec![],
            }
        };

        dijkstra.add_start(NodeIdx::new(0), 0);
        let completed = dijkstra.run(1000, graph);

        assert!(completed);
        assert_eq!(dijkstra.get_cost(NodeIdx::new(0)), 0);
        assert_eq!(dijkstra.get_cost(NodeIdx::new(1)), 10);
        assert_eq!(dijkstra.get_cost(NodeIdx::new(2)), 30);
    }

    #[test]
    fn test_multiple_paths() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);

        // Graph with multiple paths:
        //     10
        //  0 ---> 1
        //  |      |
        //  5      5
        //  v      v
        //  2 ---> 3
        //     10
        let graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.value() {
                0 => vec![(NodeIdx::new(1), 10), (NodeIdx::new(2), 5)],
                1 => vec![(NodeIdx::new(3), 5)],
                2 => vec![(NodeIdx::new(3), 10)],
                _ => vec![],
            }
        };

        dijkstra.add_start(NodeIdx::new(0), 0);
        dijkstra.run(1000, graph);

        // Best path to node 3 is 0->1->3 (cost 15) or 0->2->3 (cost 15)
        assert_eq!(dijkstra.get_cost(NodeIdx::new(3)), 15);
    }

    #[test]
    fn test_max_cost_limit() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(100);

        // Graph: 0 --10--> 1 --200--> 2 (edge exceeds max)
        let graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.value() {
                0 => vec![(NodeIdx::new(1), 10)],
                1 => vec![(NodeIdx::new(2), 200)],
                _ => vec![],
            }
        };

        dijkstra.add_start(NodeIdx::new(0), 0);
        let completed = dijkstra.run(100, graph);

        // Should not complete (max cost reached)
        assert!(!completed);
        assert!(dijkstra.max_reached());

        // Node 1 should be reachable, but not node 2
        assert_eq!(dijkstra.get_cost(NodeIdx::new(1)), 10);
        assert_eq!(dijkstra.get_cost(NodeIdx::new(2)), Cost::MAX);
    }

    #[test]
    fn test_reset() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);
        dijkstra.add_start(NodeIdx::new(0), 0);

        assert_eq!(dijkstra.num_explored(), 1);

        // Reset and verify state is cleared
        dijkstra.reset(1000);
        assert_eq!(dijkstra.get_cost(NodeIdx::new(0)), Cost::MAX);
        assert!(!dijkstra.max_reached());
        assert_eq!(dijkstra.num_explored(), 0);
    }

    #[test]
    fn test_predecessor_tracking() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);

        // Graph: 0 -> 1 -> 2
        let graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.value() {
                0 => vec![(NodeIdx::new(1), 10)],
                1 => vec![(NodeIdx::new(2), 20)],
                _ => vec![],
            }
        };

        dijkstra.add_start(NodeIdx::new(0), 0);
        dijkstra.run(1000, graph);

        // Check predecessors
        assert_eq!(dijkstra.get_predecessor(NodeIdx::new(0)), None);
        assert_eq!(
            dijkstra.get_predecessor(NodeIdx::new(1)),
            Some(NodeIdx::new(0))
        );
        assert_eq!(
            dijkstra.get_predecessor(NodeIdx::new(2)),
            Some(NodeIdx::new(1))
        );
    }

    #[test]
    fn test_path_reconstruction() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);

        // Graph: 0 -> 1 -> 2 -> 3
        let graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.value() {
                0 => vec![(NodeIdx::new(1), 10)],
                1 => vec![(NodeIdx::new(2), 20)],
                2 => vec![(NodeIdx::new(3), 30)],
                _ => vec![],
            }
        };

        dijkstra.add_start(NodeIdx::new(0), 0);
        dijkstra.run(1000, graph);

        let path = dijkstra.reconstruct_path(NodeIdx::new(3));
        assert_eq!(
            path,
            vec![
                NodeIdx::new(0),
                NodeIdx::new(1),
                NodeIdx::new(2),
                NodeIdx::new(3)
            ]
        );
    }

    #[test]
    fn test_run_to_target() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);

        // Graph: 0 -> 1 -> 2 -> 3
        let graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.value() {
                0 => vec![(NodeIdx::new(1), 10)],
                1 => vec![(NodeIdx::new(2), 20)],
                2 => vec![(NodeIdx::new(3), 30)],
                _ => vec![],
            }
        };

        dijkstra.add_start(NodeIdx::new(0), 0);
        let cost = dijkstra.run_to_target(NodeIdx::new(2), 1000, graph);

        assert_eq!(cost, Some(30)); // 0->1->2 = 10+20=30
                                    // Should not have explored node 3
        assert_eq!(dijkstra.get_cost(NodeIdx::new(3)), Cost::MAX);
    }

    #[test]
    fn test_multiple_starts() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);

        // Graph:
        //  0 --5--> 2
        //  1 --3--> 2
        let graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.value() {
                0 => vec![(NodeIdx::new(2), 5)],
                1 => vec![(NodeIdx::new(2), 3)],
                _ => vec![],
            }
        };

        // Add multiple starts
        dijkstra.add_start(NodeIdx::new(0), 0);
        dijkstra.add_start(NodeIdx::new(1), 0);
        dijkstra.run(1000, graph);

        // Should reach node 2 via node 1 (cost 3) since it's cheaper
        assert_eq!(dijkstra.get_cost(NodeIdx::new(2)), 3);
        assert_eq!(
            dijkstra.get_predecessor(NodeIdx::new(2)),
            Some(NodeIdx::new(1))
        );
    }

    #[test]
    fn test_unreachable_node() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);

        // Graph: 0 -> 1, but 2 is disconnected
        let graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.value() {
                0 => vec![(NodeIdx::new(1), 10)],
                _ => vec![],
            }
        };

        dijkstra.add_start(NodeIdx::new(0), 0);
        dijkstra.run(1000, graph);

        assert_eq!(dijkstra.get_cost(NodeIdx::new(2)), Cost::MAX);
        assert_eq!(dijkstra.get_predecessor(NodeIdx::new(2)), None);
    }

    #[test]
    fn test_diamond_graph() {
        let mut dijkstra = Dijkstra::new();
        dijkstra.reset(1000);

        // Diamond graph:
        //      10      20
        //   0 ---> 1 ---> 3
        //   |             ^
        //   5             5
        //   v             |
        //   2 ----------->
        //         10
        let graph = |node: NodeIdx| -> Vec<(NodeIdx, Cost)> {
            match node.value() {
                0 => vec![(NodeIdx::new(1), 10), (NodeIdx::new(2), 5)],
                1 => vec![(NodeIdx::new(3), 20)],
                2 => vec![(NodeIdx::new(3), 10)],
                _ => vec![],
            }
        };

        dijkstra.add_start(NodeIdx::new(0), 0);
        dijkstra.run(1000, graph);

        // Best path to 3: 0->2->3 (cost 15) vs 0->1->3 (cost 30)
        assert_eq!(dijkstra.get_cost(NodeIdx::new(3)), 15);
        assert_eq!(
            dijkstra.get_predecessor(NodeIdx::new(3)),
            Some(NodeIdx::new(2))
        );
    }
}
