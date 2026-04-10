//! Translation of osr/include/osr/routing/tracking.h
//!
//! Path tracking for routing (elevator usage, specific nodes, etc.).
//! Matches C++ implementation exactly.

use crate::routing::path::Path;
use crate::types::{NodeIdx, WayIdx};
use crate::ways::Ways;

/// Trait for tracking information during routing
///
/// This trait allows different routing profiles to track different
/// information about the path as it's being constructed.
///
/// # C++ Equivalent
/// Each tracking type has `write()` and `track()` methods matching this interface.
pub trait Tracking: Clone + Default {
    /// Write tracked information to the path
    fn write(&self, p: &mut Path);

    /// Track information from the previous label during routing
    ///
    /// # Arguments
    /// * `l` - Previous tracking state
    /// * `r` - Routing graph data (ways)
    /// * `w` - Way being traversed
    /// * `n` - Node being visited
    /// * `track` - Whether to actually track this step (for special nodes)
    fn track(&mut self, l: &Self, r: &Ways, w: WayIdx, n: NodeIdx, track: bool);
}

/// No-op tracking (does nothing)
///
/// # C++ Equivalent
/// ```cpp
/// struct noop_tracking {
///   void write(path&) const {}
///   void track(noop_tracking const&, ways::routing const&, way_idx_t, node_idx_t, bool) {}
/// };
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTracking;

impl NoopTracking {
    pub fn new() -> Self {
        Self
    }
}

impl Tracking for NoopTracking {
    fn write(&self, _p: &mut Path) {
        // No-op
    }

    fn track(&mut self, _l: &Self, _r: &Ways, _w: WayIdx, _n: NodeIdx, _track: bool) {
        // No-op
    }
}

/// Track elevator usage in a path
///
/// # C++ Equivalent
/// ```cpp
/// struct elevator_tracking {
///   void write(path& p) const { p.uses_elevator_ = uses_elevator_; }
///   void track(elevator_tracking const& l, ways::routing const& r, way_idx_t,
///              node_idx_t const n, bool) {
///     uses_elevator_ = l.uses_elevator_ || r.node_properties_[n].is_elevator();
///   }
///   bool uses_elevator_{false};
/// };
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ElevatorTracking {
    pub uses_elevator: bool,
}

impl ElevatorTracking {
    pub fn new() -> Self {
        Self {
            uses_elevator: false,
        }
    }
}

impl Tracking for ElevatorTracking {
    fn write(&self, p: &mut Path) {
        p.uses_elevator = self.uses_elevator;
    }

    fn track(&mut self, l: &Self, r: &Ways, _w: WayIdx, n: NodeIdx, _track: bool) {
        self.uses_elevator = l.uses_elevator
            || r.get_node_properties(n)
                .map_or(false, |props| props.is_elevator());
    }
}

/// Track a specific node in the path
///
/// Used for tracking the last bike/car sharing station used.
///
/// # C++ Equivalent
/// ```cpp
/// struct track_node_tracking {
///   void write(path& p) const { p.track_node_ = track_node_; }
///   void track(track_node_tracking const& l, ways::routing const&, way_idx_t,
///              node_idx_t const n, bool const track) {
///     track_node_ = track ? n : l.track_node_;
///   }
///   node_idx_t track_node_{node_idx_t::invalid()};
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TrackNodeTracking {
    pub track_node: NodeIdx,
}

impl TrackNodeTracking {
    pub fn new() -> Self {
        Self {
            track_node: NodeIdx::INVALID,
        }
    }
}

impl Default for TrackNodeTracking {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracking for TrackNodeTracking {
    fn write(&self, p: &mut Path) {
        p.track_node = self.track_node;
    }

    fn track(&mut self, l: &Self, _r: &Ways, _w: WayIdx, n: NodeIdx, track: bool) {
        self.track_node = if track { n } else { l.track_node };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OsmNodeIdx;
    use crate::ways::NodeProperties;

    fn create_test_ways() -> Ways {
        let mut ways = Ways::new();

        // Add some nodes with properties
        for i in 0..1000 {
            let props = if i == 100 {
                // Mark node 100 as elevator
                NodeProperties::new_elevator()
            } else {
                NodeProperties::default()
            };
            ways.add_node(
                OsmNodeIdx::new(i as u64),
                crate::Point::from_latlng(0.0, 0.0),
                props,
            );
        }

        ways
    }

    #[test]
    fn test_noop_tracking_write() {
        let tracking = NoopTracking::new();
        let mut path = Path::default();

        tracking.write(&mut path);
        // Should have no effect
    }

    #[test]
    fn test_noop_tracking_track() {
        let mut tracking = NoopTracking::new();
        let prev = NoopTracking::new();
        let ways = create_test_ways();

        tracking.track(&prev, &ways, WayIdx::new(0), NodeIdx::new(0), false);
        // Should have no effect
    }

    #[test]
    fn test_elevator_tracking_initial() {
        let tracking = ElevatorTracking::new();
        assert!(!tracking.uses_elevator);
    }

    #[test]
    fn test_elevator_tracking_write() {
        let mut tracking = ElevatorTracking::new();
        tracking.uses_elevator = true;

        let mut path = Path::default();
        tracking.write(&mut path);

        assert!(path.uses_elevator);
    }

    #[test]
    fn test_elevator_tracking_detects_elevator() {
        let mut tracking = ElevatorTracking::new();
        let prev = ElevatorTracking::new();
        let ways = create_test_ways();

        // Visit non-elevator node
        tracking.track(&prev, &ways, WayIdx::new(0), NodeIdx::new(50), false);
        assert!(!tracking.uses_elevator);

        // Visit elevator node
        tracking.track(&prev, &ways, WayIdx::new(0), NodeIdx::new(100), false);
        assert!(tracking.uses_elevator);
    }

    #[test]
    fn test_elevator_tracking_propagates() {
        let mut prev = ElevatorTracking::new();
        prev.uses_elevator = true;

        let mut tracking = ElevatorTracking::new();
        let ways = create_test_ways();

        // Should propagate from previous even if current node is not elevator
        tracking.track(&prev, &ways, WayIdx::new(0), NodeIdx::new(50), false);
        assert!(tracking.uses_elevator);
    }

    #[test]
    fn test_track_node_tracking_initial() {
        let tracking = TrackNodeTracking::new();
        assert_eq!(tracking.track_node, NodeIdx::INVALID);
    }

    #[test]
    fn test_track_node_tracking_write() {
        let mut tracking = TrackNodeTracking::new();
        tracking.track_node = NodeIdx::new(42);

        let mut path = Path::default();
        tracking.write(&mut path);

        assert_eq!(path.track_node, NodeIdx::new(42));
    }

    #[test]
    fn test_track_node_tracking_when_true() {
        let mut tracking = TrackNodeTracking::new();
        let prev = TrackNodeTracking::new();
        let ways = create_test_ways();

        // track = true should update the node
        tracking.track(&prev, &ways, WayIdx::new(0), NodeIdx::new(123), true);
        assert_eq!(tracking.track_node, NodeIdx::new(123));
    }

    #[test]
    fn test_track_node_tracking_when_false() {
        let mut prev = TrackNodeTracking::new();
        prev.track_node = NodeIdx::new(42);

        let mut tracking = TrackNodeTracking::new();
        let ways = create_test_ways();

        // track = false should propagate previous node
        tracking.track(&prev, &ways, WayIdx::new(0), NodeIdx::new(999), false);
        assert_eq!(tracking.track_node, NodeIdx::new(42));
    }

    #[test]
    fn test_track_node_tracking_sequence() {
        let ways = create_test_ways();

        // Start with invalid
        let t1 = TrackNodeTracking::new();
        assert_eq!(t1.track_node, NodeIdx::INVALID);

        // Track node 10
        let mut t2 = TrackNodeTracking::new();
        t2.track(&t1, &ways, WayIdx::new(0), NodeIdx::new(10), true);
        assert_eq!(t2.track_node, NodeIdx::new(10));

        // Don't track node 20, should keep 10
        let mut t3 = TrackNodeTracking::new();
        t3.track(&t2, &ways, WayIdx::new(0), NodeIdx::new(20), false);
        assert_eq!(t3.track_node, NodeIdx::new(10));

        // Track node 30, should update
        let mut t4 = TrackNodeTracking::new();
        t4.track(&t3, &ways, WayIdx::new(0), NodeIdx::new(30), true);
        assert_eq!(t4.track_node, NodeIdx::new(30));
    }
}
