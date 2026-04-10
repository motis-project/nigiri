//! Path representation for routing results.
//!
//! Translates from C++ `osr/include/osr/routing/path.h`.
//! A path consists of segments, where each segment represents movement
//! along a single way with consistent mode and level.

use crate::elevation_storage::Elevation;
use crate::routing::mode::Mode;
use crate::types::*;
use crate::Point;

/// Path segment representing movement along a single way.
///
/// Each segment has:
/// - Polyline geometry (list of points)
/// - Level information (for multi-level routing)
/// - Start/end nodes
/// - Associated way
/// - Cost and distance
/// - Elevation gain/loss
/// - Mode of transport
#[derive(Debug, Clone)]
pub struct Segment {
    /// Polyline geometry for this segment
    pub polyline: Vec<Point>,

    /// Starting level
    pub from_level: Level,

    /// Ending level
    pub to_level: Level,

    /// Starting node index
    pub from: NodeIdx,

    /// Ending node index
    pub to: NodeIdx,

    /// Way index this segment belongs to
    pub way: WayIdx,

    /// Cost of this segment (in cost units, typically seconds)
    pub cost: Cost,

    /// Distance of this segment in meters
    pub dist: Distance,

    /// Elevation gain/loss for this segment
    pub elevation: Elevation,

    /// Mode of transport for this segment
    pub mode: Mode,
}

impl Segment {
    /// Create a new segment with default/invalid values
    pub fn new() -> Self {
        Self {
            polyline: Vec::new(),
            from_level: Level::from_idx(Level::NO_LEVEL),
            to_level: Level::from_idx(Level::NO_LEVEL),
            from: NodeIdx::INVALID,
            to: NodeIdx::INVALID,
            way: WayIdx::INVALID,
            cost: K_INFEASIBLE,
            dist: 0,
            elevation: Elevation::new(0, 0),
            mode: Mode::Foot,
        }
    }

    /// Create a segment with specific values
    pub fn with_values(
        polyline: Vec<Point>,
        from_level: Level,
        to_level: Level,
        from: NodeIdx,
        to: NodeIdx,
        way: WayIdx,
        cost: Cost,
        dist: Distance,
        elevation: Elevation,
        mode: Mode,
    ) -> Self {
        Self {
            polyline,
            from_level,
            to_level,
            from,
            to,
            way,
            cost,
            dist,
            elevation,
            mode,
        }
    }

    /// Check if segment is valid (has valid nodes and way)
    pub fn is_valid(&self) -> bool {
        self.from != NodeIdx::INVALID
            && self.to != NodeIdx::INVALID
            && self.way != WayIdx::INVALID
            && self.cost != K_INFEASIBLE
    }

    /// Get number of points in polyline
    pub fn num_points(&self) -> usize {
        self.polyline.len()
    }
}

impl Default for Segment {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete routing path from origin to destination.
///
/// A path consists of:
/// - Multiple segments (each along a single way)
/// - Total cost and distance
/// - Total elevation gain/loss
/// - Whether elevator was used
/// - Tracking information
#[derive(Debug, Clone)]
pub struct Path {
    /// Total cost of the path (in cost units, typically seconds)
    pub cost: Cost,

    /// Total distance in meters
    pub dist: f64,

    /// Total elevation gain/loss
    pub elevation: Elevation,

    /// List of path segments
    pub segments: Vec<Segment>,

    /// Whether path uses an elevator
    pub uses_elevator: bool,

    /// Tracking node for partial paths
    pub track_node: NodeIdx,
}

impl Path {
    /// Create a new empty path
    pub fn new() -> Self {
        Self {
            cost: K_INFEASIBLE,
            dist: 0.0,
            elevation: Elevation::new(0, 0),
            segments: Vec::new(),
            uses_elevator: false,
            track_node: NodeIdx::INVALID,
        }
    }

    /// Add a segment to the path
    pub fn add_segment(&mut self, segment: Segment) {
        // Update totals
        if self.cost == K_INFEASIBLE {
            self.cost = segment.cost;
        } else {
            self.cost += segment.cost;
        }
        self.dist += segment.dist as f64;
        self.elevation.add(&segment.elevation);

        self.segments.push(segment);
    }

    /// Check if path is empty (no segments)
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Get number of segments
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Check if path is valid (has cost and segments)
    pub fn is_valid(&self) -> bool {
        !self.is_empty() && self.cost != K_INFEASIBLE
    }

    /// Get total number of points across all segments
    pub fn total_points(&self) -> usize {
        self.segments.iter().map(|s| s.num_points()).sum()
    }

    /// Get first segment (if exists)
    pub fn first_segment(&self) -> Option<&Segment> {
        self.segments.first()
    }

    /// Get last segment (if exists)
    pub fn last_segment(&self) -> Option<&Segment> {
        self.segments.last()
    }

    /// Get all polyline points in order
    pub fn all_points(&self) -> Vec<Point> {
        let mut points = Vec::with_capacity(self.total_points());
        for segment in &self.segments {
            points.extend_from_slice(&segment.polyline);
        }
        points
    }

    /// Clear the path
    pub fn clear(&mut self) {
        self.cost = K_INFEASIBLE;
        self.dist = 0.0;
        self.elevation = Elevation::new(0, 0);
        self.segments.clear();
        self.uses_elevator = false;
        self.track_node = NodeIdx::INVALID;
    }

    /// Reverse the path (swap start and end)
    pub fn reverse(&mut self) {
        // Reverse segments order
        self.segments.reverse();

        // Reverse each segment
        for segment in &mut self.segments {
            std::mem::swap(&mut segment.from, &mut segment.to);
            std::mem::swap(&mut segment.from_level, &mut segment.to_level);
            segment.polyline.reverse();
            segment.elevation = segment.elevation.swapped();
        }

        // Reverse total elevation
        self.elevation = self.elevation.swapped();
    }
}

impl Default for Path {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_creation() {
        let segment = Segment::new();
        assert!(!segment.is_valid());
        assert_eq!(segment.num_points(), 0);
        assert_eq!(segment.cost, K_INFEASIBLE);
    }

    #[test]
    fn test_segment_with_values() {
        let points = vec![
            Point::from_latlng(52.52, 13.405),
            Point::from_latlng(52.53, 13.415),
        ];

        let segment = Segment::with_values(
            points.clone(),
            Level::from_float(0.0),
            Level::from_float(0.0),
            NodeIdx(10),
            NodeIdx(20),
            WayIdx(100),
            120,
            1000,
            Elevation::new(10, 5),
            Mode::Bike,
        );

        assert!(segment.is_valid());
        assert_eq!(segment.num_points(), 2);
        assert_eq!(segment.cost, 120);
        assert_eq!(segment.dist, 1000);
        assert_eq!(segment.from, NodeIdx(10));
        assert_eq!(segment.to, NodeIdx(20));
        assert_eq!(segment.mode, Mode::Bike);
    }

    #[test]
    fn test_empty_path() {
        let path = Path::new();
        assert!(path.is_empty());
        assert!(!path.is_valid());
        assert_eq!(path.len(), 0);
        assert_eq!(path.dist, 0.0);
        assert_eq!(path.cost, K_INFEASIBLE);
        assert!(!path.uses_elevator);
    }

    #[test]
    fn test_add_segment_to_path() {
        let mut path = Path::new();

        let segment1 = Segment::with_values(
            vec![Point::from_latlng(52.52, 13.405)],
            Level::from_float(0.0),
            Level::from_float(0.0),
            NodeIdx(1),
            NodeIdx(2),
            WayIdx(10),
            100,
            500,
            Elevation::new(5, 0),
            Mode::Foot,
        );

        path.add_segment(segment1);

        assert!(!path.is_empty());
        assert!(path.is_valid());
        assert_eq!(path.len(), 1);
        assert_eq!(path.cost, 100);
        assert_eq!(path.dist, 500.0);
        assert_eq!(path.elevation.up, 5);
        assert_eq!(path.elevation.down, 0);
    }

    #[test]
    fn test_add_multiple_segments() {
        let mut path = Path::new();

        let segment1 = Segment::with_values(
            vec![],
            Level::from_float(0.0),
            Level::from_float(0.0),
            NodeIdx(1),
            NodeIdx(2),
            WayIdx(10),
            100,
            500,
            Elevation::new(5, 2),
            Mode::Foot,
        );

        let segment2 = Segment::with_values(
            vec![],
            Level::from_float(0.0),
            Level::from_float(0.0),
            NodeIdx(2),
            NodeIdx(3),
            WayIdx(11),
            150,
            750,
            Elevation::new(3, 7),
            Mode::Foot,
        );

        path.add_segment(segment1);
        path.add_segment(segment2);

        assert_eq!(path.len(), 2);
        assert_eq!(path.cost, 250);
        assert_eq!(path.dist, 1250.0);
        assert_eq!(path.elevation.up, 8);
        assert_eq!(path.elevation.down, 9);
    }

    #[test]
    fn test_total_points() {
        let mut path = Path::new();

        let segment1 = Segment::with_values(
            vec![
                Point::from_latlng(52.52, 13.40),
                Point::from_latlng(52.53, 13.41),
            ],
            Level::from_float(0.0),
            Level::from_float(0.0),
            NodeIdx(1),
            NodeIdx(2),
            WayIdx(10),
            100,
            500,
            Elevation::new(0, 0),
            Mode::Foot,
        );

        let segment2 = Segment::with_values(
            vec![
                Point::from_latlng(52.53, 13.41),
                Point::from_latlng(52.54, 13.42),
                Point::from_latlng(52.55, 13.43),
            ],
            Level::from_float(0.0),
            Level::from_float(0.0),
            NodeIdx(2),
            NodeIdx(3),
            WayIdx(11),
            150,
            750,
            Elevation::new(0, 0),
            Mode::Foot,
        );

        path.add_segment(segment1);
        path.add_segment(segment2);

        assert_eq!(path.total_points(), 5);
        assert_eq!(path.all_points().len(), 5);
    }

    #[test]
    fn test_first_last_segment() {
        let mut path = Path::new();

        let segment1 = Segment::with_values(
            vec![],
            Level::from_float(0.0),
            Level::from_float(0.0),
            NodeIdx(1),
            NodeIdx(2),
            WayIdx(10),
            100,
            500,
            Elevation::new(0, 0),
            Mode::Foot,
        );

        let segment2 = Segment::with_values(
            vec![],
            Level::from_float(0.0),
            Level::from_float(0.0),
            NodeIdx(2),
            NodeIdx(3),
            WayIdx(11),
            150,
            750,
            Elevation::new(0, 0),
            Mode::Bike,
        );

        path.add_segment(segment1);
        path.add_segment(segment2);

        assert_eq!(path.first_segment().unwrap().from, NodeIdx(1));
        assert_eq!(path.last_segment().unwrap().to, NodeIdx(3));
        assert_eq!(path.last_segment().unwrap().mode, Mode::Bike);
    }

    #[test]
    fn test_clear_path() {
        let mut path = Path::new();

        let segment = Segment::with_values(
            vec![],
            Level::from_float(0.0),
            Level::from_float(0.0),
            NodeIdx(1),
            NodeIdx(2),
            WayIdx(10),
            100,
            500,
            Elevation::new(0, 0),
            Mode::Foot,
        );

        path.add_segment(segment);
        assert!(!path.is_empty());

        path.clear();
        assert!(path.is_empty());
        assert_eq!(path.cost, K_INFEASIBLE);
        assert_eq!(path.dist, 0.0);
    }

    #[test]
    fn test_reverse_path() {
        let mut path = Path::new();

        let segment1 = Segment::with_values(
            vec![
                Point::from_latlng(52.52, 13.40),
                Point::from_latlng(52.53, 13.41),
            ],
            Level::from_float(0.0),
            Level::from_float(1.0),
            NodeIdx(1),
            NodeIdx(2),
            WayIdx(10),
            100,
            500,
            Elevation::new(10, 5),
            Mode::Foot,
        );

        let segment2 = Segment::with_values(
            vec![
                Point::from_latlng(52.53, 13.41),
                Point::from_latlng(52.54, 13.42),
            ],
            Level::from_float(1.0),
            Level::from_float(2.0),
            NodeIdx(2),
            NodeIdx(3),
            WayIdx(11),
            150,
            750,
            Elevation::new(8, 3),
            Mode::Foot,
        );

        path.add_segment(segment1);
        path.add_segment(segment2);

        let original_cost = path.cost;

        path.reverse();

        // Cost and distance stay the same
        assert_eq!(path.cost, original_cost);

        // Elevation is swapped
        assert_eq!(path.elevation.up, 8); // Was down: 5 + 3 = 8
        assert_eq!(path.elevation.down, 18); // Was up: 10 + 8 = 18

        // Segments are reversed
        assert_eq!(path.first_segment().unwrap().from, NodeIdx(3));
        assert_eq!(path.last_segment().unwrap().to, NodeIdx(1));

        // Levels are swapped in segments
        assert_eq!(
            path.first_segment().unwrap().from_level,
            Level::from_float(2.0)
        );
        assert_eq!(
            path.first_segment().unwrap().to_level,
            Level::from_float(1.0)
        );
    }
}
