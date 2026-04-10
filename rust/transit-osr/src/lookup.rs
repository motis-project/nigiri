//! Translation of osr/include/osr/lookup.h
//!
//! Street network lookup and map-matching functionality.
//!
//! NOTE: Full profile-based matching requires R-tree integration.
//! For production use, integrate rstar crate for spatial indexing.

use std::collections::HashSet;
use std::path::Path;

use rstar::{RTree, RTreeObject, AABB};

use crate::routing::profile::SearchProfile;
use crate::types::*;
use crate::ways::Ways;
use crate::{Direction, Level, Location, Point};

/// Way segment for R-tree spatial indexing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WaySegment {
    way: WayIdx,
    segment_idx: usize,
    // Store envelope as min/max coords for serialization
    min_lat: f64,
    min_lng: f64,
    max_lat: f64,
    max_lng: f64,
}

impl WaySegment {
    fn new(way: WayIdx, segment_idx: usize, envelope: AABB<[f64; 2]>) -> Self {
        let min = envelope.lower();
        let max = envelope.upper();
        Self {
            way,
            segment_idx,
            min_lat: min[0],
            min_lng: min[1],
            max_lat: max[0],
            max_lng: max[1],
        }
    }

    fn envelope(&self) -> AABB<[f64; 2]> {
        AABB::from_corners([self.min_lat, self.min_lng], [self.max_lat, self.max_lng])
    }
}

impl RTreeObject for WaySegment {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        WaySegment::envelope(self)
    }
}

/// Node candidate for matching
#[derive(Debug, Clone)]
pub struct NodeCandidate {
    pub lvl: Level,
    pub way_dir: Direction,
    pub node: NodeIdx,
    pub dist_to_node: f64,
    pub cost: Cost,
    pub path: Vec<(f64, f64)>, // (lat, lng) path
}

impl NodeCandidate {
    pub fn valid(&self) -> bool {
        self.node.is_valid()
    }

    pub fn new_invalid() -> Self {
        Self {
            lvl: Level::default(),
            way_dir: Direction::Forward,
            node: NodeIdx::INVALID,
            dist_to_node: 0.0,
            cost: 0,
            path: vec![],
        }
    }
}

/// Raw node candidate (before cost calculation)
#[derive(Debug, Clone)]
pub struct RawNodeCandidate {
    pub node: NodeIdx,
    pub dist_to_node: f32,
}

impl RawNodeCandidate {
    pub fn valid(&self) -> bool {
        self.node.is_valid()
    }

    pub fn new_invalid() -> Self {
        Self {
            node: NodeIdx::INVALID,
            dist_to_node: 0.0,
        }
    }
}

/// Way candidate for matching
#[derive(Debug, Clone)]
pub struct WayCandidate {
    pub dist_to_way: f64,
    pub way: WayIdx,
    pub left: NodeCandidate,
    pub right: NodeCandidate,
}

impl WayCandidate {
    pub fn new(dist_to_way: f64, way: WayIdx) -> Self {
        Self {
            dist_to_way,
            way,
            left: NodeCandidate::new_invalid(),
            right: NodeCandidate::new_invalid(),
        }
    }

    pub fn with_nodes(
        dist_to_way: f64,
        way: WayIdx,
        left: NodeCandidate,
        right: NodeCandidate,
    ) -> Self {
        Self {
            dist_to_way,
            way,
            left,
            right,
        }
    }
}

impl PartialEq for WayCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist_to_way == other.dist_to_way
    }
}

impl Eq for WayCandidate {}

impl PartialOrd for WayCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist_to_way.partial_cmp(&other.dist_to_way)
    }
}

impl Ord for WayCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Raw way candidate (before node lookup)
#[derive(Debug, Clone)]
pub struct RawWayCandidate {
    pub dist_to_way: f32,
    pub way: WayIdx,
    pub left: RawNodeCandidate,
    pub right: RawNodeCandidate,
}

impl RawWayCandidate {
    pub fn new(dist_to_way: f32, way: WayIdx) -> Self {
        Self {
            dist_to_way,
            way,
            left: RawNodeCandidate::new_invalid(),
            right: RawNodeCandidate::new_invalid(),
        }
    }
}

impl PartialEq for RawWayCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist_to_way == other.dist_to_way
    }
}

impl Eq for RawWayCandidate {}

impl PartialOrd for RawWayCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist_to_way.partial_cmp(&other.dist_to_way)
    }
}

impl Ord for RawWayCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub type Match = Vec<WayCandidate>;

/// Bounding box for spatial queries
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min_lat: f64,
    pub min_lng: f64,
    pub max_lat: f64,
    pub max_lng: f64,
}

impl BoundingBox {
    /// Create bounding box around a point with given radius in meters
    pub fn around_point(center: &Point, radius_m: f64) -> Self {
        // Rough approximation: 1 degree latitude ≈ 111km
        let lat_delta = radius_m / 111_000.0;

        // Longitude delta depends on latitude
        let (lat, lng) = center.as_latlng();
        let lng_delta = radius_m / (111_000.0 * lat.to_radians().cos());

        Self {
            min_lat: lat - lat_delta,
            max_lat: lat + lat_delta,
            min_lng: lng - lng_delta,
            max_lng: lng + lng_delta,
        }
    }

    /// Check if point is inside bbox
    pub fn contains(&self, point: &Point) -> bool {
        let (lat, lng) = point.as_latlng();
        lat >= self.min_lat && lat <= self.max_lat && lng >= self.min_lng && lng <= self.max_lng
    }
}

/// Street network lookup with spatial queries
///
/// Provides map-matching functionality to snap locations to the street network.
///
/// Uses R-tree spatial indexing for efficient location-to-way matching.
pub struct Lookup {
    rtree: RTree<WaySegment>,
}

/// Calculate squared distance from point to line segment with projection
/// Returns (squared_distance, projected_point)
fn point_to_segment_distance(point: &Point, seg_start: &Point, seg_end: &Point) -> (f64, Point) {
    let (px, py) = point.as_latlng();
    let (ax, ay) = seg_start.as_latlng();
    let (bx, by) = seg_end.as_latlng();

    // Vector from A to B
    let abx = bx - ax;
    let aby = by - ay;

    // Vector from A to P
    let apx = px - ax;
    let apy = py - ay;

    // Calculate parameter t for projection: P' = A + t * AB
    let ab_squared = abx * abx + aby * aby;

    if ab_squared < 1e-10 {
        // Segment is essentially a point
        let dist_sq = apx * apx + apy * apy;
        return (dist_sq, *seg_start);
    }

    let t = (apx * abx + apy * aby) / ab_squared;

    // Clamp t to [0, 1] to stay within segment
    let t = t.clamp(0.0, 1.0);

    // Calculate projected point
    let proj_x = ax + t * abx;
    let proj_y = ay + t * aby;
    let projected = Point::from_latlng(proj_x, proj_y);

    // Calculate squared distance from point to projection
    let dx = px - proj_x;
    let dy = py - proj_y;
    let dist_squared = dx * dx + dy * dy;

    (dist_squared, projected)
}

/// Calculate distance from point to polyline
/// Returns (squared_distance, best_projected_point, segment_index)
fn distance_to_polyline(point: &Point, polyline: &[Point]) -> (f64, Point, usize) {
    if polyline.is_empty() {
        return (f64::MAX, *point, 0);
    }

    if polyline.len() == 1 {
        let dist = point.distance_to(&polyline[0]);
        return (dist * dist, polyline[0], 0);
    }

    let mut min_dist_sq = f64::MAX;
    let mut best_point = *point;
    let mut best_segment = 0;

    for i in 0..polyline.len() - 1 {
        let (dist_sq, proj) = point_to_segment_distance(point, &polyline[i], &polyline[i + 1]);
        if dist_sq < min_dist_sq {
            min_dist_sq = dist_sq;
            best_point = proj;
            best_segment = i;
        }
    }

    (min_dist_sq, best_point, best_segment)
}

impl Lookup {
    /// Create lookup from ways and build R-tree
    pub fn new(ways: &Ways) -> Self {
        let mut lookup = Self {
            rtree: RTree::new(),
        };
        lookup.build_rtree(ways);
        lookup
    }

    /// Load lookup with R-tree from disk
    pub fn load(ways: &Ways, path: &Path) -> crate::Result<Self> {
        let rtree_path = path.join("rtree_data.bin");

        if rtree_path.exists() {
            // Load serialized R-tree
            let data = std::fs::read(&rtree_path)?;

            let segments: Vec<WaySegment> = bincode::deserialize(&data).map_err(|e| {
                crate::Error::Serialization(format!("Failed to deserialize R-tree: {}", e))
            })?;

            Ok(Self {
                rtree: RTree::bulk_load(segments),
            })
        } else {
            // Build R-tree if not found
            let lookup = Self::new(ways);
            Ok(lookup)
        }
    }

    /// Save R-tree to disk
    pub fn save(&self, path: &Path) -> crate::Result<()> {
        let rtree_path = path.join("rtree_data.bin");

        // Extract segments from R-tree
        let segments: Vec<WaySegment> = self.rtree.iter().cloned().collect();

        // Serialize
        let data = bincode::serialize(&segments).map_err(|e| {
            crate::Error::Serialization(format!("Failed to serialize R-tree: {}", e))
        })?;

        std::fs::write(&rtree_path, data)?;

        Ok(())
    }

    pub fn check_rtree(&self) {
        println!("R-tree has {} segments", self.rtree.size());
    }

    /// Build R-tree spatial index from way segments
    pub fn build_rtree(&mut self, ways: &Ways) {
        let mut segments = Vec::new();

        for way_idx in 0..ways.n_ways() {
            let way = WayIdx::new(way_idx as u32);
            let nodes = ways.get_way_nodes(way);

            // Create segments between consecutive nodes
            for i in 0..nodes.len().saturating_sub(1) {
                let pos1 = ways.get_node_pos(nodes[i]);
                let pos2 = ways.get_node_pos(nodes[i + 1]);

                let (lat1, lng1) = pos1.as_latlng();
                let (lat2, lng2) = pos2.as_latlng();

                let min_lat = lat1.min(lat2);
                let max_lat = lat1.max(lat2);
                let min_lng = lng1.min(lng2);
                let max_lng = lng1.max(lng2);

                let envelope = AABB::from_corners([min_lat, min_lng], [max_lat, max_lng]);

                segments.push(WaySegment::new(way, i, envelope));
            }
        }

        self.rtree = RTree::bulk_load(segments);
    }

    /// Find ways within bounding box using R-tree spatial index
    pub fn find_ways_in_bbox(&self, bbox: &BoundingBox) -> Vec<WayIdx> {
        let envelope =
            AABB::from_corners([bbox.min_lat, bbox.min_lng], [bbox.max_lat, bbox.max_lng]);

        let mut way_set = HashSet::new();

        for segment in self.rtree.locate_in_envelope(&envelope) {
            way_set.insert(segment.way);
        }

        way_set.into_iter().collect()
    }

    /// Find raw way candidates near location
    ///
    /// Returns ways sorted by distance with closest endpoints.
    pub fn get_raw_match(
        &self,
        ways: &Ways,
        location: &Location,
        max_distance: f64,
    ) -> Vec<RawWayCandidate> {
        let bbox = BoundingBox::around_point(&location.pos_, max_distance);
        // println!(
        //     "DEBUG lookup: bbox = [{:.6}, {:.6}] to [{:.6}, {:.6}]",
        //     bbox.min_lat, bbox.min_lng, bbox.max_lat, bbox.max_lng
        // );

        let nearby_ways = self.find_ways_in_bbox(&bbox);
        // println!(
        //     "DEBUG lookup: Found {} nearby ways in bbox",
        //     nearby_ways.len()
        // );

        let mut candidates = Vec::new();

        for way in nearby_ways {
            let nodes = ways.get_way_nodes(way);
            if nodes.is_empty() {
                continue;
            }

            // Find distance to way by checking all nodes
            let mut min_dist = f64::MAX;
            let mut closest_idx = 0;

            for (idx, &node) in nodes.iter().enumerate() {
                let pos = ways.get_node_pos(node);
                let dist = location.pos_.distance_to(&pos);
                if dist < min_dist {
                    min_dist = dist;
                    closest_idx = idx;
                }
            }

            if min_dist <= max_distance {
                // println!(
                //     "DEBUG lookup: Way {:?} - min_dist={:.2}m (threshold={:.2}m)",
                //     way, min_dist, max_distance
                // );
                // Find left and right candidates
                let left_node = if closest_idx > 0 {
                    nodes[closest_idx - 1]
                } else {
                    NodeIdx::INVALID
                };

                let right_node = if closest_idx < nodes.len() - 1 {
                    nodes[closest_idx + 1]
                } else {
                    NodeIdx::INVALID
                };

                let left_dist = if left_node.is_valid() {
                    let pos = ways.get_node_pos(left_node);
                    location.pos_.distance_to(&pos) as f32
                } else {
                    f32::MAX
                };

                let right_dist = if right_node.is_valid() {
                    let pos = ways.get_node_pos(right_node);
                    location.pos_.distance_to(&pos) as f32
                } else {
                    f32::MAX
                };

                candidates.push(RawWayCandidate {
                    dist_to_way: min_dist as f32,
                    way,
                    left: RawNodeCandidate {
                        node: left_node,
                        dist_to_node: left_dist,
                    },
                    right: RawNodeCandidate {
                        node: right_node,
                        dist_to_node: right_dist,
                    },
                });
            }
        }

        // Sort by distance
        candidates.sort();
        // println!("DEBUG lookup: Returning {} candidates", candidates.len());
        candidates
    }

    /// Match location to street network with profile-based cost calculation
    ///
    /// Finds nearby ways, calculates routing costs based on profile,
    /// and filters out infeasible candidates.
    ///
    /// # C++ Equivalent
    /// ```cpp
    /// match_t match(P::parameters const& params,
    ///               location const& query,
    ///               bool const reverse,
    ///               direction const search_dir,
    ///               double max_match_distance,
    ///               bitvec<node_idx_t> const* blocked) const;
    /// ```
    pub fn match_location(
        &self,
        ways: &Ways,
        profile: SearchProfile,
        location: &Location,
        max_distance: f64,
    ) -> Match {
        let mut distance = max_distance;
        let mut candidates = Vec::new();
        let mut iteration = 0;

        // Try up to 4 times with doubling distance (matches C++ behavior)
        while candidates.is_empty() && iteration < 4 {
            candidates = self.get_way_candidates(ways, profile, location, distance);
            if candidates.is_empty() {
                distance *= 2.0;
            }
            iteration += 1;
        }

        candidates
    }

    /// Get way candidates with profile-based cost filtering
    fn get_way_candidates(
        &self,
        ways: &Ways,
        profile: SearchProfile,
        location: &Location,
        max_distance: f64,
    ) -> Match {
        let bbox = BoundingBox::around_point(&location.pos_, max_distance);
        // println!(
        //     "DEBUG get_way_candidates: bbox = [{:.6}, {:.6}] to [{:.6}, {:.6}], max_dist={:.2}m",
        //     bbox.min_lat, bbox.min_lng, bbox.max_lat, bbox.max_lng, max_distance
        // );

        let nearby_ways = self.find_ways_in_bbox(&bbox);
        // println!(
        //     "DEBUG get_way_candidates: Found {} ways in bbox",
        //     nearby_ways.len()
        // );

        let mut candidates = Vec::new();
        let max_dist_squared = max_distance * max_distance;

        for way in nearby_ways {
            let nodes = ways.get_way_nodes(way);
            // println!(
            //     "DEBUG get_way_candidates: Processing way {:?}, nodes.len()={}",
            //     way,
            //     nodes.len()
            // );
            if nodes.is_empty() {
                // println!("  -> Skipping: no nodes");
                continue;
            }

            // Build polyline from nodes
            let polyline: Vec<Point> = nodes.iter().map(|&n| ways.get_node_pos(n)).collect();

            // Calculate proper distance to polyline with projection
            let (dist_squared, best_point, segment_idx) =
                distance_to_polyline(&location.pos_, &polyline);
            let min_dist = dist_squared.sqrt();

            // println!(
            //     "  -> min_dist={:.2}m, segment_idx={}, best_point=({:.6}, {:.6})",
            //     min_dist,
            //     segment_idx,
            //     best_point.as_latlng().0,
            //     best_point.as_latlng().1
            //);

            if dist_squared >= max_dist_squared {
                // println!(
                //     "  -> Skipping: too far ({:.2}m > {:.2}m)",
                //     min_dist, max_distance
                // );
                continue;
            }

            // Find navigable nodes on left and right of closest segment
            // println!(
            //     "  -> Finding left node (backward from segment {})...",
            //     segment_idx
            // );
            let left = self.find_next_node(
                ways,
                profile,
                way,
                segment_idx,
                Direction::Backward,
                location,
                &best_point,
                min_dist,
            );
            // println!(
            //     "    left.valid()={}, left.node={:?}",
            //     left.valid(),
            //     left.node
            // );

            // println!(
            //     "  -> Finding right node (forward from segment {})...",
            //     segment_idx
            // );
            let right = self.find_next_node(
                ways,
                profile,
                way,
                segment_idx,
                Direction::Forward,
                location,
                &best_point,
                min_dist,
            );
            // println!(
            //     "    right.valid()={}, right.node={:?}",
            //     right.valid(),
            //     right.node
            // );

            // Only add if at least one direction is feasible
            if left.valid() || right.valid() {
                // println!(
                //     "  -> ADDED candidate (left_valid={}, right_valid={})",
                //     left.valid(),
                //     right.valid()
                // );
                candidates.push(WayCandidate::with_nodes(min_dist, way, left, right));
            } else {
                // println!("  -> SKIPPED: neither left nor right is valid");
            }
        }

        // Sort by distance (closest first)
        candidates.sort();
        candidates
    }

    /// Find next navigable node in given direction from segment
    fn find_next_node(
        &self,
        ways: &Ways,
        profile: SearchProfile,
        way: WayIdx,
        segment_idx: usize,
        dir: Direction,
        location: &Location,
        projected_point: &Point,
        dist_to_way: f64,
    ) -> NodeCandidate {
        let nodes = ways.get_way_nodes(way);
        let Some(way_props) = ways.get_way_properties(way) else {
            // println!("      find_next_node: No way properties for {:?}", way);
            return NodeCandidate::new_invalid();
        };

        // Check if way is usable for this profile in this direction
        let way_cost_check = profile.way_cost(way_props, dir, 0);
        // println!(
        //     "      find_next_node: way_cost({:?}, {:?}, 0) = {}",
        //     way, dir, way_cost_check
        // );
        if way_cost_check == Cost::MAX {
            // println!(
            //     "      find_next_node: Way not usable for profile in direction {:?}",
            //     dir
            // );
            return NodeCandidate::new_invalid();
        }

        let mut dist_to_node = dist_to_way;
        let mut path = vec![projected_point.as_latlng()];

        // Traverse in the given direction to find first navigable node
        let start_idx = match dir {
            Direction::Forward => segment_idx + 1,
            Direction::Backward => segment_idx,
        };

        let indices: Vec<usize> = match dir {
            Direction::Forward => (start_idx..nodes.len()).collect(),
            Direction::Backward => (0..=start_idx.min(nodes.len().saturating_sub(1)))
                .rev()
                .collect(),
        };

        // println!(
        //     "      find_next_node: Traversing {} nodes in direction {:?}",
        //     indices.len(),
        //     dir
        // );

        for idx in indices {
            let node = nodes[idx];
            let pos = ways.get_node_pos(node);
            path.push(pos.as_latlng());

            // println!("      find_next_node: [{}] idx={}, node={:?}", i, idx, node);

            // Check if this is a routing node (not just a geometry point)
            let Some(node_props) = ways.get_node_properties(node) else {
                // println!("        -> No node properties, continuing");
                continue;
            };

            // Calculate actual routing cost
            let node_cost = profile.node_cost(node_props);
            // println!("        -> node_cost={}", node_cost);
            if node_cost == Cost::MAX {
                // println!("        -> Node not accessible, continuing");
                continue; // Node not accessible for this profile
            }

            // Calculate distance along path
            if idx > 0 {
                let prev_idx = if dir == Direction::Forward {
                    idx - 1
                } else {
                    idx + 1
                };
                if prev_idx < nodes.len() {
                    let prev_pos = ways.get_node_pos(nodes[prev_idx]);
                    dist_to_node += pos.distance_to(&prev_pos);
                }
            }

            // Calculate way cost
            let way_cost = profile.way_cost(way_props, dir, dist_to_node as u16);
            // println!("        -> way_cost({:.2}m) = {}", dist_to_node, way_cost);
            if way_cost == Cost::MAX {
                // println!("        -> Way cost is MAX, returning invalid");
                return NodeCandidate::new_invalid();
            }

            // Found a valid node!
            // println!("        -> FOUND VALID NODE! cost={}", way_cost + node_cost);
            return NodeCandidate {
                lvl: location.lvl_,
                way_dir: dir,
                node,
                dist_to_node,
                cost: way_cost + node_cost,
                path,
            };
        }

        // println!(
        //     "      find_next_node: No valid node found after traversing {} indices",
        //     indices.len()
        // );
        NodeCandidate::new_invalid()
    }

    /// Find elevator nodes within a bounding box using the R-tree spatial index.
    ///
    /// Queries the R-tree for ways in the bounding box, then checks each node
    /// of matched ways for the `is_elevator` flag. Uses a HashSet to deduplicate
    /// nodes that appear in multiple ways.
    ///
    /// # C++ Equivalent
    /// ```cpp
    /// hash_set<node_idx_t> lookup::find_elevators(geo::box const& b) const {
    ///   auto elevators = hash_set<node_idx_t>{};
    ///   find(b, [&](way_idx_t const way) {
    ///     for (auto const n : ways_.r_->way_nodes_[way]) {
    ///       if (ways_.r_->node_properties_[n].is_elevator()) {
    ///         elevators.emplace(n);
    ///       }
    ///     }
    ///   });
    ///   return elevators;
    /// }
    /// ```
    pub fn find_elevators(&self, ways: &Ways, min: &Location, max: &Location) -> HashSet<NodeIdx> {
        let mut elevators = HashSet::new();

        let (min_lat, min_lng) = min.pos_.as_latlng();
        let (max_lat, max_lng) = max.pos_.as_latlng();

        let bbox = BoundingBox {
            min_lat,
            min_lng,
            max_lat,
            max_lng,
        };

        // Use R-tree to find ways in bounding box (C++: find(b, ...))
        let found_ways = self.find_ways_in_bbox(&bbox);

        for way_idx in found_ways {
            // Check each node in the way for elevator flag
            for &node in ways.get_way_nodes(way_idx) {
                if let Some(props) = ways.get_node_properties(node) {
                    if props.is_elevator() {
                        elevators.insert(node);
                    }
                }
            }
        }

        elevators
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_candidate_valid() {
        let valid = NodeCandidate {
            lvl: Level::default(),
            way_dir: Direction::Forward,
            node: NodeIdx::new(123),
            dist_to_node: 10.0,
            cost: 100,
            path: vec![],
        };
        assert!(valid.valid());

        let invalid = NodeCandidate {
            lvl: Level::default(),
            way_dir: Direction::Forward,
            node: NodeIdx::INVALID,
            dist_to_node: 10.0,
            cost: 100,
            path: vec![],
        };
        assert!(!invalid.valid());
    }

    #[test]
    fn test_way_candidate_ordering() {
        let wc1 = WayCandidate::new(10.0, WayIdx::new(1));
        let wc2 = WayCandidate::new(20.0, WayIdx::new(2));

        assert!(wc1 < wc2);
    }

    #[test]
    fn test_raw_way_candidate_ordering() {
        let wc1 = RawWayCandidate::new(10.0, WayIdx::new(1));
        let wc2 = RawWayCandidate::new(20.0, WayIdx::new(2));

        assert!(wc1 < wc2);
    }

    #[test]
    fn test_bounding_box_contains() {
        use crate::Point;

        let center = Point::from_latlng(37.7749, -122.4194);
        let bbox = BoundingBox::around_point(&center, 1000.0);

        // Center should be inside
        assert!(bbox.contains(&center));

        // Nearby point should be inside
        let nearby = Point::from_latlng(37.7759, -122.4204);
        assert!(bbox.contains(&nearby));

        // Far point should be outside
        let far = Point::from_latlng(40.7128, -74.0060);
        assert!(!bbox.contains(&far));
    }

    #[test]
    fn test_bounding_box_around_point() {
        use crate::Point;

        let center = Point::from_latlng(37.7749, -122.4194);
        let bbox = BoundingBox::around_point(&center, 1000.0);

        // Check that bbox is centered
        let lat_center = (bbox.min_lat + bbox.max_lat) / 2.0;
        let lng_center = (bbox.min_lng + bbox.max_lng) / 2.0;

        assert!((lat_center - 37.7749).abs() < 0.0001);
        assert!((lng_center - (-122.4194)).abs() < 0.0001);
    }

    #[test]
    fn test_find_ways_in_bbox() {
        use crate::{ways::Ways, Point};

        let mut ways = Ways::new();

        // Add nodes
        let pos1 = Point::from_latlng(37.7749, -122.4194);
        let pos2 = Point::from_latlng(37.7759, -122.4204);
        let pos3 = Point::from_latlng(40.7128, -74.0060); // Far away

        ways.add_node(OsmNodeIdx(1), pos1, crate::ways::NodeProperties::default());
        ways.add_node(OsmNodeIdx(2), pos2, crate::ways::NodeProperties::default());
        ways.add_node(OsmNodeIdx(3), pos3, crate::ways::NodeProperties::default());

        // Add ways
        ways.add_way(
            OsmWayIdx(10),
            vec![NodeIdx::new(0), NodeIdx::new(1)],
            crate::ways::WayProperties::default(),
        );
        ways.add_way(
            OsmWayIdx(20),
            vec![NodeIdx::new(2)],
            crate::ways::WayProperties::default(),
        );

        let lookup = Lookup::new(&ways);

        // Search near SF
        let center = Point::from_latlng(37.7749, -122.4194);
        let bbox = BoundingBox::around_point(&center, 2000.0);
        let results = lookup.find_ways_in_bbox(&bbox);

        // Should find way 0 (SF area), not way 1 (NY)
        assert!(results.contains(&WayIdx::new(0)));
        assert!(!results.contains(&WayIdx::new(1)));
    }

    #[test]
    fn test_get_raw_match() {
        use crate::{ways::Ways, Level, Location, Point};

        let mut ways = Ways::new();

        // Add nodes in SF
        let pos1 = Point::from_latlng(37.7749, -122.4194);
        let pos2 = Point::from_latlng(37.7759, -122.4204);

        ways.add_node(OsmNodeIdx(1), pos1, crate::ways::NodeProperties::default());
        ways.add_node(OsmNodeIdx(2), pos2, crate::ways::NodeProperties::default());

        // Add way
        ways.add_way(
            OsmWayIdx(10),
            vec![NodeIdx::new(0), NodeIdx::new(1)],
            crate::ways::WayProperties::default(),
        );

        let lookup = Lookup::new(&ways);

        // Use a larger search radius (1000m instead of 100m) since R-tree might need margin
        let location = Location {
            pos_: pos1,
            lvl_: Level::default(),
        };

        let candidates = lookup.get_raw_match(&ways, &location, 1000.0);

        // Should find the way
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].way, WayIdx::new(0));
    }

    #[test]
    fn test_match_location() {
        use crate::{ways::Ways, Level, Location, Point};

        let mut ways = Ways::new();

        // Add nodes with foot accessibility
        let pos1 = Point::from_latlng(37.7749, -122.4194);
        let pos2 = Point::from_latlng(37.7759, -122.4204);

        let mut node_props = crate::ways::NodeProperties::default();
        node_props.is_foot_accessible = true;

        ways.add_node(OsmNodeIdx(1), pos1, node_props);
        ways.add_node(OsmNodeIdx(2), pos2, node_props);

        // Add way with foot accessibility
        let mut way_props = crate::ways::WayProperties::default();
        way_props.is_foot_accessible = true;

        ways.add_way(
            OsmWayIdx(10),
            vec![NodeIdx::new(0), NodeIdx::new(1)],
            way_props,
        );

        // Connect ways and finalize
        ways.connect_ways();
        ways.finalize();

        let lookup = Lookup::new(&ways);

        // Match location with larger radius
        let location = Location {
            pos_: pos1,
            lvl_: Level::default(),
        };

        let matches = lookup.match_location(&ways, SearchProfile::Foot, &location, 1000.0);

        // Should get matches
        assert!(!matches.is_empty());
        assert_eq!(matches[0].way, WayIdx::new(0));
    }

    #[test]
    fn test_lookup_ways_accessor() {
        use crate::ways::Ways;

        let ways = Ways::new();
        let lookup = Lookup::new(&ways);

        // Should have access to ways
        assert_eq!(lookup.rtree.size(), 0);
    }

    #[test]
    fn test_rtree_save_load() {
        use crate::{ways::Ways, Point};

        let mut ways = Ways::new();

        // Add some nodes
        let pos1 = Point::from_latlng(37.7749, -122.4194); // SF
        let pos2 = Point::from_latlng(37.7849, -122.4094); // Nearby

        ways.add_node(
            crate::types::OsmNodeIdx(100),
            pos1,
            crate::ways::NodeProperties::default(),
        );
        ways.add_node(
            crate::types::OsmNodeIdx(200),
            pos2,
            crate::ways::NodeProperties::default(),
        );

        // Add way
        ways.add_way(
            crate::types::OsmWayIdx(10),
            vec![NodeIdx::new(0), NodeIdx::new(1)],
            crate::ways::WayProperties::default(),
        );

        // Create lookup and save
        let lookup = Lookup::new(&ways);
        let temp_dir = std::env::temp_dir().join("osr_rtree_test");
        std::fs::create_dir_all(&temp_dir).unwrap();

        lookup.save(&temp_dir).unwrap();

        // Verify file was created
        assert!(temp_dir.join("rtree_data.bin").exists());

        // Load and verify
        let loaded_lookup = Lookup::load(&ways, &temp_dir).unwrap();

        // Should find same ways in bbox
        let center = Point::from_latlng(37.7749, -122.4194);
        let bbox = BoundingBox::around_point(&center, 2000.0);

        let original_ways = lookup.find_ways_in_bbox(&bbox);
        let loaded_ways = loaded_lookup.find_ways_in_bbox(&bbox);

        assert_eq!(original_ways, loaded_ways);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
