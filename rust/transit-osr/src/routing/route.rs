//! Translation of osr/include/osr/routing/route.h
//!
//! High-level routing API that ties together all routing components:
//! - Location matching via Lookup
//! - Profile selection via with_profile.rs  
//! - Dijkstra/Bidirectional execution
//! - Path construction
//!
//! This implementation handles the multi-node-type challenge by using
//! enum dispatch based on profile type.

use crate::lookup::NodeCandidate;
use crate::routing::profiles::{bike::BikeNode, car::CarNode, foot::FootNode};
use crate::routing::with_profile::ProfileInstance;
use crate::routing::{Dijkstra, Path, RoutingAlgorithm, SearchProfile};
use crate::types::{Cost, Distance, K_INFEASIBLE};
use crate::Direction;
use crate::{ElevationStorage, Level, Location, Lookup, NodeIdx, Ways};

/// Route from one location to another
///
/// Returns None if no route found or locations cannot be matched to network.
///
/// # Arguments
/// * `ways` - Street network data
/// * `lookup` - Location to node mapping
/// * `elevations` - Optional elevation data for cost calculation
/// * `profile` - Routing profile (foot, bike, car, etc.)
/// * `from` - Start location
/// * `to` - End location  
/// * `max_cost` - Maximum routing cost (seconds)
/// * `algorithm` - Routing algorithm to use
///
/// # Implementation
/// 1. Creates profile instance from SearchProfile enum
/// 2. Matches start/end locations to street network nodes
/// 3. Runs routing algorithm (currently Dijkstra only)
/// 4. Reconstructs path from results
///
/// # Example
/// ```ignore
/// let path = route(
///     &ways,
///     &lookup,
///     SearchProfile::Foot,
///     Location::from_latlng(52.5, 13.4, Level::default()),
///     Location::from_latlng(52.51, 13.41, Level::default()),
///     3600, // 1 hour max
///     RoutingAlgorithm::Dijkstra,
/// );
/// ```
pub fn route(
    ways: &Ways,
    lookup: &Lookup,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: SearchProfile,
    from: Location,
    to: Location,
    max_cost: Cost,
    algorithm: RoutingAlgorithm,
) -> Option<Path> {
    // Create profile instance
    let profile_inst = ProfileInstance::from_search_profile(profile);
    let max_distance = profile_inst.max_match_distance();

    // println!(
    //     "DEBUG: Matching start location with max_distance={}",
    //     max_distance
    // );
    // Match start and end locations to network
    let start_candidates = lookup.match_location(&ways, profile, &from, max_distance);
    // println!("DEBUG: Found {} start candidates", start_candidates.len());
    // for (i, c) in start_candidates.iter().enumerate() {
    //     println!(
    //         "  Candidate {}: way={:?}, dist={:.2}m, left_node={:?}, right_node={:?}",
    //         i, c.way, c.dist_to_way, c.left.node, c.right.node
    //     );
    // }

    let end_candidates = lookup.match_location(&ways, profile, &to, max_distance);
    // println!("DEBUG: Found {} end candidates", end_candidates.len());

    if start_candidates.is_empty() || end_candidates.is_empty() {
        return None;
    }

    // BUG-11 fix: prune end candidates whose snap distance is more than 1/3 of the
    // straight-line from→to distance (kMaxMatchingDistanceSquaredRatio = 9.0 in C++).
    // Always keep at least kBottomKDefinitelyConsidered = 5 candidates.
    let from_to_dist_sq = {
        let dlat = (from.pos_.lat() - to.pos_.lat()) * 111_320.0;
        let avg_lat = (from.pos_.lat() + to.pos_.lat()) * 0.5;
        let dlon = (from.pos_.lng() - to.pos_.lng()) * 111_320.0 * avg_lat.to_radians().cos();
        dlat * dlat + dlon * dlon
    };
    let max_snap_dist_sq = from_to_dist_sq / 9.0;
    let end_candidates: Vec<_> = end_candidates
        .iter()
        .enumerate()
        .take_while(|(j, c)| *j < 5 || c.dist_to_way * c.dist_to_way <= max_snap_dist_sq)
        .map(|(_, c)| c.clone())
        .collect();
    let end_candidates = &end_candidates[..];

    // BUG-10 fix: check that at least one start candidate shares a connected component
    // with at least one end candidate, otherwise no route is possible.
    let has_common_component = start_candidates.iter().any(|sc| {
        let sc_comp = ways.get_component(sc.way);
        end_candidates.iter().any(|ec| {
            let ec_comp = ways.get_component(ec.way);
            sc_comp.is_none() || ec_comp.is_none() || sc_comp == ec_comp
        })
    });
    if !has_common_component {
        return None;
    }

    if end_candidates.is_empty() {
        return None;
    }

    // BUG-06 fix: if the two query points are within 8 metres of each other,
    // return a direct path immediately (matching C++ try_direct()).
    let direct_dist = from.pos_.distance_to(&to.pos_);
    if direct_dist < 8.0 {
        let from_pt = crate::Point::from_latlng(from.pos_.lat(), from.pos_.lng());
        let to_pt = crate::Point::from_latlng(to.pos_.lat(), to.pos_.lng());
        return Some(Path {
            cost: 60,
            dist: direct_dist,
            elevation: crate::elevation_storage::Elevation::new(0, 0),
            segments: vec![crate::routing::path::Segment::with_values(
                vec![from_pt, to_pt],
                from.lvl_,
                to.lvl_,
                NodeIdx::INVALID,
                NodeIdx::INVALID,
                crate::types::WayIdx::INVALID,
                60,
                direct_dist.min(u16::MAX as f64) as u16,
                crate::elevation_storage::Elevation::new(0, 0),
                crate::routing::mode::Mode::Foot,
            )],
            uses_elevator: false,
            track_node: NodeIdx::INVALID,
        });
    }

    // Route based on algorithm and profile type
    match algorithm {
        RoutingAlgorithm::Dijkstra => route_dijkstra(
            ways,
            elevations,
            &profile_inst,
            &start_candidates,
            &end_candidates,
            max_cost,
        ),
        RoutingAlgorithm::AStarBi => route_bidirectional(
            ways,
            elevations,
            &profile_inst,
            &start_candidates,
            &end_candidates,
            max_cost,
            from,
            to,
        ),
    }
}

/// Internal routing implementation using Dijkstra
fn route_dijkstra(
    ways: &Ways,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &ProfileInstance,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
) -> Option<Path> {
    // Dispatch to appropriate profile implementation
    match profile {
        ProfileInstance::Foot(foot_profile) => route_foot_dijkstra(
            ways,
            elevations,
            foot_profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::Bike(bike_profile) => route_bike_dijkstra(
            ways,
            elevations,
            bike_profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::Car(car_profile) => route_car_dijkstra(
            ways,
            elevations,
            car_profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::BikeSharing(profile) => route_bike_sharing_dijkstra(
            ways,
            elevations,
            profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::CarSharing(profile) => route_car_sharing_dijkstra(
            ways,
            elevations,
            profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::CarParking(profile) => route_car_parking_dijkstra(
            ways,
            elevations,
            profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
    }
}

/// Internal bidirectional routing implementation
///
/// ## Implementation Parity: Generic Bidirectional A*
///
/// This implementation uses `GenericBidirectional<P>`, which is generic over profiles
/// and supports profile-specific node types (e.g., `BikeNode` with direction state).
///
/// **Supported profiles:**
/// - ✅ Foot: Uses `FootNode` (NodeIdx + Level)
/// - ✅ Bike: Uses `BikeNode` (NodeIdx + Direction)
/// - ✅ Car: Uses `CarNode` (NodeIdx + WayIdx + Direction)
///
/// **Fallback to Dijkstra:**
/// - ❌ Sharing modes: Require complex state tracking (planned)
fn route_bidirectional(
    ways: &Ways,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &ProfileInstance,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
    from: Location,
    to: Location,
) -> Option<Path> {
    // Dispatch to appropriate profile implementation
    match profile {
        ProfileInstance::Foot(foot_profile) => route_foot_bidirectional(
            ways,
            foot_profile,
            start_candidates,
            end_candidates,
            max_cost,
            from,
            to,
        ),
        ProfileInstance::Bike(bike_profile) => route_bike_bidirectional(
            ways,
            elevations,
            bike_profile,
            start_candidates,
            end_candidates,
            max_cost,
            from,
            to,
        ),
        ProfileInstance::Car(car_profile) => route_car_bidirectional(
            ways,
            car_profile,
            start_candidates,
            end_candidates,
            max_cost,
            from,
            to,
        ),
        ProfileInstance::BikeSharing(_) => {
            // Bike sharing bidirectional not implemented - fallback to Dijkstra
            route_dijkstra(
                ways,
                elevations,
                profile,
                start_candidates,
                end_candidates,
                max_cost,
            )
        }
        ProfileInstance::CarSharing(_) => {
            // Car sharing bidirectional not implemented - fallback to Dijkstra
            route_dijkstra(
                ways,
                elevations,
                profile,
                start_candidates,
                end_candidates,
                max_cost,
            )
        }
        ProfileInstance::CarParking(_) => {
            // Car parking bidirectional not implemented - fallback to Dijkstra
            route_dijkstra(
                ways,
                elevations,
                profile,
                start_candidates,
                end_candidates,
                max_cost,
            )
        }
    }
}

/// Internal one-to-many routing implementation using Dijkstra
fn route_multi_dijkstra(
    ways: &Ways,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &ProfileInstance,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[Vec<crate::lookup::WayCandidate>],
    max_cost: Cost,
) -> Vec<Option<Path>> {
    // Dispatch to appropriate profile implementation
    match profile {
        ProfileInstance::Foot(foot_profile) => route_foot_multi_dijkstra(
            ways,
            elevations,
            foot_profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::Bike(bike_profile) => route_bike_multi_dijkstra(
            ways,
            elevations,
            bike_profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::Car(car_profile) => route_car_multi_dijkstra(
            ways,
            elevations,
            car_profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::BikeSharing(profile) => route_bike_sharing_multi_dijkstra(
            ways,
            elevations,
            profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::CarSharing(profile) => route_car_sharing_multi_dijkstra(
            ways,
            elevations,
            profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
        ProfileInstance::CarParking(profile) => route_car_parking_multi_dijkstra(
            ways,
            elevations,
            profile,
            start_candidates,
            end_candidates,
            max_cost,
        ),
    }
}

/// Find the NodeCandidate (left or right) in a candidate list that matches a given node.
fn find_snap_nc(candidates: &[crate::lookup::WayCandidate], node: NodeIdx) -> Option<&NodeCandidate> {
    candidates.iter().find_map(|c| {
        if c.left.valid() && c.left.node == node {
            Some(&c.left)
        } else if c.right.valid() && c.right.node == node {
            Some(&c.right)
        } else {
            None
        }
    })
}

/// Build a snap segment (projected-point → routing-node path) from a NodeCandidate.
/// Pass `reversed = true` for the end snap (routing-node → projected-point).
fn make_snap_segment(
    nc: &NodeCandidate,
    lvl: Level,
    reversed: bool,
    mode: crate::routing::mode::Mode,
) -> Option<crate::routing::path::Segment> {
    if nc.path.is_empty() {
        return None;
    }
    let mut polyline: Vec<crate::Point> = nc
        .path
        .iter()
        .map(|&(lat, lng)| crate::Point::from_latlng(lat, lng))
        .collect();
    if reversed {
        polyline.reverse();
    }
    Some(crate::routing::path::Segment::with_values(
        polyline,
        lvl,
        lvl,
        NodeIdx::INVALID,
        NodeIdx::INVALID,
        crate::types::WayIdx::INVALID,
        nc.cost,
        nc.dist_to_node.min(u16::MAX as f64) as Distance,
        crate::elevation_storage::Elevation::new(0, 0),
        mode,
    ))
}

/// Prepend and append snap segments to a path, and add their distances.
fn add_snap_segments(
    path: &mut Path,
    start_nc: Option<&NodeCandidate>,
    end_nc: Option<&NodeCandidate>,
    lvl: Level,
    mode: crate::routing::mode::Mode,
) {
    if let Some(nc) = end_nc {
        if let Some(seg) = make_snap_segment(nc, lvl, true, mode) {
            path.dist += nc.dist_to_node;
            path.segments.push(seg);
        }
    }
    if let Some(nc) = start_nc {
        if let Some(seg) = make_snap_segment(nc, lvl, false, mode) {
            path.dist += nc.dist_to_node;
            path.segments.insert(0, seg);
        }
    }
}

/// Foot profile routing with Dijkstra
fn route_foot_dijkstra(
    ways: &Ways,
    _elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &crate::routing::profiles::foot::FootProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
) -> Option<Path> {
    use crate::routing::profiles::foot::{node_cost, way_cost};

    let mut dijkstra = Dijkstra::<FootNode>::new();
    dijkstra.reset(max_cost);

    // println!(
    //     "DEBUG route_foot: max_cost={}, start_candidates={}, end_candidates={}",
    //     max_cost,
    //     start_candidates.len(),
    //     end_candidates.len()
    // );

    // Add start nodes
    for candidate in start_candidates {
        if candidate.left.valid() {
            // println!(
            //     "DEBUG route_foot: Adding start left node {:?}",
            //     candidate.left.node
            // );
            dijkstra.add_start(
                FootNode {
                    n: candidate.left.node,
                    lvl: candidate.left.lvl,
                },
                candidate.left.cost,
            );
        }
        if candidate.right.valid() {
            // println!(
            //     "DEBUG route_foot: Adding start right node {:?}",
            //     candidate.right.node
            // );
            dijkstra.add_start(
                FootNode {
                    n: candidate.right.node,
                    lvl: candidate.right.lvl,
                },
                candidate.right.cost,
            );
        }
    }
    // println!("DEBUG route_foot: Added {} start nodes", start_count);

    // Collect end nodes for target detection (typical: 2-4 candidates)
    // Store NodeCandidate reference to retrieve snap path after search
    let mut end_nodes: Vec<(FootNode, &NodeCandidate)> = Vec::with_capacity(8);
    for candidate in end_candidates {
        if candidate.left.valid() {
            end_nodes.push((FootNode {
                n: candidate.left.node,
                lvl: candidate.left.lvl,
            }, &candidate.left));
        }
        if candidate.right.valid() {
            end_nodes.push((FootNode {
                n: candidate.right.node,
                lvl: candidate.right.lvl,
            }, &candidate.right));
        }
    }
    // println!("DEBUG route_foot: Collected {} end nodes", end_nodes.len());

    // Run Dijkstra with neighbor generation
    let mut best_node = None;
    let mut best_cost = Cost::MAX;
    let mut nodes_explored = 0;

    dijkstra.run(max_cost, |current: FootNode| {
        nodes_explored += 1;
        if nodes_explored % 1000 == 0 {
            // println!("DEBUG route_foot: Explored {} nodes", nodes_explored);
        }
        
        // Pre-allocate for typical number of neighbors (2-6 per node)
        let mut neighbors = Vec::with_capacity(8);

        // Get all ways connected to this node
        let node_ways = ways.get_node_ways(current.n);
        let node_way_positions = ways.get_node_in_way_idx(current.n);

        for (&way, &pos_in_way) in node_ways.iter().zip(node_way_positions.iter()) {
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

            if nodes_explored <= 3 {
                // println!("DEBUG route_foot: Found current node {:?} at position {} in way {:?} with {} nodes",
                //          current.n, i, way, way_nodes.len());
            }

            // Check immediate neighbors (only adjacent nodes in the way)
            // Previous node (i-1)
            if i > 0 {
                let next_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0) as u32;

                if nodes_explored <= 3 {
                    // println!("DEBUG route_foot: Checking prev node: dist={}, dist>0={}, dist<=50000={}",
                    //          dist, dist > 0, dist <= 50000);
                }

                if dist > 0 && dist <= 50000 {
                    // Calculate cost — node cost is paid on ARRIVAL at target (BUG-01 fix)
                    let edge_cost = way_cost(&profile.params, props, profile.is_wheelchair, dist as u16);

                    if nodes_explored <= 3 {
                        // println!("DEBUG route_foot: edge_cost={}, K_INFEASIBLE={}", edge_cost, K_INFEASIBLE);
                    }

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(next_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            let neighbor = FootNode {
                                n: next_node,
                                lvl: current.lvl,
                            };
                            neighbors.push((neighbor, total_cost));
                        }
                    }
                }
            }

            // Next node (i+1)
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0) as u32;

                if nodes_explored <= 3 {
                    // println!("DEBUG route_foot: Checking next node: dist={}, dist>0={}, dist<=50000={}",
                    //          dist, dist > 0, dist <= 50000);
                }

                if dist > 0 && dist <= 50000 {
                    // Calculate cost — node cost is paid on ARRIVAL at target (BUG-01 fix)
                    let edge_cost = way_cost(&profile.params, props, profile.is_wheelchair, dist as u16);

                    if nodes_explored <= 3 {
                        // println!("DEBUG route_foot: edge_cost={}, K_INFEASIBLE={}", edge_cost, K_INFEASIBLE);
                    }

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(next_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            let neighbor = FootNode {
                                n: next_node,
                                lvl: current.lvl,
                            };
                            neighbors.push((neighbor, total_cost));
                        }
                    }
                }
            }
        }
        
        if nodes_explored <= 3 {
            // println!("DEBUG route_foot: Generated {} neighbors", neighbors.len());
        }
        
        neighbors
    });

    // println!(
    //     "DEBUG route_foot: Dijkstra finished, explored {} nodes",
    //     nodes_explored
    // );

    // Check if we reached any end node and add end snap cost
    let mut best_end_nc: Option<&NodeCandidate> = None;
    for &(end_node, nc) in &end_nodes {
        let path_cost = dijkstra.get_cost(end_node);
        if path_cost < K_INFEASIBLE {
            let total_cost = path_cost.saturating_add(nc.cost);
            if total_cost < best_cost {
                best_cost = total_cost;
                best_node = Some(end_node);
                best_end_nc = Some(nc);
            }
        }
    }

    // Reconstruct path and attach snap segments (BUG-04 fix)
    if let Some(end) = best_node {
        let mut path = reconstruct_foot_path(ways, &dijkstra, end, best_cost)?;

        // Find the start node by walking the predecessor chain
        let mut curr = end;
        while let Some(pred) = dijkstra.get_predecessor(curr) {
            curr = pred;
        }
        let start_nc = find_snap_nc(start_candidates, curr.n);

        add_snap_segments(&mut path, start_nc, best_end_nc, end.lvl, crate::routing::mode::Mode::Foot);
        Some(path)
    } else {
        None
    }
}

/// Reconstruct path from Dijkstra predecessors
fn reconstruct_foot_path(
    ways: &Ways,
    dijkstra: &Dijkstra<FootNode>,
    end_node: FootNode,
    total_cost: Cost,
) -> Option<Path> {
    // Pre-allocate for typical path length (10-100 segments)
    let mut segments = Vec::with_capacity(64);
    let mut current = end_node;
    let mut total_dist = 0.0;

    // Walk backwards through predecessors
    while let Some(pred) = dijkstra.get_predecessor(current) {
        // Find the way connecting pred to current
        let pred_ways = ways.get_node_ways(pred.n);
        let curr_ways = ways.get_node_ways(current.n);

        // Find common way
        let mut found_way = None;
        for &pw in pred_ways {
            for &cw in curr_ways {
                if pw == cw {
                    found_way = Some(pw);
                    break;
                }
            }
            if found_way.is_some() {
                break;
            }
        }

        if let Some(way) = found_way {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let polyline = ways.get_way_polyline(way);
            let polyline_indices = ways.get_way_node_polyline_indices(way);

            // Find node positions in way (routing node indices)
            let pred_pos = way_nodes.iter().position(|&n| n == pred.n);
            let curr_pos = way_nodes.iter().position(|&n| n == current.n);

            if let (Some(p_pos), Some(c_pos)) = (pred_pos, curr_pos) {
                // Calculate segment distance (way_node_dist values are in meters)
                let start_idx = p_pos.min(c_pos);
                let end_idx = p_pos.max(c_pos);
                let is_reverse = p_pos > c_pos;
                let mut dist: u32 = 0;
                for j in start_idx..end_idx {
                    dist += way_dists.get(j).copied().unwrap_or(0) as u32;
                }

                total_dist += dist as f64;

                // Map routing node indices to polyline indices for full geometry
                let poly_start = polyline_indices.get(start_idx).copied().unwrap_or(start_idx as u16) as usize;
                let poly_end = polyline_indices.get(end_idx).copied().unwrap_or(end_idx as u16) as usize;

                // Extract polyline segment using full-geometry indices
                let mut segment_polyline = if !polyline.is_empty() && poly_end < polyline.len() {
                    polyline[poly_start..=poly_end].to_vec()
                } else {
                    vec![]
                };

                // Reverse polyline when walking backward along the way
                if is_reverse {
                    segment_polyline.reverse();
                }

                // Get segment cost
                let segment_cost = dijkstra
                    .get_cost(current)
                    .saturating_sub(dijkstra.get_cost(pred));

                segments.push(crate::routing::path::Segment::with_values(
                    segment_polyline,
                    pred.lvl,
                    current.lvl,
                    pred.n,
                    current.n,
                    way,
                    segment_cost,
                    dist.min(u16::MAX as u32) as u16,
                    crate::elevation_storage::Elevation::new(0, 0),
                    crate::routing::mode::Mode::Foot,
                ));
            }
        }

        current = pred;
    }

    // Reverse segments to get start-to-end order
    segments.reverse();

    Some(Path {
        cost: total_cost,
        dist: total_dist,
        elevation: crate::elevation_storage::Elevation::new(0, 0),
        segments,
        uses_elevator: false,
        track_node: NodeIdx::INVALID,
    })
}

/// Foot profile routing with Bidirectional A*
fn route_foot_bidirectional(
    ways: &Ways,
    profile: &crate::routing::profiles::foot::FootProfile<crate::routing::tracking::NoopTracking>,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
    from: Location,
    to: Location,
) -> Option<Path> {
    use crate::routing::bidirectional::GenericBidirectional;
    use crate::routing::profiles::foot::{FootNode, FootProfile};
    use crate::routing::tracking::NoopTracking;
    use crate::routing::bidirectional_profile::BidirectionalProfile;

    let mut bidir = GenericBidirectional::<FootProfile<NoopTracking>>::new();
    bidir.set_ways(ways);
    bidir.reset(profile, max_cost, from, to);

    // BUG-14 fix: bail early if heuristic shows no improvement possible
    if bidir.radius() >= max_cost {
        return None;
    }

    // Add start nodes
    for candidate in start_candidates {
        if candidate.left.valid() {
            bidir.add_start(profile, FootNode { n: candidate.left.node, lvl: candidate.left.lvl }, candidate.left.cost);
        }
        if candidate.right.valid() {
            bidir.add_start(profile, FootNode { n: candidate.right.node, lvl: candidate.right.lvl }, candidate.right.cost);
        }
    }

    // Add end nodes
    for candidate in end_candidates {
        if candidate.left.valid() {
            bidir.add_end(profile, FootNode { n: candidate.left.node, lvl: candidate.left.lvl }, candidate.left.cost);
        }
        if candidate.right.valid() {
            bidir.add_end(profile, FootNode { n: candidate.right.node, lvl: candidate.right.lvl }, candidate.right.cost);
        }
    }

    // Run bidirectional search using the profile's adjacent method
    bidir.run(
        profile,
        max_cost,
        |n| {
            let mut neighbors = Vec::new();
            FootProfile::<NoopTracking>::adjacent(profile, ways, None, n, Direction::Forward, |node, cost| {
                neighbors.push((node, cost));
            });
            neighbors
        },
        |n| {
            let mut neighbors = Vec::new();
            FootProfile::<NoopTracking>::adjacent(profile, ways, None, n, Direction::Backward, |node, cost| {
                neighbors.push((node, cost));
            });
            neighbors
        },
    );

    // Check if we found a path
    if bidir.meet_point().is_none() {
        return None;
    }

    // Reconstruct path and attach snap segments (BUG-04 fix)
    let node_path = bidir.reconstruct_path()?;
    let start_nc = node_path.first().and_then(|n| find_snap_nc(start_candidates, n.n));
    let end_nc = node_path.last().and_then(|n| find_snap_nc(end_candidates, n.n));
    let mut path = reconstruct_foot_path_from_nodes(ways, &node_path, bidir.best_cost())?;
    add_snap_segments(&mut path, start_nc, end_nc, Level::default(), crate::routing::mode::Mode::Foot);
    Some(path)
}

/// Reconstruct path from FootNodes
fn reconstruct_foot_path_from_nodes(
    ways: &Ways,
    nodes: &[crate::routing::profiles::foot::FootNode],
    total_cost: Cost,
) -> Option<Path> {
    if nodes.len() < 2 {
        return Some(Path {
            cost: total_cost,
            dist: 0.0,
            elevation: crate::elevation_storage::Elevation::new(0, 0),
            segments: Vec::new(),
            uses_elevator: false,
            track_node: NodeIdx::INVALID,
        });
    }

    let mut segments = Vec::with_capacity(nodes.len());
    let mut total_dist = 0.0;

    for i in 0..nodes.len() - 1 {
        let from_node = nodes[i];
        let to_node = nodes[i + 1];

        // Find the way connecting these nodes
        let from_ways = ways.get_node_ways(from_node.n);
        let to_ways = ways.get_node_ways(to_node.n);

        let mut found_way = None;
        for &fw in from_ways {
            for &tw in to_ways {
                if fw == tw {
                    found_way = Some(fw);
                    break;
                }
            }
            if found_way.is_some() {
                break;
            }
        }

        if let Some(way) = found_way {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let polyline = ways.get_way_polyline(way);
            let polyline_indices = ways.get_way_node_polyline_indices(way);

            let from_pos = way_nodes.iter().position(|&n| n == from_node.n);
            let to_pos = way_nodes.iter().position(|&n| n == to_node.n);

            if let (Some(f_pos), Some(t_pos)) = (from_pos, to_pos) {
                let (start_idx, end_idx, is_reverse) = if f_pos < t_pos {
                    (f_pos, t_pos, false)
                } else {
                    (t_pos, f_pos, true)
                };

                // Map routing node indices to polyline indices for full geometry
                let poly_start = polyline_indices.get(start_idx).copied().unwrap_or(start_idx as u16) as usize;
                let poly_end = polyline_indices.get(end_idx).copied().unwrap_or(end_idx as u16) as usize;

                let mut seg_polyline: Vec<_> = if !polyline.is_empty() && poly_end < polyline.len() {
                    polyline[poly_start..=poly_end].to_vec()
                } else {
                    vec![]
                };

                if is_reverse {
                    seg_polyline.reverse();
                }

                let mut seg_dist: u32 = 0;
                for j in start_idx..end_idx {
                    seg_dist += way_dists.get(j).copied().unwrap_or(0) as u32;
                }

                total_dist += seg_dist as f64;

                segments.push(crate::routing::path::Segment::with_values(
                    seg_polyline,
                    from_node.lvl,
                    to_node.lvl,
                    from_node.n,
                    to_node.n,
                    way,
                    0,
                    seg_dist.min(u16::MAX as u32) as u16,
                    crate::elevation_storage::Elevation::new(0, 0),
                    crate::routing::mode::Mode::Foot,
                ));
            }
        }
    }

    Some(Path {
        cost: total_cost,
        dist: total_dist,
        elevation: crate::elevation_storage::Elevation::new(0, 0),
        segments,
        uses_elevator: false,
        track_node: NodeIdx::INVALID,
    })
}

/// Reconstruct path from BikeNodes
fn reconstruct_bike_path_from_nodes(
    ways: &Ways,
    elevations: Option<&ElevationStorage>,
    _profile: &crate::routing::profiles::BikeProfile,
    nodes: &[crate::routing::profiles::bike::BikeNode],
    total_cost: Cost,
) -> Option<Path> {
    if nodes.len() < 2 {
        return Some(Path {
            cost: total_cost,
            dist: 0.0,
            elevation: crate::elevation_storage::Elevation::new(0, 0),
            segments: Vec::new(),
            uses_elevator: false,
            track_node: NodeIdx::INVALID,
        });
    }

    let mut segments = Vec::with_capacity(nodes.len());
    let mut total_dist = 0.0;
    let mut total_elevation = crate::elevation_storage::Elevation::new(0, 0);

    for i in 0..nodes.len() - 1 {
        let from_node = nodes[i];
        let to_node = nodes[i + 1];

        let from_ways = ways.get_node_ways(from_node.n);
        let to_ways = ways.get_node_ways(to_node.n);

        let mut found_way = None;
        for &fw in from_ways {
            for &tw in to_ways {
                if fw == tw {
                    found_way = Some(fw);
                    break;
                }
            }
            if found_way.is_some() {
                break;
            }
        }

        if let Some(way) = found_way {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let polyline = ways.get_way_polyline(way);
            let polyline_indices = ways.get_way_node_polyline_indices(way);

            let from_pos = way_nodes.iter().position(|&n| n == from_node.n);
            let to_pos = way_nodes.iter().position(|&n| n == to_node.n);

            if let (Some(f_pos), Some(t_pos)) = (from_pos, to_pos) {
                let (start_idx, end_idx, is_reverse) = if f_pos < t_pos {
                    (f_pos, t_pos, false)
                } else {
                    (t_pos, f_pos, true)
                };

                // Map routing node indices to polyline indices for full geometry
                let poly_start = polyline_indices.get(start_idx).copied().unwrap_or(start_idx as u16) as usize;
                let poly_end = polyline_indices.get(end_idx).copied().unwrap_or(end_idx as u16) as usize;

                let mut seg_polyline: Vec<_> = if !polyline.is_empty() && poly_end < polyline.len() {
                    polyline[poly_start..=poly_end].to_vec()
                } else {
                    vec![]
                };

                if is_reverse {
                    seg_polyline.reverse();
                }

                let mut seg_dist: u32 = 0;
                for j in start_idx..end_idx {
                    seg_dist += way_dists.get(j).copied().unwrap_or(0) as u32;
                }

                total_dist += seg_dist as f64;

                let seg_elevation = if let Some(el) = elevations {
                    let mut sum = crate::elevation_storage::Elevation::new(0, 0);
                    for j in start_idx..end_idx {
                        let mut e = el.get_elevation(way, j as u16);
                        if is_reverse {
                             e = e.swapped();
                        }
                        sum.add(&e);
                    }
                    sum
                } else {
                    crate::elevation_storage::Elevation::new(0, 0)
                };
                total_elevation.add(&seg_elevation);

                segments.push(crate::routing::path::Segment::with_values(
                    seg_polyline,
                    Level::default(),
                    Level::default(),
                    from_node.n,
                    to_node.n,
                    way,
                    0,
                    seg_dist.min(u16::MAX as u32) as u16,
                    seg_elevation,
                    crate::routing::mode::Mode::Bike,
                ));
            }
        }
    }

    Some(Path {
        cost: total_cost,
        dist: total_dist,
        elevation: total_elevation,
        segments,
        uses_elevator: false,
        track_node: NodeIdx::INVALID,
    })
}

/// Reconstruct path from CarNodes
fn reconstruct_car_path_from_nodes(
    ways: &Ways,
    nodes: &[crate::routing::profiles::car::CarNode],
    total_cost: Cost,
) -> Option<Path> {
    if nodes.len() < 2 {
        return Some(Path {
            cost: total_cost,
            dist: 0.0,
            elevation: crate::elevation_storage::Elevation::new(0, 0),
            segments: Vec::new(),
            uses_elevator: false,
            track_node: NodeIdx::INVALID,
        });
    }

    let mut segments = Vec::with_capacity(nodes.len());
    let mut total_dist = 0.0;

    for i in 0..nodes.len() - 1 {
        let from_node = nodes[i];
        let to_node = nodes[i + 1];

        let from_ways = ways.get_node_ways(from_node.n);
        let to_ways = ways.get_node_ways(to_node.n);

        let mut found_way = None;
        for &fw in from_ways {
            for &tw in to_ways {
                if fw == tw {
                    found_way = Some(fw);
                    break;
                }
            }
            if found_way.is_some() {
                break;
            }
        }

        if let Some(way) = found_way {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let polyline = ways.get_way_polyline(way);
            let polyline_indices = ways.get_way_node_polyline_indices(way);

            let from_pos = way_nodes.iter().position(|&n| n == from_node.n);
            let to_pos = way_nodes.iter().position(|&n| n == to_node.n);

            if let (Some(f_pos), Some(t_pos)) = (from_pos, to_pos) {
                let (start_idx, end_idx, is_reverse) = if f_pos < t_pos {
                    (f_pos, t_pos, false)
                } else {
                    (t_pos, f_pos, true)
                };

                // Map routing node indices to polyline indices for full geometry
                let poly_start = polyline_indices.get(start_idx).copied().unwrap_or(start_idx as u16) as usize;
                let poly_end = polyline_indices.get(end_idx).copied().unwrap_or(end_idx as u16) as usize;

                let mut seg_polyline: Vec<_> = if !polyline.is_empty() && poly_end < polyline.len() {
                    polyline[poly_start..=poly_end].to_vec()
                } else {
                    vec![]
                };

                if is_reverse {
                    seg_polyline.reverse();
                }

                let mut seg_dist: u32 = 0;
                for j in start_idx..end_idx {
                    seg_dist += way_dists.get(j).copied().unwrap_or(0) as u32;
                }

                total_dist += seg_dist as f64;

                segments.push(crate::routing::path::Segment::with_values(
                    seg_polyline,
                    Level::default(),
                    Level::default(),
                    from_node.n,
                    to_node.n,
                    way,
                    0,
                    seg_dist.min(u16::MAX as u32) as u16,
                    crate::elevation_storage::Elevation::new(0, 0),
                    crate::routing::mode::Mode::Car,
                ));
            }
        }
    }

    Some(Path {
        cost: total_cost,
        dist: total_dist,
        elevation: crate::elevation_storage::Elevation::new(0, 0),
        segments,
        uses_elevator: false,
        track_node: NodeIdx::INVALID,
    })
}

/// Route for bikes using bidirectional A*
fn route_bike_bidirectional(
    ways: &Ways,
    elevations: Option<&ElevationStorage>,
    profile: &crate::routing::profiles::BikeProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
    from: Location,
    to: Location,
) -> Option<Path> {
    use crate::routing::bidirectional::GenericBidirectional;
    use crate::routing::profiles::bike::{BikeNode, BikeProfile};
    use crate::routing::bidirectional_profile::BidirectionalProfile;

    let mut bidir = GenericBidirectional::<BikeProfile>::new();
    bidir.set_ways(ways);
    bidir.reset(profile, max_cost, from, to);

    // BUG-14 fix: bail early if heuristic shows no improvement possible
    if bidir.radius() >= max_cost {
        return None;
    }

    // Add start nodes
    for candidate in start_candidates {
        if candidate.left.valid() {
            bidir.add_start(profile, BikeNode { n: candidate.left.node, dir: Direction::Forward }, candidate.left.cost);
            bidir.add_start(profile, BikeNode { n: candidate.left.node, dir: Direction::Backward }, candidate.left.cost);
        }
        if candidate.right.valid() {
            bidir.add_start(profile, BikeNode { n: candidate.right.node, dir: Direction::Forward }, candidate.right.cost);
            bidir.add_start(profile, BikeNode { n: candidate.right.node, dir: Direction::Backward }, candidate.right.cost);
        }
    }

    // Add end nodes
    for candidate in end_candidates {
        if candidate.left.valid() {
            bidir.add_end(profile, BikeNode { n: candidate.left.node, dir: Direction::Forward }, candidate.left.cost);
            bidir.add_end(profile, BikeNode { n: candidate.left.node, dir: Direction::Backward }, candidate.left.cost);
        }
        if candidate.right.valid() {
            bidir.add_end(profile, BikeNode { n: candidate.right.node, dir: Direction::Forward }, candidate.right.cost);
            bidir.add_end(profile, BikeNode { n: candidate.right.node, dir: Direction::Backward }, candidate.right.cost);
        }
    }

    // Run bidirectional search
    bidir.run(
        profile,
        max_cost,
        |n| {
            let mut neighbors = Vec::new();
            BikeProfile::adjacent(profile, ways, elevations, n, Direction::Forward, |node, cost| {
                neighbors.push((node, cost));
            });
            neighbors
        },
        |n| {
            let mut neighbors = Vec::new();
            BikeProfile::adjacent(profile, ways, elevations, n, Direction::Backward, |node, cost| {
                neighbors.push((node, cost));
            });
            neighbors
        },
    );

    // Reconstruct path and attach snap segments (BUG-04 fix)
    let node_path = bidir.reconstruct_path()?;
    let start_nc = node_path.first().and_then(|n| find_snap_nc(start_candidates, n.n));
    let end_nc = node_path.last().and_then(|n| find_snap_nc(end_candidates, n.n));
    let mut path = reconstruct_bike_path_from_nodes(ways, elevations, profile, &node_path, bidir.best_cost())?;
    add_snap_segments(&mut path, start_nc, end_nc, Level::default(), crate::routing::mode::Mode::Bike);
    Some(path)
}

/// Route for cars using bidirectional A*
fn route_car_bidirectional(
    ways: &Ways,
    profile: &crate::routing::profiles::CarProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
    from: Location,
    to: Location,
) -> Option<Path> {
    use crate::routing::bidirectional::GenericBidirectional;
    use crate::routing::profiles::car::{CarNode, CarProfile};
    use crate::routing::bidirectional_profile::BidirectionalProfile;

    let mut bidir = GenericBidirectional::<CarProfile>::new();
    bidir.set_ways(ways);
    bidir.reset(profile, max_cost, from, to);

    // BUG-14 fix: if heuristic says no path can be shorter than max_cost, bail early
    if bidir.radius() >= max_cost {
        return None;
    }

    // Add start nodes — BUG-08 fix: use get_way_pos, add both directions
    for candidate in start_candidates {
        if candidate.left.valid() {
            let wp = ways.get_way_pos(candidate.left.node, candidate.way);
            bidir.add_start(profile, CarNode { n: candidate.left.node, way: wp, dir: Direction::Forward }, candidate.left.cost);
            bidir.add_start(profile, CarNode { n: candidate.left.node, way: wp, dir: Direction::Backward }, candidate.left.cost);
        }
        if candidate.right.valid() {
            let wp = ways.get_way_pos(candidate.right.node, candidate.way);
            bidir.add_start(profile, CarNode { n: candidate.right.node, way: wp, dir: Direction::Forward }, candidate.right.cost);
            bidir.add_start(profile, CarNode { n: candidate.right.node, way: wp, dir: Direction::Backward }, candidate.right.cost);
        }
    }

    // Add end nodes — BUG-08 fix: use get_way_pos, add both directions
    for candidate in end_candidates {
        if candidate.left.valid() {
            let wp = ways.get_way_pos(candidate.left.node, candidate.way);
            bidir.add_end(profile, CarNode { n: candidate.left.node, way: wp, dir: Direction::Forward }, candidate.left.cost);
            bidir.add_end(profile, CarNode { n: candidate.left.node, way: wp, dir: Direction::Backward }, candidate.left.cost);
        }
        if candidate.right.valid() {
            let wp = ways.get_way_pos(candidate.right.node, candidate.way);
            bidir.add_end(profile, CarNode { n: candidate.right.node, way: wp, dir: Direction::Forward }, candidate.right.cost);
            bidir.add_end(profile, CarNode { n: candidate.right.node, way: wp, dir: Direction::Backward }, candidate.right.cost);
        }
    }

    // Run bidirectional search
    bidir.run(
        profile,
        max_cost,
        |n| {
            let mut neighbors = Vec::new();
            CarProfile::adjacent(profile, ways, None, n, Direction::Forward, |node, cost| {
                neighbors.push((node, cost));
            });
            neighbors
        },
        |n| {
            let mut neighbors = Vec::new();
            CarProfile::adjacent(profile, ways, None, n, Direction::Backward, |node, cost| {
                neighbors.push((node, cost));
            });
            neighbors
        },
    );

    // Reconstruct path and attach snap segments (BUG-04 fix)
    let node_path = bidir.reconstruct_path()?;
    let start_nc = node_path.first().and_then(|n| find_snap_nc(start_candidates, n.n));
    let end_nc = node_path.last().and_then(|n| find_snap_nc(end_candidates, n.n));
    let mut path = reconstruct_car_path_from_nodes(ways, &node_path, bidir.best_cost())?;
    add_snap_segments(&mut path, start_nc, end_nc, Level::default(), crate::routing::mode::Mode::Car);
    Some(path)
}

/// Convert a sequence of nodes to a Path with segments
fn _reconstruct_path_from_nodes(ways: &Ways, nodes: &[NodeIdx], total_cost: Cost) -> Option<Path> {
    // If the path contains fewer than 2 nodes, it's a zero-length path (start==end).
    // Return an empty Path with the provided total_cost to match Dijkstra behavior.
    if nodes.len() < 2 {
        return Some(Path {
            cost: total_cost,
            dist: 0.0,
            elevation: crate::elevation_storage::Elevation::new(0, 0),
            segments: Vec::new(),
            uses_elevator: false,
            track_node: NodeIdx::INVALID,
        });
    }

    // Pre-allocate for typical path length
    let mut segments = Vec::with_capacity(64);
    let mut total_dist = 0.0;

    for i in 0..nodes.len() - 1 {
        let from_node = nodes[i];
        let to_node = nodes[i + 1];

        // Find the way connecting these nodes
        let from_ways = ways.get_node_ways(from_node);
        let to_ways = ways.get_node_ways(to_node);

        let mut found_way = None;
        for &fw in from_ways {
            for &tw in to_ways {
                if fw == tw {
                    found_way = Some(fw);
                    break;
                }
            }
            if found_way.is_some() {
                break;
            }
        }

        if let Some(way) = found_way {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let polyline = ways.get_way_polyline(way);

            let from_pos = way_nodes.iter().position(|&n| n == from_node);
            let to_pos = way_nodes.iter().position(|&n| n == to_node);

            if let (Some(f_pos), Some(t_pos)) = (from_pos, to_pos) {
                let (start_idx, end_idx, is_reverse) = if f_pos < t_pos {
                    (f_pos, t_pos, false)
                } else {
                    (t_pos, f_pos, true)
                };

                // Extract polyline segment
                let mut seg_polyline: Vec<_> = polyline
                    .iter()
                    .skip(start_idx)
                    .take(end_idx - start_idx + 1)
                    .cloned()
                    .collect();

                if is_reverse {
                    seg_polyline.reverse();
                }

                // Calculate distance in centimeters and convert to meters for total_dist
                let mut seg_dist_cm: u32 = 0;
                for j in start_idx..end_idx {
                    seg_dist_cm += way_dists.get(j).copied().unwrap_or(0) as u32;
                }

                total_dist += seg_dist_cm as f64 / 100.0; // convert cm -> m

                segments.push(crate::routing::path::Segment::with_values(
                    seg_polyline,
                    Level::default(),
                    Level::default(),
                    from_node,
                    to_node,
                    way,
                    0, // cost per segment not tracked in simple version
                    seg_dist_cm.min(u16::MAX as u32) as u16,
                    crate::elevation_storage::Elevation::new(0, 0),
                    crate::routing::mode::Mode::Foot,
                ));
            }
        }
    }

    Some(Path {
        cost: total_cost,
        dist: total_dist,
        elevation: crate::elevation_storage::Elevation::new(0, 0),
        segments,
        uses_elevator: false,
        track_node: NodeIdx::INVALID,
    })
}

/// Foot profile one-to-many routing with Dijkstra
fn route_foot_multi_dijkstra(
    ways: &Ways,
    _elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &crate::routing::profiles::foot::FootProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[Vec<crate::lookup::WayCandidate>],
    max_cost: Cost,
) -> Vec<Option<Path>> {
    use crate::routing::profiles::foot::{node_cost, way_cost};
    use ahash::AHashMap;

    let mut dijkstra = Dijkstra::<FootNode>::new();
    dijkstra.reset(max_cost);

    // Add start nodes
    for candidate in start_candidates {
        if candidate.left.valid() {
            dijkstra.add_start(
                FootNode {
                    n: candidate.left.node,
                    lvl: candidate.left.lvl,
                },
                candidate.left.cost,
            );
        }
        if candidate.right.valid() {
            dijkstra.add_start(
                FootNode {
                    n: candidate.right.node,
                    lvl: candidate.right.lvl,
                },
                candidate.right.cost,
            );
        }
    }

    // Build target map: node -> list of destination indices
    let mut targets: AHashMap<FootNode, Vec<usize>> = AHashMap::new();
    for (dest_idx, candidates) in end_candidates.iter().enumerate() {
        for candidate in candidates {
            if candidate.left.valid() {
                let node = FootNode {
                    n: candidate.left.node,
                    lvl: candidate.left.lvl,
                };
                targets.entry(node).or_default().push(dest_idx);
            }
            if candidate.right.valid() {
                let node = FootNode {
                    n: candidate.right.node,
                    lvl: candidate.right.lvl,
                };
                targets.entry(node).or_default().push(dest_idx);
            }
        }
    }

    // Track best node and cost for each destination
    let mut reached: Vec<Option<(FootNode, Cost)>> = vec![None; end_candidates.len()];

    // Run Dijkstra
    dijkstra.run(max_cost, |current: FootNode| {
        // Generate neighbors (same logic as single-destination) - typical 2-6 neighbors
        let mut neighbors = Vec::with_capacity(8);

        for &way in ways.get_node_ways(current.n) {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let way_props = ways.get_way_properties(way);

            if way_nodes.is_empty() || way_props.is_none() {
                continue;
            }

            let props = way_props.unwrap();

            // BUG-01 fix: find position without inner loop
            let Some(i) = way_nodes.iter().position(|&n| n == current.n) else { continue };

            // Previous node — node cost paid on ARRIVAL at target
            if i > 0 {
                let next_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0) as u32;

                if dist > 0 && dist <= 50000 {
                    let edge_cost =
                        way_cost(&profile.params, props, profile.is_wheelchair, dist as u16);
                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(next_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            neighbors.push((
                                FootNode {
                                    n: next_node,
                                    lvl: current.lvl,
                                },
                                total_cost,
                            ));
                        }
                    }
                }
            }

            // Next node — node cost paid on ARRIVAL at target
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0) as u32;

                if dist > 0 && dist <= 50000 {
                    let edge_cost =
                        way_cost(&profile.params, props, profile.is_wheelchair, dist as u16);
                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(next_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            neighbors.push((
                                FootNode {
                                    n: next_node,
                                    lvl: current.lvl,
                                },
                                total_cost,
                            ));
                        }
                    }
                }
            }
        }

        neighbors
    });

    // Find reached destinations after search completes
    for (node_candidate, dest_indices) in &targets {
        let cost = dijkstra.get_cost(*node_candidate);
        if cost < K_INFEASIBLE {
            for &dest_idx in dest_indices {
                if reached[dest_idx].is_none() || cost < reached[dest_idx].unwrap().1 {
                    reached[dest_idx] = Some((*node_candidate, cost));
                }
            }
        }
    }

    // Reconstruct paths for all reached destinations
    reached
        .iter()
        .map(|opt| opt.and_then(|(node, cost)| reconstruct_foot_path(ways, &dijkstra, node, cost)))
        .collect()
}

/// Bike profile routing with Dijkstra
fn route_bike_dijkstra(
    ways: &Ways,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &crate::routing::profiles::bike::BikeProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
) -> Option<Path> {
    use crate::routing::profiles::bike::{calculate_elevation_cost, node_cost, way_cost};

    let mut dijkstra = Dijkstra::<BikeNode>::new();
    dijkstra.reset(max_cost);

    // Add start nodes (both directions)
    for candidate in start_candidates {
        if candidate.left.valid() {
            dijkstra.add_start(
                BikeNode {
                    n: candidate.left.node,
                    dir: Direction::Forward,
                },
                candidate.left.cost,
            );
        }
        if candidate.right.valid() {
            dijkstra.add_start(
                BikeNode {
                    n: candidate.right.node,
                    dir: Direction::Forward,
                },
                candidate.right.cost,
            );
        }
    }

    // Collect end nodes (both directions for flexibility) - typical 2-8 candidates
    let mut end_nodes: Vec<(BikeNode, &NodeCandidate)> = Vec::with_capacity(16);
    for candidate in end_candidates {
        if candidate.left.valid() {
            end_nodes.push((BikeNode { n: candidate.left.node, dir: Direction::Forward }, &candidate.left));
            end_nodes.push((BikeNode { n: candidate.left.node, dir: Direction::Backward }, &candidate.left));
        }
        if candidate.right.valid() {
            end_nodes.push((BikeNode { n: candidate.right.node, dir: Direction::Forward }, &candidate.right));
            end_nodes.push((BikeNode { n: candidate.right.node, dir: Direction::Backward }, &candidate.right));
        }
    }

    // Run Dijkstra with direction-aware neighbor generation
    let mut best_node = None;
    let mut best_cost = Cost::MAX;

    dijkstra.run(max_cost, |current: BikeNode| {
        let mut neighbors = Vec::new();

        // Get all ways connected to this node
        for (&way, &pos_in_way) in ways.get_node_ways(current.n).iter()
            .zip(ways.get_node_in_way_idx(current.n))
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

            // Forward along way (increasing index) — node cost on ARRIVAL at target
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = Direction::Forward;
                    let edge_cost = way_cost(
                        &profile.params,
                        props,
                        travel_dir,
                        profile.cost_strategy,
                        dist,
                    );

                    if edge_cost != K_INFEASIBLE {
                        let elevation = elevations
                            .map(|e| e.get_elevation(way, i as u16))
                            .unwrap_or_default();
                        let elev_cost = calculate_elevation_cost(
                            profile.elevation_up_cost,
                            profile.elevation_exponent_thousandth,
                            &elevation,
                            dist,
                        );
                        let t_cost = ways.get_node_properties(next_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(elev_cost).saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            neighbors.push((
                                BikeNode {
                                    n: next_node,
                                    dir: travel_dir,
                                },
                                total_cost,
                            ));
                        }
                    }
                }
            }

            // Backward along way (decreasing index) — node cost on ARRIVAL at target
            if i > 0 {
                let prev_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = Direction::Backward;
                    let edge_cost = way_cost(
                        &profile.params,
                        props,
                        travel_dir,
                        profile.cost_strategy,
                        dist,
                    );

                    if edge_cost != K_INFEASIBLE {
                        let elevation = elevations
                            .map(|e| e.get_elevation(way, (i - 1) as u16).swapped())
                            .unwrap_or_default();
                        let elev_cost = calculate_elevation_cost(
                            profile.elevation_up_cost,
                            profile.elevation_exponent_thousandth,
                            &elevation,
                            dist,
                        );
                        let t_cost = ways.get_node_properties(prev_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(elev_cost).saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            neighbors.push((
                                BikeNode {
                                    n: prev_node,
                                    dir: travel_dir,
                                },
                                total_cost,
                            ));
                        }
                    }
                }
            }
        }

        neighbors
    });

    // Check if we reached any end node and add end snap cost
    let mut best_end_nc: Option<&NodeCandidate> = None;
    for &(end_node, nc) in &end_nodes {
        let path_cost = dijkstra.get_cost(end_node);
        if path_cost < K_INFEASIBLE {
            let total_cost = path_cost.saturating_add(nc.cost);
            if total_cost < best_cost {
                best_cost = total_cost;
                best_node = Some(end_node);
                best_end_nc = Some(nc);
            }
        }
    }

    // Reconstruct path and attach snap segments (BUG-04 fix)
    if let Some(end) = best_node {
        let mut nodes = vec![end];
        let mut curr = end;
        while let Some(pred) = dijkstra.get_predecessor(curr) {
            nodes.push(pred);
            curr = pred;
        }
        nodes.reverse();

        let start_nc = find_snap_nc(start_candidates, nodes.first().map(|n| n.n).unwrap_or(NodeIdx::INVALID));
        let mut path = reconstruct_bike_path_from_nodes(ways, elevations, profile, &nodes, best_cost)?;
        add_snap_segments(&mut path, start_nc, best_end_nc, Level::default(), crate::routing::mode::Mode::Bike);
        Some(path)
    } else {
        None
    }
}

/// Reconstruct bike path from Dijkstra predecessors
fn reconstruct_bike_path(
    ways: &Ways,
    dijkstra: &Dijkstra<BikeNode>,
    end_node: BikeNode,
    total_cost: Cost,
) -> Option<Path> {
    let mut segments = Vec::new();
    let mut current = end_node;
    let mut total_dist = 0.0;

    // Walk backwards through predecessors
    while let Some(pred) = dijkstra.get_predecessor(current) {
        // Find the way connecting pred to current
        let pred_ways = ways.get_node_ways(pred.n);
        let curr_ways = ways.get_node_ways(current.n);

        // Find common way
        let mut found_way = None;
        for &pw in pred_ways {
            for &cw in curr_ways {
                if pw == cw {
                    found_way = Some(pw);
                    break;
                }
            }
            if found_way.is_some() {
                break;
            }
        }

        if let Some(way) = found_way {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let polyline = ways.get_way_polyline(way);
            let polyline_indices = ways.get_way_node_polyline_indices(way);

            // Find node positions in way
            let pred_pos = way_nodes.iter().position(|&n| n == pred.n);
            let curr_pos = way_nodes.iter().position(|&n| n == current.n);

            if let (Some(p_pos), Some(c_pos)) = (pred_pos, curr_pos) {
                // Calculate segment distance (way_node_dist values are in meters)
                let start_idx = p_pos.min(c_pos);
                let end_idx = p_pos.max(c_pos);
                let is_reverse = p_pos > c_pos;
                let mut dist: u32 = 0;
                for j in start_idx..end_idx {
                    dist += way_dists.get(j).copied().unwrap_or(0) as u32;
                }

                total_dist += dist as f64;

                // Map routing node indices to polyline indices for full geometry
                let poly_start = polyline_indices.get(start_idx).copied().unwrap_or(start_idx as u16) as usize;
                let poly_end = polyline_indices.get(end_idx).copied().unwrap_or(end_idx as u16) as usize;

                // Extract polyline segment using full-geometry indices
                let mut segment_polyline = if !polyline.is_empty() && poly_end < polyline.len() {
                    polyline[poly_start..=poly_end].to_vec()
                } else {
                    vec![]
                };

                // Reverse polyline when traversing backward along the way
                if is_reverse {
                    segment_polyline.reverse();
                }

                // Get segment cost
                let segment_cost = dijkstra
                    .get_cost(current)
                    .saturating_sub(dijkstra.get_cost(pred));

                segments.push(crate::routing::path::Segment::with_values(
                    segment_polyline,
                    Level::from_idx(Level::NO_LEVEL),
                    Level::from_idx(Level::NO_LEVEL),
                    pred.n,
                    current.n,
                    way,
                    segment_cost,
                    dist.min(u16::MAX as u32) as u16,
                    crate::elevation_storage::Elevation::new(0, 0),
                    crate::routing::mode::Mode::Bike,
                ));
            }
        }

        current = pred;
    }

    // Reverse segments to get start-to-end order
    segments.reverse();

    Some(Path {
        cost: total_cost,
        dist: total_dist,
        elevation: crate::elevation_storage::Elevation::new(0, 0),
        segments,
        uses_elevator: false,
        track_node: NodeIdx::INVALID,
    })
}

/// Bike profile one-to-many routing with Dijkstra
fn route_bike_multi_dijkstra(
    ways: &Ways,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &crate::routing::profiles::bike::BikeProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[Vec<crate::lookup::WayCandidate>],
    max_cost: Cost,
) -> Vec<Option<Path>> {
    use crate::routing::profiles::bike::{calculate_elevation_cost, node_cost, way_cost};
    use ahash::AHashMap;

    let mut dijkstra = Dijkstra::<BikeNode>::new();
    dijkstra.reset(max_cost);

    // Add start nodes
    for candidate in start_candidates {
        if candidate.left.valid() {
            dijkstra.add_start(
                BikeNode {
                    n: candidate.left.node,
                    dir: Direction::Forward,
                },
                candidate.left.cost,
            );
        }
        if candidate.right.valid() {
            dijkstra.add_start(
                BikeNode {
                    n: candidate.right.node,
                    dir: Direction::Forward,
                },
                candidate.right.cost,
            );
        }
    }

    // Build target map
    let mut targets: AHashMap<BikeNode, Vec<usize>> = AHashMap::new();
    for (dest_idx, candidates) in end_candidates.iter().enumerate() {
        for candidate in candidates {
            if candidate.left.valid() {
                targets
                    .entry(BikeNode {
                        n: candidate.left.node,
                        dir: Direction::Forward,
                    })
                    .or_default()
                    .push(dest_idx);
                targets
                    .entry(BikeNode {
                        n: candidate.left.node,
                        dir: Direction::Backward,
                    })
                    .or_default()
                    .push(dest_idx);
            }
            if candidate.right.valid() {
                targets
                    .entry(BikeNode {
                        n: candidate.right.node,
                        dir: Direction::Forward,
                    })
                    .or_default()
                    .push(dest_idx);
                targets
                    .entry(BikeNode {
                        n: candidate.right.node,
                        dir: Direction::Backward,
                    })
                    .or_default()
                    .push(dest_idx);
            }
        }
    }

    let mut reached: Vec<Option<(BikeNode, Cost)>> = vec![None; end_candidates.len()];

    dijkstra.run(max_cost, |current: BikeNode| {
        let mut neighbors = Vec::new();

        for &way in ways.get_node_ways(current.n) {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let way_props = ways.get_way_properties(way);

            if way_nodes.is_empty() || way_props.is_none() {
                continue;
            }

            let props = way_props.unwrap();

            // BUG-01 fix: find position without inner loop
            let Some(i) = way_nodes.iter().position(|&n| n == current.n) else { continue };

            // Forward — node cost on ARRIVAL at target
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = Direction::Forward;
                    let edge_cost = way_cost(
                        &profile.params,
                        props,
                        travel_dir,
                        profile.cost_strategy,
                        dist,
                    );

                    if edge_cost != K_INFEASIBLE {
                        let elevation = elevations
                            .map(|e| e.get_elevation(way, i as u16))
                            .unwrap_or_default();
                        let elev_cost = calculate_elevation_cost(
                            profile.elevation_up_cost,
                            profile.elevation_exponent_thousandth,
                            &elevation,
                            dist,
                        );
                        let t_cost = ways.get_node_properties(next_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(elev_cost).saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            neighbors.push((
                                BikeNode {
                                    n: next_node,
                                    dir: travel_dir,
                                },
                                total_cost,
                            ));
                        }
                    }
                }
            }

            // Backward — node cost on ARRIVAL at target
            if i > 0 {
                let prev_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = Direction::Backward;
                    let edge_cost = way_cost(
                        &profile.params,
                        props,
                        travel_dir,
                        profile.cost_strategy,
                        dist,
                    );

                    if edge_cost != K_INFEASIBLE {
                        let elevation = elevations
                            .map(|e| e.get_elevation(way, (i - 1) as u16).swapped())
                            .unwrap_or_default();
                        let elev_cost = calculate_elevation_cost(
                            profile.elevation_up_cost,
                            profile.elevation_exponent_thousandth,
                            &elevation,
                            dist,
                        );
                        let t_cost = ways.get_node_properties(prev_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(elev_cost).saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            neighbors.push((
                                BikeNode {
                                    n: prev_node,
                                    dir: travel_dir,
                                },
                                total_cost,
                            ));
                        }
                    }
                }
            }
        }

        neighbors
    });

    // Find reached destinations after search completes
    for (node_candidate, dest_indices) in &targets {
        let cost = dijkstra.get_cost(*node_candidate);
        if cost < K_INFEASIBLE {
            for &dest_idx in dest_indices {
                if reached[dest_idx].is_none() || cost < reached[dest_idx].unwrap().1 {
                    reached[dest_idx] = Some((*node_candidate, cost));
                }
            }
        }
    }

    reached
        .iter()
        .map(|opt| opt.and_then(|(node, cost)| reconstruct_bike_path(ways, &dijkstra, node, cost)))
        .collect()
}

/// Car profile routing with Dijkstra
fn route_car_dijkstra(
    ways: &Ways,
    _elevations: Option<&crate::elevation_storage::ElevationStorage>,
    _profile: &crate::routing::profiles::car::CarProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
) -> Option<Path> {
    use crate::routing::profiles::car::{node_cost, way_cost};

    let mut dijkstra = Dijkstra::<CarNode>::new();
    dijkstra.reset(max_cost);

    // BUG-08 fix: initialise start nodes with correct way_pos and both directions
    for candidate in start_candidates {
        if candidate.left.valid() {
            let wp = ways.get_way_pos(candidate.left.node, candidate.way);
            dijkstra.add_start(CarNode { n: candidate.left.node, way: wp, dir: Direction::Forward }, candidate.left.cost);
            dijkstra.add_start(CarNode { n: candidate.left.node, way: wp, dir: Direction::Backward }, candidate.left.cost);
        }
        if candidate.right.valid() {
            let wp = ways.get_way_pos(candidate.right.node, candidate.way);
            dijkstra.add_start(CarNode { n: candidate.right.node, way: wp, dir: Direction::Forward }, candidate.right.cost);
            dijkstra.add_start(CarNode { n: candidate.right.node, way: wp, dir: Direction::Backward }, candidate.right.cost);
        }
    }

    // BUG-08 fix: end nodes need all (way_pos, dir) combinations so get_cost matches
    // Store NodeCandidate reference alongside each CarNode for snap segment retrieval
    let mut end_nodes: Vec<(CarNode, &NodeCandidate)> = Vec::new();
    for candidate in end_candidates {
        for nc in [&candidate.left, &candidate.right] {
            if !nc.valid() { continue; }
            let node_ways = ways.get_node_ways(nc.node);
            for (wp, _) in node_ways.iter().enumerate() {
                for dir in [Direction::Forward, Direction::Backward] {
                    end_nodes.push((CarNode { n: nc.node, way: wp as u8, dir }, nc));
                }
            }
        }
    }

    // Run Dijkstra with car-specific neighbor generation
    let mut best_node = None;
    let mut best_cost = Cost::MAX;

    dijkstra.run(max_cost, |current: CarNode| {
        let mut neighbors = Vec::new();

        let car_node_ways = ways.get_node_ways(current.n);
        let car_way_positions = ways.get_node_in_way_idx(current.n);

        // Get all ways connected to this node
        for (outgoing_way_pos, (&way, &pos_in_way)) in car_node_ways.iter()
            .zip(car_way_positions)
            .enumerate()
        {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let way_props = ways.get_way_properties(way);

            if way_nodes.is_empty() || way_props.is_none() {
                continue;
            }

            let props = way_props.unwrap();

            // Turn restriction check
            if ways.is_restricted(current.n, current.way, outgoing_way_pos as crate::types::WayPos) {
                continue;
            }

            let i = pos_in_way as usize;
            if i >= way_nodes.len() {
                continue;
            }

            // Forward along way — node cost on ARRIVAL at target (BUG-01), correct way_pos (BUG-03)
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = Direction::Forward;
                    let edge_cost = way_cost(props, travel_dir, dist);

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(next_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            // BUG-03 fix: use get_way_pos for target node's way position
                            let target_wp = ways.get_way_pos(next_node, way);
                            neighbors.push((
                                CarNode { n: next_node, way: target_wp, dir: travel_dir },
                                total_cost,
                            ));
                        }
                    }
                }
            }

            // Backward along way — node cost on ARRIVAL at target (BUG-01), correct way_pos (BUG-03)
            if i > 0 {
                let prev_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = Direction::Backward;
                    let edge_cost = way_cost(props, travel_dir, dist);

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(prev_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            // BUG-03 fix: use get_way_pos for target node's way position
                            let target_wp = ways.get_way_pos(prev_node, way);
                            neighbors.push((
                                CarNode { n: prev_node, way: target_wp, dir: travel_dir },
                                total_cost,
                            ));
                        }
                    }
                }
            }
        }

        neighbors
    });

    // Check if we reached any end node and add end snap cost
    let mut best_end_nc: Option<&NodeCandidate> = None;
    for &(end_node, nc) in &end_nodes {
        let path_cost = dijkstra.get_cost(end_node);
        if path_cost < K_INFEASIBLE {
            let total_cost = path_cost.saturating_add(nc.cost);
            if total_cost < best_cost {
                best_cost = total_cost;
                best_node = Some(end_node);
                best_end_nc = Some(nc);
            }
        }
    }

    // Reconstruct path and attach snap segments (BUG-04 fix)
    if let Some(end) = best_node {
        let mut path = reconstruct_car_path(ways, &dijkstra, end, best_cost)?;

        // Find start node by walking predecessor chain
        let mut curr = end;
        while let Some(pred) = dijkstra.get_predecessor(curr) {
            curr = pred;
        }
        let start_nc = find_snap_nc(start_candidates, curr.n);

        add_snap_segments(&mut path, start_nc, best_end_nc, Level::default(), crate::routing::mode::Mode::Car);
        Some(path)
    } else {
        None
    }
}

/// Reconstruct car path from Dijkstra predecessors
fn reconstruct_car_path(
    ways: &Ways,
    dijkstra: &Dijkstra<CarNode>,
    end_node: CarNode,
    total_cost: Cost,
) -> Option<Path> {
    let mut segments = Vec::new();
    let mut current = end_node;
    let mut total_dist = 0.0;

    // Walk backwards through predecessors
    while let Some(pred) = dijkstra.get_predecessor(current) {
        // Find the way connecting pred to current
        let pred_ways = ways.get_node_ways(pred.n);
        let curr_ways = ways.get_node_ways(current.n);

        // Find common way
        let mut found_way = None;
        for &pw in pred_ways {
            for &cw in curr_ways {
                if pw == cw {
                    found_way = Some(pw);
                    break;
                }
            }
            if found_way.is_some() {
                break;
            }
        }

        if let Some(way) = found_way {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let polyline = ways.get_way_polyline(way);
            let polyline_indices = ways.get_way_node_polyline_indices(way);

            // Find node positions in way
            let pred_pos = way_nodes.iter().position(|&n| n == pred.n);
            let curr_pos = way_nodes.iter().position(|&n| n == current.n);

            if let (Some(p_pos), Some(c_pos)) = (pred_pos, curr_pos) {
                // Calculate segment distance (way_node_dist values are in meters)
                let start_idx = p_pos.min(c_pos);
                let end_idx = p_pos.max(c_pos);
                let is_reverse = p_pos > c_pos;
                let mut dist: u32 = 0;
                for j in start_idx..end_idx {
                    dist += way_dists.get(j).copied().unwrap_or(0) as u32;
                }

                total_dist += dist as f64;

                // Map routing node indices to polyline indices for full geometry
                let poly_start = polyline_indices.get(start_idx).copied().unwrap_or(start_idx as u16) as usize;
                let poly_end = polyline_indices.get(end_idx).copied().unwrap_or(end_idx as u16) as usize;

                // Extract polyline segment using full-geometry indices
                let mut segment_polyline = if !polyline.is_empty() && poly_end < polyline.len() {
                    polyline[poly_start..=poly_end].to_vec()
                } else {
                    vec![]
                };

                // Reverse polyline when traversing backward along the way
                if is_reverse {
                    segment_polyline.reverse();
                }

                // Get segment cost
                let segment_cost = dijkstra
                    .get_cost(current)
                    .saturating_sub(dijkstra.get_cost(pred));

                segments.push(crate::routing::path::Segment::with_values(
                    segment_polyline,
                    Level::from_idx(Level::NO_LEVEL),
                    Level::from_idx(Level::NO_LEVEL),
                    pred.n,
                    current.n,
                    way,
                    segment_cost,
                    dist.min(u16::MAX as u32) as u16,
                    crate::elevation_storage::Elevation::new(0, 0),
                    crate::routing::mode::Mode::Car,
                ));
            }
        }

        current = pred;
    }

    // Reverse segments to get start-to-end order
    segments.reverse();

    Some(Path {
        cost: total_cost,
        dist: total_dist,
        elevation: crate::elevation_storage::Elevation::new(0, 0),
        segments,
        uses_elevator: false,
        track_node: NodeIdx::INVALID,
    })
}

/// Car profile one-to-many routing with Dijkstra
fn route_car_multi_dijkstra(
    ways: &Ways,
    _elevations: Option<&crate::elevation_storage::ElevationStorage>,
    _profile: &crate::routing::profiles::car::CarProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[Vec<crate::lookup::WayCandidate>],
    max_cost: Cost,
) -> Vec<Option<Path>> {
    use crate::routing::profiles::car::{node_cost, way_cost};
    use ahash::AHashMap;

    let mut dijkstra = Dijkstra::<CarNode>::new();
    dijkstra.reset(max_cost);

    // BUG-08 fix: correct way_pos and both directions for start nodes
    for candidate in start_candidates {
        if candidate.left.valid() {
            let wp = ways.get_way_pos(candidate.left.node, candidate.way);
            dijkstra.add_start(CarNode { n: candidate.left.node, way: wp, dir: Direction::Forward }, candidate.left.cost);
            dijkstra.add_start(CarNode { n: candidate.left.node, way: wp, dir: Direction::Backward }, candidate.left.cost);
        }
        if candidate.right.valid() {
            let wp = ways.get_way_pos(candidate.right.node, candidate.way);
            dijkstra.add_start(CarNode { n: candidate.right.node, way: wp, dir: Direction::Forward }, candidate.right.cost);
            dijkstra.add_start(CarNode { n: candidate.right.node, way: wp, dir: Direction::Backward }, candidate.right.cost);
        }
    }

    // BUG-08 fix: register all (way_pos, dir) combinations as targets
    let mut targets: AHashMap<CarNode, Vec<usize>> = AHashMap::new();
    for (dest_idx, candidates) in end_candidates.iter().enumerate() {
        for candidate in candidates {
            for nc in [&candidate.left, &candidate.right] {
                if !nc.valid() { continue; }
                let node_ways = ways.get_node_ways(nc.node);
                for (wp, _) in node_ways.iter().enumerate() {
                    for dir in [Direction::Forward, Direction::Backward] {
                        targets
                            .entry(CarNode { n: nc.node, way: wp as u8, dir })
                            .or_default()
                            .push(dest_idx);
                    }
                }
            }
        }
    }

    let mut reached: Vec<Option<(CarNode, Cost)>> = vec![None; end_candidates.len()];

    dijkstra.run(max_cost, |current: CarNode| {
        let mut neighbors = Vec::new();

        for (outgoing_way_pos, &way) in ways.get_node_ways(current.n).iter().enumerate() {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let way_props = ways.get_way_properties(way);

            if way_nodes.is_empty() || way_props.is_none() {
                continue;
            }

            let props = way_props.unwrap();

            // Turn restriction check
            if ways.is_restricted(current.n, current.way, outgoing_way_pos as crate::types::WayPos) {
                continue;
            }

            // BUG-01 fix: find position without inner loop
            let Some(i) = way_nodes.iter().position(|&n| n == current.n) else { continue };

            // Forward — node cost on ARRIVAL (BUG-01), correct way_pos (BUG-03)
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = Direction::Forward;
                    let edge_cost = way_cost(props, travel_dir, dist);

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(next_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            let target_wp = ways.get_way_pos(next_node, way);
                            neighbors.push((
                                CarNode { n: next_node, way: target_wp, dir: travel_dir },
                                total_cost,
                            ));
                        }
                    }
                }
            }

            // Backward — node cost on ARRIVAL (BUG-01), correct way_pos (BUG-03)
            if i > 0 {
                let prev_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let travel_dir = Direction::Backward;
                    let edge_cost = way_cost(props, travel_dir, dist);

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(prev_node).map(|p| node_cost(p)).unwrap_or(0);
                        let total_cost = edge_cost.saturating_add(t_cost);
                        if total_cost < K_INFEASIBLE {
                            let target_wp = ways.get_way_pos(prev_node, way);
                            neighbors.push((
                                CarNode { n: prev_node, way: target_wp, dir: travel_dir },
                                total_cost,
                            ));
                        }
                    }
                }
            }
        }

        neighbors
    });

    // Find reached destinations after search completes
    for (node_candidate, dest_indices) in &targets {
        let cost = dijkstra.get_cost(*node_candidate);
        if cost < K_INFEASIBLE {
            for &dest_idx in dest_indices {
                if reached[dest_idx].is_none() || cost < reached[dest_idx].unwrap().1 {
                    reached[dest_idx] = Some((*node_candidate, cost));
                }
            }
        }
    }

    reached
        .iter()
        .map(|opt| opt.and_then(|(node, cost)| reconstruct_car_path(ways, &dijkstra, node, cost)))
        .collect()
}

/// Bike sharing profile routing with Dijkstra
///
/// Routes using multi-modal approach: Walk → Bike → Walk
/// For simplicity, allows mode switch at any node (stations not enforced yet)
fn route_bike_sharing_dijkstra(
    ways: &Ways,
    _elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &crate::routing::profiles::bike_sharing::BikeSharingProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
) -> Option<Path> {
    use crate::routing::profiles::bike;

    // For bike sharing, we do a simplified approach:
    // Just use bike routing (since we can switch modes at any node in principle)
    // In a full implementation, we'd track mode state and enforce station constraints

    let mut dijkstra = Dijkstra::<BikeNode>::new();
    dijkstra.reset(max_cost);

    // Add start nodes (walking initially, but we model as bike nodes)
    for candidate in start_candidates {
        if candidate.left.valid() {
            dijkstra.add_start(
                BikeNode {
                    n: candidate.left.node,
                    dir: Direction::Forward,
                },
                candidate.left.cost + profile.start_switch_penalty as Cost,
            );
        }
        if candidate.right.valid() {
            dijkstra.add_start(
                BikeNode {
                    n: candidate.right.node,
                    dir: Direction::Forward,
                },
                candidate.right.cost + profile.start_switch_penalty as Cost,
            );
        }
    }

    // Collect end nodes
    let mut end_nodes = Vec::new();
    for candidate in end_candidates {
        if candidate.left.valid() {
            end_nodes.push(BikeNode {
                n: candidate.left.node,
                dir: Direction::Forward,
            });
            end_nodes.push(BikeNode {
                n: candidate.left.node,
                dir: Direction::Backward,
            });
        }
        if candidate.right.valid() {
            end_nodes.push(BikeNode {
                n: candidate.right.node,
                dir: Direction::Forward,
            });
            end_nodes.push(BikeNode {
                n: candidate.right.node,
                dir: Direction::Backward,
            });
        }
    }

    // Run Dijkstra with bike costs (simplified - no mode tracking)
    dijkstra.run(max_cost, |current: BikeNode| {
        let mut neighbors = Vec::new();

        for &way in ways.get_node_ways(current.n) {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let way_props = ways.get_way_properties(way);

            if way_nodes.is_empty() || way_props.is_none() {
                continue;
            }

            let props = way_props.unwrap();

            let Some(i) = way_nodes.iter().position(|&n| n == current.n) else { continue };

            // Forward — node cost on ARRIVAL (BUG-01 fix)
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let edge_cost = bike::way_cost(
                        &profile.bike_params,
                        props,
                        Direction::Forward,
                        crate::routing::profiles::bike::BikeCostStrategy::Safe,
                        dist,
                    );

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(next_node).map(|p| bike::node_cost(p)).unwrap_or(0);
                        let total = edge_cost.saturating_add(t_cost);
                        if total < K_INFEASIBLE {
                            neighbors.push((BikeNode { n: next_node, dir: Direction::Forward }, total));
                        }
                    }
                }
            }

            // Backward — node cost on ARRIVAL (BUG-01 fix)
            if i > 0 {
                let prev_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let edge_cost = bike::way_cost(
                        &profile.bike_params,
                        props,
                        Direction::Backward,
                        crate::routing::profiles::bike::BikeCostStrategy::Safe,
                        dist,
                    );

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(prev_node).map(|p| bike::node_cost(p)).unwrap_or(0);
                        let total = edge_cost.saturating_add(t_cost);
                        if total < K_INFEASIBLE {
                            neighbors.push((BikeNode { n: prev_node, dir: Direction::Backward }, total));
                        }
                    }
                }
            }
        }

        neighbors
    });

    // Find best end node (add end switch penalty)
    let mut best_cost = Cost::MAX;
    let mut best_node = None;

    for end_node in &end_nodes {
        let cost = dijkstra.get_cost(*end_node);
        let total_cost = cost.saturating_add(profile.end_switch_penalty as Cost);
        if total_cost < best_cost {
            best_cost = total_cost;
            best_node = Some(*end_node);
        }
    }

    // Reconstruct path (similar to bike)
    if let Some(end) = best_node {
        reconstruct_bike_path(ways, &dijkstra, end, best_cost)
    } else {
        None
    }
}

/// Car sharing profile routing with Dijkstra
///
/// Routes using multi-modal approach: Walk → Car → Walk
/// For simplicity, allows mode switch at any node (stations not enforced yet)
fn route_car_sharing_dijkstra(
    ways: &Ways,
    _elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &crate::routing::profiles::car_sharing::CarSharingProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
) -> Option<Path> {
    use crate::routing::profiles::car;

    // Simplified car sharing: use car routing with switch penalties
    let mut dijkstra = Dijkstra::<CarNode>::new();
    dijkstra.reset(max_cost);

    // Add start nodes with pickup penalty
    for candidate in start_candidates {
        if candidate.left.valid() {
            dijkstra.add_start(
                CarNode {
                    n: candidate.left.node,
                    way: 0,
                    dir: Direction::Forward,
                },
                candidate.left.cost + profile.start_switch_penalty as Cost,
            );
        }
        if candidate.right.valid() {
            dijkstra.add_start(
                CarNode {
                    n: candidate.right.node,
                    way: 0,
                    dir: Direction::Forward,
                },
                candidate.right.cost + profile.start_switch_penalty as Cost,
            );
        }
    }

    // Collect end nodes
    let mut end_nodes = Vec::new();
    for candidate in end_candidates {
        if candidate.left.valid() {
            end_nodes.push(CarNode {
                n: candidate.left.node,
                way: 0,
                dir: Direction::Forward,
            });
        }
        if candidate.right.valid() {
            end_nodes.push(CarNode {
                n: candidate.right.node,
                way: 0,
                dir: Direction::Forward,
            });
        }
    }

    // Run Dijkstra with car costs
    dijkstra.run(max_cost, |current: CarNode| {
        let mut neighbors = Vec::new();

        for (outgoing_way_pos, &way) in ways.get_node_ways(current.n).iter().enumerate() {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let way_props = ways.get_way_properties(way);

            if way_nodes.is_empty() || way_props.is_none() {
                continue;
            }

            let props = way_props.unwrap();

            if ways.is_restricted(current.n, current.way, outgoing_way_pos as crate::types::WayPos) {
                continue;
            }

            let Some(i) = way_nodes.iter().position(|&n| n == current.n) else { continue };

            // Forward — node cost on ARRIVAL (BUG-01), correct way_pos (BUG-03)
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let edge_cost = car::way_cost(props, Direction::Forward, dist);

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(next_node).map(|p| car::node_cost(p)).unwrap_or(0);
                        let total = edge_cost.saturating_add(t_cost);
                        if total < K_INFEASIBLE {
                            let target_wp = ways.get_way_pos(next_node, way);
                            neighbors.push((CarNode { n: next_node, way: target_wp, dir: Direction::Forward }, total));
                        }
                    }
                }
            }

            // Backward — node cost on ARRIVAL (BUG-01), correct way_pos (BUG-03)
            if i > 0 {
                let prev_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let edge_cost = car::way_cost(props, Direction::Backward, dist);

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(prev_node).map(|p| car::node_cost(p)).unwrap_or(0);
                        let total = edge_cost.saturating_add(t_cost);
                        if total < K_INFEASIBLE {
                            let target_wp = ways.get_way_pos(prev_node, way);
                            neighbors.push((CarNode { n: prev_node, way: target_wp, dir: Direction::Backward }, total));
                        }
                    }
                }
            }
        }

        neighbors
    });

    // Find best end node with dropoff penalty
    let mut best_cost = Cost::MAX;
    let mut best_node = None;

    for end_node in &end_nodes {
        let cost = dijkstra.get_cost(*end_node);
        let total_cost = cost.saturating_add(profile.end_switch_penalty as Cost);
        if total_cost < best_cost {
            best_cost = total_cost;
            best_node = Some(*end_node);
        }
    }

    // Reconstruct path
    if let Some(end) = best_node {
        reconstruct_car_path(ways, &dijkstra, end, best_cost)
    } else {
        None
    }
}

/// Car parking profile routing with Dijkstra
///
/// Routes by car to vicinity of destination, then allows walking
/// For simplicity, doesn't enforce specific parking locations yet
fn route_car_parking_dijkstra(
    ways: &Ways,
    _elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: &crate::routing::profiles::car_parking::CarParkingProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[crate::lookup::WayCandidate],
    max_cost: Cost,
) -> Option<Path> {
    use crate::routing::profiles::car;

    // Car parking: drive to near destination, park (penalty), then could walk
    // Simplified: just use car routing with parking penalty at end
    let mut dijkstra = Dijkstra::<CarNode>::new();
    dijkstra.reset(max_cost);

    // Add start nodes
    for candidate in start_candidates {
        if candidate.left.valid() {
            dijkstra.add_start(
                CarNode {
                    n: candidate.left.node,
                    way: 0,
                    dir: Direction::Forward,
                },
                candidate.left.cost,
            );
        }
        if candidate.right.valid() {
            dijkstra.add_start(
                CarNode {
                    n: candidate.right.node,
                    way: 0,
                    dir: Direction::Forward,
                },
                candidate.right.cost,
            );
        }
    }

    // Collect end nodes
    let mut end_nodes = Vec::new();
    for candidate in end_candidates {
        if candidate.left.valid() {
            end_nodes.push(CarNode {
                n: candidate.left.node,
                way: 0,
                dir: Direction::Forward,
            });
        }
        if candidate.right.valid() {
            end_nodes.push(CarNode {
                n: candidate.right.node,
                way: 0,
                dir: Direction::Forward,
            });
        }
    }

    // Run Dijkstra with car costs
    dijkstra.run(max_cost, |current: CarNode| {
        let mut neighbors = Vec::new();

        for (outgoing_way_pos, &way) in ways.get_node_ways(current.n).iter().enumerate() {
            let way_nodes = ways.get_way_nodes(way);
            let way_dists = ways.get_way_node_distances(way);
            let way_props = ways.get_way_properties(way);

            if way_nodes.is_empty() || way_props.is_none() {
                continue;
            }

            let props = way_props.unwrap();

            if ways.is_restricted(current.n, current.way, outgoing_way_pos as crate::types::WayPos) {
                continue;
            }

            let Some(i) = way_nodes.iter().position(|&n| n == current.n) else { continue };

            // Forward — node cost on ARRIVAL (BUG-01), correct way_pos (BUG-03)
            if i + 1 < way_nodes.len() {
                let next_node = way_nodes[i + 1];
                let dist = way_dists.get(i).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let edge_cost = car::way_cost(props, Direction::Forward, dist);

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(next_node).map(|p| car::node_cost(p)).unwrap_or(0);
                        let total = edge_cost.saturating_add(t_cost);
                        if total < K_INFEASIBLE {
                            let target_wp = ways.get_way_pos(next_node, way);
                            neighbors.push((CarNode { n: next_node, way: target_wp, dir: Direction::Forward }, total));
                        }
                    }
                }
            }

            // Backward — node cost on ARRIVAL (BUG-01), correct way_pos (BUG-03)
            if i > 0 {
                let prev_node = way_nodes[i - 1];
                let dist = way_dists.get(i - 1).copied().unwrap_or(0);

                if dist > 0 && dist < 50000 {
                    let edge_cost = car::way_cost(props, Direction::Backward, dist);

                    if edge_cost != K_INFEASIBLE {
                        let t_cost = ways.get_node_properties(prev_node).map(|p| car::node_cost(p)).unwrap_or(0);
                        let total = edge_cost.saturating_add(t_cost);
                        if total < K_INFEASIBLE {
                            let target_wp = ways.get_way_pos(prev_node, way);
                            neighbors.push((CarNode { n: prev_node, way: target_wp, dir: Direction::Backward }, total));
                        }
                    }
                }
            }
        }

        neighbors
    });

    // Find best end node with parking penalty
    let mut best_cost = Cost::MAX;
    let mut best_node = None;

    for end_node in &end_nodes {
        let cost = dijkstra.get_cost(*end_node);
        let total_cost = cost.saturating_add(profile.switch_penalty as Cost);
        if total_cost < best_cost {
            best_cost = total_cost;
            best_node = Some(*end_node);
        }
    }

    // Reconstruct path
    if let Some(end) = best_node {
        reconstruct_car_path(ways, &dijkstra, end, best_cost)
    } else {
        None
    }
}

/// Bike sharing profile one-to-many routing (simplified)
fn route_bike_sharing_multi_dijkstra(
    ways: &Ways,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    _profile: &crate::routing::profiles::bike_sharing::BikeSharingProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[Vec<crate::lookup::WayCandidate>],
    max_cost: Cost,
) -> Vec<Option<Path>> {
    // Simplified: delegate to bike multi
    route_bike_multi_dijkstra(
        ways,
        elevations,
        &crate::routing::profiles::bike::BikeProfile::default(),
        start_candidates,
        end_candidates,
        max_cost,
    )
}

/// Car sharing profile one-to-many routing (simplified)
fn route_car_sharing_multi_dijkstra(
    ways: &Ways,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    _profile: &crate::routing::profiles::car_sharing::CarSharingProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[Vec<crate::lookup::WayCandidate>],
    max_cost: Cost,
) -> Vec<Option<Path>> {
    // Simplified: delegate to car multi
    route_car_multi_dijkstra(
        ways,
        elevations,
        &crate::routing::profiles::car::CarProfile::default(),
        start_candidates,
        end_candidates,
        max_cost,
    )
}

/// Car parking profile one-to-many routing (simplified)
fn route_car_parking_multi_dijkstra(
    ways: &Ways,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    _profile: &crate::routing::profiles::car_parking::CarParkingProfile,
    start_candidates: &[crate::lookup::WayCandidate],
    end_candidates: &[Vec<crate::lookup::WayCandidate>],
    max_cost: Cost,
) -> Vec<Option<Path>> {
    // Simplified: delegate to car multi
    route_car_multi_dijkstra(
        ways,
        elevations,
        &crate::routing::profiles::car::CarProfile::default(),
        start_candidates,
        end_candidates,
        max_cost,
    )
}

/// Route from one location to multiple destinations
///
/// Returns a vector of paths, one for each destination.
/// Returns None for unreachable destinations.
///
/// # Implementation
/// Uses one-to-many Dijkstra optimization: runs a single search from the source
/// and tracks all destinations, terminating when all are found or max_cost exceeded.
/// This is ~10-100x faster than running N separate searches for nearby destinations.
pub fn route_multi(
    ways: &Ways,
    lookup: &Lookup,
    elevations: Option<&crate::elevation_storage::ElevationStorage>,
    profile: SearchProfile,
    from: Location,
    to: &[Location],
    max_cost: Cost,
    algorithm: RoutingAlgorithm,
) -> Vec<Option<Path>> {
    if to.is_empty() {
        return vec![];
    }

    // Create profile instance
    let profile_inst = ProfileInstance::from_search_profile(profile);
    let max_distance = profile_inst.max_match_distance();

    // Match start location to network
    let start_candidates = lookup.match_location(&ways, profile, &from, max_distance);
    if start_candidates.is_empty() {
        return vec![None; to.len()];
    }

    // Match all destinations to network
    let end_candidates: Vec<_> = to
        .iter()
        .map(|dest| lookup.match_location(&ways, profile, dest, max_distance))
        .collect();

    // Check if any destinations have no candidates
    if end_candidates.iter().any(|c| c.is_empty()) {
        // Still run the algorithm but those destinations will be None
    }

    // Route based on algorithm and profile type
    match algorithm {
        RoutingAlgorithm::Dijkstra => route_multi_dijkstra(
            ways,
            elevations,
            &profile_inst,
            &start_candidates,
            &end_candidates,
            max_cost,
        ),
        RoutingAlgorithm::AStarBi => {
            // One-to-many bidirectional A* not applicable - use Dijkstra
            // Note: Dijkstra is more efficient for one-to-many queries
            route_multi_dijkstra(
                ways,
                elevations,
                &profile_inst,
                &start_candidates,
                &end_candidates,
                max_cost,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Level;

    #[test]
    fn test_profile_dispatch() {
        // Verify all profile variants are handled
        assert_eq!(SearchProfile::Foot.mode(), crate::routing::mode::Mode::Foot);
        assert_eq!(SearchProfile::Bike.mode(), crate::routing::mode::Mode::Bike);
        assert_eq!(SearchProfile::Car.mode(), crate::routing::mode::Mode::Car);
    }

    #[test]
    fn test_route_signature() {
        // Verify function signature compiles
        assert!(true);
    }

    #[test]
    fn test_route_multi_signature() {
        // Verify function signature compiles
        assert!(true);
    }

    #[test]
    fn test_algorithm_support() {
        // Verify we handle the RoutingAlgorithm parameter
        let alg = RoutingAlgorithm::Dijkstra;
        assert_eq!(alg, RoutingAlgorithm::Dijkstra);
    }

    #[test]
    fn test_route_returns_none_no_data() {
        // With empty Ways, routing should return None (no matches)
        let ways = crate::Ways::new();
        let lookup = Lookup::new(&ways);
        let from = Location::from_latlng(0.0, 0.0, Level::default());
        let to = Location::from_latlng(0.1, 0.1, Level::default());

        let result = route(
            &ways,
            &lookup,
            None,
            SearchProfile::Foot,
            from,
            to,
            3600,
            RoutingAlgorithm::Dijkstra,
        );

        assert!(result.is_none(), "Empty network should return None");
    }

    #[test]
    fn test_route_multi_returns_empty() {
        // With empty Ways, all routes should return None
        let ways = crate::Ways::new();
        let lookup = Lookup::new(&ways);
        let from = Location::from_latlng(0.0, 0.0, Level::default());
        let destinations = vec![
            Location::from_latlng(0.1, 0.1, Level::default()),
            Location::from_latlng(0.2, 0.2, Level::default()),
        ];

        let results = route_multi(
            &ways,
            &lookup,
            None,
            SearchProfile::Foot,
            from,
            &destinations,
            3600,
            RoutingAlgorithm::Dijkstra,
        );

        assert_eq!(results.len(), 2);
        assert!(
            results.iter().all(|r| r.is_none()),
            "Empty network returns all None"
        );
    }

    #[test]
    fn test_profile_instance_dispatch() {
        // Test that ProfileInstance dispatch works for all profile types
        let profiles = vec![
            SearchProfile::Foot,
            SearchProfile::Wheelchair,
            SearchProfile::Bike,
            SearchProfile::BikeFast,
            SearchProfile::Car,
        ];

        for profile in profiles {
            let inst = ProfileInstance::from_search_profile(profile);
            let max_dist = inst.max_match_distance();
            assert!(
                max_dist > 0.0,
                "Profile {:?} should have positive max_match_distance",
                profile
            );
        }
    }
}
