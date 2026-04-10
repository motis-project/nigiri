//! Car sharing routing profile.
//!
//! This profile handles multi-modal routing with car sharing:
//! 1. Walk to car station (foot mode)
//! 2. Pick up shared car
//! 3. Drive car (car mode)
//! 4. Drop off car at station
//! 5. Walk to destination (foot mode)

use crate::routing::{CarParameters, FootParameters, Mode};

/// Node type in car sharing routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// Initial walking to car station
    InitialFoot,
    /// Driving shared car (rental)
    Rental,
    /// Final walking from car station
    TrailingFoot,
}

/// A car share pod location loaded from OSM data.
///
/// Loaded from OSM tags:
/// - `amenity=car_sharing`
/// - `name=*`
/// - `operator=*`
/// - `capacity=*`
#[derive(Debug, Clone)]
pub struct CarSharePod {
    /// Unique identifier (OSM node id)
    pub id: String,
    /// Geographic location (lat, lng)
    pub location: (f64, f64),
    /// Human-readable location name
    pub name: String,
    /// Number of available vehicles (u32::MAX = unknown)
    pub available_vehicles: u32,
    /// Operator / network name (e.g. "GoGet", "Zipcar")
    pub operator: String,
}

impl CarSharePod {
    /// Returns `true` if the pod has at least one available vehicle.
    ///
    /// Returns `true` when availability is unknown (permissive default).
    #[inline]
    pub fn has_available_vehicle(&self) -> bool {
        self.available_vehicles == u32::MAX || self.available_vehicles > 0
    }
}

/// Car sharing routing profile.
///
/// Combines walking and driving with station stops.
/// Used for car-share systems like Zipcar, Car2Go, etc.
#[derive(Debug, Clone)]
pub struct CarSharingProfile {
    /// Parameters for foot segments
    pub foot_params: FootParameters,

    /// Parameters for car segments
    pub car_params: CarParameters,

    /// Penalty for picking up car (seconds)
    pub start_switch_penalty: u16,

    /// Penalty for dropping off car (seconds)
    pub end_switch_penalty: u16,

    /// Maximum walking distance to/from a car share pod (meters)
    pub max_walk_to_pod: f64,

    /// Cost per minute of driving (local currency units)
    pub cost_per_minute: f64,

    /// Cost per kilometre of driving (local currency units)
    pub cost_per_km: f64,
}

impl CarSharingProfile {
    /// Create a new car sharing profile with default penalties.
    ///
    /// # Arguments
    /// * `foot_params` - Walking parameters
    /// * `car_params`  - Driving parameters
    pub fn new(foot_params: FootParameters, car_params: CarParameters) -> Self {
        Self {
            foot_params,
            car_params,
            start_switch_penalty: 30, // 30 seconds to pick up car
            end_switch_penalty: 30,   // 30 seconds to drop off car
            max_walk_to_pod: 600.0,   // search within 600 m
            cost_per_minute: 0.0,
            cost_per_km: 0.0,
        }
    }

    /// Create with custom switch penalties.
    pub fn with_penalties(
        foot_params: FootParameters,
        car_params: CarParameters,
        start_penalty: u16,
        end_penalty: u16,
    ) -> Self {
        Self {
            foot_params,
            car_params,
            start_switch_penalty: start_penalty,
            end_switch_penalty: end_penalty,
            max_walk_to_pod: 600.0,
            cost_per_minute: 0.0,
            cost_per_km: 0.0,
        }
    }

    /// Get the mode for a given node type.
    pub fn mode_for_type(&self, node_type: NodeType) -> Mode {
        match node_type {
            NodeType::InitialFoot | NodeType::TrailingFoot => Mode::Foot,
            NodeType::Rental => Mode::Car,
        }
    }

    /// Get switch penalty when transitioning between modes.
    pub fn switch_penalty(&self, from: NodeType, to: NodeType) -> u16 {
        match (from, to) {
            (NodeType::InitialFoot, NodeType::Rental) => self.start_switch_penalty,
            (NodeType::Rental, NodeType::TrailingFoot) => self.end_switch_penalty,
            _ => 0,
        }
    }

    /// Maximum distance to match to street network (uses car's larger range).
    pub fn max_match_distance(&self) -> f64 {
        200.0 // meters (car range)
    }

    /// Calculate cost for a segment based on node type.
    pub fn calculate_cost(
        &self,
        node_type: NodeType,
        distance_meters: f64,
        speed_limit_kmh: Option<f64>,
    ) -> u16 {
        match node_type {
            NodeType::InitialFoot | NodeType::TrailingFoot => {
                // Walking cost
                (distance_meters / self.foot_params.speed_meters_per_second as f64) as u16
            }
            NodeType::Rental => {
                // Driving cost (respects speed limits)
                let speed = speed_limit_kmh.unwrap_or(50.0);
                let speed_ms = speed / 3.6;
                (distance_meters / speed_ms) as u16
            }
        }
    }

    /// Trip cost in currency units for a driving segment.
    ///
    /// Formula: `drive_seconds / 60 × cost_per_minute + distance_km × cost_per_km`
    ///
    /// # Arguments
    /// * `drive_seconds`   - Driving duration in seconds
    /// * `distance_meters` - Driving distance in metres
    pub fn trip_cost(&self, drive_seconds: u32, distance_meters: f64) -> f64 {
        let minutes = drive_seconds as f64 / 60.0;
        let km = distance_meters / 1000.0;
        minutes * self.cost_per_minute + km * self.cost_per_km
    }
}

impl Default for CarSharingProfile {
    fn default() -> Self {
        Self::new(FootParameters::default(), CarParameters::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_car_sharing_creation() {
        let profile = CarSharingProfile::default();
        assert_eq!(profile.start_switch_penalty, 30);
        assert_eq!(profile.end_switch_penalty, 30);
        assert_eq!(profile.max_walk_to_pod, 600.0);
        assert_eq!(profile.cost_per_minute, 0.0);
        assert_eq!(profile.cost_per_km, 0.0);
    }

    #[test]
    fn test_mode_for_type() {
        let profile = CarSharingProfile::default();
        assert_eq!(profile.mode_for_type(NodeType::InitialFoot), Mode::Foot);
        assert_eq!(profile.mode_for_type(NodeType::Rental), Mode::Car);
        assert_eq!(profile.mode_for_type(NodeType::TrailingFoot), Mode::Foot);
    }

    #[test]
    fn test_switch_penalties() {
        let profile = CarSharingProfile::default();
        assert_eq!(
            profile.switch_penalty(NodeType::InitialFoot, NodeType::Rental),
            30
        );
        assert_eq!(
            profile.switch_penalty(NodeType::Rental, NodeType::TrailingFoot),
            30
        );
        assert_eq!(
            profile.switch_penalty(NodeType::InitialFoot, NodeType::TrailingFoot),
            0
        );
    }

    #[test]
    fn test_trip_cost_time_only() {
        let mut profile = CarSharingProfile::default();
        profile.cost_per_minute = 0.5; // $0.50/min
        // 10 minutes driving, 0 km
        let cost = profile.trip_cost(600, 0.0);
        assert!((cost - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_trip_cost_distance_only() {
        let mut profile = CarSharingProfile::default();
        profile.cost_per_km = 0.3; // $0.30/km
        // 0 minutes, 5 km
        let cost = profile.trip_cost(0, 5000.0);
        assert!((cost - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_trip_cost_combined() {
        let mut profile = CarSharingProfile::default();
        profile.cost_per_minute = 0.4;
        profile.cost_per_km = 0.2;
        // 5 minutes (300s), 2000 m (2 km)
        // cost = 5*0.4 + 2*0.2 = 2.0 + 0.4 = 2.4
        let cost = profile.trip_cost(300, 2000.0);
        assert!((cost - 2.4).abs() < 0.001);
    }

    // --- CarSharePod tests ---

    fn make_pod(available: u32) -> CarSharePod {
        CarSharePod {
            id: "pod_42".to_string(),
            location: (-27.4698, 153.0251),
            name: "CBD Station".to_string(),
            available_vehicles: available,
            operator: "GoGet".to_string(),
        }
    }

    #[test]
    fn test_pod_has_available_vehicle() {
        assert!(make_pod(2).has_available_vehicle());
        assert!(!make_pod(0).has_available_vehicle());
        assert!(make_pod(u32::MAX).has_available_vehicle()); // unknown = available
    }

    #[test]
    fn test_pod_fields() {
        let pod = make_pod(3);
        assert_eq!(pod.id, "pod_42");
        assert_eq!(pod.name, "CBD Station");
        assert_eq!(pod.operator, "GoGet");
        assert_eq!(pod.available_vehicles, 3);
    }
}
