//! Bike sharing routing profile.
//!
//! This profile handles multi-modal routing with bike sharing:
//! 1. Walk to bike station (foot mode)
//! 2. Pick up shared bike
//! 3. Ride bike (bike mode)
//! 4. Drop off bike at station
//! 5. Walk to destination (foot mode)

use crate::routing::{BikeParameters, FootParameters, Mode};

/// Node type in bike sharing routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// Initial walking to bike station
    InitialFoot,
    /// Riding shared bike
    Bike,
    /// Final walking from bike station
    TrailingFoot,
}

/// A bike share station loaded from OSM or GBFS data.
///
/// Loaded from OSM tags:
/// - `amenity=bicycle_rental`
/// - `name=*`
/// - `capacity=*`
/// - `operator=*`
///
/// Real-time fields (`available_bikes`, `available_docks`) are optionally
/// populated from a GBFS `station_status.json` feed.
#[derive(Debug, Clone)]
pub struct BikeShareStation {
    /// Unique identifier (OSM node id or GBFS station_id)
    pub id: String,
    /// Geographic location (lat, lng)
    pub location: (f64, f64),
    /// Human-readable station name
    pub name: String,
    /// Total dock capacity (0 = unknown)
    pub capacity: u32,
    /// Number of bikes currently available (u32::MAX = unknown)
    pub available_bikes: u32,
    /// Number of empty docks currently available (u32::MAX = unknown)
    pub available_docks: u32,
    /// Operator / network name (e.g. "CityCycle")
    pub operator: String,
}

impl BikeShareStation {
    /// Returns `true` if the station has at least one available bike.
    ///
    /// Returns `true` when availability is unknown (permissive default).
    #[inline]
    pub fn has_available_bike(&self) -> bool {
        self.available_bikes == u32::MAX || self.available_bikes > 0
    }

    /// Returns `true` if the station has at least one free dock.
    ///
    /// Returns `true` when availability is unknown (permissive default).
    #[inline]
    pub fn has_available_dock(&self) -> bool {
        self.available_docks == u32::MAX || self.available_docks > 0
    }
}

/// Bike sharing routing profile.
///
/// Combines walking and biking with station stops.
/// Used for bike-share systems like Citi Bike, Divvy, etc.
#[derive(Debug, Clone)]
pub struct BikeSharingProfile {
    /// Parameters for foot segments
    pub foot_params: FootParameters,

    /// Parameters for bike segments
    pub bike_params: BikeParameters,

    /// Penalty for picking up bike (seconds)
    pub start_switch_penalty: u16,

    /// Penalty for dropping off bike (seconds)
    pub end_switch_penalty: u16,

    /// Maximum walking distance to/from a bike share station (meters)
    pub max_walk_to_station: f64,

    /// Per-trip or per-day rental cost (in local currency units).
    /// Used for informational purposes and future cost-weighted routing.
    pub rental_cost: f64,
}

impl BikeSharingProfile {
    /// Create a new bike sharing profile with default penalties.
    ///
    /// # Arguments
    /// * `foot_params` - Walking parameters
    /// * `bike_params` - Biking parameters
    pub fn new(foot_params: FootParameters, bike_params: BikeParameters) -> Self {
        Self {
            foot_params,
            bike_params,
            start_switch_penalty: 30, // 30 seconds to pick up bike
            end_switch_penalty: 30,   // 30 seconds to drop off bike
            max_walk_to_station: 400.0, // search within 400 m
            rental_cost: 0.0,
        }
    }

    /// Create with custom switch penalties.
    pub fn with_penalties(
        foot_params: FootParameters,
        bike_params: BikeParameters,
        start_penalty: u16,
        end_penalty: u16,
    ) -> Self {
        Self {
            foot_params,
            bike_params,
            start_switch_penalty: start_penalty,
            end_switch_penalty: end_penalty,
            max_walk_to_station: 400.0,
            rental_cost: 0.0,
        }
    }

    /// Get the mode for a given node type.
    pub fn mode_for_type(&self, node_type: NodeType) -> Mode {
        match node_type {
            NodeType::InitialFoot | NodeType::TrailingFoot => Mode::Foot,
            NodeType::Bike => Mode::Bike,
        }
    }

    /// Get switch penalty when transitioning between modes.
    pub fn switch_penalty(&self, from: NodeType, to: NodeType) -> u16 {
        match (from, to) {
            (NodeType::InitialFoot, NodeType::Bike) => self.start_switch_penalty,
            (NodeType::Bike, NodeType::TrailingFoot) => self.end_switch_penalty,
            _ => 0,
        }
    }

    /// Maximum distance to match to street network (uses bike's larger range).
    pub fn max_match_distance(&self) -> f64 {
        100.0 // meters (bike range)
    }

    /// Calculate cost for a segment based on node type.
    pub fn calculate_cost(
        &self,
        node_type: NodeType,
        distance_meters: f64,
        elevation_gain: f64,
    ) -> u16 {
        match node_type {
            NodeType::InitialFoot | NodeType::TrailingFoot => {
                // Walking cost
                (distance_meters / self.foot_params.speed_meters_per_second as f64) as u16
            }
            NodeType::Bike => {
                // Biking cost (with elevation penalty)
                let base_cost = distance_meters / self.bike_params.speed_meters_per_second as f64;
                let elevation_cost = elevation_gain * 2.0; // Penalty for hills
                (base_cost + elevation_cost) as u16
            }
        }
    }
}

impl Default for BikeSharingProfile {
    fn default() -> Self {
        Self::new(FootParameters::default(), BikeParameters::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bike_sharing_creation() {
        let profile = BikeSharingProfile::default();
        assert_eq!(profile.start_switch_penalty, 30);
        assert_eq!(profile.end_switch_penalty, 30);
        assert_eq!(profile.max_walk_to_station, 400.0);
        assert_eq!(profile.rental_cost, 0.0);
    }

    #[test]
    fn test_mode_for_type() {
        let profile = BikeSharingProfile::default();
        assert_eq!(profile.mode_for_type(NodeType::InitialFoot), Mode::Foot);
        assert_eq!(profile.mode_for_type(NodeType::Bike), Mode::Bike);
        assert_eq!(profile.mode_for_type(NodeType::TrailingFoot), Mode::Foot);
    }

    #[test]
    fn test_switch_penalties() {
        let profile = BikeSharingProfile::default();
        assert_eq!(
            profile.switch_penalty(NodeType::InitialFoot, NodeType::Bike),
            30
        );
        assert_eq!(
            profile.switch_penalty(NodeType::Bike, NodeType::TrailingFoot),
            30
        );
        assert_eq!(
            profile.switch_penalty(NodeType::InitialFoot, NodeType::TrailingFoot),
            0
        );
    }

    #[test]
    fn test_cost_calculation() {
        let profile = BikeSharingProfile::default();

        // Walking: 100m at 1.2 m/s ≈ 83 seconds
        let walk_cost = profile.calculate_cost(NodeType::InitialFoot, 100.0, 0.0);
        assert!(walk_cost >= 80 && walk_cost <= 86);

        // Biking: 100m at 4.2 m/s = 24 seconds (faster)
        let bike_cost = profile.calculate_cost(NodeType::Bike, 100.0, 0.0);
        assert!(bike_cost > 20 && bike_cost < 30);
        assert!(bike_cost < walk_cost);
    }

    #[test]
    fn test_custom_penalties() {
        let profile = BikeSharingProfile::with_penalties(
            FootParameters::default(),
            BikeParameters::default(),
            60,
            45,
        );
        assert_eq!(profile.start_switch_penalty, 60);
        assert_eq!(profile.end_switch_penalty, 45);
    }

    // --- BikeShareStation tests ---

    fn make_station(available_bikes: u32, available_docks: u32) -> BikeShareStation {
        BikeShareStation {
            id: "stn_001".to_string(),
            location: (-27.4698, 153.0251),
            name: "Central Station".to_string(),
            capacity: 20,
            available_bikes,
            available_docks,
            operator: "CityCycle".to_string(),
        }
    }

    #[test]
    fn test_station_has_available_bike() {
        assert!(make_station(5, 15).has_available_bike());
        assert!(!make_station(0, 20).has_available_bike());
        // Unknown availability is treated as available
        assert!(make_station(u32::MAX, 0).has_available_bike());
    }

    #[test]
    fn test_station_has_available_dock() {
        assert!(make_station(10, 10).has_available_dock());
        assert!(!make_station(20, 0).has_available_dock());
        // Unknown availability is treated as available
        assert!(make_station(0, u32::MAX).has_available_dock());
    }

    #[test]
    fn test_station_fields() {
        let s = make_station(3, 17);
        assert_eq!(s.id, "stn_001");
        assert_eq!(s.name, "Central Station");
        assert_eq!(s.capacity, 20);
        assert_eq!(s.operator, "CityCycle");
    }
}
