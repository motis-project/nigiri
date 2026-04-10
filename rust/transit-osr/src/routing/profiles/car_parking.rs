//! Car parking routing profile.
//!
//! This profile handles routing to a destination with parking:
//! 1. Drive to parking location (car mode)
//! 2. Park car
//! 3. Walk to final destination (foot mode)
//!
//! Also supports wheelchair-accessible routing with parking.

use crate::routing::{CarParameters, FootParameters, Mode};

/// Node type in car parking routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// Driving car
    Car,
    /// Walking after parking
    Foot,
}

/// Type of parking facility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParkingType {
    /// On-street parking
    Street,
    /// Multi-level parking garage
    ParkingGarage,
    /// Surface parking lot
    ParkingLot,
    /// Park-and-ride facility
    ParkAndRide,
}

impl Default for ParkingType {
    fn default() -> Self {
        ParkingType::Street
    }
}

/// A parking facility loaded from OSM data.
///
/// Loaded from OSM tags:
/// - `amenity=parking`
/// - `parking=street_side|multi-storey|surface|park_ride`
/// - `capacity=<number>`
/// - `fee=yes|no`
#[derive(Debug, Clone)]
pub struct ParkingFacility {
    /// Unique identifier (from OSM node/way id)
    pub id: String,
    /// Geographic location (lat, lng)
    pub location: (f64, f64),
    /// Total parking capacity (0 = unknown)
    pub capacity: u32,
    /// Hourly cost in local currency units (0.0 = free or unknown)
    pub hourly_rate: f64,
    /// Type of parking facility
    pub facility_type: ParkingType,
}

/// Car parking routing profile.
///
/// Routes by car to a parking location, then walks to destination.
/// Supports wheelchair accessibility.
#[derive(Debug, Clone)]
pub struct CarParkingProfile {
    /// Parameters for car driving
    pub car_params: CarParameters,

    /// Parameters for walking after parking
    pub foot_params: FootParameters,

    /// Whether this is wheelchair-accessible routing
    pub is_wheelchair: bool,

    /// Penalty for parking car (seconds)
    pub switch_penalty: u16,

    /// Maximum distance to search for a parking facility (meters)
    pub max_parking_distance: f64,

    /// Weight applied to parking cost when optimising drive+walk+cost.
    /// Higher values penalise paid parking more strongly.
    pub parking_cost_weight: f64,
}

impl CarParkingProfile {
    /// Create a new car parking profile.
    ///
    /// # Arguments
    /// * `car_params`     - Driving parameters
    /// * `foot_params`    - Walking parameters
    /// * `is_wheelchair`  - Whether to require wheelchair accessibility
    pub fn new(
        car_params: CarParameters,
        foot_params: FootParameters,
        is_wheelchair: bool,
    ) -> Self {
        Self {
            car_params,
            foot_params,
            is_wheelchair,
            switch_penalty: 200,          // 200 seconds (~3 minutes) to park
            max_parking_distance: 500.0,  // search within 500 m of destination
            parking_cost_weight: 1.0,     // neutral weight by default
        }
    }

    /// Create with custom parking penalty.
    pub fn with_penalty(
        car_params: CarParameters,
        foot_params: FootParameters,
        is_wheelchair: bool,
        penalty: u16,
    ) -> Self {
        Self {
            car_params,
            foot_params,
            is_wheelchair,
            switch_penalty: penalty,
            max_parking_distance: 500.0,
            parking_cost_weight: 1.0,
        }
    }

    /// Get the mode for a given node type.
    pub fn mode(&self, node_type: NodeType) -> Mode {
        match (node_type, self.is_wheelchair) {
            (NodeType::Car, _) => Mode::Car,
            (NodeType::Foot, true) => Mode::Wheelchair,
            (NodeType::Foot, false) => Mode::Foot,
        }
    }

    /// Get switch penalty when parking.
    pub fn parking_penalty(&self, from: NodeType, to: NodeType) -> u16 {
        match (from, to) {
            (NodeType::Car, NodeType::Foot) => self.switch_penalty,
            _ => 0,
        }
    }

    /// Maximum distance to match to street network (uses car's range).
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
            NodeType::Car => {
                // Driving cost (respects speed limits)
                let speed = speed_limit_kmh.unwrap_or(50.0);
                let speed_ms = speed / 3.6;
                (distance_meters / speed_ms) as u16
            }
            NodeType::Foot => {
                // Walking cost (potentially slower for wheelchair)
                (distance_meters / self.foot_params.speed_meters_per_second as f64) as u16
            }
        }
    }

    /// Combined routing cost for drive + park + walk.
    ///
    /// Optimises: `drive_seconds + switch_penalty + walk_seconds + hourly_rate *
    /// parking_cost_weight`.
    ///
    /// # Arguments
    /// * `drive_seconds`  - Time to drive to the parking facility
    /// * `walk_seconds`   - Time to walk from parking to destination
    /// * `hourly_rate`    - Cost per hour at the parking facility
    pub fn combined_cost(
        &self,
        drive_seconds: u32,
        walk_seconds: u32,
        hourly_rate: f64,
    ) -> f64 {
        let time_cost = drive_seconds as f64
            + self.switch_penalty as f64
            + walk_seconds as f64;
        let money_cost = hourly_rate * self.parking_cost_weight;
        time_cost + money_cost
    }

    /// Check if parking is allowed at a location.
    ///
    /// In a full implementation this would check parking restrictions,
    /// time limits, costs, etc.
    pub fn is_parking_allowed(&self, _has_parking: bool) -> bool {
        // Simplified: allow parking anywhere for now
        true
    }
}

impl Default for CarParkingProfile {
    fn default() -> Self {
        Self::new(CarParameters::default(), FootParameters::default(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_car_parking_creation() {
        let profile = CarParkingProfile::default();
        assert_eq!(profile.switch_penalty, 200);
        assert!(!profile.is_wheelchair);
        assert_eq!(profile.max_parking_distance, 500.0);
        assert_eq!(profile.parking_cost_weight, 1.0);
    }

    #[test]
    fn test_wheelchair_mode() {
        let profile =
            CarParkingProfile::new(CarParameters::default(), FootParameters::default(), true);
        assert_eq!(profile.mode(NodeType::Car), Mode::Car);
        assert_eq!(profile.mode(NodeType::Foot), Mode::Wheelchair);
    }

    #[test]
    fn test_normal_mode() {
        let profile = CarParkingProfile::default();
        assert_eq!(profile.mode(NodeType::Car), Mode::Car);
        assert_eq!(profile.mode(NodeType::Foot), Mode::Foot);
    }

    #[test]
    fn test_parking_penalty() {
        let profile = CarParkingProfile::default();
        assert_eq!(profile.parking_penalty(NodeType::Car, NodeType::Foot), 200);
        assert_eq!(profile.parking_penalty(NodeType::Foot, NodeType::Car), 0);
    }

    #[test]
    fn test_combined_cost_free_parking() {
        let profile = CarParkingProfile::default();
        // 120s drive + 200s park penalty + 60s walk + free parking
        let cost = profile.combined_cost(120, 60, 0.0);
        assert_eq!(cost, 380.0);
    }

    #[test]
    fn test_combined_cost_paid_parking() {
        let profile = CarParkingProfile::default(); // weight = 1.0
        // 120s drive + 200s penalty + 60s walk + 3.0 cost-unit hourly
        let cost = profile.combined_cost(120, 60, 3.0);
        assert_eq!(cost, 383.0);
    }

    #[test]
    fn test_parking_type_default() {
        assert_eq!(ParkingType::default(), ParkingType::Street);
    }

    #[test]
    fn test_parking_facility_fields() {
        let facility = ParkingFacility {
            id: "n123456".to_string(),
            location: (-27.4698, 153.0251),
            capacity: 50,
            hourly_rate: 2.5,
            facility_type: ParkingType::ParkingGarage,
        };
        assert_eq!(facility.id, "n123456");
        assert_eq!(facility.capacity, 50);
        assert_eq!(facility.facility_type, ParkingType::ParkingGarage);
    }

    #[test]
    fn test_parking_facility_street_free() {
        let facility = ParkingFacility {
            id: "w789".to_string(),
            location: (0.0, 0.0),
            capacity: 0,
            hourly_rate: 0.0,
            facility_type: ParkingType::Street,
        };
        assert_eq!(facility.hourly_rate, 0.0);
        assert_eq!(facility.facility_type, ParkingType::Street);
    }

    #[test]
    fn test_park_and_ride_type() {
        let facility = ParkingFacility {
            id: "park_ride_1".to_string(),
            location: (0.0, 0.0),
            capacity: 200,
            hourly_rate: 0.0,
            facility_type: ParkingType::ParkAndRide,
        };
        assert_eq!(facility.facility_type, ParkingType::ParkAndRide);
        assert_eq!(facility.capacity, 200);
    }
}
