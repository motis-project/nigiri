//! Translation of osr/include/osr/routing/profile.h
//!
//! Routing profiles for different modes and preferences.

use std::fmt;

use crate::routing::mode::Mode;
use crate::routing::parameters::{BikeParameters, FootParameters};
use crate::routing::profiles::{self, BikeCostStrategy};
use crate::types::{Cost, Direction};
use crate::ways::{NodeProperties, WayProperties};

/// Search profile selection
///
/// Determines which routing algorithm and cost function to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SearchProfile {
    Foot = 0,
    Wheelchair = 1,
    Bike = 2,
    BikeFast = 3,
    BikeElevationLow = 4,
    BikeElevationHigh = 5,
    Car = 6,
    CarDropOff = 7,
    CarDropOffWheelchair = 8,
    CarParking = 9,
    CarParkingWheelchair = 10,
    BikeSharing = 11,
    CarSharing = 12,
}

impl SearchProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            SearchProfile::Foot => "foot",
            SearchProfile::Wheelchair => "wheelchair",
            SearchProfile::Bike => "bike",
            SearchProfile::BikeFast => "bike_fast",
            SearchProfile::BikeElevationLow => "bike_elevation_low",
            SearchProfile::BikeElevationHigh => "bike_elevation_high",
            SearchProfile::Car => "car",
            SearchProfile::CarDropOff => "car_drop_off",
            SearchProfile::CarDropOffWheelchair => "car_drop_off_wheelchair",
            SearchProfile::CarParking => "car_parking",
            SearchProfile::CarParkingWheelchair => "car_parking_wheelchair",
            SearchProfile::BikeSharing => "bike_sharing",
            SearchProfile::CarSharing => "car_sharing",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "foot" => Some(SearchProfile::Foot),
            "wheelchair" => Some(SearchProfile::Wheelchair),
            "bike" => Some(SearchProfile::Bike),
            "bike_fast" => Some(SearchProfile::BikeFast),
            "bike_elevation_low" => Some(SearchProfile::BikeElevationLow),
            "bike_elevation_high" => Some(SearchProfile::BikeElevationHigh),
            "car" => Some(SearchProfile::Car),
            "car_drop_off" => Some(SearchProfile::CarDropOff),
            "car_drop_off_wheelchair" => Some(SearchProfile::CarDropOffWheelchair),
            "car_parking" => Some(SearchProfile::CarParking),
            "car_parking_wheelchair" => Some(SearchProfile::CarParkingWheelchair),
            "bike_sharing" => Some(SearchProfile::BikeSharing),
            "car_sharing" => Some(SearchProfile::CarSharing),
            _ => None,
        }
    }

    /// Check if this is a rental/sharing profile
    pub fn is_rental(&self) -> bool {
        matches!(self, SearchProfile::BikeSharing | SearchProfile::CarSharing)
    }

    /// Get the base mode for this profile
    pub fn mode(&self) -> Mode {
        match self {
            SearchProfile::Foot | SearchProfile::Wheelchair => Mode::Foot,
            SearchProfile::Bike
            | SearchProfile::BikeFast
            | SearchProfile::BikeElevationLow
            | SearchProfile::BikeElevationHigh
            | SearchProfile::BikeSharing => Mode::Bike,
            SearchProfile::Car
            | SearchProfile::CarDropOff
            | SearchProfile::CarDropOffWheelchair
            | SearchProfile::CarParking
            | SearchProfile::CarParkingWheelchair
            | SearchProfile::CarSharing => Mode::Car,
        }
    }

    /// Calculate way cost in seconds for this profile
    ///
    /// # Arguments
    /// * `props` - Way properties (accessibility, speeds, etc.)
    /// * `dir` - Direction of travel on the way
    /// * `dist` - Distance in meters (rounded to u16)
    ///
    /// # Returns
    /// Cost in seconds, or Cost::MAX if way cannot be used
    pub fn way_cost(&self, props: &WayProperties, dir: Direction, dist: u16) -> Cost {
        match self {
            Self::Foot => {
                let params = FootParameters::default();
                profiles::foot::way_cost(&params, props, false, dist)
            }
            Self::Wheelchair => {
                let params = FootParameters::wheelchair();
                profiles::foot::way_cost(&params, props, true, dist)
            }
            Self::Bike => {
                let params = BikeParameters::default();
                profiles::bike::way_cost(&params, props, dir, BikeCostStrategy::Safe, dist)
            }
            Self::BikeFast => {
                let params = BikeParameters::default();
                profiles::bike::way_cost(&params, props, dir, BikeCostStrategy::Fast, dist)
            }
            Self::BikeElevationLow => {
                let params = BikeParameters::default();
                profiles::bike::way_cost(&params, props, dir, BikeCostStrategy::Safe, dist)
            }
            Self::BikeElevationHigh => {
                let params = BikeParameters::default();
                profiles::bike::way_cost(&params, props, dir, BikeCostStrategy::Safe, dist)
            }
            Self::Car | Self::CarDropOff | Self::CarDropOffWheelchair => {
                profiles::car::way_cost(props, dir, dist)
            }
            Self::CarParking | Self::CarParkingWheelchair => {
                // Car parking drives to parking, then walks. The driving phase uses
                // standard car costs; the mode-switch happens inside
                // route_car_parking_dijkstra() via CarParkingProfile::switch_penalty.
                profiles::car::way_cost(props, dir, dist)
            }
            Self::BikeSharing => {
                // Bike sharing walks to station, cycles, then walks. This returns
                // bike cost as the primary motor; the walk legs are handled by the
                // multi-modal Dijkstra in route_bike_sharing_dijkstra().
                let params = BikeParameters::default();
                profiles::bike::way_cost(&params, props, dir, BikeCostStrategy::Safe, dist)
            }
            Self::CarSharing => {
                // Car sharing walks to pod, drives, then walks. This returns
                // car cost as the primary motor; the walk legs are handled by the
                // multi-modal Dijkstra in route_car_sharing_dijkstra().
                profiles::car::way_cost(props, dir, dist)
            }
        }
    }

    /// Calculate node cost in seconds for this profile
    ///
    /// # Arguments
    /// * `props` - Node properties (accessibility, elevator, etc.)
    ///
    /// # Returns
    /// Cost in seconds, or Cost::MAX if node cannot be used
    pub fn node_cost(&self, props: &NodeProperties) -> Cost {
        match self {
            Self::Foot | Self::Wheelchair => profiles::foot::node_cost(props),
            Self::Bike
            | Self::BikeFast
            | Self::BikeElevationLow
            | Self::BikeElevationHigh
            | Self::BikeSharing => profiles::bike::node_cost(props),
            Self::Car
            | Self::CarDropOff
            | Self::CarDropOffWheelchair
            | Self::CarParking
            | Self::CarParkingWheelchair
            | Self::CarSharing => profiles::car::node_cost(props),
        }
    }

    /// Maximum distance to match a location to the network (meters)
    pub fn max_match_distance(&self) -> f64 {
        match self {
            Self::Foot | Self::Wheelchair => profiles::foot::MAX_MATCH_DISTANCE as f64,
            Self::Bike
            | Self::BikeFast
            | Self::BikeElevationLow
            | Self::BikeElevationHigh
            | Self::BikeSharing => profiles::bike::BikeProfile::MAX_MATCH_DISTANCE as f64,
            Self::Car
            | Self::CarDropOff
            | Self::CarDropOffWheelchair
            | Self::CarParking
            | Self::CarParkingWheelchair
            | Self::CarSharing => profiles::car::CarProfile::MAX_MATCH_DISTANCE as f64,
        }
    }
}

impl fmt::Display for SearchProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for SearchProfile {
    fn default() -> Self {
        SearchProfile::Foot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_to_str() {
        assert_eq!(SearchProfile::Foot.as_str(), "foot");
        assert_eq!(
            SearchProfile::BikeElevationHigh.as_str(),
            "bike_elevation_high"
        );
    }

    #[test]
    fn test_profile_from_str() {
        assert_eq!(
            SearchProfile::from_str("wheelchair"),
            Some(SearchProfile::Wheelchair)
        );
        assert_eq!(SearchProfile::from_str("invalid"), None);
    }

    #[test]
    fn test_is_rental() {
        assert!(SearchProfile::BikeSharing.is_rental());
        assert!(SearchProfile::CarSharing.is_rental());
        assert!(!SearchProfile::Bike.is_rental());
        assert!(!SearchProfile::Car.is_rental());
    }

    #[test]
    fn test_mode() {
        assert_eq!(SearchProfile::Foot.mode(), Mode::Foot);
        assert_eq!(SearchProfile::Wheelchair.mode(), Mode::Foot);
        assert_eq!(SearchProfile::Bike.mode(), Mode::Bike);
        assert_eq!(SearchProfile::BikeFast.mode(), Mode::Bike);
        assert_eq!(SearchProfile::Car.mode(), Mode::Car);
    }

    #[test]
    fn test_default() {
        assert_eq!(SearchProfile::default(), SearchProfile::Foot);
    }
}
