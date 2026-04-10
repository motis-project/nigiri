//! Profile dispatch utility.
//!
//! This module provides a way to dispatch routing operations to the
//! appropriate profile implementation based on a SearchProfile enum.
//!
//! In C++, this is done with template metaprogramming. In Rust, we use
//! an enum to represent the different profile instances and provide
//! methods to work with them.

use crate::routing::profiles::{
    bike::{
        BikeCostStrategy, BikeProfile, ELEVATION_HIGH_COST, ELEVATION_LOW_COST,
        ELEVATION_NO_COST,
    },
    car::CarProfile,
    foot::FootProfile,
    BikeSharingProfile, CarParkingProfile, CarSharingProfile,
};
use crate::routing::{BikeParameters, CarParameters, FootParameters, SearchProfile};

/// Runtime representation of a routing profile.
///
/// This enum wraps all possible profile types and provides a uniform
/// interface for routing operations.
#[derive(Debug, Clone)]
pub enum ProfileInstance {
    Foot(FootProfile),
    Bike(BikeProfile),
    Car(CarProfile),
    BikeSharing(BikeSharingProfile),
    CarSharing(CarSharingProfile),
    CarParking(CarParkingProfile),
}

impl ProfileInstance {
    /// Create a profile instance from a search profile enum.
    ///
    /// # Arguments
    /// * `profile` - The search profile to instantiate
    ///
    /// # Returns
    /// A ProfileInstance with appropriate parameters
    pub fn from_search_profile(profile: SearchProfile) -> Self {
        match profile {
            SearchProfile::Foot => ProfileInstance::Foot(FootProfile::new(false)),
            SearchProfile::Wheelchair => ProfileInstance::Foot(FootProfile::new(true)),
            SearchProfile::Bike => ProfileInstance::Bike(BikeProfile::new(
                BikeCostStrategy::Safe,
                ELEVATION_NO_COST,
                1000,
            )),
            SearchProfile::BikeFast => ProfileInstance::Bike(BikeProfile::new(
                BikeCostStrategy::Fast,
                ELEVATION_NO_COST,
                1000,
            )),
            SearchProfile::BikeElevationLow => ProfileInstance::Bike(BikeProfile::new(
                BikeCostStrategy::Safe,
                ELEVATION_LOW_COST,
                2100,
            )),
            SearchProfile::BikeElevationHigh => ProfileInstance::Bike(BikeProfile::new(
                BikeCostStrategy::Safe,
                ELEVATION_HIGH_COST,
                2100,
            )),
            SearchProfile::Car => ProfileInstance::Car(CarProfile::new()),
            SearchProfile::CarDropOff => {
                // Drop-off: no parking needed, not wheelchair
                ProfileInstance::CarParking(CarParkingProfile::new(
                    CarParameters::default(),
                    FootParameters::default(),
                    false,
                ))
            }
            SearchProfile::CarDropOffWheelchair => {
                // Drop-off with wheelchair accessibility
                ProfileInstance::CarParking(CarParkingProfile::new(
                    CarParameters::default(),
                    FootParameters::default(),
                    true,
                ))
            }
            SearchProfile::CarParking => {
                // Parking: includes parking search, not wheelchair
                ProfileInstance::CarParking(CarParkingProfile::new(
                    CarParameters::default(),
                    FootParameters::default(),
                    false,
                ))
            }
            SearchProfile::CarParkingWheelchair => {
                // Parking with wheelchair accessibility
                ProfileInstance::CarParking(CarParkingProfile::new(
                    CarParameters::default(),
                    FootParameters::default(),
                    true,
                ))
            }
            SearchProfile::BikeSharing => ProfileInstance::BikeSharing(BikeSharingProfile::new(
                FootParameters::default(),
                BikeParameters::default(),
            )),
            SearchProfile::CarSharing => ProfileInstance::CarSharing(CarSharingProfile::new(
                FootParameters::default(),
                CarParameters::default(),
            )),
        }
    }

    /// Get the maximum match distance for this profile.
    pub fn max_match_distance(&self) -> f64 {
        match self {
            ProfileInstance::Foot(p) => p.max_match_distance() as f64,
            ProfileInstance::Bike(p) => p.max_match_distance() as f64,
            ProfileInstance::Car(p) => p.max_match_distance() as f64,
            ProfileInstance::BikeSharing(p) => p.max_match_distance(),
            ProfileInstance::CarSharing(p) => p.max_match_distance(),
            ProfileInstance::CarParking(p) => p.max_match_distance(),
        }
    }
}

/// Execute a closure with the appropriate profile instance.
///
/// This is the Rust equivalent of the C++ `with_profile` template function.
/// It creates the appropriate profile instance and passes it to the closure.
///
/// # Arguments
/// * `profile` - The search profile to use
/// * `f` - Closure to execute with the profile instance
///
/// # Returns
/// The result of the closure
///
/// # Example
/// ```ignore
/// let result = with_profile(SearchProfile::Foot, |profile| {
///     // Use profile for routing
///     profile.max_match_distance()
/// });
/// ```
pub fn with_profile<F, R>(profile: SearchProfile, f: F) -> R
where
    F: FnOnce(ProfileInstance) -> R,
{
    let instance = ProfileInstance::from_search_profile(profile);
    f(instance)
}

#[cfg(test)]
mod tests {
    use crate::routing::profiles::ElevationCost;

    use super::*;

    #[test]
    fn test_foot_profile_creation() {
        let instance = ProfileInstance::from_search_profile(SearchProfile::Foot);
        assert!(matches!(instance, ProfileInstance::Foot(_)));
        assert_eq!(instance.max_match_distance(), 100.0);
    }

    #[test]
    fn test_wheelchair_profile_creation() {
        let instance = ProfileInstance::from_search_profile(SearchProfile::Wheelchair);
        if let ProfileInstance::Foot(foot) = instance {
            assert!(foot.is_wheelchair);
        } else {
            panic!("Expected FootProfile");
        }
    }


    #[test]
    fn test_bike_profiles() {
        use crate::routing::profiles::ElevationCost;

    // Regular bike
        let bike = ProfileInstance::from_search_profile(SearchProfile::Bike);
        assert!(matches!(bike, ProfileInstance::Bike(_)));
        assert_eq!(bike.max_match_distance(), 100.0);

        // Fast bike
        let fast = ProfileInstance::from_search_profile(SearchProfile::BikeFast);
        if let ProfileInstance::Bike(b) = fast {
            assert_eq!(b.cost_strategy, BikeCostStrategy::Fast);
        } else {
            panic!("Expected BikeProfile");
        }
        // With elevation
        let elev = ProfileInstance::from_search_profile(SearchProfile::BikeElevationLow);
        if let ProfileInstance::Bike(b) = elev {
            assert_eq!(b.elevation_cost, ElevationCost::Low);
        } else {
            panic!("Expected BikeProfile");
        }
    }

    #[test]
    fn test_car_profiles() {
        let car = ProfileInstance::from_search_profile(SearchProfile::Car);
        assert!(matches!(car, ProfileInstance::Car(_)));
        assert_eq!(car.max_match_distance(), 200.0);
    }

    #[test]
    fn test_car_parking_profiles() {
        let parking = ProfileInstance::from_search_profile(SearchProfile::CarParking);
        if let ProfileInstance::CarParking(p) = parking {
            assert!(!p.is_wheelchair);
        } else {
            panic!("Expected CarParkingProfile");
        }

        let wheelchair = ProfileInstance::from_search_profile(SearchProfile::CarParkingWheelchair);
        if let ProfileInstance::CarParking(p) = wheelchair {
            assert!(p.is_wheelchair);
        } else {
            panic!("Expected CarParkingProfile");
        }
    }

    #[test]
    fn test_sharing_profiles() {
        let bike_share = ProfileInstance::from_search_profile(SearchProfile::BikeSharing);
        assert!(matches!(bike_share, ProfileInstance::BikeSharing(_)));

        let car_share = ProfileInstance::from_search_profile(SearchProfile::CarSharing);
        assert!(matches!(car_share, ProfileInstance::CarSharing(_)));
    }

    #[test]
    fn test_with_profile_closure() {
        let distance = with_profile(SearchProfile::Bike, |profile| profile.max_match_distance());
        assert_eq!(distance, 100.0);
    }

    #[test]
    fn test_all_profiles_instantiate() {
        // Ensure all profiles can be created without panicking
        for profile in [
            SearchProfile::Foot,
            SearchProfile::Wheelchair,
            SearchProfile::Bike,
            SearchProfile::BikeFast,
            SearchProfile::BikeElevationLow,
            SearchProfile::BikeElevationHigh,
            SearchProfile::Car,
            SearchProfile::CarDropOff,
            SearchProfile::CarDropOffWheelchair,
            SearchProfile::CarParking,
            SearchProfile::CarParkingWheelchair,
            SearchProfile::BikeSharing,
            SearchProfile::CarSharing,
        ] {
            let instance = ProfileInstance::from_search_profile(profile);
            // Just verify it has a reasonable match distance
            let dist = instance.max_match_distance();
            assert!(dist > 0.0 && dist <= 300.0);
        }
    }
}
