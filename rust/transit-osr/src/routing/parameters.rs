//! Translation of osr/src/routing/parameters.h + parameters.cc
//!
//! Routing parameters for different profiles.
//! This module mirrors the C++ profile_parameters variant and get_parameters.

use crate::routing::profile::SearchProfile;

/// Default walking speed for foot routing (meters/second).
pub const FOOT_SPEED_MPS: f32 = 1.2;

/// Default walking speed for wheelchair routing (meters/second).
pub const WHEELCHAIR_SPEED_MPS: f32 = 0.8;

/// Default bike speed (meters/second).
pub const BIKE_SPEED_MPS: f32 = 4.2;

/// Foot routing parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FootParameters {
    pub speed_meters_per_second: f32,
}

impl FootParameters {
    pub fn foot() -> Self {
        Self {
            speed_meters_per_second: FOOT_SPEED_MPS,
        }
    }

    pub fn wheelchair() -> Self {
        Self {
            speed_meters_per_second: WHEELCHAIR_SPEED_MPS,
        }
    }
}

impl Default for FootParameters {
    fn default() -> Self {
        Self::foot()
    }
}

/// Bike routing parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BikeParameters {
    pub speed_meters_per_second: f32,
}

impl Default for BikeParameters {
    fn default() -> Self {
        Self {
            speed_meters_per_second: BIKE_SPEED_MPS,
        }
    }
}

/// Car routing parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CarParameters;

impl Default for CarParameters {
    fn default() -> Self {
        CarParameters
    }
}

/// Bike sharing parameters (bike + foot).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BikeSharingParameters {
    pub bike: BikeParameters,
    pub foot: FootParameters,
}

impl Default for BikeSharingParameters {
    fn default() -> Self {
        Self {
            bike: BikeParameters::default(),
            foot: FootParameters::default(),
        }
    }
}

/// Car sharing parameters (car + foot).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarSharingParameters {
    pub car: CarParameters,
    pub foot: FootParameters,
}

impl Default for CarSharingParameters {
    fn default() -> Self {
        Self {
            car: CarParameters::default(),
            foot: FootParameters::default(),
        }
    }
}

/// Car parking parameters (car + foot, plus template flags).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarParkingParameters {
    pub car: CarParameters,
    pub foot: FootParameters,
    pub is_wheelchair: bool,
    pub use_parking: bool,
}

impl CarParkingParameters {
    pub fn drop_off() -> Self {
        Self {
            car: CarParameters::default(),
            foot: FootParameters::default(),
            is_wheelchair: false,
            use_parking: false,
        }
    }

    pub fn drop_off_wheelchair() -> Self {
        Self {
            car: CarParameters::default(),
            foot: FootParameters::wheelchair(),
            is_wheelchair: true,
            use_parking: false,
        }
    }

    pub fn parking() -> Self {
        Self {
            car: CarParameters::default(),
            foot: FootParameters::default(),
            is_wheelchair: false,
            use_parking: true,
        }
    }

    pub fn parking_wheelchair() -> Self {
        Self {
            car: CarParameters::default(),
            foot: FootParameters::wheelchair(),
            is_wheelchair: true,
            use_parking: true,
        }
    }
}

/// Parameters for all supported routing profiles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProfileParameters {
    Foot(FootParameters),
    Wheelchair(FootParameters),
    FootElevator(FootParameters),
    WheelchairElevator(FootParameters),
    BikeSafeNoElevation(BikeParameters),
    BikeFastNoElevation(BikeParameters),
    BikeSafeLowElevation(BikeParameters),
    BikeSafeHighElevation(BikeParameters),
    Car(CarParameters),
    CarDropOff(CarParkingParameters),
    CarDropOffWheelchair(CarParkingParameters),
    CarParking(CarParkingParameters),
    CarParkingWheelchair(CarParkingParameters),
    BikeSharing(BikeSharingParameters),
    CarSharing(CarSharingParameters),
}

/// Get the parameters for a given search profile.
pub fn get_parameters(profile: SearchProfile) -> ProfileParameters {
    match profile {
        SearchProfile::Foot => ProfileParameters::FootElevator(FootParameters::foot()),
        SearchProfile::Wheelchair => {
            ProfileParameters::WheelchairElevator(FootParameters::wheelchair())
        }
        SearchProfile::Bike => ProfileParameters::BikeSafeNoElevation(BikeParameters::default()),
        SearchProfile::BikeFast => {
            ProfileParameters::BikeFastNoElevation(BikeParameters::default())
        }
        SearchProfile::BikeElevationLow => {
            ProfileParameters::BikeSafeLowElevation(BikeParameters::default())
        }
        SearchProfile::BikeElevationHigh => {
            ProfileParameters::BikeSafeHighElevation(BikeParameters::default())
        }
        SearchProfile::Car => ProfileParameters::Car(CarParameters::default()),
        SearchProfile::CarDropOff => {
            ProfileParameters::CarDropOff(CarParkingParameters::drop_off())
        }
        SearchProfile::CarDropOffWheelchair => {
            ProfileParameters::CarDropOffWheelchair(CarParkingParameters::drop_off_wheelchair())
        }
        SearchProfile::CarParking => ProfileParameters::CarParking(CarParkingParameters::parking()),
        SearchProfile::CarParkingWheelchair => {
            ProfileParameters::CarParkingWheelchair(CarParkingParameters::parking_wheelchair())
        }
        SearchProfile::BikeSharing => {
            ProfileParameters::BikeSharing(BikeSharingParameters::default())
        }
        SearchProfile::CarSharing => ProfileParameters::CarSharing(CarSharingParameters::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foot_parameters_defaults() {
        let foot = FootParameters::default();
        let wheelchair = FootParameters::wheelchair();
        assert_eq!(foot.speed_meters_per_second, FOOT_SPEED_MPS);
        assert_eq!(wheelchair.speed_meters_per_second, WHEELCHAIR_SPEED_MPS);
    }

    #[test]
    fn test_get_parameters_variants() {
        assert!(matches!(
            get_parameters(SearchProfile::Foot),
            ProfileParameters::FootElevator(_)
        ));
        assert!(matches!(
            get_parameters(SearchProfile::BikeElevationHigh),
            ProfileParameters::BikeSafeHighElevation(_)
        ));
        assert!(matches!(
            get_parameters(SearchProfile::CarParkingWheelchair),
            ProfileParameters::CarParkingWheelchair(_)
        ));
        assert!(matches!(
            get_parameters(SearchProfile::CarSharing),
            ProfileParameters::CarSharing(_)
        ));
    }
}
