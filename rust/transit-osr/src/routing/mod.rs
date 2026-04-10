//! Routing algorithms and data structures

pub mod additional_edge;
pub mod algorithms;
pub mod bidirectional;
pub mod bidirectional_profile;
pub mod dial;
pub mod dijkstra;
pub mod mode;
pub mod parameters;
pub mod path;
pub mod profile;
pub mod profiles;
pub mod route;
pub mod sharing_data;
pub mod tracking;
pub mod with_profile;

// Re-exports
pub use additional_edge::AdditionalEdge;
pub use algorithms::RoutingAlgorithm;
pub use bidirectional::Bidirectional;
pub use dial::Dial;
pub use dijkstra::{Dijkstra, Label};
pub use mode::Mode;
pub use parameters::{
    get_parameters, BikeParameters, BikeSharingParameters, CarParameters, CarParkingParameters,
    CarSharingParameters, FootParameters, ProfileParameters,
};
pub use path::{Path, Segment};
pub use profile::SearchProfile;
pub use route::{route, route_multi};
pub use sharing_data::{is_allowed, SharingData};
pub use tracking::{ElevatorTracking, NoopTracking, TrackNodeTracking, Tracking};
pub use with_profile::{with_profile, ProfileInstance};
