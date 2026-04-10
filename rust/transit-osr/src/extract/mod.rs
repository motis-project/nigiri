//! OSM data extraction module

pub mod extract;
pub mod tags;

pub use extract::{extract, get_speed_limit, is_big_street};
pub use tags::{
    is_accessible, BikeProfile, CarProfile, FootProfile, OsmObjType, Override, Profile, Tags,
};
