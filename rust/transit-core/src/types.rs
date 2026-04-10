use serde::{Deserialize, Serialize};
use std::fmt;

/// Macro for generating strongly-typed index newtypes over `u32`.
///
/// Each generated type implements: Debug, Clone, Copy, PartialEq, Eq, Hash,
/// PartialOrd, Ord, Display, From<u32>, Serialize, Deserialize.
macro_rules! define_idx {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u32);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<u32> for $name {
            fn from(v: u32) -> Self {
                Self(v)
            }
        }

        impl From<$name> for u32 {
            fn from(v: $name) -> Self {
                v.0
            }
        }

        impl $name {
            pub const INVALID: Self = Self(u32::MAX);

            pub fn is_valid(self) -> bool {
                self != Self::INVALID
            }
        }
    };
}

define_idx!(LocationIdx, "Index into the location (stop/station) array.");
define_idx!(RouteIdx, "Index into the route array.");
define_idx!(
    TransportIdx,
    "Index into the transport (trip-instance) array."
);
define_idx!(TripIdx, "Index into the trip array.");
define_idx!(SourceIdx, "Index identifying the data source/tag.");
define_idx!(DayIdx, "Day index within the timetable interval.");
define_idx!(
    MinutesAfterMidnight,
    "Minutes after midnight for time representation."
);

// --- Geo types ---

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub lat: f64,
    pub lon: f64,
}

impl Position {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
}

impl BoundingBox {
    pub fn contains(&self, pos: &Position) -> bool {
        pos.lat >= self.min_lat
            && pos.lat <= self.max_lat
            && pos.lon >= self.min_lon
            && pos.lon <= self.max_lon
    }
}

// --- GraphQL shared types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteColor {
    pub background: String,
    pub text: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PageInfo {
    pub page: u32,
    pub page_size: u32,
    pub total_items: u32,
    pub total_pages: u32,
}

impl PageInfo {
    pub fn new(page: u32, page_size: u32, total_items: u32) -> Self {
        let total_pages = if page_size == 0 {
            0
        } else {
            total_items.div_ceil(page_size)
        };
        Self {
            page,
            page_size,
            total_items,
            total_pages,
        }
    }
}

// --- Enums matching GraphQL schema ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RouteType {
    Tram,
    Subway,
    Rail,
    Bus,
    Ferry,
    CableTram,
    AerialLift,
    Funicular,
    Trolleybus,
    Monorail,
    HighSpeedRail,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WheelchairAccessible {
    NoInformation,
    Accessible,
    NotAccessible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OccupancyStatus {
    Empty,
    ManySeatsAvailable,
    FewSeatsAvailable,
    StandingRoomOnly,
    CrushedStandingRoomOnly,
    Full,
    NotAcceptingPassengers,
    NoDataAvailable,
    NotBoardable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LocationType {
    Stop,
    Station,
    Entrance,
    GenericNode,
    BoardingArea,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idx_display_and_conversion() {
        let loc = LocationIdx(42);
        assert_eq!(format!("{loc}"), "42");
        assert_eq!(u32::from(loc), 42);
        assert_eq!(LocationIdx::from(42), loc);
    }

    #[test]
    fn idx_invalid_sentinel() {
        assert!(!LocationIdx::INVALID.is_valid());
        assert!(LocationIdx(0).is_valid());
    }

    #[test]
    fn position_basics() {
        let p = Position::new(48.1351, 11.5820);
        assert_eq!(p.lat, 48.1351);
        assert_eq!(p.lon, 11.5820);
    }

    #[test]
    fn bounding_box_contains() {
        let bb = BoundingBox {
            min_lat: 48.0,
            min_lon: 11.0,
            max_lat: 49.0,
            max_lon: 12.0,
        };
        assert!(bb.contains(&Position::new(48.5, 11.5)));
        assert!(!bb.contains(&Position::new(50.0, 11.5)));
    }

    #[test]
    fn page_info_calculation() {
        let pi = PageInfo::new(1, 10, 25);
        assert_eq!(pi.total_pages, 3);
    }

    #[test]
    fn route_type_serde() {
        let json = serde_json::to_string(&RouteType::HighSpeedRail).unwrap();
        assert_eq!(json, "\"HIGH_SPEED_RAIL\"");
    }
}
