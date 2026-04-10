use async_graphql::{Enum, SimpleObject, ID};

// --- Scalars ---

/// Geographic position (latitude/longitude).
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlPosition {
    pub lat: f64,
    pub lon: f64,
}

/// Color scheme for transit route branding.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlColor {
    pub background: String,
    pub text: String,
}

/// Pagination metadata.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlPageInfo {
    pub current_page: i32,
    pub page_size: i32,
    pub total_items: i32,
    pub total_pages: i32,
    pub has_next_page: bool,
    pub has_previous_page: bool,
}

impl GqlPageInfo {
    pub fn new(page: u32, page_size: u32, total_items: u32) -> Self {
        let total_pages = if page_size == 0 {
            0
        } else {
            total_items.div_ceil(page_size)
        };
        Self {
            current_page: page as i32,
            page_size: page_size as i32,
            total_items: total_items as i32,
            total_pages: total_pages as i32,
            has_next_page: page < total_pages,
            has_previous_page: page > 1,
        }
    }
}

// --- Enums ---

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GqlLocationType {
    Stop,
    Station,
    Entrance,
    GenericNode,
    BoardingArea,
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GqlRouteType {
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

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GqlWheelchairAccessible {
    NoInformation,
    Accessible,
    NotAccessible,
}

// --- Stop ---

/// A transit stop or platform.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlStop {
    pub id: ID,
    pub source: String,
    pub gtfs_id: String,
    pub name: String,
    pub code: Option<String>,
    pub position: GqlPosition,
    pub location_type: GqlLocationType,
    pub route_type: Option<GqlRouteType>,
    pub wheelchair_boarding: GqlWheelchairAccessible,
    pub distance: Option<f64>,
    // Phase 3+: departures, routes, alerts, parentStation resolved lazily
}

/// A transit station (group of stops).
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlStation {
    pub id: ID,
    pub source: String,
    pub gtfs_id: String,
    pub name: String,
    pub position: GqlPosition,
    pub location_type: GqlLocationType,
    pub child_stops: Vec<GqlStop>,
    // Phase 3+: routes, departures resolved lazily
}

/// Lightweight stop reference for use in route stop lists.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlStopReference {
    pub id: ID,
    pub source: String,
    pub gtfs_id: String,
    pub name: String,
    pub code: Option<String>,
    pub position: GqlPosition,
    pub location_type: GqlLocationType,
    pub distance: Option<f64>,
    pub platform_code: Option<String>,
}

// --- Route ---

/// A transit route.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlRoute {
    pub id: ID,
    pub source: String,
    pub gtfs_route_ids: Vec<String>,
    pub short_name: Option<String>,
    pub long_name: String,
    #[graphql(name = "type")]
    pub route_type: GqlRouteType,
    pub color: Option<GqlColor>,
    pub timezone: String,
    pub agency: Option<GqlAgencyReference>,
    pub network: Option<String>,
    // Phase 3+: itineraries, stops, trips, alerts resolved lazily
}

/// Lightweight route reference.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlRouteReference {
    pub id: ID,
    pub source: String,
    pub gtfs_id: String,
    pub short_name: Option<String>,
    pub long_name: String,
    #[graphql(name = "type")]
    pub route_type: GqlRouteType,
    pub color: Option<GqlColor>,
}

/// Agency (operator) reference.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlAgencyReference {
    pub id: ID,
    pub source: String,
    pub gtfs_id: String,
    pub name: String,
}

// --- Connections ---

/// Paginated list of stops.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlStopsConnection {
    pub items: Vec<GqlStop>,
    pub page_info: GqlPageInfo,
}

/// Paginated list of routes.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlRoutesConnection {
    pub items: Vec<GqlRoute>,
    pub page_info: GqlPageInfo,
}

// --- Phase 3: Journey/Departure/Trip types ---

/// A planned journey from origin to destination.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlJourney {
    pub departure_time: String,
    pub arrival_time: String,
    pub duration_seconds: i32,
    pub transfers: i32,
    pub legs: Vec<GqlLeg>,
}

/// A leg of a journey — either transit or walking.
#[derive(async_graphql::Union, Debug, Clone)]
pub enum GqlLeg {
    Transit(GqlTransitLeg),
    Walk(GqlWalkLeg),
}

/// A transit leg (riding a vehicle).
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlTransitLeg {
    pub from: GqlPosition,
    pub from_name: String,
    pub from_id: String,
    pub to: GqlPosition,
    pub to_name: String,
    pub to_id: String,
    pub departure_time: String,
    pub arrival_time: String,
    pub route: GqlRouteReference,
    pub headsign: String,
    pub intermediate_stops: Vec<GqlIntermediateStop>,
    pub trip_id: Option<String>,
}

/// A walking/transfer leg.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlWalkLeg {
    pub from: GqlPosition,
    pub from_name: String,
    pub to: GqlPosition,
    pub to_name: String,
    pub duration_seconds: i32,
}

/// An intermediate stop on a transit leg.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlIntermediateStop {
    pub name: String,
    pub position: GqlPosition,
    pub arrival_time: Option<String>,
    pub departure_time: Option<String>,
}

/// A departure from a stop.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlDeparture {
    pub aimed_departure_time: String,
    pub aimed_arrival_time: String,
    pub estimated_departure_time: Option<String>,
    pub estimated_arrival_time: Option<String>,
    pub is_cancelled: Option<bool>,
    pub is_real_time: bool,
    pub delay: Option<i32>,
    pub trip_id: Option<String>,
    pub route: GqlRouteReference,
    pub headsign: String,
    pub direction: i32,
    pub accessible: GqlWheelchairAccessible,
}

/// Full trip detail with stop times.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlTrip {
    pub id: ID,
    pub route: GqlRouteReference,
    pub headsign: String,
    pub direction: i32,
    pub stop_times: Vec<GqlStopTime>,
}

/// A stop time within a trip.
#[derive(SimpleObject, Debug, Clone)]
pub struct GqlStopTime {
    pub stop_sequence: i32,
    pub stop_id: String,
    pub stop_name: String,
    pub position: GqlPosition,
    pub arrival_time: Option<String>,
    pub departure_time: Option<String>,
}

// --- Conversion helpers ---

/// Convert nigiri clasz (0-9) to GraphQL RouteType.
pub fn clasz_to_route_type(clasz: u8) -> GqlRouteType {
    match clasz {
        0 => GqlRouteType::HighSpeedRail,
        1 => GqlRouteType::Rail, // long distance
        2 => GqlRouteType::Rail, // night
        3 => GqlRouteType::Rail, // regional fast
        4 => GqlRouteType::Rail, // regional
        5 => GqlRouteType::Subway,
        6 => GqlRouteType::Tram,
        7 => GqlRouteType::Bus,
        8 => GqlRouteType::Ferry,
        9 => GqlRouteType::Other,
        _ => GqlRouteType::Other,
    }
}

/// Convert nigiri location_type (0=generated track, 1=track, 2=station) to GraphQL.
pub fn nigiri_location_type(lt: u8) -> GqlLocationType {
    match lt {
        2 => GqlLocationType::Station,
        _ => GqlLocationType::Stop,
    }
}

/// Convert a u32 color (0xRRGGBB) to a hex string.
pub fn color_to_hex(color: u32) -> String {
    format!("{:06X}", color & 0xFFFFFF)
}
