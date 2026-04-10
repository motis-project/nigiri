/// A planned journey from origin to destination.
#[derive(Debug, Clone)]
pub struct Journey {
    pub departure_time: i64,
    pub arrival_time: i64,
    pub transfers: u16,
    pub legs: Vec<Leg>,
}

/// A single leg of a journey.
#[derive(Debug, Clone)]
pub enum Leg {
    Transit(TransitLeg),
    Walk(WalkLeg),
}

/// A transit leg (riding a vehicle).
#[derive(Debug, Clone)]
pub struct TransitLeg {
    pub from_location_idx: u32,
    pub to_location_idx: u32,
    pub departure_time: i64,
    pub arrival_time: i64,
    pub transport_idx: u32,
    pub route_idx: u32,
    pub day_idx: u16,
    pub from_stop_idx: u16,
    pub to_stop_idx: u16,
    pub headsign: String,
    pub intermediate_stops: Vec<IntermediateStop>,
}

/// A walking/transfer leg.
#[derive(Debug, Clone)]
pub struct WalkLeg {
    pub from_location_idx: u32,
    pub to_location_idx: u32,
    pub duration_minutes: u32,
}

/// An intermediate stop on a transit leg.
#[derive(Debug, Clone)]
pub struct IntermediateStop {
    pub location_idx: u32,
    pub arrival_mam: Option<i16>,
    pub departure_mam: Option<i16>,
}

/// A departure from a stop.
#[derive(Debug, Clone)]
pub struct Departure {
    pub scheduled_departure: i64,
    pub scheduled_arrival: i64,
    pub transport_idx: u32,
    pub route_idx: u32,
    pub day_idx: u16,
    pub stop_idx: u16,
    pub headsign: String,
    pub clasz: u8,
    pub is_real_time: bool,
    pub delay_seconds: Option<i32>,
    pub is_cancelled: bool,
}

/// Trip detail (full stop sequence).
#[derive(Debug, Clone)]
pub struct TripDetail {
    pub transport_idx: u32,
    pub day_idx: u16,
    pub route_idx: u32,
    pub headsign: String,
    pub clasz: u8,
    pub stop_times: Vec<TripStopTime>,
}

/// A stop time within a trip.
#[derive(Debug, Clone)]
pub struct TripStopTime {
    pub location_idx: u32,
    pub arrival_time: Option<i64>,
    pub departure_time: Option<i64>,
    pub stop_sequence: u16,
}
