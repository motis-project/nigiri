# Phase 13 — Journey Formatting (→ GraphQL Types)

**Priority**: Fourteenth (evolves continuously from Phase 3 onward)  
**Type**: Pure Rust  
**Depends on**: Phase 3 (routing), Phase 4 (OSR), Phase 7 (GBFS), Phase 8 (flex), Phase 9 (ODM), Phase 12 (shapes)  
**Crate**: `transit-server` (resolver layer)  
**Estimated effort**: Large (but incremental — grows as each mode is added)

## Objective

Convert raw RAPTOR routing output (nigiri `journey` structs with typed legs) into the GraphQL `Itinerary` response type. This is the formatting/presentation layer that maps internal routing data to the API schema.

## What motis Does (`src/journey_to_response.cc`, ~1,200+ LOC)

### Overview

The journey-to-response conversion handles:

1. **Leg type dispatch** — Each RAPTOR leg type (RUN, FOOTPATH, OFFSET) maps to a different GraphQL leg
2. **Place resolution** — Convert locations to `Place` objects with coordinates, name, stop ID
3. **Transit leg enrichment** — Extract mode, agency, headsign, route info, intermediate stops
4. **Shape extraction** — Get polyline from shapes storage for each transit segment
5. **Alert attachment** — Lookup GTFS-RT service alerts for affected routes/trips/stops
6. **Fare computation** — nigiri's fare engine for pricing information
7. **Rental annotation** — GBFS rental info for sharing legs
8. **Flex annotation** — Pickup/dropoff windows for demand-responsive legs
9. **ODM annotation** — Taxi/rideshare leg metadata
10. **Timezone handling** — Propagate timezone from route to all stops in the leg

### RAPTOR Leg Types

```rust
pub enum RaptorLegType {
    /// Transit ride: a run (route + trip)
    Run {
        route_idx: RouteIdx,
        trip_idx: TripIdx,
        from_stop_idx: u16,
        to_stop_idx: u16,
    },
    /// Walking transfer between stops
    Footpath {
        from: LocationIdx,
        to: LocationIdx,
        duration_minutes: i16,
    },
    /// First/last-mile offset (walk/bike/car from coordinates to stop)
    Offset {
        from: Position,
        to: LocationIdx,
        duration_minutes: i16,
        mode: LegMode,
        transport_mode_id: Option<u32>,  // GBFS/flex/ODM mode encoding
    },
}
```

## Rust Implementation

### Journey Formatter

```rust
// transit-server/src/format/journey.rs

use crate::graphql::types::*;

pub struct JourneyFormatter<'a> {
    tt: &'a nigiri::Timetable,
    rt: Option<&'a nigiri::RtTimetable>,
    osr: &'a transit_osr::OsrData,
    shapes: &'a ShapesCache,
    gbfs: Option<&'a GbfsData>,
    tags: &'a TagLookup,
}

impl<'a> JourneyFormatter<'a> {
    pub fn format_journey(&self, journey: &nigiri::Journey) -> Itinerary {
        let legs: Vec<Leg> = journey.legs.iter()
            .map(|leg| self.format_leg(leg))
            .collect();

        let departure = legs.first().map(|l| l.start_time()).unwrap_or_default();
        let arrival = legs.last().map(|l| l.end_time()).unwrap_or_default();

        Itinerary {
            duration: (arrival - departure).num_seconds() as i32,
            start_time: departure,
            end_time: arrival,
            legs,
            transfers: count_transfers(&journey.legs),
        }
    }

    fn format_leg(&self, leg: &nigiri::JourneyLeg) -> Leg {
        match leg.leg_type() {
            RaptorLegType::Run { route_idx, trip_idx, from_stop_idx, to_stop_idx } => {
                self.format_transit_leg(route_idx, trip_idx, from_stop_idx, to_stop_idx, leg)
            }
            RaptorLegType::Footpath { from, to, duration_minutes } => {
                self.format_walk_leg(from, to, duration_minutes)
            }
            RaptorLegType::Offset { from, to, duration_minutes, mode, transport_mode_id } => {
                self.format_offset_leg(from, to, duration_minutes, mode, transport_mode_id)
            }
        }
    }
}
```

### Transit Leg Formatting

```rust
// transit-server/src/format/transit_leg.rs

impl<'a> JourneyFormatter<'a> {
    fn format_transit_leg(
        &self,
        route_idx: RouteIdx,
        trip_idx: TripIdx,
        from_stop_idx: u16,
        to_stop_idx: u16,
        leg: &nigiri::JourneyLeg,
    ) -> Leg {
        let route = self.tt.get_route(route_idx);
        let trip = self.tt.get_trip(trip_idx);

        // Resolve stops
        let from_loc_idx = route.stop_at(from_stop_idx);
        let to_loc_idx = route.stop_at(to_stop_idx);
        let from_loc = self.tt.get_location(from_loc_idx);
        let to_loc = self.tt.get_location(to_loc_idx);

        // Get real-time times if available
        let (dep_time, arr_time) = if let Some(rt) = self.rt {
            (
                rt.event_time(trip_idx, from_stop_idx, EventType::Departure)
                    .unwrap_or(leg.departure()),
                rt.event_time(trip_idx, to_stop_idx, EventType::Arrival)
                    .unwrap_or(leg.arrival()),
            )
        } else {
            (leg.departure(), leg.arrival())
        };

        // Intermediate stops
        let intermediates: Vec<IntermediateStop> = (from_stop_idx + 1..to_stop_idx)
            .map(|i| {
                let loc_idx = route.stop_at(i);
                let loc = self.tt.get_location(loc_idx);
                IntermediateStop {
                    stop: self.format_stop(loc_idx, &loc),
                    scheduled_arrival: self.tt.event_time(trip_idx, i, EventType::Arrival),
                    scheduled_departure: self.tt.event_time(trip_idx, i, EventType::Departure),
                    realtime_arrival: self.rt.and_then(|rt|
                        rt.event_time(trip_idx, i, EventType::Arrival)),
                    realtime_departure: self.rt.and_then(|rt|
                        rt.event_time(trip_idx, i, EventType::Departure)),
                }
            })
            .collect();

        // Shape polyline
        let shape = self.shapes.get_shape(route_idx, Some(trip_idx));

        // Timezone from route
        let timezone = route.timezone().unwrap_or("UTC");

        Leg::Transit(TransitLeg {
            mode: transport_class_to_mode(route.transport_class()),
            route: RouteInfo {
                id: self.tags.route_id(route_idx),
                short_name: route.short_name().map(|s| s.to_string()),
                long_name: route.long_name().map(|s| s.to_string()),
                color: route.color().map(|c| format!("#{:06X}", c)),
                text_color: route.text_color().map(|c| format!("#{:06X}", c)),
                agency: route.agency().map(|a| AgencyInfo {
                    name: a.name.to_string(),
                    url: a.url.map(|u| u.to_string()),
                }),
            },
            trip_id: self.tags.trip_id(trip_idx),
            headsign: trip.headsign().map(|s| s.to_string()),
            from: self.format_place_stop(from_loc_idx, &from_loc),
            to: self.format_place_stop(to_loc_idx, &to_loc),
            departure: dep_time,
            arrival: arr_time,
            scheduled_departure: leg.departure(),
            scheduled_arrival: leg.arrival(),
            intermediate_stops: intermediates,
            shape,
            realtime: self.rt.is_some(),
            timezone: timezone.to_string(),
        })
    }
}
```

### Walk/Offset Leg Formatting

```rust
// transit-server/src/format/street_leg.rs

impl<'a> JourneyFormatter<'a> {
    fn format_walk_leg(
        &self,
        from: LocationIdx,
        to: LocationIdx,
        duration_minutes: i16,
    ) -> Leg {
        let from_loc = self.tt.get_location(from);
        let to_loc = self.tt.get_location(to);

        // Try to get walk polyline from OSR
        let shape = transit_osr::routing::route::route(
            &self.osr.ways, &self.osr.lookup,
            Some(&self.osr.elevations),
            SearchProfile::Foot,
            Location::from_latlng_no_level(from_loc.lat, from_loc.lon),
            Location::from_latlng_no_level(to_loc.lat, to_loc.lon),
            (duration_minutes as u32) * 60 * 2, // generous timeout
            RoutingAlgorithm::Dijkstra,
        ).map(|p| path_to_polyline(&p));

        Leg::Walk(WalkLeg {
            from: self.format_place_stop(from, &from_loc),
            to: self.format_place_stop(to, &to_loc),
            duration: duration_minutes as i32 * 60,
            distance: shape.as_ref().map(|_| 0.0), // TODO: compute from path
            shape,
            mode: LegMode::Walk,
        })
    }

    fn format_offset_leg(
        &self,
        from: Position,
        to: LocationIdx,
        duration_minutes: i16,
        mode: LegMode,
        transport_mode_id: Option<u32>,
    ) -> Leg {
        let to_loc = self.tt.get_location(to);

        // Check if this is a GBFS/flex/ODM leg
        if let Some(mode_id) = transport_mode_id {
            if FlexModeId::is_flex(mode_id) {
                return self.format_flex_offset_leg(from, to, mode_id, duration_minutes);
            }
            if mode_id >= GBFS_TRANSPORT_MODE_ID_OFFSET {
                return self.format_gbfs_offset_leg(from, to, mode_id, duration_minutes);
            }
            if mode_id == ODM_TRANSPORT_MODE_ID {
                return self.format_odm_offset_leg(from, to, duration_minutes);
            }
        }

        // Regular walk/bike/car offset
        let profile = leg_mode_to_profile(mode);
        let shape = transit_osr::routing::route::route(
            &self.osr.ways, &self.osr.lookup,
            Some(&self.osr.elevations),
            profile,
            Location::from_latlng_no_level(from.lat, from.lon),
            Location::from_latlng_no_level(to_loc.lat, to_loc.lon),
            (duration_minutes as u32) * 60 * 2,
            RoutingAlgorithm::Dijkstra,
        ).map(|p| path_to_polyline(&p));

        Leg::Walk(WalkLeg {
            from: Place::Coordinate(from),
            to: self.format_place_stop(to, &to_loc),
            duration: duration_minutes as i32 * 60,
            distance: None,
            shape,
            mode,
        })
    }
}
```

### Place Resolution

```rust
// transit-server/src/format/place.rs

impl<'a> JourneyFormatter<'a> {
    fn format_place_stop(&self, idx: LocationIdx, loc: &LocationInfo) -> Place {
        Place::Stop(StopPlace {
            id: self.tags.location_id(idx),
            name: loc.name.clone(),
            lat: loc.lat,
            lon: loc.lon,
            platform: loc.platform.clone(),
            parent_station: loc.parent.map(|p| {
                let parent = self.tt.get_location(p);
                StopRef { id: self.tags.location_id(p), name: parent.name.clone() }
            }),
        })
    }
}
```

### Alert Attachment

```rust
// transit-server/src/format/alerts.rs

impl<'a> JourneyFormatter<'a> {
    pub fn attach_alerts(&self, itinerary: &mut Itinerary) {
        if let Some(rt) = self.rt {
            for leg in &mut itinerary.legs {
                if let Leg::Transit(ref mut transit) = leg {
                    let alerts = rt.get_alerts_for_trip(transit.trip_id.as_deref());
                    let route_alerts = rt.get_alerts_for_route(
                        &transit.route.id.as_deref().unwrap_or("")
                    );

                    transit.alerts = alerts.into_iter()
                        .chain(route_alerts)
                        .map(|a| ServiceAlert {
                            header: a.header_text.clone(),
                            description: a.description_text.clone(),
                            url: a.url.clone(),
                            effect: a.effect.map(|e| format!("{:?}", e)),
                            cause: a.cause.map(|c| format!("{:?}", c)),
                        })
                        .collect();
                }
            }
        }
    }
}
```

### Mode Mapping

```rust
// transit-server/src/format/mode.rs

pub fn transport_class_to_mode(clasz: TransportClass) -> TransitMode {
    match clasz {
        TransportClass::Air => TransitMode::Airplane,
        TransportClass::HighSpeed => TransitMode::HighSpeedRail,
        TransportClass::LongDistance => TransitMode::LongDistanceRail,
        TransportClass::Coach => TransitMode::Coach,
        TransportClass::Night => TransitMode::NightRail,
        TransportClass::RegionalFast => TransitMode::RegionalRail,
        TransportClass::Regional => TransitMode::RegionalRail,
        TransportClass::Metro => TransitMode::Metro,
        TransportClass::Subway => TransitMode::Subway,
        TransportClass::Tram => TransitMode::Tram,
        TransportClass::Bus => TransitMode::Bus,
        TransportClass::Ship => TransitMode::Ferry,
        TransportClass::Other => TransitMode::Other,
    }
}
```

## GraphQL Types

```graphql
type Itinerary {
  duration: Int!
  startTime: DateTime!
  endTime: DateTime!
  transfers: Int!
  legs: [Leg!]!
}

union Leg = TransitLeg | WalkLeg | RentalLeg | TaxiLeg

type TransitLeg {
  mode: TransitMode!
  route: Route!
  tripId: String
  headsign: String
  from: Place!
  to: Place!
  departure: DateTime!
  arrival: DateTime!
  scheduledDeparture: DateTime!
  scheduledArrival: DateTime!
  intermediateStops: [IntermediateStop!]!
  shape: String
  realtime: Boolean!
  alerts: [ServiceAlert!]!
}

type WalkLeg {
  from: Place!
  to: Place!
  duration: Int!
  distance: Float
  shape: String
  mode: LegMode!
}

union Place = StopPlace | CoordinatePlace

type StopPlace {
  id: String!
  name: String!
  lat: Float!
  lon: Float!
  platform: String
  parentStation: StopRef
}
```

## Incremental Build Strategy

Journey formatting evolves with each phase:

| Phase | Leg Types Supported |
|---|---|
| Phase 3 | Transit (RUN) + Walk (FOOTPATH) — basic mode, no shapes |
| Phase 4 | + Street offset legs with polyline |
| Phase 7 | + GBFS rental legs with provider info |
| Phase 8 | + Flex legs with pickup/dropoff windows |
| Phase 9 | + ODM taxi/rideshare legs |
| Phase 10 | Wheelchair-aware footpath rendering |
| Phase 12 | + Route shapes on transit legs |
| Phase 13 | Full formatting with alerts, fares, all leg types |

## Acceptance Criteria

1. Transit legs include route, trip, headsign, agency, intermediate stops
2. Real-time departure/arrival times used when available
3. Walk legs include OSR-routed polyline geometry
4. Offset legs correctly dispatch to GBFS/flex/ODM formatters
5. Service alerts attached to affected transit legs
6. Timezone propagated correctly on all stop times
7. Transfers counted correctly
8. All GraphQL `Itinerary` fields populated per schema

## motis Source Reference

| File | Lines | Key Pattern |
|---|---|---|
| `src/journey_to_response.cc` | ~1,200 | Main formatting: leg dispatch, place resolution, shape extraction |
| `src/endpoints/routing.cc` | ~500 | Query orchestration → response formatting |
| `include/motis/journey_to_response.h` | ~80 | Output interface + leg formatters |
| `include/motis/gbfs/gbfs_output.h` | ~45 | GBFS rental leg formatting |
| `include/motis/flex/flex_output.h` | ~45 | Flex leg formatting |
