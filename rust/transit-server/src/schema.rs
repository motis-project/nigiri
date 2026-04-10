use async_graphql::{Context, EmptyMutation, EmptySubscription, Object, Result, Schema, ID};
use transit_core::{SourceIdx, TagLookup};

use crate::app_data::AppData;
use crate::gql_types::*;

pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Health check — returns true when the server is running.
    async fn health(&self) -> bool {
        true
    }

    /// Get stop by ID (format: "tag:gtfs_id" or bare "gtfs_id").
    async fn stop(&self, ctx: &Context<'_>, id: ID) -> Result<Option<GqlStop>> {
        let data = ctx.data::<AppData>()?;
        let tt = match &data.timetable {
            Some(tt) => tt,
            None => return Ok(None),
        };

        let idx = resolve_location_id(tt, data.tags.as_ref(), &id)?;
        let idx = match idx {
            Some(idx) => idx,
            None => return Ok(None),
        };

        Ok(Some(build_stop(tt, data.tags.as_ref(), idx, None)?))
    }

    /// Get station by ID.
    async fn station(&self, ctx: &Context<'_>, id: ID) -> Result<Option<GqlStation>> {
        let data = ctx.data::<AppData>()?;
        let tt = match &data.timetable {
            Some(tt) => tt,
            None => return Ok(None),
        };

        let idx = resolve_location_id(tt, data.tags.as_ref(), &id)?;
        let idx = match idx {
            Some(idx) => idx,
            None => return Ok(None),
        };

        build_station(tt, data.tags.as_ref(), idx)
    }

    /// Get route by ID (format: "tag:gtfs_route_id" or bare "route_idx").
    async fn route(&self, ctx: &Context<'_>, id: ID) -> Result<Option<GqlRoute>> {
        let data = ctx.data::<AppData>()?;
        let tt = match &data.timetable {
            Some(tt) => tt,
            None => return Ok(None),
        };

        // Try tag:id format first, fall back to numeric route_idx
        let route_idx = resolve_route_id(tt, data.tags.as_ref(), &id)?;
        let route_idx = match route_idx {
            Some(idx) => idx,
            None => return Ok(None),
        };

        Ok(Some(build_route(tt, data.tags.as_ref(), route_idx)?))
    }

    /// Find stops near a location.
    async fn nearby_stops(
        &self,
        ctx: &Context<'_>,
        lat: f64,
        lon: f64,
        #[graphql(default = 500.0)] radius: f64,
    ) -> Result<Vec<GqlStop>> {
        let data = ctx.data::<AppData>()?;
        let tt = match &data.timetable {
            Some(tt) => tt,
            None => return Ok(vec![]),
        };
        let rtree = match &data.location_rtree {
            Some(rt) => rt,
            None => return Ok(vec![]),
        };

        let radius_deg = transit_import::rtree::meters_to_degrees(radius, lat);
        let nearby = transit_import::rtree::locations_in_radius(rtree, lat, lon, radius_deg);

        let mut stops = Vec::with_capacity(nearby.len());
        for entry in nearby {
            let distance = haversine_distance(lat, lon, entry.lat, entry.lon);
            if let Ok(stop) = build_stop(tt, data.tags.as_ref(), entry.idx.0, Some(distance)) {
                stops.push(stop);
            }
        }
        stops.sort_by(|a, b| {
            a.distance
                .unwrap_or(f64::MAX)
                .partial_cmp(&b.distance.unwrap_or(f64::MAX))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(stops)
    }

    /// Search stops with pagination.
    async fn stops(
        &self,
        ctx: &Context<'_>,
        phrase: Option<String>,
        #[graphql(default = 1)] page: i32,
        #[graphql(default = 50)] page_size: i32,
    ) -> Result<GqlStopsConnection> {
        let data = ctx.data::<AppData>()?;
        let tt = match &data.timetable {
            Some(tt) => tt,
            None => {
                return Ok(GqlStopsConnection {
                    items: vec![],
                    page_info: GqlPageInfo::new(1, page_size as u32, 0),
                });
            }
        };

        let page = page.max(1) as u32;
        let page_size = page_size.clamp(1, 1000) as u32;

        // Collect matching stops
        let mut all_stops = Vec::new();
        let loc_count = tt.location_count();
        for i in 0..loc_count {
            if let Ok(detail) = tt.get_location_detail(i) {
                // Skip special/zero-coord locations
                if detail.lat == 0.0 && detail.lon == 0.0 {
                    continue;
                }
                // Filter by phrase if provided
                if let Some(ref phrase) = phrase {
                    let lower_phrase = phrase.to_lowercase();
                    if !detail.name.to_lowercase().contains(&lower_phrase)
                        && !detail.id.to_lowercase().contains(&lower_phrase)
                    {
                        continue;
                    }
                }
                all_stops.push(i);
            }
        }

        let total = all_stops.len() as u32;
        let start = ((page - 1) * page_size) as usize;
        let end = (start + page_size as usize).min(all_stops.len());

        let tags = data.tags.as_ref();
        let items: Vec<GqlStop> = all_stops[start..end]
            .iter()
            .filter_map(|&idx| build_stop(tt, tags, idx, None).ok())
            .collect();

        Ok(GqlStopsConnection {
            items,
            page_info: GqlPageInfo::new(page, page_size, total),
        })
    }

    /// Search routes with pagination.
    async fn routes(
        &self,
        ctx: &Context<'_>,
        phrase: Option<String>,
        #[graphql(default = 1)] page: i32,
        #[graphql(default = 50)] page_size: i32,
    ) -> Result<GqlRoutesConnection> {
        let data = ctx.data::<AppData>()?;
        let tt = match &data.timetable {
            Some(tt) => tt,
            None => {
                return Ok(GqlRoutesConnection {
                    items: vec![],
                    page_info: GqlPageInfo::new(1, page_size as u32, 0),
                });
            }
        };

        let page = page.max(1) as u32;
        let page_size = page_size.clamp(1, 1000) as u32;

        let mut all_routes = Vec::new();
        let route_count = tt.route_count();
        for i in 0..route_count {
            if let Ok(detail) = tt.get_route_detail(i) {
                if let Some(ref phrase) = phrase {
                    let lower = phrase.to_lowercase();
                    if !detail.short_name.to_lowercase().contains(&lower)
                        && !detail.long_name.to_lowercase().contains(&lower)
                    {
                        continue;
                    }
                }
                all_routes.push(i);
            }
        }

        let total = all_routes.len() as u32;
        let start = ((page - 1) * page_size) as usize;
        let end = (start + page_size as usize).min(all_routes.len());

        let tags = data.tags.as_ref();
        let items: Vec<GqlRoute> = all_routes[start..end]
            .iter()
            .filter_map(|&idx| build_route(tt, tags, idx).ok())
            .collect();

        Ok(GqlRoutesConnection {
            items,
            page_info: GqlPageInfo::new(page, page_size, total),
        })
    }

    // --- Phase 3: Routing resolvers ---

    /// Plan a journey between two stops.
    async fn plan_journey(
        &self,
        ctx: &Context<'_>,
        from: ID,
        to: ID,
        time: String,
        #[graphql(default = false)] arrive_by: bool,
    ) -> Result<Vec<GqlJourney>> {
        let data = ctx.data::<AppData>()?;
        let tt = match &data.timetable {
            Some(tt) => tt,
            None => return Err("no timetable loaded".into()),
        };
        let tags = data.tags.as_ref().ok_or("no tag lookup")?;

        let from_idx = resolve_location_id(tt, Some(tags), &from)?.ok_or_else(|| {
            async_graphql::Error::new(format!("origin not found: {}", from.as_str()))
        })?;
        let to_idx = resolve_location_id(tt, Some(tags), &to)?.ok_or_else(|| {
            async_graphql::Error::new(format!("destination not found: {}", to.as_str()))
        })?;

        let time_ts = parse_datetime(&time)?;

        let router = transit_routing::TransitRouter::new(tt, tags);
        let journeys = router
            .plan_journey(from_idx, to_idx, time_ts, arrive_by)
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;

        journeys
            .into_iter()
            .map(|j| convert_journey(tt, tags, j))
            .collect()
    }

    /// Get upcoming departures from a stop.
    async fn stop_departures(
        &self,
        ctx: &Context<'_>,
        stop_id: ID,
        #[graphql(default = 10)] limit: i32,
        start_time: Option<String>,
    ) -> Result<Vec<GqlDeparture>> {
        let data = ctx.data::<AppData>()?;
        let tt = match &data.timetable {
            Some(tt) => tt,
            None => return Ok(vec![]),
        };
        let tags = data.tags.as_ref().ok_or("no tag lookup")?;

        let loc_idx = resolve_location_id(tt, Some(tags), &stop_id)?.ok_or_else(|| {
            async_graphql::Error::new(format!("stop not found: {}", stop_id.as_str()))
        })?;

        let start_ts = match &start_time {
            Some(t) => parse_datetime(t)?,
            None => chrono::Utc::now().timestamp(),
        };

        let router = transit_routing::TransitRouter::new(tt, tags);
        let deps = router
            .stop_departures(loc_idx, start_ts, limit.max(1) as u32)
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;

        deps.into_iter()
            .map(|d| convert_departure(tt, tags, d))
            .collect()
    }

    /// Get trip details by transport_idx and day_idx.
    async fn trip(&self, ctx: &Context<'_>, id: ID) -> Result<Option<GqlTrip>> {
        let data = ctx.data::<AppData>()?;
        let tt = match &data.timetable {
            Some(tt) => tt,
            None => return Ok(None),
        };
        let tags = data.tags.as_ref().ok_or("no tag lookup")?;

        // Parse trip ID: "transport_idx:day_idx" (simple format for now)
        let parts: Vec<&str> = id.split(':').collect();
        if parts.len() < 2 {
            return Ok(None);
        }
        let transport_idx: u32 = parts[0].parse().map_err(|_| "invalid transport_idx")?;
        let day_idx: u16 = parts[1].parse().map_err(|_| "invalid day_idx")?;

        let router = transit_routing::TransitRouter::new(tt, tags);
        let detail = router
            .trip_detail(transport_idx, day_idx)
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;

        Ok(Some(convert_trip(tt, tags, detail)?))
    }
}

// --- ID resolution helpers ---

fn resolve_location_id(
    tt: &nigiri::Timetable,
    tags: Option<&TagLookup>,
    id: &str,
) -> Result<Option<u32>, async_graphql::Error> {
    // If we have tags and the ID contains ':', try tag:gtfs_id format
    if let Some(tags) = tags {
        if let Ok((_src, gtfs_id)) = tags.parse_entity_id(id) {
            return Ok(tt.find_location(gtfs_id));
        }
    }
    // Fall back to bare GTFS ID
    Ok(tt.find_location(id))
}

fn resolve_route_id(
    tt: &nigiri::Timetable,
    tags: Option<&TagLookup>,
    id: &str,
) -> Result<Option<u32>, async_graphql::Error> {
    // Try tag:gtfs_route_id format
    if let Some(tags) = tags {
        if let Ok((_src, gtfs_route_id)) = tags.parse_entity_id(id) {
            // Search routes by GTFS ID
            for i in 0..tt.route_count() {
                if let Some(rid) = tt.route_gtfs_id(i) {
                    if rid == gtfs_route_id {
                        return Ok(Some(i));
                    }
                }
            }
            return Ok(None);
        }
    }
    // Fall back to numeric route index
    if let Ok(idx) = id.parse::<u32>() {
        if idx < tt.route_count() {
            return Ok(Some(idx));
        }
    }
    // Try bare GTFS route_id
    for i in 0..tt.route_count() {
        if let Some(rid) = tt.route_gtfs_id(i) {
            if rid == id {
                return Ok(Some(i));
            }
        }
    }
    Ok(None)
}

// --- Builder helpers ---

fn get_tag_and_id(
    tags: Option<&TagLookup>,
    src_idx: u32,
    gtfs_id: &str,
) -> (String, String, String) {
    if let Some(tags) = tags {
        let tag = tags
            .tag_for_source(SourceIdx(src_idx))
            .unwrap_or("")
            .to_string();
        let full_id = tags
            .format_location_id(SourceIdx(src_idx), gtfs_id)
            .unwrap_or_else(|| gtfs_id.to_string());
        (full_id, tag, gtfs_id.to_string())
    } else {
        (gtfs_id.to_string(), String::new(), gtfs_id.to_string())
    }
}

fn build_stop(
    tt: &nigiri::Timetable,
    tags: Option<&TagLookup>,
    idx: u32,
    distance: Option<f64>,
) -> Result<GqlStop, async_graphql::Error> {
    let detail = tt
        .get_location_detail(idx)
        .map_err(|e| async_graphql::Error::new(e.to_string()))?;

    let (full_id, source, gtfs_id) = get_tag_and_id(tags, detail.src_idx, &detail.id);

    Ok(GqlStop {
        id: ID(full_id),
        source,
        gtfs_id,
        name: detail.name,
        code: None,
        position: GqlPosition {
            lat: detail.lat,
            lon: detail.lon,
        },
        location_type: nigiri_location_type(detail.location_type),
        route_type: None,
        wheelchair_boarding: GqlWheelchairAccessible::NoInformation,
        distance,
    })
}

fn build_station(
    tt: &nigiri::Timetable,
    tags: Option<&TagLookup>,
    idx: u32,
) -> Result<Option<GqlStation>, async_graphql::Error> {
    let detail = tt
        .get_location_detail(idx)
        .map_err(|e| async_graphql::Error::new(e.to_string()))?;

    let station_idx = if detail.location_type == 2 {
        idx
    } else if let Some(parent) = detail.parent_idx {
        parent
    } else {
        return Ok(None);
    };

    let station_detail = tt
        .get_location_detail(station_idx)
        .map_err(|e| async_graphql::Error::new(e.to_string()))?;

    let (full_id, source, gtfs_id) =
        get_tag_and_id(tags, station_detail.src_idx, &station_detail.id);

    let mut children = Vec::new();
    let loc_count = tt.location_count();
    for i in 0..loc_count {
        if let Ok(child_detail) = tt.get_location_detail(i) {
            if child_detail.parent_idx == Some(station_idx) && i != station_idx {
                if let Ok(stop) = build_stop(tt, tags, i, None) {
                    children.push(stop);
                }
            }
        }
    }

    Ok(Some(GqlStation {
        id: ID(full_id),
        source,
        gtfs_id,
        name: station_detail.name,
        position: GqlPosition {
            lat: station_detail.lat,
            lon: station_detail.lon,
        },
        location_type: GqlLocationType::Station,
        child_stops: children,
    }))
}

fn build_route(
    tt: &nigiri::Timetable,
    tags: Option<&TagLookup>,
    idx: u32,
) -> Result<GqlRoute, async_graphql::Error> {
    let detail = tt
        .get_route_detail(idx)
        .map_err(|e| async_graphql::Error::new(e.to_string()))?;

    // Get route GTFS ID and tag info
    let gtfs_route_id = tt.route_gtfs_id(idx).unwrap_or_default();
    let route_src = tt.transport_source(0).unwrap_or(0); // approximate
    let (full_id, source, _) = get_tag_and_id(tags, route_src, &gtfs_route_id);

    let color = match (detail.color, detail.text_color) {
        (Some(bg), Some(txt)) => Some(GqlColor {
            background: color_to_hex(bg),
            text: color_to_hex(txt),
        }),
        (Some(bg), None) => Some(GqlColor {
            background: color_to_hex(bg),
            text: "000000".to_string(),
        }),
        _ => None,
    };

    let agency = if !detail.agency_name.is_empty() {
        Some(GqlAgencyReference {
            id: ID(detail.agency_id.clone()),
            source: source.clone(),
            gtfs_id: detail.agency_id.clone(),
            name: detail.agency_name,
        })
    } else {
        None
    };

    Ok(GqlRoute {
        id: ID(full_id),
        source,
        gtfs_route_ids: vec![gtfs_route_id],
        short_name: if detail.short_name.is_empty() {
            None
        } else {
            Some(detail.short_name)
        },
        long_name: if detail.long_name.is_empty() {
            format!("Route {idx}")
        } else {
            detail.long_name
        },
        route_type: clasz_to_route_type(detail.clasz),
        color,
        timezone: String::new(), // Phase 2: from provider timezone
        agency,
        network: None,
    })
}

/// Haversine distance in meters between two lat/lon points.
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0; // Earth's radius in meters
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();

    let a = (d_lat / 2.0).sin().powi(2) + lat1_r.cos() * lat2_r.cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    r * c
}

// --- Phase 3: Conversion helpers ---

fn parse_datetime(s: &str) -> Result<i64, async_graphql::Error> {
    // Try ISO 8601 format
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Ok(dt.timestamp());
    }
    // Try "YYYY-MM-DDTHH:MM:SS"
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Ok(dt.and_utc().timestamp());
    }
    // Try Unix timestamp
    if let Ok(ts) = s.parse::<i64>() {
        return Ok(ts);
    }
    Err(async_graphql::Error::new(format!("invalid datetime: {s}")))
}

fn unix_to_iso(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_default()
}

fn build_route_ref(
    tt: &nigiri::Timetable,
    tags: Option<&TagLookup>,
    route_idx: u32,
) -> GqlRouteReference {
    let detail = tt.get_route_detail(route_idx).ok();
    let gtfs_id = tt.route_gtfs_id(route_idx).unwrap_or_default();
    let src = tt.transport_source(0).unwrap_or(0);
    let (full_id, source, _) = get_tag_and_id(tags, src, &gtfs_id);

    let color = detail.as_ref().and_then(|d| match (d.color, d.text_color) {
        (Some(bg), Some(txt)) => Some(GqlColor {
            background: color_to_hex(bg),
            text: color_to_hex(txt),
        }),
        (Some(bg), None) => Some(GqlColor {
            background: color_to_hex(bg),
            text: "000000".to_string(),
        }),
        _ => None,
    });

    GqlRouteReference {
        id: ID(full_id),
        source,
        gtfs_id,
        short_name: detail.as_ref().and_then(|d| {
            if d.short_name.is_empty() {
                None
            } else {
                Some(d.short_name.clone())
            }
        }),
        long_name: detail
            .as_ref()
            .map(|d| {
                if d.long_name.is_empty() {
                    format!("Route {route_idx}")
                } else {
                    d.long_name.clone()
                }
            })
            .unwrap_or_else(|| format!("Route {route_idx}")),
        route_type: detail
            .as_ref()
            .map(|d| clasz_to_route_type(d.clasz))
            .unwrap_or(GqlRouteType::Other),
        color,
    }
}

fn loc_info(tt: &nigiri::Timetable, loc_idx: u32) -> (String, GqlPosition) {
    if let Ok(d) = tt.get_location_detail(loc_idx) {
        (
            d.name,
            GqlPosition {
                lat: d.lat,
                lon: d.lon,
            },
        )
    } else {
        (String::new(), GqlPosition { lat: 0.0, lon: 0.0 })
    }
}

fn convert_journey(
    tt: &nigiri::Timetable,
    tags: &TagLookup,
    journey: transit_routing::Journey,
) -> Result<GqlJourney, async_graphql::Error> {
    let mut gql_legs = Vec::with_capacity(journey.legs.len());

    for leg in &journey.legs {
        match leg {
            transit_routing::Leg::Transit(tl) => {
                let (from_name, from_pos) = loc_info(tt, tl.from_location_idx);
                let (to_name, to_pos) = loc_info(tt, tl.to_location_idx);
                let from_detail = tt.get_location_detail(tl.from_location_idx).ok();
                let to_detail = tt.get_location_detail(tl.to_location_idx).ok();
                let from_id = from_detail.map(|d| d.id).unwrap_or_default();
                let to_id = to_detail.map(|d| d.id).unwrap_or_default();

                let dep_mam = tt
                    .event_mam(tl.transport_idx, tl.from_stop_idx, false)
                    .unwrap_or(0);
                let arr_mam = tt
                    .event_mam(tl.transport_idx, tl.to_stop_idx, true)
                    .unwrap_or(0);
                let dep_ts = tt.to_unixtime(tl.day_idx, dep_mam as u16);
                let arr_ts = tt.to_unixtime(tl.day_idx, arr_mam as u16);

                let intermediates: Vec<GqlIntermediateStop> = tl
                    .intermediate_stops
                    .iter()
                    .map(|is| {
                        let (name, pos) = loc_info(tt, is.location_idx);
                        GqlIntermediateStop {
                            name,
                            position: pos,
                            arrival_time: is
                                .arrival_mam
                                .map(|m| unix_to_iso(tt.to_unixtime(tl.day_idx, m as u16))),
                            departure_time: is
                                .departure_mam
                                .map(|m| unix_to_iso(tt.to_unixtime(tl.day_idx, m as u16))),
                        }
                    })
                    .collect();

                let trip_id = tt.transport_trip_id(tl.transport_idx).and_then(|tid| {
                    let src = tt.transport_source(tl.transport_idx)?;
                    let date = tt.day_to_date_str(tl.day_idx)?;
                    let mam = tt.transport_first_dep_mam(tl.transport_idx)?;
                    let time = format!("{}:{:02}", mam / 60, mam % 60);
                    tags.format_trip_id(SourceIdx(src), &date, &time, &tid)
                });

                gql_legs.push(GqlLeg::Transit(GqlTransitLeg {
                    from: from_pos,
                    from_name,
                    from_id,
                    to: to_pos,
                    to_name,
                    to_id,
                    departure_time: unix_to_iso(dep_ts),
                    arrival_time: unix_to_iso(arr_ts),
                    route: build_route_ref(tt, Some(tags), tl.route_idx),
                    headsign: tl.headsign.clone(),
                    intermediate_stops: intermediates,
                    trip_id,
                }));
            }
            transit_routing::Leg::Walk(wl) => {
                let (from_name, from_pos) = loc_info(tt, wl.from_location_idx);
                let (to_name, to_pos) = loc_info(tt, wl.to_location_idx);

                gql_legs.push(GqlLeg::Walk(GqlWalkLeg {
                    from: from_pos,
                    from_name,
                    to: to_pos,
                    to_name,
                    duration_seconds: (wl.duration_minutes * 60) as i32,
                }));
            }
        }
    }

    Ok(GqlJourney {
        departure_time: unix_to_iso(journey.departure_time),
        arrival_time: unix_to_iso(journey.arrival_time),
        duration_seconds: (journey.arrival_time - journey.departure_time) as i32,
        transfers: journey.transfers as i32,
        legs: gql_legs,
    })
}

fn convert_departure(
    tt: &nigiri::Timetable,
    tags: &TagLookup,
    dep: transit_routing::Departure,
) -> Result<GqlDeparture, async_graphql::Error> {
    let trip_id = tt.transport_trip_id(dep.transport_idx).and_then(|tid| {
        let src = tt.transport_source(dep.transport_idx)?;
        let date = tt.day_to_date_str(dep.day_idx)?;
        let mam = tt.transport_first_dep_mam(dep.transport_idx)?;
        let time = format!("{}:{:02}", mam / 60, mam % 60);
        tags.format_trip_id(SourceIdx(src), &date, &time, &tid)
    });

    Ok(GqlDeparture {
        aimed_departure_time: unix_to_iso(dep.scheduled_departure),
        aimed_arrival_time: unix_to_iso(dep.scheduled_arrival),
        estimated_departure_time: None,
        estimated_arrival_time: None,
        is_cancelled: Some(dep.is_cancelled),
        is_real_time: dep.is_real_time,
        delay: dep.delay_seconds,
        trip_id,
        route: build_route_ref(tt, Some(tags), dep.route_idx),
        headsign: dep.headsign,
        direction: 0,
        accessible: GqlWheelchairAccessible::NoInformation,
    })
}

fn convert_trip(
    tt: &nigiri::Timetable,
    tags: &TagLookup,
    detail: transit_routing::TripDetail,
) -> Result<GqlTrip, async_graphql::Error> {
    let trip_id = tt
        .transport_trip_id(detail.transport_idx)
        .and_then(|tid| {
            let src = tt.transport_source(detail.transport_idx)?;
            let date = tt.day_to_date_str(detail.day_idx)?;
            let mam = tt.transport_first_dep_mam(detail.transport_idx)?;
            let time = format!("{}:{:02}", mam / 60, mam % 60);
            tags.format_trip_id(SourceIdx(src), &date, &time, &tid)
        })
        .unwrap_or_else(|| format!("{}:{}", detail.transport_idx, detail.day_idx));

    let stop_times: Vec<GqlStopTime> = detail
        .stop_times
        .iter()
        .map(|st| {
            let (name, pos) = loc_info(tt, st.location_idx);
            let loc_detail = tt.get_location_detail(st.location_idx).ok();
            let stop_id = loc_detail.map(|d| d.id).unwrap_or_default();

            GqlStopTime {
                stop_sequence: st.stop_sequence as i32,
                stop_id,
                stop_name: name,
                position: pos,
                arrival_time: st.arrival_time.map(unix_to_iso),
                departure_time: st.departure_time.map(unix_to_iso),
            }
        })
        .collect();

    Ok(GqlTrip {
        id: ID(trip_id),
        route: build_route_ref(tt, Some(tags), detail.route_idx),
        headsign: detail.headsign,
        direction: 0,
        stop_times,
    })
}

pub type AppSchema = Schema<QueryRoot, EmptyMutation, EmptySubscription>;

pub fn build_schema(data: AppData) -> AppSchema {
    Schema::build(QueryRoot, EmptyMutation, EmptySubscription)
        .data(data)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use transit_core::Config;

    #[tokio::test]
    async fn health_query() {
        let data = AppData::empty(Config::default(), PathBuf::from("."));
        let schema = build_schema(data);
        let res = schema.execute("{ health }").await;
        assert_eq!(res.data.to_string(), r#"{health: true}"#);
    }

    #[tokio::test]
    async fn stop_query_no_timetable() {
        let data = AppData::empty(Config::default(), PathBuf::from("."));
        let schema = build_schema(data);
        let res = schema.execute(r#"{ stop(id: "123") { id name } }"#).await;
        assert!(res.errors.is_empty());
        assert_eq!(res.data.to_string(), r#"{stop: null}"#);
    }

    #[tokio::test]
    async fn stops_query_no_timetable() {
        let data = AppData::empty(Config::default(), PathBuf::from("."));
        let schema = build_schema(data);
        let res = schema
            .execute(r#"{ stops { items { id } pageInfo { totalItems } } }"#)
            .await;
        assert!(res.errors.is_empty());
    }
}
