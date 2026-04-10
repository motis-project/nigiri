use transit_core::{TagLookup, TransitError, TransitResult};

use crate::types::*;

/// Transit routing orchestrator.
///
/// Wraps nigiri's RAPTOR engine and provides journey planning,
/// departure boards, and trip detail queries.
pub struct TransitRouter<'a> {
    pub tt: &'a nigiri::Timetable,
    pub tags: &'a TagLookup,
}

impl<'a> TransitRouter<'a> {
    pub fn new(tt: &'a nigiri::Timetable, tags: &'a TagLookup) -> Self {
        Self { tt, tags }
    }

    /// Plan journeys between two stops using RAPTOR.
    pub fn plan_journey(
        &self,
        from_idx: u32,
        to_idx: u32,
        time: i64,
        is_arrive_by: bool,
    ) -> TransitResult<Vec<Journey>> {
        let raw_journeys = self
            .tt
            .get_journeys(from_idx, to_idx, time, is_arrive_by)
            .map_err(|e| TransitError::Nigiri(format!("routing failed: {e}")))?;

        let journeys = raw_journeys
            .into_iter()
            .map(|j| self.convert_journey(j))
            .collect();

        Ok(journeys)
    }

    /// Get upcoming departures from a stop.
    pub fn stop_departures(
        &self,
        location_idx: u32,
        start_time: i64,
        limit: u32,
    ) -> TransitResult<Vec<Departure>> {
        let routes = self.tt.location_routes(location_idx);
        let mut departures = Vec::new();

        let day_count = self.tt.day_count();

        for &route_idx in &routes {
            let stop_idx = match self.tt.stop_idx_in_route(route_idx, location_idx) {
                Some(idx) => idx,
                None => continue,
            };

            let (transport_from, transport_to) = match self.tt.route_transport_range(route_idx) {
                Some(range) => range,
                None => continue,
            };

            // Get route clasz
            let clasz = self
                .tt
                .get_route_detail(route_idx)
                .map(|d| d.clasz)
                .unwrap_or(9);

            for transport_idx in transport_from..transport_to {
                let dep_mam = match self.tt.event_mam(transport_idx, stop_idx, false) {
                    Some(mam) => mam,
                    None => continue,
                };

                // Check each day in the timetable
                for day in 0..day_count {
                    if !self.tt.is_transport_active(transport_idx, day) {
                        continue;
                    }

                    let dep_ts = self.tt.to_unixtime(day, dep_mam as u16);
                    if dep_ts < start_time {
                        continue;
                    }

                    // Get arrival at next stop (for display)
                    let n_stops = self.tt.route_stop_count(route_idx);
                    let arr_ts = if stop_idx + 1 < n_stops {
                        let arr_mam = self.tt.event_mam(transport_idx, stop_idx + 1, true);
                        arr_mam.map(|m| self.tt.to_unixtime(day, m as u16))
                    } else {
                        None
                    };

                    let headsign = self.tt.transport_display_name(transport_idx);

                    departures.push(Departure {
                        scheduled_departure: dep_ts,
                        scheduled_arrival: arr_ts.unwrap_or(dep_ts),
                        transport_idx,
                        route_idx,
                        day_idx: day,
                        stop_idx,
                        headsign,
                        clasz,
                        is_real_time: false,
                        delay_seconds: None,
                        is_cancelled: false,
                    });
                }
            }
        }

        // Sort by departure time and limit
        departures.sort_by_key(|d| d.scheduled_departure);
        departures.truncate(limit as usize);

        Ok(departures)
    }

    /// Get full trip detail (all stop times).
    pub fn trip_detail(&self, transport_idx: u32, day_idx: u16) -> TransitResult<TripDetail> {
        let route_idx = self.tt.transport_route(transport_idx);
        let clasz = self
            .tt
            .get_route_detail(route_idx)
            .map(|d| d.clasz)
            .unwrap_or(9);
        let headsign = self.tt.transport_display_name(transport_idx);

        let stop_times = self.tt.transport_stop_times(transport_idx).ok_or_else(|| {
            TransitError::NotFound(format!("transport {transport_idx} stop times"))
        })?;

        let trip_stop_times: Vec<TripStopTime> = stop_times
            .iter()
            .enumerate()
            .map(|(i, st)| {
                let arr_time = st
                    .arrival_mam
                    .map(|m| self.tt.to_unixtime(day_idx, m as u16));
                let dep_time = st
                    .departure_mam
                    .map(|m| self.tt.to_unixtime(day_idx, m as u16));
                TripStopTime {
                    location_idx: st.location_idx,
                    arrival_time: arr_time,
                    departure_time: dep_time,
                    stop_sequence: i as u16,
                }
            })
            .collect();

        Ok(TripDetail {
            transport_idx,
            day_idx,
            route_idx,
            headsign,
            clasz,
            stop_times: trip_stop_times,
        })
    }

    /// Convert a raw nigiri journey to our domain type.
    fn convert_journey(&self, raw: nigiri::Journey) -> Journey {
        let mut legs = Vec::with_capacity(raw.legs.len());
        let mut transfers = 0u16;

        for (i, leg) in raw.legs.iter().enumerate() {
            if leg.is_footpath {
                legs.push(Leg::Walk(WalkLeg {
                    from_location_idx: leg.from_location_idx,
                    to_location_idx: leg.to_location_idx,
                    duration_minutes: leg.duration,
                }));
            } else {
                if i > 0 {
                    transfers += 1;
                }
                let route_idx = self.tt.transport_route(leg.transport_idx);
                let headsign = self.tt.transport_display_name(leg.transport_idx);

                // Build intermediate stops
                let intermediates =
                    self.build_intermediates(leg.transport_idx, leg.from_stop_idx, leg.to_stop_idx);

                legs.push(Leg::Transit(TransitLeg {
                    from_location_idx: leg.from_location_idx,
                    to_location_idx: leg.to_location_idx,
                    departure_time: raw.start_time, // approximate per-leg
                    arrival_time: raw.dest_time,
                    transport_idx: leg.transport_idx,
                    route_idx,
                    day_idx: leg.day_idx,
                    from_stop_idx: leg.from_stop_idx,
                    to_stop_idx: leg.to_stop_idx,
                    headsign,
                    intermediate_stops: intermediates,
                }));
            }
        }

        Journey {
            departure_time: raw.start_time,
            arrival_time: raw.dest_time,
            transfers,
            legs,
        }
    }

    fn build_intermediates(
        &self,
        transport_idx: u32,
        from_stop: u16,
        to_stop: u16,
    ) -> Vec<IntermediateStop> {
        if to_stop <= from_stop + 1 {
            return vec![];
        }
        let stop_times = match self.tt.transport_stop_times(transport_idx) {
            Some(st) => st,
            None => return vec![],
        };

        ((from_stop + 1) as usize..to_stop as usize)
            .filter_map(|i| stop_times.get(i))
            .map(|st| IntermediateStop {
                location_idx: st.location_idx,
                arrival_mam: st.arrival_mam,
                departure_mam: st.departure_mam,
            })
            .collect()
    }
}
