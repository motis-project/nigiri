//! Safe Rust wrapper for the nigiri transit routing C++ library.
//!
//! Provides safe, idiomatic Rust types around the C FFI layer in `nigiri-sys`.
//! All C++ memory management is handled automatically via `Drop` implementations.
//!
//! # Example
//! ```no_run
//! use nigiri::Timetable;
//!
//! let tt = Timetable::load("/path/to/gtfs", 1_700_000_000, 1_700_100_000).unwrap();
//! println!("Locations: {}", tt.location_count());
//! println!("Routes: {}", tt.route_count());
//!
//! let journeys = tt.get_journeys(0, 100, 1_700_050_000, false).unwrap();
//! for j in &journeys {
//!     println!("depart {} -> arrive {}", j.start_time, j.dest_time);
//! }
//! ```

/// A loaded timetable with optional real-time data.
///
/// Wraps the opaque `nigiri_timetable_t` pointer. The underlying C++ timetable
/// is memory-mapped (zero-copy) so this handle is cheap — just two shared pointers.
pub struct Timetable {
    raw: *mut nigiri_sys::nigiri_timetable_t,
}

// SAFETY: The C++ timetable is immutable after load and the rt_timetable
// mutations go through the C ABI which handles its own synchronization.
unsafe impl Send for Timetable {}
// SAFETY: All read-only accessors on the C++ timetable are thread-safe.
// The mutable RT operations go through the C ABI with its own synchronization.
unsafe impl Sync for Timetable {}

impl Timetable {
    /// Load a GTFS/HRD timetable from a directory or zip file.
    ///
    /// `from_ts` and `to_ts` are Unix timestamps bounding the date range.
    pub fn load(path: &str, from_ts: i64, to_ts: i64) -> Result<Self, Error> {
        let c_path = std::ffi::CString::new(path).map_err(|_| Error::InvalidPath)?;
        let raw = unsafe { nigiri_sys::nigiri_load(c_path.as_ptr(), from_ts, to_ts) };
        if raw.is_null() {
            return Err(Error::LoadFailed);
        }
        Ok(Self { raw })
    }

    /// Load with a custom stop-linking distance (meters).
    pub fn load_linking_stops(
        path: &str,
        from_ts: i64,
        to_ts: i64,
        link_stop_distance: u32,
    ) -> Result<Self, Error> {
        let c_path = std::ffi::CString::new(path).map_err(|_| Error::InvalidPath)?;
        let raw = unsafe {
            nigiri_sys::nigiri_load_linking_stops(
                c_path.as_ptr(),
                from_ts,
                to_ts,
                link_stop_distance,
            )
        };
        if raw.is_null() {
            return Err(Error::LoadFailed);
        }
        Ok(Self { raw })
    }

    /// Unix timestamp of the first internal day.
    pub fn start_day_ts(&self) -> i64 {
        unsafe { nigiri_sys::nigiri_get_start_day_ts(self.raw) }
    }

    /// Number of days in the internal interval.
    pub fn day_count(&self) -> u16 {
        unsafe { nigiri_sys::nigiri_get_day_count(self.raw) }
    }

    /// Total number of transports (trips).
    pub fn transport_count(&self) -> u32 {
        unsafe { nigiri_sys::nigiri_get_transport_count(self.raw) }
    }

    /// Total number of routes.
    pub fn route_count(&self) -> u32 {
        unsafe { nigiri_sys::nigiri_get_route_count(self.raw) }
    }

    /// Total number of locations (stops/stations).
    pub fn location_count(&self) -> u32 {
        unsafe { nigiri_sys::nigiri_get_location_count(self.raw) }
    }

    /// Get transport metadata by index.
    pub fn get_transport(&self, idx: u32) -> Result<Transport, Error> {
        let raw = unsafe { nigiri_sys::nigiri_get_transport(self.raw, idx) };
        if raw.is_null() {
            return Err(Error::IndexOutOfBounds);
        }
        Ok(Transport { raw })
    }

    /// Check whether a transport is active on a given day.
    pub fn is_transport_active(&self, transport_idx: u32, day_idx: u16) -> bool {
        unsafe { nigiri_sys::nigiri_is_transport_active(self.raw, transport_idx, day_idx) }
    }

    /// Get route metadata by index.
    pub fn get_route(&self, idx: u32) -> Result<Route, Error> {
        let raw = unsafe { nigiri_sys::nigiri_get_route(self.raw, idx) };
        if raw.is_null() {
            return Err(Error::IndexOutOfBounds);
        }
        Ok(Route { raw })
    }

    /// Get location (stop/station) by index.
    pub fn get_location(&self, idx: u32) -> Result<Location, Error> {
        let raw = unsafe { nigiri_sys::nigiri_get_location(self.raw, idx) };
        if raw.is_null() {
            return Err(Error::IndexOutOfBounds);
        }
        Ok(Location { raw })
    }

    /// Get location with footpath data.
    pub fn get_location_with_footpaths(&self, idx: u32, incoming: bool) -> Result<Location, Error> {
        let raw =
            unsafe { nigiri_sys::nigiri_get_location_with_footpaths(self.raw, idx, incoming) };
        if raw.is_null() {
            return Err(Error::IndexOutOfBounds);
        }
        Ok(Location { raw })
    }

    /// Run a routing query between two locations.
    ///
    /// `time` is a Unix timestamp. If `backward` is true, searches backward
    /// from `time` (latest arrival); otherwise forward (earliest departure).
    pub fn get_journeys(
        &self,
        start_location_idx: u32,
        dest_location_idx: u32,
        time: i64,
        backward: bool,
    ) -> Result<Vec<Journey>, Error> {
        let raw = unsafe {
            nigiri_sys::nigiri_get_journeys(
                self.raw,
                start_location_idx,
                dest_location_idx,
                time,
                backward,
            )
        };
        if raw.is_null() {
            return Err(Error::QueryFailed);
        }

        let pareto = unsafe { &*raw };
        let mut journeys = Vec::with_capacity(pareto.n_journeys as usize);

        for i in 0..pareto.n_journeys as usize {
            let j = unsafe { &*pareto.journeys.add(i) };
            let mut legs = Vec::with_capacity(j.n_legs as usize);

            for k in 0..j.n_legs as usize {
                let l = unsafe { &*j.legs.add(k) };
                legs.push(Leg {
                    is_footpath: l.is_footpath,
                    transport_idx: l.transport_idx,
                    day_idx: l.day_idx,
                    from_stop_idx: l.from_stop_idx,
                    from_location_idx: l.from_location_idx,
                    to_stop_idx: l.to_stop_idx,
                    to_location_idx: l.to_location_idx,
                    duration: l.duration,
                });
            }

            journeys.push(Journey {
                legs,
                start_time: j.start_time,
                dest_time: j.dest_time,
            });
        }

        unsafe { nigiri_sys::nigiri_destroy_journeys(raw) };
        Ok(journeys)
    }

    /// Apply a GTFS-RT protobuf update from a file.
    ///
    /// The callback receives each event change as it is applied.
    pub fn update_with_rt<F>(&self, gtfsrt_pb_path: &str, mut callback: F)
    where
        F: FnMut(EventChange),
    {
        let c_path = std::ffi::CString::new(gtfsrt_pb_path).expect("path contains null byte");

        unsafe extern "C" fn trampoline<F: FnMut(EventChange)>(
            change: nigiri_sys::nigiri_event_change_t,
            context: *mut std::ffi::c_void,
        ) {
            let cb = unsafe { &mut *(context as *mut F) };
            cb(EventChange::from_raw(&change));
        }

        let context = &mut callback as *mut F as *mut std::ffi::c_void;
        unsafe {
            nigiri_sys::nigiri_update_with_rt(
                self.raw,
                c_path.as_ptr(),
                Some(trampoline::<F>),
                context,
            );
        }
    }

    /// Get the raw pointer (for advanced interop).
    pub fn as_raw(&self) -> *mut nigiri_sys::nigiri_timetable_t {
        self.raw
    }

    // --- Extended accessors ---

    /// Start of the schedule date range (Unix timestamp).
    pub fn external_interval_start(&self) -> i64 {
        unsafe { nigiri_sys::nigiri_get_external_interval_start(self.raw) }
    }

    /// End of the schedule date range (Unix timestamp).
    pub fn external_interval_end(&self) -> i64 {
        unsafe { nigiri_sys::nigiri_get_external_interval_end(self.raw) }
    }

    /// Number of stops in a given route.
    pub fn route_stop_count(&self, route_idx: u32) -> u16 {
        unsafe { nigiri_sys::nigiri_get_route_stop_count(self.raw, route_idx) }
    }

    /// Get a transport's name without allocating a full Transport struct.
    pub fn transport_name(&self, transport_idx: u32) -> String {
        // First call to get length
        let len = unsafe {
            nigiri_sys::nigiri_get_transport_name(self.raw, transport_idx, std::ptr::null_mut(), 0)
        };
        if len == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; len as usize];
        unsafe {
            nigiri_sys::nigiri_get_transport_name(
                self.raw,
                transport_idx,
                buf.as_mut_ptr() as *mut i8,
                len,
            );
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Get the route index for a given transport.
    pub fn transport_route(&self, transport_idx: u32) -> u32 {
        unsafe { nigiri_sys::nigiri_get_transport_route(self.raw, transport_idx) }
    }

    /// Number of real-time transports currently tracked.
    pub fn rt_transport_count(&self) -> u32 {
        unsafe { nigiri_sys::nigiri_get_rt_transport_count(self.raw) }
    }

    /// Find a location by its string ID. Returns `None` if not found.
    pub fn find_location(&self, id: &str) -> Option<u32> {
        let result = unsafe {
            nigiri_sys::nigiri_find_location(self.raw, id.as_ptr() as *const i8, id.len() as u32)
        };
        if result == u32::MAX {
            None
        } else {
            Some(result)
        }
    }

    /// Convert a Unix timestamp to a day index within the timetable.
    pub fn to_day_idx(&self, unix_ts: i64) -> u16 {
        unsafe { nigiri_sys::nigiri_to_day_idx(self.raw, unix_ts) }
    }

    /// Convert a day index and minutes-after-midnight to a Unix timestamp.
    pub fn to_unixtime(&self, day_idx: u16, minutes_after_midnight: u16) -> i64 {
        unsafe { nigiri_sys::nigiri_to_unixtime(self.raw, day_idx, minutes_after_midnight) }
    }

    // --- Phase 1: Detail accessors ---

    /// Number of data sources (datasets) in the timetable.
    pub fn source_count(&self) -> u32 {
        unsafe { nigiri_sys::nigiri_get_source_count(self.raw) }
    }

    /// Get detailed location metadata (type, wheelchair, parent, source).
    pub fn get_location_detail(&self, idx: u32) -> Result<LocationDetail, Error> {
        let mut detail = std::mem::MaybeUninit::<nigiri_sys::nigiri_location_detail_t>::zeroed();
        let ok =
            unsafe { nigiri_sys::nigiri_get_location_detail(self.raw, idx, detail.as_mut_ptr()) };
        if !ok {
            return Err(Error::IndexOutOfBounds);
        }
        let d = unsafe { detail.assume_init() };
        Ok(LocationDetail {
            lat: d.lat,
            lon: d.lon,
            name: unsafe { str_from_ptr(d.name, d.name_len) }.to_string(),
            id: unsafe { str_from_ptr(d.id, d.id_len) }.to_string(),
            location_type: d.location_type,
            parent_idx: if d.parent_idx == u32::MAX {
                None
            } else {
                Some(d.parent_idx)
            },
            src_idx: d.src_idx,
            transfer_time: d.transfer_time,
        })
    }

    /// Get detailed route metadata (names, agency, colors).
    pub fn get_route_detail(&self, idx: u32) -> Result<RouteDetail, Error> {
        let mut detail = std::mem::MaybeUninit::<nigiri_sys::nigiri_route_detail_t>::zeroed();
        let ok = unsafe { nigiri_sys::nigiri_get_route_detail(self.raw, idx, detail.as_mut_ptr()) };
        if !ok {
            return Err(Error::IndexOutOfBounds);
        }
        let d = unsafe { detail.assume_init() };
        Ok(RouteDetail {
            short_name: unsafe { str_from_ptr(d.short_name, d.short_name_len) }.to_string(),
            long_name: unsafe { str_from_ptr(d.long_name, d.long_name_len) }.to_string(),
            agency_name: unsafe { str_from_ptr(d.agency_name, d.agency_name_len) }.to_string(),
            agency_id: unsafe { str_from_ptr(d.agency_id, d.agency_id_len) }.to_string(),
            clasz: d.clasz,
            color: if d.color == 0 { None } else { Some(d.color) },
            text_color: if d.text_color == 0 {
                None
            } else {
                Some(d.text_color)
            },
        })
    }

    // --- Phase 2: Tag lookup support ---

    /// Get the GTFS trip_id string for a transport.
    pub fn transport_trip_id(&self, transport_idx: u32) -> Option<String> {
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let mut len: u32 = 0;
        let ok = unsafe {
            nigiri_sys::nigiri_get_transport_trip_id(self.raw, transport_idx, &mut ptr, &mut len)
        };
        if !ok {
            return None;
        }
        Some(unsafe { str_from_ptr(ptr, len) }.to_string())
    }

    /// Get the source index for a transport.
    pub fn transport_source(&self, transport_idx: u32) -> Option<u32> {
        let result = unsafe { nigiri_sys::nigiri_get_transport_source(self.raw, transport_idx) };
        if result == u32::MAX {
            None
        } else {
            Some(result)
        }
    }

    /// Get the first departure MAM (minutes after midnight) for a transport.
    pub fn transport_first_dep_mam(&self, transport_idx: u32) -> Option<i16> {
        let result =
            unsafe { nigiri_sys::nigiri_get_transport_first_dep_mam(self.raw, transport_idx) };
        if result < 0 { None } else { Some(result) }
    }

    /// Get the GTFS route_id string for a route.
    pub fn route_gtfs_id(&self, route_idx: u32) -> Option<String> {
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let mut len: u32 = 0;
        let ok = unsafe {
            nigiri_sys::nigiri_get_route_gtfs_id(self.raw, route_idx, &mut ptr, &mut len)
        };
        if !ok {
            return None;
        }
        Some(unsafe { str_from_ptr(ptr, len) }.to_string())
    }

    /// Convert a day index to a date string "YYYYMMDD".
    pub fn day_to_date_str(&self, day_idx: u16) -> Option<String> {
        let mut buf = [0u8; 9];
        let ok = unsafe {
            nigiri_sys::nigiri_day_to_date_str(self.raw, day_idx, buf.as_mut_ptr() as *mut i8)
        };
        if !ok {
            return None;
        }
        Some(String::from_utf8_lossy(&buf[..8]).to_string())
    }

    // --- Phase 3: Routing support ---

    /// Get routes serving a location.
    pub fn location_routes(&self, location_idx: u32) -> Vec<u32> {
        let count = unsafe {
            nigiri_sys::nigiri_get_location_routes(self.raw, location_idx, std::ptr::null_mut(), 0)
        };
        if count == 0 {
            return vec![];
        }
        let mut routes = vec![0u32; count as usize];
        unsafe {
            nigiri_sys::nigiri_get_location_routes(
                self.raw,
                location_idx,
                routes.as_mut_ptr(),
                count,
            );
        }
        routes
    }

    /// Find the stop_idx of a location within a route.
    pub fn stop_idx_in_route(&self, route_idx: u32, location_idx: u32) -> Option<u16> {
        let idx =
            unsafe { nigiri_sys::nigiri_get_stop_idx_in_route(self.raw, route_idx, location_idx) };
        if idx == u16::MAX { None } else { Some(idx) }
    }

    /// Get departure/arrival MAM for a transport at a stop.
    pub fn event_mam(&self, transport_idx: u32, stop_idx: u16, is_arrival: bool) -> Option<i16> {
        let mam = unsafe {
            nigiri_sys::nigiri_get_event_mam(self.raw, transport_idx, stop_idx, is_arrival)
        };
        if mam < 0 { None } else { Some(mam) }
    }

    /// Get the transport index range [from, to) for a route.
    pub fn route_transport_range(&self, route_idx: u32) -> Option<(u32, u32)> {
        let mut from = 0u32;
        let mut to = 0u32;
        let ok = unsafe {
            nigiri_sys::nigiri_get_route_transport_range(self.raw, route_idx, &mut from, &mut to)
        };
        if ok { Some((from, to)) } else { None }
    }

    /// Get all stop times for a transport (locations + dep/arr MAMs).
    pub fn transport_stop_times(&self, transport_idx: u32) -> Option<Vec<StopTime>> {
        // First get the route to know the stop count
        let route_idx = self.transport_route(transport_idx);
        let n_stops = self.route_stop_count(route_idx);
        if n_stops == 0 {
            return None;
        }

        let mut locations = vec![0u32; n_stops as usize];
        let mut dep_mams = vec![-1i16; n_stops as usize];
        let mut arr_mams = vec![-1i16; n_stops as usize];

        let actual = unsafe {
            nigiri_sys::nigiri_get_transport_stop_times(
                self.raw,
                transport_idx,
                locations.as_mut_ptr(),
                dep_mams.as_mut_ptr(),
                arr_mams.as_mut_ptr(),
                n_stops,
            )
        };
        if actual == 0 {
            return None;
        }

        let mut stop_times = Vec::with_capacity(actual as usize);
        for i in 0..actual as usize {
            stop_times.push(StopTime {
                location_idx: locations[i],
                departure_mam: if dep_mams[i] >= 0 {
                    Some(dep_mams[i])
                } else {
                    None
                },
                arrival_mam: if arr_mams[i] >= 0 {
                    Some(arr_mams[i])
                } else {
                    None
                },
            });
        }
        Some(stop_times)
    }

    /// Get the display name (headsign) for a transport.
    pub fn transport_display_name(&self, transport_idx: u32) -> String {
        let len = unsafe {
            nigiri_sys::nigiri_get_transport_display_name(
                self.raw,
                transport_idx,
                std::ptr::null_mut(),
                0,
            )
        };
        if len == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; len as usize];
        unsafe {
            nigiri_sys::nigiri_get_transport_display_name(
                self.raw,
                transport_idx,
                buf.as_mut_ptr() as *mut i8,
                len,
            );
        }
        String::from_utf8_lossy(&buf).into_owned()
    }
}

/// Stop time data for a single stop in a transport's schedule.
#[derive(Debug, Clone)]
pub struct StopTime {
    pub location_idx: u32,
    pub departure_mam: Option<i16>,
    pub arrival_mam: Option<i16>,
}

impl Drop for Timetable {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { nigiri_sys::nigiri_destroy(self.raw) };
        }
    }
}

// --- Transport ---

/// Transport (trip) metadata.
pub struct Transport {
    raw: *mut nigiri_sys::nigiri_transport_t,
}

impl Transport {
    pub fn route_idx(&self) -> u32 {
        unsafe { (*self.raw).route_idx }
    }

    /// Departure/arrival times in minutes-after-midnight.
    /// Layout: [stop0_dep, stop1_arr, stop1_dep, stop2_arr, ...].
    pub fn event_mams(&self) -> &[i16] {
        unsafe {
            let t = &*self.raw;
            if t.event_mams.is_null() || t.n_event_mams == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(t.event_mams, t.n_event_mams as usize)
            }
        }
    }

    pub fn name(&self) -> &str {
        unsafe {
            let t = &*self.raw;
            if t.name.is_null() || t.name_len == 0 {
                ""
            } else {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    t.name as *const u8,
                    t.name_len as usize,
                ))
            }
        }
    }
}

impl Drop for Transport {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { nigiri_sys::nigiri_destroy_transport(self.raw) };
        }
    }
}

// --- Route ---

/// Route metadata.
pub struct Route {
    raw: *mut nigiri_sys::nigiri_route_t,
}

/// A stop within a route.
#[derive(Debug, Clone, Copy)]
pub struct RouteStop {
    pub location_idx: u32,
    pub in_allowed: bool,
    pub out_allowed: bool,
    pub in_allowed_wheelchair: bool,
    pub out_allowed_wheelchair: bool,
}

impl Route {
    pub fn clasz(&self) -> u16 {
        unsafe { (*self.raw).clasz }
    }

    pub fn stops(&self) -> Vec<RouteStop> {
        unsafe {
            let r = &*self.raw;
            let mut stops = Vec::with_capacity(r.n_stops as usize);
            for i in 0..r.n_stops as usize {
                let s = &*r.stops.add(i);
                stops.push(RouteStop {
                    location_idx: s.location_idx(),
                    in_allowed: s.in_allowed() != 0,
                    out_allowed: s.out_allowed() != 0,
                    in_allowed_wheelchair: s.in_allowed_wheelchair() != 0,
                    out_allowed_wheelchair: s.out_allowed_wheelchair() != 0,
                });
            }
            stops
        }
    }
}

impl Drop for Route {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { nigiri_sys::nigiri_destroy_route(self.raw) };
        }
    }
}

// --- Location ---

/// A stop or station.
pub struct Location {
    raw: *mut nigiri_sys::nigiri_location_t,
}

/// A walking connection to a nearby location.
#[derive(Debug, Clone, Copy)]
pub struct Footpath {
    pub target_location_idx: u32,
    pub duration_minutes: u32,
}

impl Location {
    pub fn id(&self) -> &str {
        unsafe {
            let loc = &*self.raw;
            if loc.id.is_null() || loc.id_len == 0 {
                ""
            } else {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    loc.id as *const u8,
                    loc.id_len as usize,
                ))
            }
        }
    }

    pub fn name(&self) -> &str {
        unsafe {
            let loc = &*self.raw;
            if loc.name.is_null() || loc.name_len == 0 {
                ""
            } else {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    loc.name as *const u8,
                    loc.name_len as usize,
                ))
            }
        }
    }

    pub fn lat(&self) -> f64 {
        unsafe { (*self.raw).lat }
    }

    pub fn lon(&self) -> f64 {
        unsafe { (*self.raw).lon }
    }

    pub fn transfer_time(&self) -> u16 {
        unsafe { (*self.raw).transfer_time }
    }

    pub fn parent(&self) -> Option<u32> {
        let p = unsafe { (*self.raw).parent };
        if p == 0 { None } else { Some(p) }
    }

    pub fn footpaths(&self) -> Vec<Footpath> {
        unsafe {
            let loc = &*self.raw;
            let mut fps = Vec::with_capacity(loc.n_footpaths as usize);
            for i in 0..loc.n_footpaths as usize {
                let fp = &*loc.footpaths.add(i);
                fps.push(Footpath {
                    target_location_idx: fp.target_location_idx(),
                    duration_minutes: fp.duration(),
                });
            }
            fps
        }
    }
}

impl Drop for Location {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { nigiri_sys::nigiri_destroy_location(self.raw) };
        }
    }
}

// --- Journey / Leg ---

/// A complete journey (connection) from origin to destination.
#[derive(Debug, Clone)]
pub struct Journey {
    pub legs: Vec<Leg>,
    /// Unix timestamp of departure.
    pub start_time: i64,
    /// Unix timestamp of arrival.
    pub dest_time: i64,
}

/// A single leg of a journey — either a transit ride or a footpath.
#[derive(Debug, Clone, Copy)]
pub struct Leg {
    pub is_footpath: bool,
    pub transport_idx: u32,
    pub day_idx: u16,
    pub from_stop_idx: u16,
    pub from_location_idx: u32,
    pub to_stop_idx: u16,
    pub to_location_idx: u32,
    /// Duration in minutes.
    pub duration: u32,
}

// --- Event Change ---

/// A real-time event change from a GTFS-RT update.
#[derive(Debug, Clone, Copy)]
pub struct EventChange {
    pub transport_idx: u32,
    pub day_idx: u16,
    pub stop_idx: u16,
    pub is_departure: bool,
    pub stop_change: bool,
    pub stop_location_idx: u32,
    pub stop_in_out_allowed: bool,
    pub delay: i16,
}

impl EventChange {
    fn from_raw(c: &nigiri_sys::nigiri_event_change_t) -> Self {
        Self {
            transport_idx: c.transport_idx,
            day_idx: c.day_idx,
            stop_idx: c.stop_idx,
            is_departure: c.is_departure,
            stop_change: c.stop_change,
            stop_location_idx: c.stop_location_idx,
            stop_in_out_allowed: c.stop_in_out_allowed,
            delay: c.delay,
        }
    }
}

// --- Error ---

#[derive(Debug, Clone)]
pub enum Error {
    InvalidPath,
    LoadFailed,
    IndexOutOfBounds,
    QueryFailed,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidPath => write!(f, "invalid path (contains null byte)"),
            Error::LoadFailed => write!(f, "failed to load timetable"),
            Error::IndexOutOfBounds => write!(f, "index out of bounds"),
            Error::QueryFailed => write!(f, "routing query failed"),
        }
    }
}

impl std::error::Error for Error {}

// --- Detail types (Phase 1) ---

/// Detailed location metadata including type and accessibility.
#[derive(Debug, Clone)]
pub struct LocationDetail {
    pub lat: f64,
    pub lon: f64,
    pub name: String,
    pub id: String,
    /// 0=generated track, 1=track, 2=station
    pub location_type: u8,
    /// Parent station index, if any.
    pub parent_idx: Option<u32>,
    /// Source dataset index.
    pub src_idx: u32,
    /// Transfer time in minutes.
    pub transfer_time: u16,
}

/// Detailed route metadata including names, agency, and colors.
#[derive(Debug, Clone)]
pub struct RouteDetail {
    pub short_name: String,
    pub long_name: String,
    pub agency_name: String,
    pub agency_id: String,
    /// Route class (GTFS route_type mapped to clasz enum).
    pub clasz: u8,
    /// Route color as 0xRRGGBB, None if not set.
    pub color: Option<u32>,
    /// Text color as 0xRRGGBB, None if not set.
    pub text_color: Option<u32>,
}

/// Helper to convert a C string pointer + length to a &str.
///
/// # Safety
/// `ptr` must be valid for `len` bytes or null.
unsafe fn str_from_ptr<'a>(ptr: *const std::ffi::c_char, len: u32) -> &'a str {
    if ptr.is_null() || len == 0 {
        ""
    } else {
        unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                ptr as *const u8,
                len as usize,
            ))
        }
    }
}
