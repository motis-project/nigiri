#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nigiri_timetable;
typedef struct nigiri_timetable nigiri_timetable_t;

struct nigiri_transport {
  uint32_t route_idx;
  uint16_t n_event_mams;
  int16_t* event_mams;
  char const* name;
  uint32_t name_len;
};
typedef struct nigiri_transport nigiri_transport_t;

static uint32_t const kTargetBits = 23U;
static uint32_t const kDurationBits = 8 * sizeof(uint32_t) - kTargetBits;
struct nigiri_footpath {
  unsigned int target_location_idx : kTargetBits;
  unsigned int duration : kDurationBits;
};
typedef struct nigiri_footpath nigiri_footpath_t;

struct nigiri_location {
  char const* id;
  uint32_t id_len;
  char const* name;
  uint32_t name_len;
  double lon;
  double lat;
  uint16_t transfer_time;
  nigiri_footpath_t* footpaths;
  uint32_t n_footpaths;
  uint32_t parent;
};
typedef struct nigiri_location nigiri_location_t;

struct nigiri_route_stop {
  unsigned int location_idx : 28;
  unsigned int in_allowed : 1;
  unsigned int out_allowed : 1;
  unsigned int in_allowed_wheelchair : 1;
  unsigned int out_allowed_wheelchair : 1;
};
typedef struct nigiri_route_stop nigiri_route_stop_t;

struct nigiri_route {
  uint16_t n_stops;
  nigiri_route_stop_t* stops;
  uint16_t clasz;
};
typedef struct nigiri_route nigiri_route_t;

struct nigiri_event_change {
  uint32_t transport_idx;
  uint16_t day_idx;
  uint16_t stop_idx;
  bool is_departure;
  bool stop_change;
  uint32_t stop_location_idx;  // ignore if UINT_MAX or stop_change == false
  bool stop_in_out_allowed;  // ignore if stop_change == false
  int16_t delay;  // ignore if stop_change == true
};
typedef struct nigiri_event_change nigiri_event_change_t;

struct nigiri_leg {
  bool is_footpath;
  uint32_t transport_idx;
  uint16_t day_idx;
  uint16_t from_stop_idx;
  uint32_t from_location_idx;
  uint16_t to_stop_idx;
  uint32_t to_location_idx;
  uint32_t duration;
};
typedef struct nigiri_leg nigiri_leg_t;

struct nigiri_journey {
  uint16_t n_legs;
  nigiri_leg_t* legs;
  int64_t start_time;
  int64_t dest_time;
};
typedef struct nigiri_journey nigiri_journey_t;

struct nigiri_pareto_set {
  uint16_t n_journeys;
  nigiri_journey_t* journeys;
};
typedef struct nigiri_pareto_set nigiri_pareto_set_t;

nigiri_timetable_t* nigiri_load(char const* path,
                                int64_t from_ts,
                                int64_t to_ts);
nigiri_timetable_t* nigiri_load_linking_stops(char const* path,
                                              int64_t from_ts,
                                              int64_t to_ts,
                                              unsigned link_stop_distance);

void nigiri_destroy(nigiri_timetable_t const* t);
int64_t nigiri_get_start_day_ts(nigiri_timetable_t const* t);
uint16_t nigiri_get_day_count(nigiri_timetable_t const* t);
uint32_t nigiri_get_transport_count(nigiri_timetable_t const* t);
nigiri_transport_t* nigiri_get_transport(nigiri_timetable_t const* t,
                                         uint32_t idx);
void nigiri_destroy_transport(nigiri_transport_t const* transport);
bool nigiri_is_transport_active(nigiri_timetable_t const* t,
                                uint32_t const transport_idx,
                                uint16_t day_idx);
uint32_t nigiri_get_route_count(nigiri_timetable_t const* t);
nigiri_route_t* nigiri_get_route(nigiri_timetable_t const* t, uint32_t idx);
void nigiri_destroy_route(nigiri_route_t const* route);
uint32_t nigiri_get_location_count(nigiri_timetable_t const* t);
nigiri_location_t* nigiri_get_location(nigiri_timetable_t const* t,
                                       uint32_t idx);
nigiri_location_t* nigiri_get_location_with_footpaths(
    nigiri_timetable_t const* t, uint32_t idx, bool incoming_footpaths);
void nigiri_destroy_location(nigiri_location_t const* location);

void nigiri_update_with_rt(nigiri_timetable_t const* t,
                           char const* gtfsrt_pb_path,
                           void (*callback)(nigiri_event_change_t,
                                            void* context),
                           void* context);

nigiri_pareto_set_t* nigiri_get_journeys(nigiri_timetable_t const* t,
                                         uint32_t start_location_idx,
                                         uint32_t destination_location_idx,
                                         int64_t time,
                                         bool backward_search);

void nigiri_destroy_journeys(nigiri_pareto_set_t const* journeys);

// --- Extended accessors for Rust/FFI consumers ---

// Schedule date range as Unix timestamps.
int64_t nigiri_get_external_interval_start(nigiri_timetable_t const* t);
int64_t nigiri_get_external_interval_end(nigiri_timetable_t const* t);

// Number of stops in a route.
uint16_t nigiri_get_route_stop_count(nigiri_timetable_t const* t,
                                     uint32_t route_idx);

// Transport name (without allocating the full nigiri_transport_t).
// Writes up to buf_len bytes into buf and returns the actual name length.
uint32_t nigiri_get_transport_name(nigiri_timetable_t const* t,
                                   uint32_t transport_idx,
                                   char* buf,
                                   uint32_t buf_len);

// Route index for a given transport.
uint32_t nigiri_get_transport_route(nigiri_timetable_t const* t,
                                    uint32_t transport_idx);

// Number of RT transports currently tracked (0 if no RT data).
uint32_t nigiri_get_rt_transport_count(nigiri_timetable_t const* t);

// Find location by string ID. Returns UINT32_MAX if not found.
uint32_t nigiri_find_location(nigiri_timetable_t const* t,
                              char const* id,
                              uint32_t id_len);

// Convert between Unix timestamp and day_idx + minutes after midnight.
uint16_t nigiri_to_day_idx(nigiri_timetable_t const* t, int64_t unix_ts);
int64_t nigiri_to_unixtime(nigiri_timetable_t const* t,
                           uint16_t day_idx,
                           uint16_t minutes_after_midnight);

// --- Phase 1: Detail accessors ---

// Number of data sources (datasets).
uint32_t nigiri_get_source_count(nigiri_timetable_t const* t);

// Location detail with type, wheelchair, parent info.
struct nigiri_location_detail {
  double lat;
  double lon;
  char const* name;
  uint32_t name_len;
  char const* id;
  uint32_t id_len;
  uint8_t location_type;     // 0=track(generated), 1=track, 2=station
  uint32_t parent_idx;       // parent station index (UINT32_MAX = none)
  uint32_t src_idx;          // source dataset index
  uint16_t transfer_time;    // transfer time in minutes
};
typedef struct nigiri_location_detail nigiri_location_detail_t;

// Fill out with location details. Returns false if idx is out of range.
bool nigiri_get_location_detail(nigiri_timetable_t const* t,
                                uint32_t location_idx,
                                nigiri_location_detail_t* out);

// Route detail with names, agency, colors.
struct nigiri_route_detail {
  char const* short_name;
  uint32_t short_name_len;
  char const* long_name;
  uint32_t long_name_len;
  char const* agency_name;
  uint32_t agency_name_len;
  char const* agency_id;
  uint32_t agency_id_len;
  uint8_t clasz;             // route class (clasz enum value)
  uint32_t color;            // 0xRRGGBB (0 = not set)
  uint32_t text_color;       // 0xRRGGBB (0 = not set)
};
typedef struct nigiri_route_detail nigiri_route_detail_t;

// Fill out with route details. Returns false if idx is out of range.
bool nigiri_get_route_detail(nigiri_timetable_t const* t,
                             uint32_t route_idx,
                             nigiri_route_detail_t* out);

// --- Phase 2: Tag lookup support ---

// Get the GTFS trip_id string for a transport.
// Returns false if transport_idx is out of range.
bool nigiri_get_transport_trip_id(nigiri_timetable_t const* t,
                                  uint32_t transport_idx,
                                  char const** id_out,
                                  uint32_t* id_len_out);

// Get the source index for a transport.
// Returns UINT32_MAX if out of range.
uint32_t nigiri_get_transport_source(nigiri_timetable_t const* t,
                                     uint32_t transport_idx);

// Get the first departure time in minutes after midnight for a transport.
int16_t nigiri_get_transport_first_dep_mam(nigiri_timetable_t const* t,
                                           uint32_t transport_idx);

// Get the GTFS route_id string for a route (via the first trip's route_id).
bool nigiri_get_route_gtfs_id(nigiri_timetable_t const* t,
                               uint32_t route_idx,
                               char const** id_out,
                               uint32_t* id_len_out);

// Convert day_idx to date string "YYYYMMDD". buf must be at least 9 bytes.
bool nigiri_day_to_date_str(nigiri_timetable_t const* t,
                            uint16_t day_idx,
                            char* buf_out);

// --- Phase 3: Routing support ---

// Get routes serving a location. Returns route count written.
// If routes_out is NULL, just returns the count.
uint32_t nigiri_get_location_routes(nigiri_timetable_t const* t,
                                    uint32_t location_idx,
                                    uint32_t* routes_out,
                                    uint32_t max_routes);

// Get the stop_idx of a location within a route.
// Returns UINT16_MAX if location is not on this route.
uint16_t nigiri_get_stop_idx_in_route(nigiri_timetable_t const* t,
                                      uint32_t route_idx,
                                      uint32_t location_idx);

// Get the departure/arrival time for a transport at a given stop.
// Returns minutes-after-midnight (MAM). Negative on error.
int16_t nigiri_get_event_mam(nigiri_timetable_t const* t,
                             uint32_t transport_idx,
                             uint16_t stop_idx,
                             bool is_arrival);

// Get transport range for a route.
// Sets *from_out and *to_out to the transport index range [from, to).
bool nigiri_get_route_transport_range(nigiri_timetable_t const* t,
                                      uint32_t route_idx,
                                      uint32_t* from_out,
                                      uint32_t* to_out);

// Get the number of intermediate stops for a transport leg.
// Returns stop times for all stops in the route of a transport.
// stop_locations_out: location indices, n_stops entries
// dep_mams_out: departure MAMs, n_stops entries (last is -1)
// arr_mams_out: arrival MAMs, n_stops entries (first is -1)
// Returns number of stops, or 0 on error.
uint16_t nigiri_get_transport_stop_times(
    nigiri_timetable_t const* t,
    uint32_t transport_idx,
    uint32_t* stop_locations_out,
    int16_t* dep_mams_out,
    int16_t* arr_mams_out,
    uint16_t max_stops);

// Get transport display name (headsign/direction).
uint32_t nigiri_get_transport_display_name(
    nigiri_timetable_t const* t,
    uint32_t transport_idx,
    char* buf,
    uint32_t buf_len);

#ifdef __cplusplus
}
#endif
