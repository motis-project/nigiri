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

#ifdef __cplusplus
}
#endif
