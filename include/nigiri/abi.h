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
  const char* name;
  uint32_t name_len;
};
typedef struct nigiri_transport nigiri_transport_t;

struct nigiri_location {
  const char* id;
  uint32_t id_len;
  const char* name;
  uint32_t name_len;
  double lon;
  double lat;
  uint16_t transfer_time;
  uint32_t parent;
};
typedef struct nigiri_location nigiri_location_t;

struct nigiri_route_stop {
  unsigned int location_idx : 30;
  unsigned int in_allowed : 1;
  unsigned int out_allowed : 1;
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
  uint32_t stop_idx;
  bool is_departure;
  int16_t delay;
  bool cancelled;
};
typedef struct nigiri_event_change nigiri_event_change_t;

nigiri_timetable_t* nigiri_load(const char* path,
                                int64_t from_ts,
                                int64_t to_ts);
void nigiri_destroy(const nigiri_timetable_t* t);
int64_t nigiri_get_start_day_ts(const nigiri_timetable_t* t);
uint16_t nigiri_get_day_count(const nigiri_timetable_t* t);
uint32_t nigiri_get_transport_count(const nigiri_timetable_t* t);
nigiri_transport_t* nigiri_get_transport(const nigiri_timetable_t* t,
                                         uint32_t idx);
void nigiri_destroy_transport(const nigiri_transport_t* transport);
bool nigiri_is_transport_active(const nigiri_timetable_t* t,
                                const uint32_t transport_idx,
                                uint16_t day_idx);
nigiri_route_t* nigiri_get_route(const nigiri_timetable_t* t, uint32_t idx);
void nigiri_destroy_route(const nigiri_route_t* route);
uint32_t nigiri_get_location_count(const nigiri_timetable_t* t);
nigiri_location_t* nigiri_get_location(const nigiri_timetable_t* t,
                                       uint32_t idx);
void nigiri_destroy_location(const nigiri_location_t* location);

void nigiri_update_with_rt(const nigiri_timetable_t* t,
                           const char* gtfsrt_pb_path,
                           void (*callback)(nigiri_event_change_t,
                                            void* context),
                           void* context);

#ifdef __cplusplus
}
#endif
