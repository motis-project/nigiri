#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nigiri_timetable;
typedef struct nigiri_timetable nigiri_timetable_t;

struct nigiri_transport {
    //bitfield_idx_t bitfield_idx_;
    uint32_t route_idx;
    uint32_t n_event_mams;
    int16_t *event_mams;
    const char* name;
};
typedef struct nigiri_transport nigiri_transport_t;

struct nigiri_stop {
    char *id;
    char *name;
    double lon;
    double lat;
    uint32_t transfer_time;
    int32_t parent;
};
typedef struct nigiri_stop nigiri_stop_t;

struct nigiri_route {
    uint32_t n_stops;
    uint32_t *stops;
    uint32_t clasz;
};
typedef struct nigiri_route nigiri_route_t;


nigiri_timetable_t *nigiri_load(const char* path, int64_t from_ts, int64_t to_ts);
void nigiri_destroy(const nigiri_timetable_t *t);
uint32_t nigiri_get_transport_count(const nigiri_timetable_t *t);
nigiri_transport_t *nigiri_get_transport(const nigiri_timetable_t *t, uint32_t idx);
void nigiri_destroy_transport(const nigiri_transport_t *transport);
nigiri_route_t *nigiri_get_route(const nigiri_timetable_t *t, uint32_t idx);
void nigiri_destroy_route(const nigiri_route_t *route);
uint32_t nigiri_get_stop_count(const nigiri_timetable_t *t);
nigiri_stop_t *nigiri_get_stop(const nigiri_timetable_t *t, uint32_t idx);
void nigiri_destroy_stop(const nigiri_stop_t *stop);

void update_with_rt(nigiri_timetable_t *t);

#ifdef __cplusplus
}
#endif
