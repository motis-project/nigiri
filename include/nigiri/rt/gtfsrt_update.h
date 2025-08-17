#pragma once

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "date/date.h"

#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

struct statistics {
  friend std::ostream& operator<<(std::ostream& out, statistics const& s);

  bool parser_error_{false};
  bool no_header_{false};
  int total_entities_{0};
  int total_entities_success_{0};
  int total_entities_fail_{0};
  int total_alerts_{0};
  int total_vehicles_{0};
  int alert_total_informed_entities_{0};
  int alert_total_resolve_success_{0};
  int alert_trip_not_found_{0};
  int alert_empty_selector_{0};
  int alert_stop_not_found_{0};
  int alert_direction_without_route_{0};
  int alert_route_id_not_found_{0};
  int alert_agency_id_not_found_{0};
  int alert_invalid_route_type_{0};
  int unsupported_deleted_{0};
  int unsupported_no_trip_id_{0};
  int no_vehicle_position_{0};
  int vehicle_position_without_position_{0};
  int vehicle_position_without_trip_{0};
  int vehicle_position_trip_without_trip_id_{0};
  int vehicle_position_illegal_stop_{0};
  int vehicle_position_position_not_at_stop_{0};
  int no_trip_update_{0};
  int trip_update_without_trip_{0};
  int trip_resolve_error_{0};
  int unsupported_schedule_relationship_{0};
  date::sys_seconds feed_timestamp_{};
};

statistics gtfsrt_update_msg(timetable const&,
                             rt_timetable&,
                             source_idx_t const,
                             std::string_view tag,
                             transit_realtime::FeedMessage const&,
                             bool use_vehicle_position = false);

statistics gtfsrt_update_buf(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             std::string_view protobuf,
                             transit_realtime::FeedMessage& msg,
                             bool use_vehicle_position = false);

statistics gtfsrt_update_buf(timetable const&,
                             rt_timetable&,
                             source_idx_t const,
                             std::string_view tag,
                             std::string_view protobuf,
                             bool use_vehicle_position = false);

}  // namespace nigiri::rt
