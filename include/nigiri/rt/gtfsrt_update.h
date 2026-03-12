#pragma once

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "nigiri/delay_prediction.h"
#include "nigiri/types.h"

#include "date/date.h"
#include "gtfsrt/gtfs-realtime.pb.h"

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
  int vehicle_position_position_not_at_stop_{0};
  int vehicle_position_without_matching_run_{0};
  int no_trip_update_{0};
  int trip_update_without_trip_{0};
  int trip_resolve_error_{0};
  int unsupported_schedule_relationship_{0};
  date::sys_seconds feed_timestamp_{};
};

enum class algorithm { kSimple, kIntelligent };
enum class hist_trip_mode { kSameDay, kPrevDays };
struct delay_prediction {
  explicit delay_prediction(algorithm const a,
                            hist_trip_mode const m,
                            uint32_t const np,
                            uint32_t const nh,
                            delay_prediction_storage* dps,
                            hist_trip_times_storage* htts,
                            vehicle_trip_matching* vtm)
      : algo{a},
        mode{m},
        number_of_predecessors{np},
        number_of_hist_trips{nh},
        delay_prediction_store{dps},
        hist_trip_time_store{htts},
        vehicle_trip_match{vtm} {}

  explicit delay_prediction() {};

  algorithm const algo = algorithm::kSimple;
  hist_trip_mode const mode = hist_trip_mode::kSameDay;
  uint32_t number_of_predecessors = 1;
  uint32_t number_of_hist_trips = 5;

  delay_prediction_storage* delay_prediction_store = nullptr;
  hist_trip_times_storage* hist_trip_time_store = nullptr;
  vehicle_trip_matching* vehicle_trip_match = nullptr;
};

statistics gtfsrt_update_msg(timetable const&,
                             rt_timetable&,
                             source_idx_t const,
                             std::string_view tag,
                             transit_realtime::FeedMessage const&,
                             delay_prediction* = nullptr);

statistics gtfsrt_update_buf(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             std::string_view protobuf,
                             transit_realtime::FeedMessage& msg,
                             delay_prediction* = nullptr);

statistics gtfsrt_update_buf(timetable const&,
                             rt_timetable&,
                             source_idx_t const,
                             std::string_view tag,
                             std::string_view protobuf,
                             delay_prediction* = nullptr);

}  // namespace nigiri::rt
