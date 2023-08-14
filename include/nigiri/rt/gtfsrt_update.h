#pragma once

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

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
  int unsupported_deleted_{0};
  int unsupported_vehicle_{0};
  int unsupported_alert_{0};
  int unsupported_no_trip_id_{0};
  int no_trip_update_{0};
  int trip_update_without_trip_{0};
  int trip_resolve_error_{0};
  int unsupported_schedule_relationship_{0};
};

statistics gtfsrt_update_msg(timetable const&,
                             rt_timetable&,
                             source_idx_t const,
                             std::string_view tag,
                             transit_realtime::FeedMessage const&);

statistics gtfsrt_update_buf(timetable const&,
                             rt_timetable&,
                             source_idx_t const,
                             std::string_view tag,
                             std::string_view protobuf);

}  // namespace nigiri::rt
